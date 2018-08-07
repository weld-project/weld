//! Implements a wrapper for the `groupmerger` builder.
//!
//! This module resembles `llvm2::dict`, in that most of the code here calls out into an external
//! implementation, and the data structures here facilitate making those calls.

extern crate lazy_static;
extern crate llvm_sys;
extern crate libc;

use libc::{c_char};

use std::ffi::CString;

use error::*;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;

use codegen::llvm2::llvm_exts::LLVMExtAttribute::*;
use codegen::llvm2::llvm_exts::*;

use super::CodeGenExt;

/// Default capacity in number of keys.
pub const DEFAULT_CAPACITY: i64 = 16;

pub struct GroupMerger {
    pub name: String,
    pub groupmerger_ty: LLVMTypeRef,
    pub key_ty: LLVMTypeRef,
    pub val_ty: LLVMTypeRef,
    pub key_comparator: LLVMValueRef,
    context: LLVMContextRef,
    module: LLVMModuleRef,
    new: Option<LLVMValueRef>,
    merge: Option<LLVMValueRef>,
    result: Option<LLVMValueRef>,
}

impl CodeGenExt for GroupMerger {
    fn module(&self) -> LLVMModuleRef {
        self.module
    }

    fn context(&self) -> LLVMContextRef {
        self.context
    }
}

pub struct Intrinsics {
    new: Option<LLVMValueRef>,
    merge: Option<LLVMValueRef>,
    result: Option<LLVMValueRef>,
    context: LLVMContextRef,
    module: LLVMModuleRef,
}

impl CodeGenExt for Intrinsics {
    fn context(&self) -> LLVMContextRef {
        self.context
    }

    fn module(&self) -> LLVMModuleRef {
        self.module
    }
}

impl Intrinsics {
    pub unsafe fn new(context: LLVMContextRef, module: LLVMModuleRef) -> Intrinsics {
        let mut intrinsics = Intrinsics {
            new: None,
            merge: None,
            result: None,
            context: context,
            module: module,
        };
        intrinsics.populate();
        intrinsics
    }

    /// Generate a call to the `new` intrinsic.
    pub unsafe fn call_new(&self, builder: LLVMBuilderRef,
                           run: LLVMValueRef,
                           key_size: LLVMValueRef,
                           val_size: LLVMValueRef,
                           comparator: LLVMValueRef,
                           capacity: LLVMValueRef,
                           name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run, key_size, val_size, comparator, capacity];
        LLVMBuildCall(builder, self.new.unwrap(), args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Generate a call to the `merge` intrinsic.
    pub unsafe fn call_merge(&self, builder: LLVMBuilderRef,
                           run: LLVMValueRef,
                           groupmerger: LLVMValueRef,
                           key: LLVMValueRef,
                           hash: LLVMValueRef,
                           value: LLVMValueRef,
                           name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run, groupmerger, key, hash, value];
        LLVMBuildCall(builder, self.merge.unwrap(), args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Generate a call to the `result` intrinsic.
    pub unsafe fn call_result(&self,
                              builder: LLVMBuilderRef,
                              run: LLVMValueRef,
                              groupmerger: LLVMValueRef,
                              name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run, groupmerger];
        LLVMBuildCall(builder, self.result.unwrap(), args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Add a function to the module.
    unsafe fn declare(&mut self,
                      name: &str,
                      params: &mut Vec<LLVMTypeRef>,
                      ret: LLVMTypeRef) -> LLVMValueRef {
        let name = CString::new(name).unwrap();
        let fn_type = LLVMFunctionType(ret, params.as_mut_ptr(), params.len() as u32, 0);
        LLVMAddFunction(self.module, name.as_ptr(), fn_type)
    }

    /// Populate `self` with the groupmerger intrinsics.
    ///
    /// These intrinsic definitions must be consistent with the C++ equivalent, so this function
    /// should be modified with care.
    unsafe fn populate(&mut self) {
        // Common types.
        let gm_type = LLVMPointerType(self.i8_type(), 0);
        let hash_type = self.i32_type();

        let ref mut params = vec![
            self.run_handle_type(),     // run
            self.i32_type(),            // key size 
            self.i32_type(),            // val size
            self.opaque_cmp_type(),     // key comparator function
            self.i64_type()             // initial capacity (power of 2)
        ];
        let ret = gm_type;
        let function = self.declare("weld_st_gb_new", params, ret);

        LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, NonNull], 0);
        LLVMExtAddAttrsOnReturn(self.context, function, &[NoAlias]);
        self.new = Some(function);

        let ref mut params = vec![
            self.run_handle_type(),     // run
            gm_type,                    // groupmerger
            self.void_pointer_type(),   // key
            hash_type,                  // hash
            self.void_pointer_type(),   // value
        ];
        let ret = self.void_type();
        let function = self.declare("weld_st_gb_merge", params, ret);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 0);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull], 1);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 2);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 4);
        self.merge = Some(function);

        let ref mut params = vec![
            self.run_handle_type(),   // run
            gm_type                   // groupmerger
        ];
        let ret = self.void_pointer_type();
        let function = self.declare("weld_st_gb_result", params, ret);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 0);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 1);
        self.result = Some(function);
    }
}

impl GroupMerger {
    pub unsafe fn define<T: AsRef<str>>(name: T,
                                key_ty: LLVMTypeRef,
                                val_ty: LLVMTypeRef,
                                key_comparator: LLVMValueRef,
                                context: LLVMContextRef,
                                module: LLVMModuleRef) -> GroupMerger {
        let c_name = CString::new(name.as_ref()).unwrap();
        let mut layout = [LLVMInt8TypeInContext(context)];
        let dummy_struct_ty = LLVMStructCreateNamed(context, c_name.as_ptr());
        LLVMStructSetBody(dummy_struct_ty, layout.as_mut_ptr(), layout.len() as u32, 0);

        // A groupmerger is just an opaque pointer with a name.
        let groupmerger_ty = LLVMPointerType(dummy_struct_ty, 0);
        let name = c_name.into_string().unwrap();

        GroupMerger {
            name: name,
            groupmerger_ty: groupmerger_ty,
            key_ty: key_ty,
            val_ty: val_ty,
            key_comparator: key_comparator,
            context: context,
            module: module,
            new: None,
            merge: None,
            result: None,
        }
    }

    /// Returns the type of a hash code.
    pub unsafe fn hash_type(&self) -> LLVMTypeRef {
        self.i32_type()
    }

    /// Generates the `new` method.
    pub unsafe fn gen_new(&mut self,
                          builder: LLVMBuilderRef,
                          intrinsics: &Intrinsics,
                          run: LLVMValueRef,
                          capacity: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.new.is_none() {
            let mut arg_tys = [self.i64_type(), self.run_handle_type()];
            let ret_ty = self.groupmerger_ty;

            let name = format!("{}.new", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let capacity = LLVMGetParam(function, 0);
            let run = LLVMGetParam(function, 1);
            let key_size = LLVMConstTruncOrBitCast(self.size_of(self.key_ty), self.i32_type());
            let val_size = LLVMConstTruncOrBitCast(self.size_of(self.val_ty), self.i32_type());
            let result = intrinsics.call_new(builder, run, key_size, val_size, self.key_comparator, capacity, None);

            // Wrap the pointer in the struct.
            let result = LLVMBuildBitCast(builder, result, self.groupmerger_ty, c_str!(""));

            LLVMBuildRet(builder, result);

            self.new = Some(function);
            LLVMDisposeBuilder(builder);
        }
        let mut args = [capacity, run];
        return Ok(LLVMBuildCall(builder, self.new.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    pub unsafe fn gen_merge(&mut self,
                          builder: LLVMBuilderRef,
                          intrinsics: &Intrinsics,
                          run: LLVMValueRef,
                          groupmerger: LLVMValueRef,
                          key: LLVMValueRef,
                          hash: LLVMValueRef,
                          value: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.merge.is_none() {
            let mut arg_tys = [
                self.groupmerger_ty,
                LLVMPointerType(self.key_ty, 0),
                self.hash_type(),
                LLVMPointerType(self.val_ty, 0),
                self.run_handle_type()
            ];
            let ret_ty = self.void_type();

            let name = format!("{}.merge", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let groupmerger = LLVMGetParam(function, 0);
            let key = LLVMGetParam(function, 1);
            let hash = LLVMGetParam(function, 2);
            let value = LLVMGetParam(function, 3);
            let run = LLVMGetParam(function, 4);

            let groupmerger = LLVMBuildBitCast(builder, groupmerger, self.void_pointer_type(), c_str!(""));
            let key = LLVMBuildBitCast(builder, key, self.void_pointer_type(), c_str!(""));
            let value = LLVMBuildBitCast(builder, value, self.void_pointer_type(), c_str!(""));
            let _ = intrinsics.call_merge(builder, run, groupmerger, key, hash, value, None);

            LLVMBuildRetVoid(builder);

            self.merge = Some(function);
            LLVMDisposeBuilder(builder);
        }
        let mut args = [groupmerger, key, hash, value, run];
        return Ok(LLVMBuildCall(builder, self.merge.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    pub unsafe fn gen_result(&mut self,
                          builder: LLVMBuilderRef,
                          intrinsics: &Intrinsics,
                          run: LLVMValueRef,
                          groupmerger: LLVMValueRef,
                          result_ty: LLVMTypeRef) -> WeldResult<LLVMValueRef> {
        if self.result.is_none() {
            let mut arg_tys = [self.groupmerger_ty, self.run_handle_type()];
            let ret_ty = result_ty;

            let name = format!("{}.result", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let groupmerger = LLVMGetParam(function, 0);
            let run = LLVMGetParam(function, 1);

            let groupmerger = LLVMBuildBitCast(builder, groupmerger, self.void_pointer_type(), c_str!(""));
            let result = intrinsics.call_result(builder, run, groupmerger, None);
            let result = LLVMBuildBitCast(builder, result, result_ty, c_str!(""));

            LLVMBuildRet(builder, result);

            self.result = Some(function);
            LLVMDisposeBuilder(builder);
        }
        let mut args = [groupmerger, run];
        return Ok(LLVMBuildCall(builder, self.result.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }
}
