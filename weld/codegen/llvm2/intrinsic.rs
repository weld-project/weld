//! Defines intrinsics in the LLVM IR.
//!
//! An intrinsic is any function that is defined but not generated in the current module. This
//! module provides utilities for calling some default intrinsics, as well as a utility for adding
//! an calling new ones.
//!
//! For documentation on the default intrinsics, see `weld::runtime::strt`.

extern crate libc;
extern crate llvm_sys;
extern crate fnv;

use fnv::FnvHashMap;

use libc::c_char;

use ast::ScalarKind;
use error::*;

use std::ffi::CString;

use super::llvm_exts::*;

use super::CodeGenExt;
use super::LLVM_VECTOR_WIDTH;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;

/// Intrinsics defined in the code generator.
///
/// An intrinsic is any function that appears without a definition in the generated module. Code
/// generators must ensure that instrinics are properly linked upon compilation.
pub struct Intrinsics {
    context: LLVMContextRef,
    module: LLVMModuleRef,
    intrinsics: FnvHashMap<String, LLVMValueRef>, 
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
    pub unsafe fn defaults(context: LLVMContextRef, module: LLVMModuleRef) -> Intrinsics {
        let mut intrinsics = Intrinsics {
            context: context,
            module: module,
            intrinsics: FnvHashMap::default(),
        };

        intrinsics.populate_defaults();
        intrinsics
    }

    /// Returns a string name for a numeric type's LLVM intrinsic.
    pub fn llvm_numeric<T: AsRef<str>>(name: T, kind: ScalarKind, simd: bool) -> String {
        use ast::ScalarKind::*;
        let mut result = format!("llvm.{}.", name.as_ref());
        if simd {
            result.push_str(&format!("v{}", LLVM_VECTOR_WIDTH));
        }

        result.push_str(match kind {
            Bool => "i1",
            I8  => "i32",
            I16 => "i16",
            I32 => "i32",
            I64 => "i64",
            U8  => "i32",
            U16 => "i16",
            U32 => "i32",
            U64 => "i64",
            F32 => "f32",
            F64 => "f64",
        });
        result
    }

    /// Get the intrinsic function with the given name.
    pub fn get<T: AsRef<str>>(&self, key: T) -> Option<LLVMValueRef> {
        return self.intrinsics.get(key.as_ref()).map(|r| *r)
    }

    /// Add a new intrinsic function with the given name, return type, and argument types.
    ///
    /// Returns true if the function was added or false if it was already registered.
    pub unsafe fn add<T: AsRef<str>>(&mut self, name: T, ret_ty: LLVMTypeRef, arg_tys: &mut [LLVMTypeRef]) -> bool {
        if !self.intrinsics.contains_key(name.as_ref()) {
            let name = CString::new(name.as_ref()).unwrap();
            let fn_type = LLVMFunctionType(ret_ty, arg_tys.as_mut_ptr(), arg_tys.len() as u32, 0);
            let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
            self.intrinsics.insert(name.into_string().unwrap(), function);
            true
        } else {
            false
        }
    }

    /// Generate code to call an intrinsic function with the given name and arguments.
    ///
    /// If the intrinsic is not defined, this function throws an error.
    pub unsafe fn call<T: AsRef<str>>(&mut self,
                                      builder: LLVMBuilderRef,
                                      name: T,
                                      args: &mut [LLVMValueRef]) -> WeldResult<LLVMValueRef> {
        if let Some(func) = self.intrinsics.get(name.as_ref()) {
            if args.len() != LLVMCountParams(*func) as usize {
                unreachable!()
            }
            Ok(LLVMBuildCall(builder, *func, args.as_mut_ptr(), args.len() as u32, c_str!("")))
        } else {
            unreachable!()
        }
    }

    /// Convinience wrapper for calling the `weld_run_init` intrinsic.
    pub unsafe fn call_weld_run_init(&mut self,
                                          builder: LLVMBuilderRef,
                                          nworkers: LLVMValueRef,
                                          memlimit: LLVMValueRef,
                                          name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [nworkers, memlimit];
        LLVMBuildCall(builder,
                      self.get("weld_runst_init").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Convinience wrapper for calling the `weld_run_get_result` intrinsic.
    pub unsafe fn call_weld_run_get_result(&mut self,
                                          builder: LLVMBuilderRef,
                                          run: LLVMValueRef,
                                          name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run];
        LLVMBuildCall(builder,
                      self.get("weld_runst_get_result").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Convinience wrapper for calling the `weld_run_set_result` intrinsic.
    pub unsafe fn call_weld_run_set_result(&mut self,
                                          builder: LLVMBuilderRef,
                                          run: LLVMValueRef,
                                          pointer: LLVMValueRef,
                                          name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run, pointer];
        LLVMBuildCall(builder,
                      self.get("weld_runst_set_result").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Convinience wrapper for calling the `weld_run_malloc` intrinsic.
    pub unsafe fn call_weld_run_malloc(&mut self,
                                      builder: LLVMBuilderRef,
                                      run: LLVMValueRef,
                                      size: LLVMValueRef,
                                      name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run, size];
        LLVMBuildCall(builder,
                      self.get("weld_runst_malloc").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Convinience wrapper for calling the `weld_run_remalloc` intrinsic.
    pub unsafe fn call_weld_run_realloc(&mut self,
                                      builder: LLVMBuilderRef,
                                      run: LLVMValueRef,
                                      pointer: LLVMValueRef,
                                      size: LLVMValueRef,
                                      name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run, pointer, size];
        LLVMBuildCall(builder,
                      self.get("weld_runst_realloc").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Convinience wrapper for calling the `weld_run_free` intrinsic.
    pub unsafe fn call_weld_run_free(&mut self,
                                      builder: LLVMBuilderRef,
                                      run: LLVMValueRef,
                                      pointer: LLVMValueRef) -> LLVMValueRef {
        let mut args = [run, pointer];
        LLVMBuildCall(builder,
                      self.get("weld_runst_free").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, c_str!(""))
    }

    /// Convinience wrapper for calling the `weld_run_get_errno` intrinsic.
    pub unsafe fn call_weld_run_get_errno(&mut self,
                                      builder: LLVMBuilderRef,
                                      run: LLVMValueRef,
                                      name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run];
        LLVMBuildCall(builder,
                      self.get("weld_runst_get_errno").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Convinience wrapper for calling the `weld_run_set_errno` intrinsic.
    pub unsafe fn call_weld_run_set_errno(&mut self,
                                      builder: LLVMBuilderRef,
                                      run: LLVMValueRef,
                                      errno: LLVMValueRef,
                                      name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run, errno];
        LLVMBuildCall(builder,
                      self.get("weld_runst_set_errno").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Convinience wrapper for calling the `weld_run_print` intrinsic.
    pub unsafe fn call_weld_run_print(&mut self,
                                      builder: LLVMBuilderRef,
                                      run: LLVMValueRef,
                                      string: LLVMValueRef) -> LLVMValueRef {
        let mut args = [run, string];
        LLVMBuildCall(builder,
                      self.get("weld_runst_print").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, c_str!(""))
    }

    /// Convinience wrapper for calling `memcpy`.
    ///
    /// This assumes the `memcpy` is non-volatile and uses an default alignment value of 8.
    pub unsafe fn call_memcpy(&mut self,
                                      builder: LLVMBuilderRef,
                                      dst: LLVMValueRef,
                                      src: LLVMValueRef,
                                      size: LLVMValueRef) -> LLVMValueRef {
        let mut args = [dst, src, size, self.i32(8), self.i1(false)];
        LLVMBuildCall(builder,
                      self.get("llvm.memcpy.p0i8.p0i8.i64").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, c_str!(""))
    }

    /// Convinience wrapper for calling `memset` with 0 bytes value.
    ///
    /// This assumes the `memset` is non-volatile.
    pub unsafe fn call_memset_zero(&mut self,
                                      builder: LLVMBuilderRef,
                                      dst: LLVMValueRef,
                                      size: LLVMValueRef) -> LLVMValueRef {
        let mut args = [dst, self.i8(0), size, self.i32(8), self.i1(false)];
        LLVMBuildCall(builder,
                      self.get("llvm.memset.p0i8.i64").unwrap(),
                      args.as_mut_ptr(), args.len() as u32, c_str!(""))
    }
}

/// Private methods.
impl Intrinsics {
    /// Populate the default intrinsics.
    ///
    /// By default, the code generator adds the Weld Run API (functions prefixed with `weld_run`)
    /// and a few other utility functions, such as `memcpy`.
    unsafe fn populate_defaults(&mut self) {
        use super::llvm_exts::LLVMExtAttribute::*;

        let int8p = LLVMPointerType(self.i8_type(), 0);

        // Defines the default intrinsics used by the Weld runtime.
        let mut params = vec![self.i32_type(), self.i64_type()];
        let name = CString::new("weld_runst_init").unwrap();
        let fn_type = LLVMFunctionType(self.run_handle_type(), params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        self.intrinsics.insert(name.into_string().unwrap(), function);

        let mut params = vec![self.run_handle_type()];
        let name = CString::new("weld_runst_get_result").unwrap();
        let fn_type = LLVMFunctionType(int8p, params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        LLVMExtAddAttrsOnFunction(self.context, function, &[NoUnwind]);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 0);
        self.intrinsics.insert(name.into_string().unwrap(), function);

        let mut params = vec![self.run_handle_type(), int8p];
        let name = CString::new("weld_runst_set_result").unwrap();
        let fn_type = LLVMFunctionType(self.void_type(), params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        LLVMExtAddAttrsOnFunction(self.context, function, &[NoUnwind]);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull], 0);
        self.intrinsics.insert(name.into_string().unwrap(), function);

        let mut params = vec![self.run_handle_type(), self.i64_type()];
        let name = CString::new("weld_runst_malloc").unwrap();
        let fn_type = LLVMFunctionType(int8p, params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull], 0);
        LLVMExtAddAttrsOnReturn(self.context, function, &[NoAlias]);
        self.intrinsics.insert(name.into_string().unwrap(), function);

        let mut params = vec![self.run_handle_type(), int8p, self.i64_type()];
        let name = CString::new("weld_runst_realloc").unwrap();
        let fn_type = LLVMFunctionType(int8p, params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull], 0);
        LLVMExtAddAttrsOnReturn(self.context, function, &[NoAlias]);
        self.intrinsics.insert(name.into_string().unwrap(), function);

        let mut params = vec![self.run_handle_type(), int8p];
        let name = CString::new("weld_runst_free").unwrap();
        let fn_type = LLVMFunctionType(self.void_type(), params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull], 0);
        self.intrinsics.insert(name.into_string().unwrap(), function);

        let mut params = vec![self.run_handle_type()];
        let name = CString::new("weld_runst_get_errno").unwrap();
        let fn_type = LLVMFunctionType(self.i64_type(), params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        LLVMExtAddAttrsOnFunction(self.context, function, &[NoUnwind]);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 0);
        self.intrinsics.insert(name.into_string().unwrap(), function);

        let mut params = vec![self.run_handle_type(), self.i64_type()];
        let name = CString::new("weld_runst_set_errno").unwrap();
        let fn_type = LLVMFunctionType(self.void_type(), params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        LLVMExtAddAttrsOnFunction(self.context, function, &[NoReturn]);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull], 0);
        self.intrinsics.insert(name.into_string().unwrap(), function);

        let mut params = vec![self.run_handle_type(), int8p];
        let name = CString::new("weld_runst_print").unwrap();
        let fn_type = LLVMFunctionType(self.void_type(), params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 0);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 1);
        self.intrinsics.insert(name.into_string().unwrap(), function);

        let mut params = vec![int8p, int8p, self.i64_type(), self.i32_type(), self.i1_type()];
        let name = CString::new("llvm.memcpy.p0i8.p0i8.i64").unwrap();
        let fn_type = LLVMFunctionType(self.void_type(), params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        // LLVM sets attributes on `memcpy` automatically.
        self.intrinsics.insert(name.into_string().unwrap(), function);

        let mut params = vec![int8p, self.i8_type(), self.i64_type(), self.i32_type(), self.i1_type()];
        let name = CString::new("llvm.memset.p0i8.i64").unwrap();
        let fn_type = LLVMFunctionType(self.void_type(), params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        // LLVM sets attributes on `memset` automatically.
        self.intrinsics.insert(name.into_string().unwrap(), function);
    }
}
