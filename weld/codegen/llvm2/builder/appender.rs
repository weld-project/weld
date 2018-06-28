//! Code generation for the merger builder type.

extern crate llvm_sys;

use std::ffi::CString;

use error::*;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;
// use self::llvm_sys::LLVMTypeKind;

use codegen::llvm2::intrinsic::Intrinsics;
use codegen::llvm2::CodeGenExt;
// use codegen::llvm2::LLVM_VECTOR_WIDTH;

const POINTER_INDEX: u32 = 0;
const SIZE_INDEX: u32 = 1;
const CAPACITY_INDEX: u32 = 2;

pub const DEFAULT_CAPACITY: i64 = 16;

pub struct Appender {
    pub appender_ty: LLVMTypeRef,
    pub elem_ty: LLVMTypeRef,
    pub name: String,
    context: LLVMContextRef,
    module: LLVMModuleRef,
    new: Option<LLVMValueRef>,
}

impl CodeGenExt for Appender {
    fn module(&self) -> LLVMModuleRef {
        self.module
    }

    fn context(&self) -> LLVMContextRef {
        self.context
    }
}

impl Appender {
    pub unsafe fn define<T: AsRef<str>>(name: T,
                                elem_ty: LLVMTypeRef,
                                context: LLVMContextRef,
                                module: LLVMModuleRef) -> Appender {
        let c_name = CString::new(name.as_ref()).unwrap();
        // An appender is struct with a pointer, size, and capacity.
        let mut layout = [
            LLVMPointerType(elem_ty, 0),
            LLVMInt64TypeInContext(context),
            LLVMInt64TypeInContext(context)
        ];
        let appender = LLVMStructCreateNamed(context, c_name.as_ptr());
        LLVMStructSetBody(appender, layout.as_mut_ptr(), layout.len() as u32, 0);
        Appender {
            appender_ty: appender,
            elem_ty: elem_ty,
            name: c_name.into_string().unwrap(),
            context: context,
            module: module,
            new: None,
        }
    }

    /// Returns a pointer to the `index`th element in the appender.
    ///
    /// If the `index` is `None`, thie method returns the base pointer. This method does not
    /// perform any bounds checking.
    unsafe fn gen_index(&mut self,
                        builder: LLVMBuilderRef,
                        appender: LLVMValueRef,
                        index: Option<LLVMValueRef>) -> WeldResult<LLVMValueRef> {
        let pointer = LLVMBuildStructGEP(builder, appender, POINTER_INDEX, c_str!(""));
        if let Some(index) = index { 
            Ok(LLVMBuildGEP(builder, pointer, [index].as_mut_ptr(), 1, c_str!("")))
        } else {
            Ok(pointer)
        }
    }
                        
    /// Generates code for a new appender.
    pub unsafe fn gen_new(&mut self,
                               builder: LLVMBuilderRef,
                               intrinsics: &mut Intrinsics,
                               run: LLVMValueRef,
                               capacity: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.new.is_none() {
            let mut arg_tys = [self.i64_type(), self.run_handle_type()];
            let ret_ty = self.appender_ty;

            let name = format!("{}.new", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let capacity = LLVMGetParam(function, 0);
            let run = LLVMGetParam(function, 1);
            let elem_size = self.size_of(self.elem_ty);
            let alloc_size = LLVMBuildMul(builder, elem_size, capacity, c_str!("capacity"));
            let bytes = intrinsics.call_weld_run_malloc(builder, run, alloc_size, Some(c_str!("bytes")));
            let elements = LLVMBuildBitCast(builder, bytes, LLVMPointerType(self.elem_ty, 0), c_str!("elements"));

            let mut result = LLVMGetUndef(self.appender_ty);
            result = LLVMBuildInsertValue(builder, result, elements, POINTER_INDEX, c_str!(""));
            result = LLVMBuildInsertValue(builder, result, self.i64(0), SIZE_INDEX, c_str!(""));
            result = LLVMBuildInsertValue(builder, result, capacity, CAPACITY_INDEX, c_str!(""));
            LLVMBuildRet(builder, result);

            self.new = Some(function);
            LLVMDisposeBuilder(builder);
        }
        let mut args = [capacity, run];
        Ok(LLVMBuildCall(builder, self.new.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }
}
