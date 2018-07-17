//! A wrapper for dictionaries in Weld.
#[allow(dead_code)]

extern crate llvm_sys;

use std::ffi::CString;

use ast::Type;
use error::*;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;

use super::LLVM_VECTOR_WIDTH;
use super::CodeGenExt;
use super::FunctionContext;
use super::LlvmGenerator;
use super::intrinsic::Intrinsics;

pub struct Dict {
    pub name: String,
    pub dict_ty: LLVMTypeRef,
    pub key_ty: LLVMTypeRef,
    pub val_ty: LLVMTypeRef,
    context: LLVMContextRef,
    module: LLVMModuleRef,
    new: Option<LLVMValueRef>,
    get: Option<LLVMValueRef>,
    put: Option<LLVMValueRef>,
    key_exists: Option<LLVMValueRef>,
    size: Option<LLVMValueRef>,
    to_vec: Option<LLVMValueRef>,
}

impl CodeGenExt for Dict {
    fn module(&self) -> LLVMModuleRef {
        self.module
    }

    fn context(&self) -> LLVMContextRef {
        self.context
    }
}

impl Dict {
    pub unsafe fn define<T: AsRef<str>>(name: T,
                                key_ty: LLVMTypeRef,
                                val_ty: LLVMTypeRef,
                                context: LLVMContextRef,
                                module: LLVMModuleRef) -> Dict {
        let c_name = CString::new(name.as_ref()).unwrap();
        let mut layout = [LLVMPointerType(LLVMInt8TypeInContext(context), 0)];
        let dict_ty = LLVMStructCreateNamed(context, c_name.as_ptr());
        LLVMStructSetBody(dict_ty, layout.as_mut_ptr(), layout.len() as u32, 0);
        Dict {
            name: c_name.into_string().unwrap(),
            dict_ty: dict_ty,
            key_ty: key_ty,
            val_ty: val_ty,
            context: context,
            module: module,
            new: None,
            get: None,
            put: None,
            key_exists: None,
            size: None,
            to_vec: None,
        }
    }

    pub unsafe fn gen_new(&mut self,
                          builder: LLVMBuilderRef,
                          capacity: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        unimplemented!()
    }

    pub unsafe fn gen_get(&mut self,
                          builder: LLVMBuilderRef,
                          dict: LLVMValueRef,
                          key: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        unimplemented!()
    }

    pub unsafe fn gen_put(&mut self,
                          builder: LLVMBuilderRef,
                          dict: LLVMValueRef,
                          key: LLVMValueRef,
                          val: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        unimplemented!()
    }

    pub unsafe fn gen_key_exists(&mut self,
                          builder: LLVMBuilderRef,
                          dict: LLVMValueRef,
                          key: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        unimplemented!()
    }

    pub unsafe fn gen_size(&mut self,
                          builder: LLVMBuilderRef,
                          dict: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        unimplemented!()
    }

    pub unsafe fn gen_to_vec(&mut self,
                          builder: LLVMBuilderRef,
                          dict: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        unimplemented!()
    }
}
