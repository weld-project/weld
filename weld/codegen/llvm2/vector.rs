//! Extensions to generate vectors and vector-related builders.
//!
//! This trait specifically provides code generation for:
//!
//! * The `vec[T]` type
//! * The `appender` builder
//! * The `vecmerger` builder

extern crate llvm_sys;

use std::ffi::CString;

use error::*;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;

/// A vector type.
pub struct Vector {
    pub vector_ty: LLVMTypeRef,
    pub name: String,
    pub elem_ty: LLVMTypeRef,
    context: LLVMContextRef,
    module: LLVMModuleRef,
    new: Option<LLVMValueRef>,
    clone: Option<LLVMValueRef>,
    at: Option<LLVMValueRef>,
}

impl Vector {
    pub unsafe fn define<T: AsRef<str>>(name: T,
                                elem_ty: LLVMTypeRef,
                                context: LLVMContextRef,
                                module: LLVMModuleRef) -> Vector {
        let c_name = CString::new(name.as_ref()).unwrap();
        let mut layout = [LLVMPointerType(elem_ty, 0), LLVMInt64TypeInContext(context)];
        let vector = LLVMStructCreateNamed(context, c_name.as_ptr());
        LLVMStructSetBody(vector, layout.as_mut_ptr(), layout.len() as u32, 0);
        Vector {
            name: c_name.into_string().unwrap(),
            context: context,
            module: module,
            vector_ty: vector,
            elem_ty: elem_ty,
            new: None,
            clone: None,
            at: None,
        }
    }

    /// Define a function with the given return type and argument type.
    ///
    /// Returns a reference to the function, a builder used to build the function body, and the
    /// entry basic block of the function. The builder is positioned at the end of the entry basic block.
    unsafe fn define_function<T: AsRef<str>>(&mut self,
                                                ret_ty: LLVMTypeRef,
                                                arg_tys: &mut [LLVMTypeRef],
                                                name: T) -> (LLVMValueRef, LLVMBuilderRef, LLVMBasicBlockRef) {

        let func_ty = LLVMFunctionType(ret_ty, arg_tys.as_mut_ptr(), arg_tys.len() as u32, 0);
        let name = CString::new(format!("{}.{}", self.name, name.as_ref())).unwrap();
        let function = LLVMAddFunction(self.module, name.as_ptr(), func_ty); 
        let builder = LLVMCreateBuilderInContext(self.context);
        let block = LLVMAppendBasicBlockInContext(self.context, function, c_str!("entry"));
        LLVMPositionBuilderAtEnd(builder, block);

        (function, builder, block)
    }

    pub unsafe fn generate_new(&mut self, builder: LLVMBuilderRef, size: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        let mut args = [size];
        if let Some(function) = self.new {
            return Ok(LLVMBuildCall(builder, function, args.as_mut_ptr(), args.len() as u32, c_str!("")))
        }

        // Generate the new() call.
        unimplemented!()
    }

    pub unsafe fn generate_at(&mut self, builder: LLVMBuilderRef, vector: LLVMValueRef, index: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.at.is_none() {
            let mut arg_tys = [self.vector_ty, LLVMInt64TypeInContext(self.context)];
            let ret_ty = self.vector_ty;

            let (function, builder, block) = self.define_function(ret_ty, &mut arg_tys, "at");

            let vector = LLVMGetParam(function, 0);
            let index = LLVMGetParam(function, 1);
            let pointer = LLVMBuildExtractValue(builder, vector, 0, c_str!(""));
            let value_pointer = LLVMBuildGEP(builder, pointer, [index].as_mut_ptr(), 1, c_str!(""));
            LLVMBuildRet(builder, value_pointer);

            self.at = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [vector, index];
        Ok(LLVMBuildCall(builder, self.at.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }
}
