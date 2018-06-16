//! Extensions to generate vectors and vector-related builders.
//!
//! This trait specifically provides code generation for:
//!
//! * The `vec[T]` type
//! * The `appender` builder
//! * The `vecmerger` builder

use error::*;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;

/// A vector type.
pub struct Vector {
    context: LLVMContextRef,
    module: LLVMModuleRef,
    vector_ty: LLVMTypeRef,
    elem_ty: LLVMTypeRef,
    new: Option<LLVMValueRef>,
    clone: Option<LLVMValueRef>,
    at: Option<LLVMValueRef>,
}

impl Vector {
    fn define(name: CString, ty: LLVMTypeRef, context: LLVMContextRef, module: LLVMModuleRef) -> Vector {
        let mut layout = [LLVMPointerType(ty, 0), LLVMIntIn64TypeInContext(context)];
        let vector = LLVMStructCreateNamed(self.context, name.as_ptr());
        LLVMStructSetBody(vector, layout.as_mut_ptr(), layout.len() as u32, 0);
        Vector {
            context: context,
            module: module,
            vector_ty: vector,
            elem_ty: ty,
            new: None,
            clone: None,
            at: None,
        }
    }

    unsafe fn generate_new(&mut self, builder: LLVMBuilderRef, size: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        let mut args = [size];
        if let Some(function) = self.new {
            return Ok(LLVMBuildCall(builder, function, args.as_mut_ptr(), args.len(), c_str!("")))
        }

        // Generate the new() call.
        unimplemented!()
    }

    unsafe fn generate_at(&mut self, builder: LLVMBuilderRef, index: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        let mut args = [index];
        if let Some(function) = self.at {
            return Ok(LLVMBuildCall(builder, function, args.as_mut_ptr(), args.len(), c_str!("")))
        }

        // TODO prefix this with a unique name...
        let mut arg_tys = [LLVMInt64TypeInContext(self.context)];
        let func_ty = LLVMFunctionType(self.ty, arg_tys.as_mut_ptr(), arg_tys.len() as u32, 0);
        let name = CString::new("at").unwrap();
        let function = LLVMAddFunction(self.module, name.as_ptr(), func_ty); 

        let builder = LLVMCreateBuilderInContext(self.context);
        let block = LLVMAppendBasicBlockInContext(self.context, function, c_str!("entry"));
        LLVMPositionBuilderAtEnd(builder, block);

        // Build the at function
    }
}
