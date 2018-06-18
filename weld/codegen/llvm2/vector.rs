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

use super::CodeGenExt;
use super::intrinsic::Intrinsics;

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
    size: Option<LLVMValueRef>,
}

impl CodeGenExt for Vector {
    fn module(&self) -> LLVMModuleRef {
        self.module
    }

    fn context(&self) -> LLVMContextRef {
        self.context
    }
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
            size: None,
        }
    }

    pub unsafe fn generate_new(&mut self, builder: LLVMBuilderRef,
                               intrinsics: &Intrinsics,
                               size: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.new.is_none() {
            let int64 = LLVMInt64TypeInContext(self.context);
            let mut arg_tys = [int64];
            let ret_ty = self.vector_ty;

            let name = format!("{}.new", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let size = LLVMGetParam(function, 0);
            debug!("got param...");
            let mut indices = [LLVMConstInt(LLVMInt32TypeInContext(self.context), 1, 1)];
            let elem_size_ptr = LLVMConstGEP(LLVMConstPointerNull(LLVMPointerType(self.elem_ty, 0)),
                                            indices.as_mut_ptr(),
                                            indices.len() as u32);
            debug!("constructed gep");
             let elem_size = LLVMBuildPtrToInt(builder, elem_size_ptr, int64, c_str!("elemSize"));
            debug!("constructed ptrtoint");
            let alloc_size = LLVMBuildMul(builder, elem_size, size, c_str!("size"));
            debug!("constructed mul");
            let run_id = LLVMBuildCall(builder,
                                       intrinsics.get("weld_rt_get_run_id").unwrap(),
                                       [].as_mut_ptr(), 0, c_str!("runId"));
            debug!("constructed call");
            let bytes = LLVMBuildCall(builder,
                                      intrinsics.get("weld_rt_malloc").unwrap(),
                                      [run_id, alloc_size].as_mut_ptr(), 2, c_str!("bytes"));
            debug!("constructed call");
            let elements = LLVMBuildBitCast(builder, bytes, LLVMPointerType(self.elem_ty, 0), c_str!("elements"));
            debug!("constructed bitcast");
            let one = LLVMBuildInsertValue(builder,
                                           LLVMGetUndef(self.vector_ty),
                                           elements, 0, c_str!(""));
            debug!("constructed insertvalue");
            let result = LLVMBuildInsertValue(builder,
                                           one,
                                           size, 1, c_str!(""));
            debug!("constructed insertvalue");
            LLVMBuildRet(builder, result);
            debug!("constructed ret");

            self.new = Some(function);
            LLVMDisposeBuilder(builder);
        }
        
        let mut args = [size];
        return Ok(LLVMBuildCall(builder, self.new.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    pub unsafe fn generate_at(&mut self, builder: LLVMBuilderRef, vector: LLVMValueRef, index: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.at.is_none() {
            let mut arg_tys = [self.vector_ty, LLVMInt64TypeInContext(self.context)];
            let ret_ty = LLVMPointerType(self.elem_ty, 0) ;

            let name = format!("{}.at", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

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

    pub unsafe fn generate_size(&mut self, builder: LLVMBuilderRef, vector: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.size.is_none() {
            let mut arg_tys = [self.vector_ty];
            let ret_ty = LLVMInt64TypeInContext(self.context);

            let name = format!("{}.size", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let vector = LLVMGetParam(function, 0);
            let size = LLVMBuildExtractValue(builder, vector, 1, c_str!(""));
            LLVMBuildRet(builder, size);

            self.size = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [vector];
        Ok(LLVMBuildCall(builder, self.size.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }
}
