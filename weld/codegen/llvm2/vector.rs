//! Extensions to generate vectors and vector-related builders.
//!
//! This trait specifically provides code generation for:
//!
//! * The `vec[T]` type
//! * The `appender` builder
//! * The `vecmerger` builder

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

/// Extensions for generating methods on vectors.
///
/// This provides convinience wrappers for calling methods on vectors.
pub trait VectorExt {
    unsafe fn gen_new(&mut self,
               ctx: &mut FunctionContext,
               vector_type: &Type,
               size: LLVMValueRef) -> WeldResult<LLVMValueRef>;
    unsafe fn gen_at(&mut self,
              ctx: &mut FunctionContext,
              vector_type: &Type,
              vec: LLVMValueRef,
              index: LLVMValueRef) -> WeldResult<LLVMValueRef>;
    unsafe fn gen_vat(&mut self,
               ctx: &mut FunctionContext,
               vector_type: &Type,
               vec: LLVMValueRef,
               index: LLVMValueRef) -> WeldResult<LLVMValueRef>;
    unsafe fn gen_size(&mut self,
                ctx: &mut FunctionContext,
                vector_type: &Type,
                vec: LLVMValueRef) -> WeldResult<LLVMValueRef>;
}

impl VectorExt for LlvmGenerator {
    unsafe fn gen_new(&mut self,
               ctx: &mut FunctionContext,
               vector_type: &Type,
               size: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if let Type::Vector(ref elem_type) = *vector_type {
            let mut methods = self.vectors.get_mut(elem_type).unwrap();
            methods.gen_new(ctx.builder, &mut self.intrinsics, ctx.get_run(), size)
        } else {
            unreachable!()
        }
    }

    unsafe fn gen_at(&mut self,
              ctx: &mut FunctionContext,
              vector_type: &Type,
              vector: LLVMValueRef,
              index: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if let Type::Vector(ref elem_type) = *vector_type {
            let mut methods = self.vectors.get_mut(elem_type).unwrap();
            methods.gen_at(ctx.builder, vector, index)
        } else {
            unreachable!()
        }
    }

    unsafe fn gen_vat(&mut self,
               ctx: &mut FunctionContext,
               vector_type: &Type,
               vector: LLVMValueRef,
               index: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if let Type::Vector(ref elem_type) = *vector_type {
            let mut methods = self.vectors.get_mut(elem_type).unwrap();
            methods.gen_vat(ctx.builder, vector, index)
        } else {
            unreachable!()
        }
    }

    unsafe fn gen_size(&mut self,
                ctx: &mut FunctionContext,
                vector_type: &Type,
                vector: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if let Type::Vector(ref elem_type) = *vector_type {
            let mut methods = self.vectors.get_mut(elem_type).unwrap();
            methods.gen_size(ctx.builder, vector)
        } else {
            unreachable!()
        }
    }
}

/// A vector type and its associated methods.
pub struct Vector {
    pub vector_ty: LLVMTypeRef,
    pub name: String,
    pub elem_ty: LLVMTypeRef,
    context: LLVMContextRef,
    module: LLVMModuleRef,
    new: Option<LLVMValueRef>,
    at: Option<LLVMValueRef>,
    vat: Option<LLVMValueRef>,
    size: Option<LLVMValueRef>,
    slice: Option<LLVMValueRef>,
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
            at: None,
            vat: None,
            size: None,
            slice: None,
        }
    }

    pub unsafe fn gen_new(&mut self,
                               builder: LLVMBuilderRef,
                               intrinsics: &mut Intrinsics,
                               run: LLVMValueRef,
                               size: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.new.is_none() {
            let mut arg_tys = [self.i64_type(), self.run_handle_type()];
            let ret_ty = self.vector_ty;

            let name = format!("{}.new", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let size = LLVMGetParam(function, 0);
            let elem_size = self.size_of(self.elem_ty);
            let alloc_size = LLVMBuildMul(builder, elem_size, size, c_str!("size"));
            let run = LLVMGetParam(function, 1);
            let bytes = intrinsics.call_weld_run_malloc(builder, run, alloc_size, Some(c_str!("bytes")));
            let elements = LLVMBuildBitCast(builder, bytes, LLVMPointerType(self.elem_ty, 0), c_str!("elements"));
            let one = LLVMBuildInsertValue(builder,
                                           LLVMGetUndef(self.vector_ty),
                                           elements, 0, c_str!(""));
            let result = LLVMBuildInsertValue(builder,
                                           one,
                                           size, 1, c_str!(""));
            LLVMBuildRet(builder, result);

            self.new = Some(function);
            LLVMDisposeBuilder(builder);
        }
        
        let mut args = [size, run];
        return Ok(LLVMBuildCall(builder, self.new.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    pub unsafe fn gen_at(&mut self,
                              builder: LLVMBuilderRef,
                              vector: LLVMValueRef,
                              index: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.at.is_none() {
            let mut arg_tys = [self.vector_ty, self.i64_type()];
            let ret_ty = LLVMPointerType(self.elem_ty, 0);

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

    pub unsafe fn gen_slice(&mut self,
                            builder: LLVMBuilderRef,
                            vector: LLVMValueRef,
                            index: LLVMValueRef,
                            size: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        use self::llvm_sys::LLVMIntPredicate::LLVMIntUGT;
        if self.slice.is_none() {
            let mut arg_tys = [self.vector_ty, self.i64_type(), self.i64_type()];
            let ret_ty = self.vector_ty;

            let name = format!("{}.slice", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let vector = LLVMGetParam(function, 0);
            let index = LLVMGetParam(function, 1);
            let size = LLVMGetParam(function, 2);

            // Compute the size of the array. We use the remaining size if the new size does not
            // accomodate the vector starting at the given index.
            let cur_size = LLVMBuildExtractValue(builder, vector, 1, c_str!(""));
            let remaining = LLVMBuildSub(builder, cur_size, index, c_str!(""));
            let size_cmp = LLVMBuildICmp(builder, LLVMIntUGT, size, remaining, c_str!(""));
            let new_size = LLVMBuildSelect(builder, size_cmp, remaining, size, c_str!(""));

            let elements = LLVMBuildExtractValue(builder, vector, 0, c_str!(""));
            let new_elements = LLVMBuildGEP(builder, elements, [index].as_mut_ptr(), 1, c_str!(""));

            let mut result = LLVMGetUndef(self.vector_ty);
            result = LLVMBuildInsertValue(builder, result, new_elements, 0, c_str!(""));
            result = LLVMBuildInsertValue(builder, result, new_size, 1, c_str!(""));
            LLVMBuildRet(builder, result);

            self.slice = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [vector, index, size];
        Ok(LLVMBuildCall(builder, self.slice.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    pub unsafe fn gen_vat(&mut self,
                              builder: LLVMBuilderRef,
                              vector: LLVMValueRef,
                              index: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.vat.is_none() {
            let mut arg_tys = [self.vector_ty, self.i64_type()];
            let ret_ty = LLVMPointerType(LLVMVectorType(self.elem_ty, LLVM_VECTOR_WIDTH), 0);

            let name = format!("{}.vat", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let vector = LLVMGetParam(function, 0);
            let index = LLVMGetParam(function, 1);
            let pointer = LLVMBuildExtractValue(builder, vector, 0, c_str!(""));
            let value_pointer = LLVMBuildGEP(builder, pointer, [index].as_mut_ptr(), 1, c_str!(""));
            let value_pointer = LLVMBuildBitCast(builder, value_pointer, ret_ty, c_str!(""));
            LLVMBuildRet(builder, value_pointer);

            self.vat = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [vector, index];
        Ok(LLVMBuildCall(builder, self.vat.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    pub unsafe fn gen_size(&mut self,
                                builder: LLVMBuilderRef,
                                vector: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.size.is_none() {
            let mut arg_tys = [self.vector_ty];
            let ret_ty = self.i64_type();

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
