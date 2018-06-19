//! Code generation for the merger builder type.

extern crate llvm_sys;

use std::ffi::CString;

use ast::BinOpKind;
use ast::ScalarKind;
use error::*;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;
use self::llvm_sys::LLVMTypeKind;

use super::CodeGenExt;
use super::intrinsic::Intrinsics;
use super::numeric::gen_binop;
use super::LLVM_VECTOR_WIDTH;

const SCALAR_INDEX: u32 = 0;
const VECTOR_INDEX: u32 = 1;

/// A potential builder API.
pub trait Builder {
    fn new(&mut self) -> LLVMValueRef;
    fn merge(&mut self, value: LLVMValueRef) -> LLVMValueRef;
    fn scatter(&mut self) -> LLVMValueRef;
    fn gather(&mut self, values: &[LLVMValueRef]) -> LLVMValueRef;
    fn result(&mut self) -> LLVMValueRef;
}

/// The merger type.
pub struct Merger {
    pub merger_ty: LLVMTypeRef,
    pub name: String,
    pub elem_ty: LLVMTypeRef,
    pub scalar_kind: ScalarKind,
    pub op: BinOpKind,
    context: LLVMContextRef,
    module: LLVMModuleRef,
    new: Option<LLVMValueRef>,
    merge: Option<LLVMValueRef>,
    vmerge: Option<LLVMValueRef>,
}

impl CodeGenExt for Merger {
    fn module(&self) -> LLVMModuleRef {
        self.module
    }

    fn context(&self) -> LLVMContextRef {
        self.context
    }
}

impl Merger {
    pub unsafe fn define<T: AsRef<str>>(name: T,
                                op: BinOpKind,
                                elem_ty: LLVMTypeRef,
                                scalar_kind: ScalarKind,
                                context: LLVMContextRef,
                                module: LLVMModuleRef) -> Merger {
        let c_name = CString::new(name.as_ref()).unwrap();
        let mut layout = [elem_ty, LLVMVectorType(elem_ty, LLVM_VECTOR_WIDTH)];
        let merger = LLVMStructCreateNamed(context, c_name.as_ptr());
        LLVMStructSetBody(merger, layout.as_mut_ptr(), layout.len() as u32, 0);
        Merger {
            name: c_name.into_string().unwrap(),
            op: op,
            merger_ty: merger,
            elem_ty: elem_ty,
            scalar_kind: scalar_kind,
            context: context,
            module: module,
            new: None,
            merge: None,
            vmerge: None,
        }
    }

    pub unsafe fn generate_new(&mut self,
                               builder: LLVMBuilderRef,
                               init: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.new.is_none() {
            let ret_ty = self.merger_ty;
            let mut arg_tys = [self.elem_ty];
            let name = format!("{}.new", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let identity = self.binop_identity(self.op, self.elem_ty, self.scalar_kind.is_signed())?;
            let mut vector_elems = [identity; LLVM_VECTOR_WIDTH as usize];
            let vector_identity = LLVMConstVector(vector_elems.as_mut_ptr(), vector_elems.len() as u32);

            let one = LLVMBuildInsertValue(builder,
                                           LLVMGetUndef(self.merger_ty),
                                           LLVMGetParam(function, 0),
                                           SCALAR_INDEX, c_str!(""));
            let result = LLVMBuildInsertValue(builder, one, vector_identity, VECTOR_INDEX, c_str!(""));

            LLVMBuildRet(builder, result);

            self.new = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [init];
        return Ok(LLVMBuildCall(builder, self.new.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    pub unsafe fn generate_merge(&mut self,
                                 llvm_builder: LLVMBuilderRef,
                                 builder: LLVMValueRef,
                                 value: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        // TODO this is somewhat messy right now...lots of repeated logic!
        use ast::Type::{Scalar, Simd};
        let vectorized = LLVMGetTypeKind(LLVMTypeOf(value)) == LLVMTypeKind::LLVMVectorTypeKind;
        if vectorized {
            if self.vmerge.is_none() {
                let ret_ty = LLVMVoidTypeInContext(self.context);
                let mut arg_tys = [LLVMPointerType(self.merger_ty, 0), LLVMVectorType(self.elem_ty, LLVM_VECTOR_WIDTH as u32)];
                let name = format!("{}.vmerge", self.name);
                let (function, fn_builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

                // Load the vector element, apply the binary operator, and then store it back.
                let elem_pointer = LLVMBuildStructGEP(fn_builder, LLVMGetParam(function, 0), VECTOR_INDEX, c_str!(""));
                let elem = LLVMBuildLoad(fn_builder, elem_pointer, c_str!(""));
                let result = gen_binop(fn_builder, self.op, elem, LLVMGetParam(function, 1), &Simd(self.scalar_kind))?;
                LLVMBuildStore(fn_builder, result, elem_pointer);
                LLVMBuildRetVoid(fn_builder);

                self.vmerge = Some(function);
                LLVMDisposeBuilder(fn_builder);
            }
            let mut args = [builder, value];
            Ok(LLVMBuildCall(llvm_builder, self.vmerge.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
        } else {
            if self.merge.is_none() {
                let ret_ty = LLVMVoidTypeInContext(self.context);
                let mut arg_tys = [LLVMPointerType(self.merger_ty, 0), self.elem_ty];
                let name = format!("{}.merge", self.name);
                let (function, fn_builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

                // Load the scalar element, apply the binary operator, and then store it back.
                let elem_pointer = LLVMBuildStructGEP(fn_builder, LLVMGetParam(function, 0), SCALAR_INDEX, c_str!(""));
                let elem = LLVMBuildLoad(fn_builder, elem_pointer, c_str!(""));
                let result = gen_binop(fn_builder, self.op, elem, LLVMGetParam(function, 1), &Scalar(self.scalar_kind))?;
                LLVMBuildStore(fn_builder, result, elem_pointer);
                LLVMBuildRetVoid(fn_builder);

                self.merge = Some(function);
                LLVMDisposeBuilder(fn_builder);
            }
            let mut args = [builder, value];
            Ok(LLVMBuildCall(llvm_builder, self.merge.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
        }
    }
}


