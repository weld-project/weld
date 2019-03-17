//! Code generation for the merger builder type.

use llvm_sys;

use std::ffi::CString;

use crate::ast::BinOpKind;
use crate::ast::ScalarKind;
use crate::ast::Type::{Scalar, Simd};
use crate::error::*;

use self::llvm_sys::core::*;
use self::llvm_sys::prelude::*;
use self::llvm_sys::LLVMTypeKind;

use crate::codegen::llvm2::llvm_exts::*;
use crate::codegen::llvm2::numeric::gen_binop;
use crate::codegen::llvm2::CodeGenExt;
use crate::codegen::llvm2::LLVM_VECTOR_WIDTH;

const SCALAR_INDEX: u32 = 0;
const VECTOR_INDEX: u32 = 1;

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
    result: Option<LLVMValueRef>,
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
    pub unsafe fn define<T: AsRef<str>>(
        name: T,
        op: BinOpKind,
        elem_ty: LLVMTypeRef,
        scalar_kind: ScalarKind,
        context: LLVMContextRef,
        module: LLVMModuleRef,
    ) -> Merger {
        let c_name = CString::new(name.as_ref()).unwrap();
        let mut layout = [elem_ty, LLVMVectorType(elem_ty, LLVM_VECTOR_WIDTH)];
        let merger = LLVMStructCreateNamed(context, c_name.as_ptr());
        LLVMStructSetBody(merger, layout.as_mut_ptr(), layout.len() as u32, 0);
        Merger {
            name: c_name.into_string().unwrap(),
            op,
            merger_ty: merger,
            elem_ty,
            scalar_kind,
            context,
            module,
            new: None,
            merge: None,
            vmerge: None,
            result: None,
        }
    }

    pub unsafe fn gen_new(
        &mut self,
        builder: LLVMBuilderRef,
        init: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef> {
        if self.new.is_none() {
            let ret_ty = self.merger_ty;
            let mut arg_tys = [self.elem_ty];
            let name = format!("{}.new", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let identity = self.binop_identity(self.op, self.scalar_kind)?;
            let mut vector_elems = [identity; LLVM_VECTOR_WIDTH as usize];
            let vector_identity =
                LLVMConstVector(vector_elems.as_mut_ptr(), vector_elems.len() as u32);
            let one = LLVMBuildInsertValue(
                builder,
                LLVMGetUndef(self.merger_ty),
                LLVMGetParam(function, 0),
                SCALAR_INDEX,
                c_str!(""),
            );
            let result =
                LLVMBuildInsertValue(builder, one, vector_identity, VECTOR_INDEX, c_str!(""));

            LLVMBuildRet(builder, result);

            self.new = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [init];
        Ok(LLVMBuildCall(
            builder,
            self.new.unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        ))
    }

    /// Builds the `Merge` function and returns a reference to the function.
    ///
    /// The merge function is similar for the scalar and vector varianthe `gep_index determines
    /// which one is generated.
    unsafe fn gen_merge_internal(
        &mut self,
        name: String,
        arguments: &mut [LLVMTypeRef],
        gep_index: u32,
    ) -> WeldResult<LLVMValueRef> {
        let ret_ty = LLVMVoidTypeInContext(self.context);
        let (function, fn_builder, _) = self.define_function(ret_ty, arguments, name);

        LLVMExtAddAttrsOnFunction(self.context, function, &[LLVMExtAttribute::AlwaysInline]);

        // Load the vector element, apply the binary operator, and then store it back.
        let elem_pointer =
            LLVMBuildStructGEP(fn_builder, LLVMGetParam(function, 0), gep_index, c_str!(""));
        let elem = LLVMBuildLoad(fn_builder, elem_pointer, c_str!(""));
        let result = gen_binop(
            fn_builder,
            self.op,
            elem,
            LLVMGetParam(function, 1),
            &Simd(self.scalar_kind),
        )?;
        LLVMBuildStore(fn_builder, result, elem_pointer);
        LLVMBuildRetVoid(fn_builder);
        LLVMDisposeBuilder(fn_builder);
        Ok(function)
    }

    pub unsafe fn gen_merge(
        &mut self,
        llvm_builder: LLVMBuilderRef,
        builder: LLVMValueRef,
        value: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef> {
        let vectorized = LLVMGetTypeKind(LLVMTypeOf(value)) == LLVMTypeKind::LLVMVectorTypeKind;
        if vectorized {
            if self.vmerge.is_none() {
                let mut arg_tys = [
                    LLVMPointerType(self.merger_ty, 0),
                    LLVMVectorType(self.elem_ty, LLVM_VECTOR_WIDTH as u32),
                ];
                let name = format!("{}.vmerge", self.name);
                self.vmerge = Some(self.gen_merge_internal(name, &mut arg_tys, VECTOR_INDEX)?);
            }
            let mut args = [builder, value];
            Ok(LLVMBuildCall(
                llvm_builder,
                self.vmerge.unwrap(),
                args.as_mut_ptr(),
                args.len() as u32,
                c_str!(""),
            ))
        } else {
            if self.merge.is_none() {
                let mut arg_tys = [LLVMPointerType(self.merger_ty, 0), self.elem_ty];
                let name = format!("{}.merge", self.name);
                self.merge = Some(self.gen_merge_internal(name, &mut arg_tys, SCALAR_INDEX)?);
            }
            let mut args = [builder, value];
            Ok(LLVMBuildCall(
                llvm_builder,
                self.merge.unwrap(),
                args.as_mut_ptr(),
                args.len() as u32,
                c_str!(""),
            ))
        }
    }

    pub unsafe fn gen_result(
        &mut self,
        llvm_builder: LLVMBuilderRef,
        builder: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef> {
        if self.result.is_none() {
            let ret_ty = self.elem_ty;
            let mut arg_tys = [LLVMPointerType(self.merger_ty, 0)];
            let name = format!("{}.result", self.name);
            let (function, fn_builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            // Load the scalar element, apply the binary operator, and then store it back.
            let builder_pointer = LLVMGetParam(function, 0);
            let scalar_pointer =
                LLVMBuildStructGEP(fn_builder, builder_pointer, SCALAR_INDEX, c_str!(""));
            let mut result = LLVMBuildLoad(fn_builder, scalar_pointer, c_str!(""));

            let vector_pointer =
                LLVMBuildStructGEP(fn_builder, builder_pointer, VECTOR_INDEX, c_str!(""));
            let vector = LLVMBuildLoad(fn_builder, vector_pointer, c_str!(""));

            for i in 0..LLVM_VECTOR_WIDTH {
                let vector_element =
                    LLVMBuildExtractElement(fn_builder, vector, self.i32(i as i32), c_str!(""));
                result = gen_binop(
                    fn_builder,
                    self.op,
                    result,
                    vector_element,
                    &Scalar(self.scalar_kind),
                )?;
            }

            LLVMBuildRet(fn_builder, result);

            self.result = Some(function);
            LLVMDisposeBuilder(fn_builder);
        }
        let mut args = [builder];
        Ok(LLVMBuildCall(
            llvm_builder,
            self.result.unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        ))
    }
}
