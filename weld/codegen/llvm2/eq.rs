//! Generates code to compare values for equality.

extern crate llvm_sys;

use std::ffi::CStr;

use ast::Type;
use error::*;

use super::llvm_exts::*;
use super::llvm_exts::LLVMExtAttribute::*;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;
use self::llvm_sys::LLVMLinkage;

use super::CodeGenExt;
use super::LlvmGenerator;

/// Trait for generating equality-comparison code.
pub trait GenEq {
    /// Generates an equality function for a type.
    ///
    /// The equality functions are over pointers of the type, e.g., an equality function for `i32`
    /// has the type signature `i1 (i32*, i32*)`. This method returns the generated function.
    unsafe fn gen_eq_fn(&mut self, ty: &Type) -> WeldResult<LLVMValueRef>; 

    /// Generates an opaque equality function for a type.
    ///
    /// Opaque equality functions are linked externally and operate over opaque (i.e., `void*`)
    /// arguments. These are used in, e.g., the dictionary implementation. This method returns the
    /// generated function.
    unsafe fn gen_opaque_eq_fn(&mut self, ty: &Type) -> WeldResult<LLVMValueRef>;
}

impl GenEq for LlvmGenerator {
    /// Generates an equality function for a type.
    unsafe fn gen_eq_fn(&mut self, ty: &Type) -> WeldResult<LLVMValueRef> {
        use ast::Type::*;
        let result = self.eq_fns.get(ty).cloned();
        if let Some(result) = result {
            return Ok(result)
        }

        let llvm_ty = self.llvm_type(ty)?;
        // XXX Do we need the run handle?
        let mut arg_tys = [LLVMPointerType(llvm_ty, 0), LLVMPointerType(llvm_ty, 0)];
        let ret_ty = self.bool_type();

        let c_prefix = LLVMPrintTypeToString(llvm_ty);
        let prefix = CStr::from_ptr(c_prefix);
        let prefix = prefix.to_str().unwrap();
        let name = format!("{}.eq", prefix);
        // Free the allocated string.
        LLVMDisposeMessage(c_prefix);

        let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

        LLVMExtAddAttrsOnFunction(self.context, function, &[InlineHint]);
        LLVMExtAddAttrsOnParameter(self.context, function, &[ReadOnly, NoAlias, NonNull, NoCapture], 0);
        LLVMExtAddAttrsOnParameter(self.context, function, &[ReadOnly, NoAlias, NonNull, NoCapture], 1);

        let left = LLVMGetParam(function, 0);
        let right = LLVMGetParam(function, 1);

        let result = match *ty {
            Builder(_, _) => unreachable!(),
            Dict(_, _) => unimplemented!(),
            Scalar(_) | Simd(_) => {
                use ast::BinOpKind::Equal;
                use codegen::llvm2::numeric::gen_binop;
                let left = self.load(builder, left)?;
                let right = self.load(builder, right)?;
                gen_binop(builder, Equal, left, right, ty)?
            }
            Struct(ref elems) => {
                let mut result = self.bool(true);
                for (i, elem) in elems.iter().enumerate() {
                    let func = self.gen_eq_fn(elem)?;
                    let field_left = LLVMBuildStructGEP(builder, left, i as u32, c_str!(""));
                    let field_right = LLVMBuildStructGEP(builder, right, i as u32, c_str!(""));
                    let mut args = [field_left, field_right];
                    let field_result = LLVMBuildCall(builder, func, args.as_mut_ptr(), args.len() as u32, c_str!(""));
                    result = LLVMBuildAnd(builder, result, field_result, c_str!(""));
                }
                result
            }
            Vector(_) => {
                // XXX Does it make sense to put this here or in `vector.rs` as a method?
                unimplemented!()
            }
            Function(_,_) | Unknown => unreachable!()
        };

        LLVMBuildRet(builder, result);
        LLVMDisposeBuilder(builder);

        self.eq_fns.insert(ty.clone(), function);
        Ok(function)
    }

    /// Generates an opaque equality function for a type.
    unsafe fn gen_opaque_eq_fn(&mut self, ty: &Type) -> WeldResult<LLVMValueRef> {
        let result = self.opaque_eq_fns.get(ty).cloned();
        if let Some(result) = result {
            return Ok(result)
        }

        let llvm_ty = self.llvm_type(ty)?;
        let mut arg_tys = [self.void_pointer_type(), self.void_pointer_type()];
        let ret_ty = self.i32_type();

        let c_prefix = LLVMPrintTypeToString(llvm_ty);
        let prefix = CStr::from_ptr(c_prefix);
        let prefix = prefix.to_str().unwrap();
        let name = format!("{}.opaque_eq", prefix);

        // Free the allocated string.
        LLVMDisposeMessage(c_prefix);

        let (function, builder, _) = self.define_function_with_visability(ret_ty,
                                                                          &mut arg_tys,
                                                                          LLVMLinkage::LLVMExternalLinkage,
                                                                          name);

        LLVMExtAddAttrsOnFunction(self.context, function, &[InlineHint]);
        LLVMExtAddAttrsOnParameter(self.context, function, &[ReadOnly, NoAlias, NonNull, NoCapture], 0);
        LLVMExtAddAttrsOnParameter(self.context, function, &[ReadOnly, NoAlias, NonNull, NoCapture], 1);

        let left = LLVMGetParam(function, 0);
        let right = LLVMGetParam(function, 1);
        let left = LLVMBuildBitCast(builder, left, LLVMPointerType(llvm_ty, 0), c_str!(""));
        let right = LLVMBuildBitCast(builder, right, LLVMPointerType(llvm_ty, 0), c_str!(""));

        let func = self.gen_eq_fn(ty)?;
        let mut args = [left, right];
        let result = LLVMBuildCall(builder, func, args.as_mut_ptr(), args.len() as u32, c_str!(""));
        let result = LLVMBuildZExt(builder, result, self.i32_type(), c_str!(""));
        LLVMBuildRet(builder, result);
        LLVMDisposeBuilder(builder);

        self.opaque_eq_fns.insert(ty.clone(), function);
        Ok(function)
    }
}
