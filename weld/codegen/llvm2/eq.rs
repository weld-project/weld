//! Generates code to compare values for equality.

extern crate llvm_sys;

use std::ffi::CStr;

use ast::Type;
use super::vector::VectorExt;
use ast::ScalarKind::{I32, I64};
use error::*;

use super::llvm_exts::*;
use super::llvm_exts::LLVMExtAttribute::*;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;
use self::llvm_sys::LLVMLinkage;
use self::llvm_sys::LLVMIntPredicate::*;

use super::CodeGenExt;
use super::LlvmGenerator;

use ast::BinOpKind::Equal;
use codegen::llvm2::numeric::gen_binop;

/// Returns whether a value can be compared with libc's `memcmp`.
///
/// XXX For now, this returns true if `memcmp` can be used for equality.
trait SupportsMemCmpEq {
    fn supports_memcmp_eq(&self) -> bool;
}

impl SupportsMemCmpEq for Type {
    fn supports_memcmp_eq(&self) -> bool {
        use ast::Type::*;
        // Structs do not support memcmp because they may be padded.
        match *self {
            Scalar(ref kind) => kind.is_integer(),
            _ => false,
        }
    }
}

/// Trait for generating equality-comparison code.
pub trait GenEq {
    /// Generates an equality function for a type.
    ///
    /// The equality functions are over pointers of the type, e.g., an equality function for `i32`
    /// has the type signature `i1 (i32*, i32*)`. This method returns the generated function. The
    /// function returns `true` if the two operands are equal.
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
        let ret_ty = self.i1_type();

        let c_prefix = LLVMPrintTypeToString(llvm_ty);
        let prefix = CStr::from_ptr(c_prefix);
        let prefix = prefix.to_str().unwrap();
        let name = format!("{}.eq", prefix);
        // Free the allocated string.
        LLVMDisposeMessage(c_prefix);

        let (function, builder, entry_block) = self.define_function(ret_ty, &mut arg_tys, name);

        LLVMExtAddAttrsOnFunction(self.context, function, &[InlineHint]);
        LLVMExtAddAttrsOnParameter(self.context, function, &[ReadOnly, NoAlias, NonNull, NoCapture], 0);
        LLVMExtAddAttrsOnParameter(self.context, function, &[ReadOnly, NoAlias, NonNull, NoCapture], 1);

        let left = LLVMGetParam(function, 0);
        let right = LLVMGetParam(function, 1);

        let result = match *ty {
            Builder(_, _) => unreachable!(),
            Dict(_, _) => unimplemented!(), // Dictionary Equality
            Simd(_) => unimplemented!(),
            Scalar(_) => {
                let left = self.load(builder, left)?;
                let right = self.load(builder, right)?;
                gen_binop(builder, Equal, left, right, ty)?
            }
            Struct(ref elems) => {
                let mut result = self.i1(true);
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
            // Vectors comprised of integers or structs of integers can be compared for equality using `memcmp`.
            //
            // XXX Note eventually when we support comparison for sorting, this won't work! We can
            // then only support unsigned integers and booleans (and structs thereof).
            Vector(ref elem) if elem.supports_memcmp_eq() => {
                let compare_data_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!(""));
                let done_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!(""));

                let left_vector = self.load(builder, left)?;
                let right_vector = self.load(builder, right)?;
                let left_size = self.gen_size(builder, ty, left_vector)?;
                let right_size = self.gen_size(builder, ty, right_vector)?;
                let size_eq = gen_binop(builder, Equal, left_size, right_size, &Scalar(I64))?;
                LLVMBuildCondBr(builder, size_eq, compare_data_block, done_block);

                LLVMPositionBuilderAtEnd(builder, compare_data_block);
                let zero = self.i64(0);
                let left_data = self.gen_at(builder, ty, left_vector, zero)?;
                let right_data = self.gen_at(builder, ty, right_vector, zero)?;
                let left_data = LLVMBuildBitCast(builder, left_data, self.void_pointer_type(), c_str!(""));
                let right_data = LLVMBuildBitCast(builder, right_data, self.void_pointer_type(), c_str!(""));
                let elem_ty = self.llvm_type(elem)?;
                let elem_size = self.size_of(elem_ty);
                let bytes = LLVMBuildNSWMul(builder, left_size, elem_size, c_str!(""));

                // Call memcmp
                let name = "memcmp";
                let ret_ty = self.i32_type();
                let ref mut arg_tys = [self.void_pointer_type(), self.void_pointer_type(), self.i64_type()];
                if self.intrinsics.add(name, ret_ty, arg_tys) {
                    let memcmp = self.intrinsics.get(name).unwrap();
                    LLVMExtAddAttrsOnParameter(self.context, memcmp, &[ReadOnly, NoCapture], 0);
                    LLVMExtAddAttrsOnParameter(self.context, memcmp, &[ReadOnly, NoCapture], 1);
                }
                let ref mut args = [left_data, right_data, bytes];
                let memcmp_result = self.intrinsics.call(builder, name, args)?;

                // If MemCmp returns 0, the two buffers are equal.
                let data_eq = gen_binop(builder, Equal, memcmp_result, self.i32(0), &Scalar(I32))?;
                LLVMBuildBr(builder, done_block);

                // Finished - build a PHI node to select the result.
                LLVMPositionBuilderAtEnd(builder, done_block);
                let result = LLVMBuildPhi(builder, self.i1_type(), c_str!(""));

                let mut incoming_blocks = [entry_block, compare_data_block];
                let mut incoming_values = [size_eq, data_eq];
                LLVMAddIncoming(result,
                                incoming_values.as_mut_ptr(),
                                incoming_blocks.as_mut_ptr(),
                                incoming_blocks.len() as u32);
                result
            }
            Vector(ref elem) => {
                // Compare vectors with a loop. Check size like before.
                let compare_data_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!(""));
                let loop_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));
                let done_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!(""));

                let left_vector = self.load(builder, left)?;
                let right_vector = self.load(builder, right)?;
                let left_size = self.gen_size(builder, ty, left_vector)?;
                let right_size = self.gen_size(builder, ty, right_vector)?;
                let size_eq = gen_binop(builder, Equal, left_size, right_size, &Scalar(I64))?;
                LLVMBuildCondBr(builder, size_eq, compare_data_block, done_block);

                LLVMPositionBuilderAtEnd(builder, compare_data_block);
                // Check if there are any elements to loop over.
                let check = LLVMBuildICmp(builder, LLVMIntNE, left_size, self.i64(0), c_str!(""));

                LLVMBuildCondBr(builder, check, loop_block, done_block);
                LLVMPositionBuilderAtEnd(builder, loop_block);

                // Index variable.
                let phi_i = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));

                let func = self.gen_eq_fn(elem)?;
                let left_value = self.gen_at(builder, ty, left_vector, phi_i)?;
                let right_value = self.gen_at(builder, ty, right_vector, phi_i)?;
                let mut args = [left_value, right_value];
                let eq_result = LLVMBuildCall(builder, func, args.as_mut_ptr(), args.len() as u32, c_str!(""));
                let neq = LLVMBuildNot(builder, eq_result, c_str!(""));

                let updated_i = LLVMBuildNSWAdd(builder, phi_i, self.i64(1), c_str!(""));
                let check1 = LLVMBuildICmp(builder, LLVMIntEQ, updated_i, left_size, c_str!(""));
                let check2 = LLVMBuildOr(builder, check1, neq, c_str!(""));
                // End loop if i == size || left != right
                LLVMBuildCondBr(builder, check2, done_block, loop_block);

                let mut blocks = [compare_data_block, loop_block];
                let mut values = [self.i64(0), updated_i];
                LLVMAddIncoming(phi_i, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);

                // Finish block - set result and compute number of consumed bits.
                LLVMPositionBuilderAtEnd(builder, done_block);
                let result = LLVMBuildPhi(builder, self.i1_type(), c_str!(""));
                let mut blocks = [entry_block, compare_data_block, loop_block];
                let mut values = [size_eq, self.i1(true), eq_result];
                LLVMAddIncoming(result, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);
                result
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

        let (function, builder, _) = self.define_function_with_visibility(ret_ty,
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
