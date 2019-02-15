//! Generates code to compare values using a comparison function.

extern crate llvm_sys;

use std::ffi::CStr;

use ast::BinOpKind::*;
use ast::Type;
use ast::ScalarKind::I64;
use codegen::llvm2::numeric::gen_binop;
use codegen::llvm2::SIR_FUNC_CALL_CONV;
use error::*;
use sir::FunctionId;

use super::vector;
use super::vector::VectorExt;

use super::llvm_exts::*;
use super::llvm_exts::LLVMExtAttribute::*;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;
use self::llvm_sys::LLVMLinkage;
use self::llvm_sys::LLVMIntPredicate::*;

use super::CodeGenExt;
use super::LlvmGenerator;

/// Returns whether a value can be compared with libc's `memcmp`.
trait SupportsMemCmp {
    fn supports_memcmp(&self) -> bool;
}

/// Only i8 or smaller values can be compared using `memcmp`.
impl SupportsMemCmp for Type {
    fn supports_memcmp(&self) -> bool {
        use ast::Type::*;
        // Structs do not support memcmp because they may be padded.
        match *self {
            Scalar(ref kind) => {
                use ast::ScalarKind::*;
                match *kind {
                    Bool | I8 | U8 | U16 | U32 | U64 => true,
                    _ => false,
                }
            },
            _ => false,
        }
    }
}

/// Trait for generating comparison code.
pub trait GenCmp {
    /// Generates a comparator for a type.
    ///
    /// The comparators are over pointers of the type, e.g., a comparator for `i32`
    /// has the type signature `i1 (i32*, i32*)`. This method returns the generated function. The
    /// function returns a value <0 if the first element is smaller, 0 if the elements are equal, and >0 if
    /// the first element is larger.
    unsafe fn gen_cmp_fn(&mut self, ty: &Type) -> WeldResult<LLVMValueRef>; 

    /// Generates an opaque comparator using the specified comparator function.
    ///
    /// The comparator should return a value < 0 if the first element is smaller, 0 if the elements are equal,
    /// and >0 if the first element is larger.
    ///
    /// # Portability Notes
    ///
    /// The opaque comparator is generated with a function signature compatible with the `libc`
    /// `qsort_r` function. Note that the function signature is slightly different for Linux
    /// platforms and FreeBSD platforms: as such, the code generated will differ by platform as
    /// well.
    unsafe fn gen_custom_cmp(&mut self,
                             elem_ty: LLVMTypeRef,
                             cf_id: FunctionId,
                             cmpfunc: LLVMValueRef) -> WeldResult<LLVMValueRef>;
}

impl GenCmp for LlvmGenerator {
    /// Generates a default comparison function for a type.
    ///
    /// For scalars, the comparison function returns -1 if left < right, 1 if left > right,
    /// and 0 otherwise.
    /// For flat aggregate types (i.e. vectors), the comparison function returns
    /// the scalar comparison value for the first element where the values are not equal, or 0 if
    /// all values are equal.
    /// For nested aggregate types (i.e. structs), the comparison function is applied
    /// recursively to the elements.
    unsafe fn gen_cmp_fn(&mut self, ty: &Type) -> WeldResult<LLVMValueRef> {
        use ast::Type::*;
        let result = self.cmp_fns.get(ty).cloned();
        if let Some(result) = result {
            return Ok(result)
        }

        let llvm_ty = self.llvm_type(ty)?;
        // XXX Do we need the run handle?
        let mut arg_tys = [LLVMPointerType(llvm_ty, 0), LLVMPointerType(llvm_ty, 0)];
        let ret_ty = self.i32_type();

        let c_prefix = LLVMPrintTypeToString(llvm_ty);
        let prefix = CStr::from_ptr(c_prefix);
        let prefix = prefix.to_str().unwrap();
        let name = format!("{}.cmp", prefix);
        // Free the allocated string.
        LLVMDisposeMessage(c_prefix);

        let (function, builder, entry_block) = self.define_function(ret_ty, &mut arg_tys, name);

        LLVMExtAddAttrsOnParameter(self.context, function, &[ReadOnly, NoAlias, NonNull, NoCapture], 0);
        LLVMExtAddAttrsOnParameter(self.context, function, &[ReadOnly, NoAlias, NonNull, NoCapture], 1);

        let left = LLVMGetParam(function, 0);
        let right = LLVMGetParam(function, 1);

        match *ty {
            Builder(_, _) => unreachable!(),
            Dict(_, _) => unimplemented!(), // dictionary comparison
            Simd(_) => unimplemented!(),
            Scalar(_) => {
                let left = self.load(builder, left)?;
                let right = self.load(builder, right)?;
                let on_geq_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!(""));
                let done_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!(""));
                
                // if lt
                let cond = gen_binop(builder, LessThan, left, right, ty)?;
                LLVMBuildCondBr(builder, cond, done_block, on_geq_block);

                LLVMPositionBuilderAtEnd(builder, on_geq_block);
                // else if equal
                // this is less likely to occur during sort so we don't check it if the lt branch passes
                // TODO: special case booleans for Select
                let eq = gen_binop(builder, Equal, left, right, ty)?;
                let on_geq = LLVMBuildSelect(builder, eq, self.i32(0), self.i32(1), c_str!(""));
                LLVMBuildBr(builder, done_block);
                
                // Finish block - set result.
                LLVMPositionBuilderAtEnd(builder, done_block);
                let result = LLVMBuildPhi(builder, ret_ty, c_str!(""));
                let mut blocks = [entry_block, on_geq_block];
                let mut values = [self.i32(-1), on_geq];
                LLVMAddIncoming(result, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);

                LLVMBuildRet(builder, result);
            }
            Struct(ref elems) => { // recursively apply cmp to elements
                let mut result = self.i32(0);
                for (i, elem) in elems.iter().enumerate() {
                    let func = self.gen_cmp_fn(elem)?;
                    let next_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!(""));
                    let done_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!(""));

                    let field_left = LLVMBuildStructGEP(builder, left, i as u32, c_str!(""));
                    let field_right = LLVMBuildStructGEP(builder, right, i as u32, c_str!(""));

                    let mut args = [field_left, field_right];
                    // Compare struct field.
                    let field_result = LLVMBuildCall(builder, func, args.as_mut_ptr(), args.len() as u32, c_str!(""));
                    let cond = LLVMBuildICmp(builder, LLVMIntEQ, field_result, self.i32(0), c_str!(""));

                    // Continue to next only if the field result returns 0 (fields are equal).
                    LLVMBuildCondBr(builder, cond, next_block, done_block);
                    
                    // If field result is nonzero, return field result.
                    LLVMPositionBuilderAtEnd(builder, done_block);
                    LLVMBuildRet(builder, field_result);
                    
                    LLVMPositionBuilderAtEnd(builder, next_block);
                }
                
                // All blocks equal, return 0.
                LLVMBuildRet(builder, result); 
            }
            // Vectors comprised of unsigned chars or booleans can be compared for using `memcmp`.
            Vector(ref elem) if elem.supports_memcmp() => {
                let left_data_ptr = LLVMBuildStructGEP(builder, left, vector::POINTER_INDEX, c_str!(""));
                let left_data = self.load(builder, left_data_ptr)?;
                let left_cast = LLVMBuildBitCast(builder, left_data,
                                                 self.void_pointer_type(), c_str!(""));
                let right_data_ptr = LLVMBuildStructGEP(builder, right, vector::POINTER_INDEX, c_str!(""));
                let right_data = self.load(builder, right_data_ptr)?;
                let right_cast = LLVMBuildBitCast(builder, right_data,
                                                  self.void_pointer_type(), c_str!(""));
                
                let left_size_ptr = LLVMBuildStructGEP(builder, left, vector::SIZE_INDEX, c_str!(""));
                let left_size = self.load(builder, left_size_ptr)?;
                let right_size_ptr = LLVMBuildStructGEP(builder, right, vector::SIZE_INDEX, c_str!(""));
                let right_size = self.load(builder, right_size_ptr)?;

                // memcmp will run off the end of the smaller buffer,
                // so emulate strcmp semantics by stopping at the end of the smaller buffer (and then comparing sizes).
                // Note that this only works when both vectors have elements of the same type.
                let min_size = gen_binop(builder, Min, left_size, right_size, &Scalar(I64))?;

                let elem_ty = self.llvm_type(elem)?;
                let elem_size = self.size_of(elem_ty);
                let bytes = LLVMBuildNSWMul(builder, min_size, elem_size, c_str!(""));

                // Call memcmp
                let name = "memcmp";
                let ret_ty = self.i32_type();
                let ref mut arg_tys = [self.void_pointer_type(), self.void_pointer_type(), self.i64_type()];
                if self.intrinsics.add(name, ret_ty, arg_tys) {
                    let memcmp = self.intrinsics.get(name).unwrap();
                    LLVMExtAddAttrsOnParameter(self.context, memcmp, &[ReadOnly, NoCapture], 0);
                    LLVMExtAddAttrsOnParameter(self.context, memcmp, &[ReadOnly, NoCapture], 1);
                }
                let ref mut args = [left_cast, right_cast, bytes];
                let memcmp_result = self.intrinsics.call(builder, name, args)?;

                // If all compared bytes were equal but sizes were not equal, the smaller vector is the lesser element.
                let func = self.gen_cmp_fn(&Scalar(I64))?;
                let mut args = [left_size_ptr, right_size_ptr];
                let size_eq = LLVMBuildCall(builder, func, args.as_mut_ptr(), args.len() as u32, c_str!(""));
                
                let bytes_equal = LLVMBuildICmp(builder, LLVMIntEQ, memcmp_result, self.i32(0), c_str!(""));
                let result = LLVMBuildSelect(builder, bytes_equal, size_eq, memcmp_result, c_str!(""));
                
                LLVMBuildRet(builder, result);
            }
            Vector(ref elem) => {
                // Compare vectors with a loop. Check size like before.
                let loop_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));
                let done_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!(""));

                let left_vector = self.load(builder, left)?;
                let right_vector = self.load(builder, right)?;
                
                let left_size_ptr = LLVMBuildStructGEP(builder, left, vector::SIZE_INDEX, c_str!(""));
                let left_size = self.load(builder, left_size_ptr)?;
                let right_size_ptr = LLVMBuildStructGEP(builder, right, vector::SIZE_INDEX, c_str!(""));
                let right_size = self.load(builder, right_size_ptr)?;

                let min_size = gen_binop(builder, Min, left_size, right_size, &Scalar(I64))?;

                // Check if there are any elements to loop over.
                let check = LLVMBuildICmp(builder, LLVMIntNE, min_size, self.i64(0), c_str!(""));

                LLVMBuildCondBr(builder, check, loop_block, done_block);
                LLVMPositionBuilderAtEnd(builder, loop_block);

                // Index variable.
                let phi_i = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));

                let func = self.gen_cmp_fn(elem)?;
                let left_value = self.gen_at(builder, ty, left_vector, phi_i)?;
                let right_value = self.gen_at(builder, ty, right_vector, phi_i)?;
                let mut args = [left_value, right_value];
                let cmp_result = LLVMBuildCall(builder, func, args.as_mut_ptr(), args.len() as u32, c_str!(""));
                let neq = LLVMBuildICmp(builder, LLVMIntNE, cmp_result, self.i32(0), c_str!(""));

                let updated_i = LLVMBuildNSWAdd(builder, phi_i, self.i64(1), c_str!(""));
                let check1 = LLVMBuildICmp(builder, LLVMIntEQ, updated_i, min_size, c_str!(""));
                let check2 = LLVMBuildOr(builder, check1, neq, c_str!(""));
                // End loop if i == size || left != right
                LLVMBuildCondBr(builder, check2, done_block, loop_block);

                let mut blocks = [entry_block, loop_block];
                let mut values = [self.i64(0), updated_i];
                LLVMAddIncoming(phi_i, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);

                // Finish block - set result and compute number of consumed bits.
                LLVMPositionBuilderAtEnd(builder, done_block);
                let result = LLVMBuildPhi(builder, self.i32_type(), c_str!(""));
                let mut blocks = [entry_block, loop_block];
                let mut values = [self.i32(0), cmp_result];
                LLVMAddIncoming(result, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);

                // If all compared bytes were equal but sizes were not equal, the smaller vector is the lesser element.
                let func = self.gen_cmp_fn(&Scalar(I64))?;
                let mut args = [left_size_ptr, right_size_ptr];
                let size_eq = LLVMBuildCall(builder, func, args.as_mut_ptr(), args.len() as u32, c_str!(""));
                
                let bytes_equal = LLVMBuildICmp(builder, LLVMIntEQ, result, self.i32(0), c_str!(""));
                let result = LLVMBuildSelect(builder, bytes_equal, size_eq, result, c_str!(""));
                
                LLVMBuildRet(builder, result);
            }
            Function(_,_) | Unknown | Alias(_, _)=> {
                unreachable!()
            }
        };

        LLVMDisposeBuilder(builder);

        self.cmp_fns.insert(ty.clone(), function);
        Ok(function)
    }

    unsafe fn gen_custom_cmp(&mut self,
                             elem_ty: LLVMTypeRef,
                             cf_id: FunctionId,
                             cmpfunc: LLVMValueRef) -> WeldResult<LLVMValueRef> {

        // Annoyingly, Linux and MacOS pass these in different orders as well...
        let mut arg_tys = if cfg!(target_os = "macos") {
            [self.run_handle_type(), self.void_pointer_type(), self.void_pointer_type()]
        } else if cfg!(target_os = "linux") {
            [self.void_pointer_type(), self.void_pointer_type(), self.run_handle_type()]
        } else {
            unimplemented!()
        };

        let ret_ty = self.i32_type();

        let name = format!("{}.custom_cmp", cf_id);

        let (function, builder, _) = self.define_function_with_visibility(ret_ty,
                                                                          &mut arg_tys,
                                                                          LLVMLinkage::LLVMExternalLinkage,
                                                                          name);

        LLVMExtAddAttrsOnFunction(self.context, function, &[InlineHint]);

        let (left, right, run) = if cfg!(target_os = "macos") {
            LLVMExtAddAttrsOnParameter(self.context, function, &[ReadOnly, NoAlias, NonNull, NoCapture], 1);
            LLVMExtAddAttrsOnParameter(self.context, function, &[ReadOnly, NoAlias, NonNull, NoCapture], 2);

            let run  = LLVMGetParam(function, 0);
            let left = LLVMGetParam(function, 1);
            let right   = LLVMGetParam(function, 2);

            (left, right, run)
        } else if cfg!(target_os = "linux") {
            LLVMExtAddAttrsOnParameter(self.context, function, &[ReadOnly, NoAlias, NonNull, NoCapture], 0);
            LLVMExtAddAttrsOnParameter(self.context, function, &[ReadOnly, NoAlias, NonNull, NoCapture], 1);

            let left  = LLVMGetParam(function, 0);
            let right = LLVMGetParam(function, 1);
            let run   = LLVMGetParam(function, 2);

            (left, right, run)
        } else {
            unimplemented!()
        };

        let left  = LLVMBuildBitCast(builder, left,  LLVMPointerType(elem_ty, 0), c_str!(""));
        let right = LLVMBuildBitCast(builder, right, LLVMPointerType(elem_ty, 0), c_str!(""));

        // Load arguments.
        let left_value  = self.load(builder, left)?;
        let right_value = self.load(builder, right)?;
        let mut cmp_args  = [left_value, right_value, run];

        // Call the comparator.
        let result = LLVMBuildCall(builder, cmpfunc, cmp_args.as_mut_ptr(),
                                   cmp_args.len() as u32, c_str!(""));
        LLVMSetInstructionCallConv(result, SIR_FUNC_CALL_CONV);

        let result = LLVMBuildZExt(builder, result, self.i32_type(), c_str!(""));
        LLVMBuildRet(builder, result);
        LLVMDisposeBuilder(builder);

        Ok(function)
    }
}
