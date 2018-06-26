//! Traits for code-generating numeric expressions.
//!
//! Specifically, this module provides code generation for the following SIR statements:
//! * `UnaryOp`
//! * `BinaryOp`
//! * `AssignLiteral`
//! * `Cast`
//! * `Negate`

extern crate time;
extern crate libc;
extern crate llvm_sys;

use ast::*;
use error::*;
use sir::*;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;

use codegen::llvm2::intrinsic::Intrinsics;

use super::{LlvmGenerator, CodeGenExt, FunctionContext, LLVM_VECTOR_WIDTH};

/// Generates numeric expresisons.
pub trait NumericExpressionGen {
    /// Generates code for a numeric unary operator.
    ///
    /// This method supports operators over both scalar and SIMD values.
    unsafe fn gen_unaryop(&mut self, ctx: &mut FunctionContext, statement: &Statement) -> WeldResult<()>;
    /// Generates code for a numeric binary operator.
    ///
    /// This method supports operators over both scalar and SIMD values.
    unsafe fn gen_binop(&mut self, ctx: &mut FunctionContext, statement: &Statement) -> WeldResult<()>;
    /// Generates a literal.
    ///
    /// This method supports both scalar and SIMD values.
    unsafe fn gen_assign_literal(&mut self, ctx: &mut FunctionContext, statement: &Statement) -> WeldResult<()>;
    /// Generates a cast expression.
    unsafe fn gen_cast(&mut self, ctx: &mut FunctionContext, statement: &Statement) -> WeldResult<()>;
}

/// Helper trait for generating numeric code.
trait NumericExpressionGenInternal {
    /// Generates the math `Pow` operator.
    unsafe fn gen_pow(&mut self,
               ctx: &mut FunctionContext,
               left: LLVMValueRef,
               right: LLVMValueRef,
               ty: &Type) -> WeldResult<LLVMValueRef>; 
}

impl NumericExpressionGenInternal for LlvmGenerator {
    unsafe fn gen_pow(&mut self,
               ctx: &mut FunctionContext,
               left: LLVMValueRef,
               right: LLVMValueRef,
               ty: &Type) -> WeldResult<LLVMValueRef> {
        use ast::Type::{Scalar, Simd};
        match *ty {
            Scalar(kind) if kind.is_float() => {
                let name = Intrinsics::llvm_numeric("pow", kind, false); 
                let ret_ty = LLVMTypeOf(left);
                let mut arg_tys = [ret_ty, ret_ty];
                self.intrinsics.add(&name, ret_ty, &mut arg_tys);
                self.intrinsics.call(ctx.builder, name, &mut [left, right])
            }
            Simd(kind) if kind.is_float() => {
                let name = Intrinsics::llvm_numeric("pow", kind, false); 
                let ret_ty = self.llvm_type(&Scalar(kind))?;
                let mut arg_tys = [ret_ty, ret_ty];
                self.intrinsics.add(&name, ret_ty, &mut arg_tys);
                // Unroll vector and apply function to each element.
                let mut result = LLVMGetUndef(LLVMVectorType(ret_ty, LLVM_VECTOR_WIDTH));
                for i in 0..LLVM_VECTOR_WIDTH {
                    let base = LLVMBuildExtractElement(ctx.builder, left, self.i32(i as i32), c_str!(""));
                    let power = LLVMBuildExtractElement(ctx.builder, right, self.i32(i as i32), c_str!(""));
                    let value = self.intrinsics.call(ctx.builder, &name, &mut [base, power])?;
                    result = LLVMBuildInsertElement(ctx.builder, result, value, self.i32(i as i32), c_str!(""));
                }
                Ok(result)
            }
            _ => unreachable!(),
        }
    }
}

trait UnaryOpSupport {
    fn llvm_intrinsic(&self) -> Option<&'static str>;
}

impl UnaryOpSupport for UnaryOpKind {
    fn llvm_intrinsic(&self) -> Option<&'static str> {
        use ast::UnaryOpKind::*;
        match *self {
            Exp => Some("exp"),
            Log => Some("log"),
            Sqrt => Some("sqrt"),
            Sin => Some("sin"),
            Cos => Some("cos"),
            _ => None
        }
    }
}

impl NumericExpressionGen for LlvmGenerator {
    unsafe fn gen_unaryop(&mut self, ctx: &mut FunctionContext, statement: &Statement) -> WeldResult<()> {
        use ast::Type::{Scalar, Simd};
        use self::UnaryOpSupport;
        use sir::StatementKind::UnaryOp;
        if let UnaryOp { op, ref child } = statement.kind {
            let ty = ctx.sir_function.symbol_type(child)?;
            let (kind, simd) = match *ty {
                Scalar(kind) => (kind, false),
                Simd(kind) => (kind, true),
                _ => unreachable!(),
            };
            let child = self.load(ctx.builder, ctx.get_value(child)?)?;
            let result = if let Some(name) = op.llvm_intrinsic() {
                let name = Intrinsics::llvm_numeric(name, kind, simd); 
                let ret_ty = LLVMTypeOf(child);
                let mut arg_tys = [ret_ty];
                self.intrinsics.add(&name, ret_ty, &mut arg_tys);
                self.intrinsics.call(ctx.builder, name, &mut [child])?
            } else {
                use ast::UnaryOpKind::*;
                use ast::ScalarKind::{F32, F64};
                let name = match (op, kind) {
                    (Tan, F32) => "tanf",
                    (ASin, F32) => "asinf",
                    (ACos, F32) => "acosf",
                    (ATan, F32) => "atanf",
                    (Sinh, F32) => "sinhf",
                    (Cosh, F32) => "coshf",
                    (Tanh, F32) => "tanhf",
                    (Erf, F32) => "erff",
                    (Tan, F64) => "tan",
                    (ASin, F64) => "asin",
                    (ACos, F64) => "acos",
                    (ATan, F64) => "atan",
                    (Sinh, F64) => "sinh",
                    (Cosh, F64) => "cosh",
                    (Tanh, F64) => "tanh",
                    (Erf, F64) => "erf",
                    _ => unreachable!(),
                };
                let ret_ty = self.llvm_type(&Scalar(kind))?;
                let mut arg_tys = [ret_ty];
                self.intrinsics.add(&name, ret_ty, &mut arg_tys);
                // If the value is a scalar, just call the intrinsic. If it's a SIMD value, unroll
                // the vector and apply the intrinsic to each element.
                if !simd {
                    self.intrinsics.call(ctx.builder, name, &mut [child])?
                } else {
                    let mut result = LLVMGetUndef(LLVMVectorType(ret_ty, LLVM_VECTOR_WIDTH));
                    for i in 0..LLVM_VECTOR_WIDTH {
                        let element = LLVMBuildExtractElement(ctx.builder, child, self.i32(i as i32), c_str!(""));
                        let value = self.intrinsics.call(ctx.builder, &name, &mut [element])?;
                        result = LLVMBuildInsertElement(ctx.builder, result, value, self.i32(i as i32), c_str!(""));
                    }
                    result
                }
            };
            let output = ctx.get_value(statement.output.as_ref().unwrap())?;
            LLVMBuildStore(ctx.builder, result, output);
            Ok(())
        } else {
            unreachable!()
        }
    }

    unsafe fn gen_binop(&mut self, ctx: &mut FunctionContext, statement: &Statement) -> WeldResult<()> {
        use ast::BinOpKind;
        use ast::Type::{Scalar, Simd, Vector};
        use sir::StatementKind::BinOp;
        if let BinOp { op, ref left, ref right } = statement.kind {
            let llvm_left = self.load(ctx.builder, ctx.get_value(left)?)?;
            let llvm_right = self.load(ctx.builder, ctx.get_value(right)?)?;
            let ty = ctx.sir_function.symbol_type(left)?;
            let result = match ty {
                &Scalar(_) | &Simd(_) => { 
                    match op {
                        BinOpKind::Pow => self.gen_pow(ctx, llvm_left, llvm_right, ty)?,
                        _ => gen_binop(ctx.builder, op, llvm_left, llvm_right, ty)?,
                    }
                }
                &Vector(_) if op.is_comparison() => {
                    unimplemented!()
                }
                _ => unreachable!(),
            };
            let output = ctx.get_value(statement.output.as_ref().unwrap())?;
            let _ = LLVMBuildStore(ctx.builder, result, output);
            Ok(())
        } else {
            unreachable!()
        }
    }

    unsafe fn gen_assign_literal(&mut self, ctx: &mut FunctionContext, statement: &Statement) -> WeldResult<()> {
        use sir::StatementKind::AssignLiteral;
        if let AssignLiteral(ref value) = statement.kind {
            let output = statement.output.as_ref().unwrap();
            let output_type = ctx.sir_function.symbol_type(output)?;
            let mut result = self.scalar_literal(value);
            if let Type::Simd(_) = output_type {
                result = LLVMConstVector([result; LLVM_VECTOR_WIDTH as usize].as_mut_ptr(), LLVM_VECTOR_WIDTH)
            }
            let pointer = ctx.get_value(output)?;
            let _ = LLVMBuildStore(ctx.builder, result, pointer);
            Ok(())
        } else {
            unreachable!()
        }
    }

    unsafe fn gen_cast(&mut self, ctx: &mut FunctionContext, statement: &Statement) -> WeldResult<()> {
        use sir::StatementKind::Cast;
        let ref output = statement.output.clone().unwrap();
        let output_pointer = ctx.get_value(output)?;
        let output_type = ctx.sir_function.symbol_type(output)?;
        if let Cast(ref child, _) = statement.kind {
            let child_type = ctx.sir_function.symbol_type(child)?;
            let child_value = self.load(ctx.builder, ctx.get_value(child)?)?;
            let result = gen_cast(ctx.builder, child_value, child_type, output_type, self.llvm_type(output_type)?)?;
            let _ = LLVMBuildStore(ctx.builder, result, output_pointer);
            Ok(())
        } else {
            unreachable!()
        }
    }
}

unsafe fn gen_cast(builder: LLVMBuilderRef,
                   value: LLVMValueRef,
                   from: &Type,
                   to: &Type,
                   to_ll: LLVMTypeRef) -> WeldResult<LLVMValueRef> {
    use ast::Type::Scalar;
    use ast::ScalarKind::*;
    let result = match (from, to) {
        (&Scalar(s1), &Scalar(s2)) => {
            match (s1, s2) {
                // Floating point extension and truncation.
                (F32, F64) => LLVMBuildFPExt(builder, value, to_ll, c_str!("")),
                (F64, F32) => LLVMBuildFPTrunc(builder, value, to_ll, c_str!("")),

                // Floating point to signed integer
                (_, _) if s1.is_float() && s2.is_signed_integer() => {
                    LLVMBuildFPToSI(builder, value, to_ll, c_str!(""))
                }

                // Floating point to unsigned integer
                (_, _) if s1.is_float() && s2.is_unsigned_integer() => {
                    LLVMBuildFPToUI(builder, value, to_ll, c_str!(""))
                }

                // Signed integer to floating point
                (_, _) if s1.is_signed_integer() && s2.is_float() => {
                    LLVMBuildSIToFP(builder, value, to_ll, c_str!(""))
                }

                // Unsigned integer to floating point
                (_, _) if s1.is_unsigned_integer() && s2.is_float() => {
                    LLVMBuildUIToFP(builder, value, to_ll, c_str!(""))
                }

                // Boolean to other integers.
                (Bool, _) if s2.is_integer() => LLVMBuildZExt(builder, value, to_ll, c_str!("")),

                // Zero-extension.
                (_, _) if s1.is_unsigned_integer() && s2.bits() > s1.bits() => {
                    LLVMBuildZExt(builder, value, to_ll, c_str!(""))
                }

                // Sign-extension.
                (_, _) if s1.is_signed_integer() && s2.bits() > s1.bits() => {
                    LLVMBuildSExt(builder, value, to_ll, c_str!(""))
                }

                // Truncation
                (_, _) if s2.bits() < s1.bits() => LLVMBuildTrunc(builder, value, to_ll, c_str!("")),

                // Bitcast
                (_, _) if s2.bits() == s1.bits() => LLVMBuildBitCast(builder, value, to_ll, c_str!("")),

                 _ => {
                     return compile_err!("Cannot cast {} to {}", from, to)
                 }
            }
        }
        _ => {
            return compile_err!("Cannot cast {} to {}", from, to)
        }
    };
    Ok(result)
}

/// Generates a binary op instruction without intrinsics.
///
/// This function supports code generation for both scalar and SIMD values.
pub unsafe fn gen_binop(builder: LLVMBuilderRef,
             op: BinOpKind,
             left: LLVMValueRef,
             right: LLVMValueRef, ty: &Type) -> WeldResult<LLVMValueRef> {
    use ast::Type::*;
    use ast::BinOpKind::*;
    use self::llvm_sys::LLVMIntPredicate::*;
    use self::llvm_sys::LLVMRealPredicate::*;
    let name = c_str!("");
    let result = match ty {
        &Scalar(s) | &Simd(s) => {
            match op {
                Add if s.is_integer() => LLVMBuildAdd(builder, left, right, name),
                Add if s.is_float() => LLVMBuildFAdd(builder, left, right, name),

                Subtract if s.is_integer() => LLVMBuildSub(builder, left, right, name),
                Subtract if s.is_float() => LLVMBuildFSub(builder, left, right, name),

                Multiply if s.is_integer() => LLVMBuildMul(builder, left, right, name),
                Multiply if s.is_float() => LLVMBuildFMul(builder, left, right, name),

                Divide if s.is_signed_integer() => LLVMBuildSDiv(builder, left, right, name),
                Divide if s.is_unsigned_integer() => LLVMBuildUDiv(builder, left, right, name),
                Divide if s.is_float() => LLVMBuildFDiv(builder, left, right, name),

                Modulo if s.is_signed_integer() => LLVMBuildSRem(builder, left, right, name),
                Modulo if s.is_unsigned_integer() => LLVMBuildURem(builder, left, right, name),
                Modulo if s.is_float() => LLVMBuildFRem(builder, left, right, name),

                Equal if s.is_integer() || s.is_bool() => LLVMBuildICmp(builder, LLVMIntEQ, left, right, name),
                Equal if s.is_float() => LLVMBuildFCmp(builder, LLVMRealOEQ, left, right, name),

                NotEqual if s.is_integer() || s.is_bool() => LLVMBuildICmp(builder, LLVMIntNE, left, right, name),
                NotEqual if s.is_float() => LLVMBuildFCmp(builder, LLVMRealONE, left, right, name),

                LessThan if s.is_signed_integer() => LLVMBuildICmp(builder, LLVMIntSLT, left, right, name),
                LessThan if s.is_unsigned_integer() => LLVMBuildICmp(builder, LLVMIntULT, left, right, name),
                LessThan if s.is_float() => LLVMBuildFCmp(builder, LLVMRealOLT, left, right, name),

                LessThanOrEqual if s.is_signed_integer() => LLVMBuildICmp(builder, LLVMIntSLE, left, right, name),
                LessThanOrEqual if s.is_unsigned_integer() => LLVMBuildICmp(builder, LLVMIntULE, left, right, name),
                LessThanOrEqual if s.is_float() => LLVMBuildFCmp(builder, LLVMRealOLE, left, right, name),

                GreaterThan if s.is_signed_integer() => LLVMBuildICmp(builder, LLVMIntSGT, left, right, name),
                GreaterThan if s.is_unsigned_integer() => LLVMBuildICmp(builder, LLVMIntUGT, left, right, name),
                GreaterThan if s.is_float() => LLVMBuildFCmp(builder, LLVMRealOGT, left, right, name),

                GreaterThanOrEqual if s.is_signed_integer() => LLVMBuildICmp(builder, LLVMIntSGE, left, right, name),
                GreaterThanOrEqual if s.is_unsigned_integer() => LLVMBuildICmp(builder, LLVMIntUGE, left, right, name),
                GreaterThanOrEqual if s.is_float() => LLVMBuildFCmp(builder, LLVMRealOGE, left, right, name),

                LogicalAnd if s.is_bool() => LLVMBuildAdd(builder, left, right, name),
                BitwiseAnd if s.is_integer() || s.is_bool() =>LLVMBuildAdd(builder, left, right, name),

                LogicalOr if s.is_bool() => LLVMBuildOr(builder, left, right, name),
                BitwiseOr if s.is_integer() || s.is_bool() => LLVMBuildOr(builder, left, right, name),

                Xor if s.is_integer() || s.is_bool() => LLVMBuildXor(builder, left, right, name),

                Max => {
                    let compare = gen_binop(builder, GreaterThanOrEqual, left, right, ty)?;
                    LLVMBuildSelect(builder, compare, left, right, c_str!(""))
                }

                Min => {
                    let compare = gen_binop(builder, LessThanOrEqual, left, right, ty)?;
                    LLVMBuildSelect(builder, compare, left, right, c_str!(""))
                }

                _ => return compile_err!("Unsupported binary op: {} on {}", op, ty)
            }
        }
        _ => return compile_err!("Unsupported binary op: {} on {}", op, ty)
    };
    Ok(result)
}
