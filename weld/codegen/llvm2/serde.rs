//! Implements serialization and deserialization of Weld types.
//!
//! This module provides code generation for the following SIR statements:
//!
//! * `Serialize`
//! * `Deserialize`


extern crate time;
extern crate libc;
extern crate llvm_sys;
extern crate lazy_static;

use ast::*;
use error::*;
use sir::*;

use std::ffi::CString;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;

use codegen::llvm2::intrinsic::Intrinsics;

use super::{LlvmGenerator, CodeGenExt, FunctionContext, LLVM_VECTOR_WIDTH};

lazy_static! {
    // The serialized type, which is a vec[i8].
    static ref SER_TY: Type = Type::Vector(Box::new(Type::Scalar(ScalarKind::I8)));
}

/// Trait for generating serialization and deserialization code.
pub trait SerDeGen {
    unsafe fn gen_serialize(&mut self, ctx: &mut FunctionContext, statement: &Statement) -> WeldResult<()>;
    unsafe fn gen_deserialize(&mut self, ctx: &mut FunctionContext, statement: &Statement) -> WeldResult<()>;
}

/// Specifies whether a type contains a pointer.
trait HasPointer {
    fn has_pointer(&self) -> bool;
}

impl HasPointer for Type {
    fn has_pointer(&self) -> bool {
        use ast::Type::*;
        match *self {
            Scalar(_) => false,
            Simd(_) => false,
            Vector(_) => true,
            Dict(_, _) => true,
            Builder(_, _) => true,
            Struct(ref tys) => tys.iter().any(|ref t| t.has_pointer()),
            Function(_, _) | Unknown => unreachable!(),
        }
    }
}

impl SerDeGen for LlvmGenerator {
    unsafe fn gen_serialize(&mut self,
                            ctx: &mut FunctionContext,
                            statement: &Statement) -> WeldResult<()> {
        use sir::StatementKind::Serialize;
        use codegen::llvm2::vector::VectorExt;
        if let Serialize(ref child) = statement.kind {
            // TODO(shoumik): Provide a size hint here!
            let initial_size = self.i64(16);
            let zero = self.i64(0);
            let buffer = self.gen_new(ctx, &SER_TY, initial_size)?;
            let child_ty = ctx.sir_function.symbol_type(child)?;
            let child = self.load(ctx.builder, ctx.get_value(child)?)?;
            let serialized = self.gen_serialize_helper(ctx,
                                                       &mut SerializePosition::new(zero),
                                                       child,
                                                       child_ty,
                                                       buffer)?;

            // XXX Update the size - the size in the returned vector is the capacity.
            // The position holds the actual size.
            let output = statement.output.as_ref().unwrap();
            LLVMBuildStore(ctx.builder, serialized, ctx.get_value(output)?);
            Ok(())
        } else {
            unreachable!()
        }
    }

    unsafe fn gen_deserialize(&mut self, ctx: &mut FunctionContext, statement: &Statement) -> WeldResult<()> {
        use sir::StatementKind::Deserialize;
        if let Deserialize(ref child) = statement.kind {
            let output = statement.output.as_ref().unwrap();
            let output_ty = ctx.sir_function.symbol_type(output)?;
            Ok(())
        } else {
            unreachable!()
        }
    }
}

struct SerializePosition {
    index: LLVMValueRef,
}

impl SerializePosition {
    fn new(index: LLVMValueRef) -> SerializePosition {
        SerializePosition {
            index: index,
        }
    }
}

/// Helper for serialization.
trait SerDeHelper {
    /// Copy a value into the serialization buffer.
    unsafe fn gen_put_value(&mut self,
                           ctx: &mut FunctionContext,
                           value: LLVMValueRef,
                           buffer: LLVMValueRef,
                           position: &mut SerializePosition) -> WeldResult<LLVMValueRef>;

    /// Copy a typed buffer of values into the serialization buffer.
    ///
    /// The buffer should have `size` objects, and the objects should not contain any nested pointers.
    unsafe fn gen_put_values(&mut self,
                           ctx: &mut FunctionContext,
                           ptr: LLVMValueRef,
                           size: LLVMValueRef,
                           buffer: LLVMValueRef,
                           position: &mut SerializePosition) -> WeldResult<LLVMValueRef>;

    /// A recursive function for serializing a value.
    ///
    /// The serialized value is written into buffer, with the next byte being written at `index`.
    /// The function updates `index` to point to the last byte in the buffer if necessary.
    unsafe fn gen_serialize_helper(&mut self,
                           ctx: &mut FunctionContext,
                           position: &mut SerializePosition,
                           value: LLVMValueRef,
                           ty: &Type,
                           buffer: LLVMValueRef) -> WeldResult<LLVMValueRef>;
}

impl SerDeHelper for LlvmGenerator {

    /// Copy a value into the serialization buffer.
    unsafe fn gen_put_value(&mut self,
                           ctx: &mut FunctionContext,
                           value: LLVMValueRef,
                           buffer: LLVMValueRef,
                           position: &mut SerializePosition) -> WeldResult<LLVMValueRef> {

        use codegen::llvm2::vector::VectorExt;
        let size = self.size_of(LLVMTypeOf(value));

        // Grow the vector to the required capacity.
        let buffer = self.gen_extend(ctx, &SER_TY, buffer, size)?;

        let ty = LLVMTypeOf(value);
        let pointer_ty = LLVMPointerType(ty, 0);
        let pointer = self.gen_at(ctx, &SER_TY, buffer, position.index)?;
        
        // Write the value.
        let pointer_typed = LLVMBuildBitCast(ctx.builder, pointer, pointer_ty, c_str!(""));
        LLVMBuildStore(ctx.builder, value, pointer_typed);

        // Update the position.
        position.index = LLVMBuildAdd(ctx.builder, position.index, size, c_str!(""));
        Ok(buffer)
    }

    /// Copy a typed buffer of values into the serialization buffer.
    ///
    /// The buffer should have `size` objects, and the objects should not contain any nested pointers.
    unsafe fn gen_put_values(&mut self,
                           ctx: &mut FunctionContext,
                           ptr: LLVMValueRef,
                           size: LLVMValueRef,
                           buffer: LLVMValueRef,
                           position: &mut SerializePosition) -> WeldResult<LLVMValueRef> {
        use codegen::llvm2::vector::VectorExt;

        let elem_size = self.size_of(LLVMGetElementType(LLVMTypeOf(ptr)));
        let size = LLVMBuildMul(ctx.builder, size, elem_size, c_str!(""));
        let buffer = self.gen_extend(ctx, &SER_TY, buffer, size)?;
        let pointer = self.gen_at(ctx, &SER_TY, buffer, position.index)?;
        
        // Write the value.
        let pointer_untyped = LLVMBuildBitCast(ctx.builder,
                                               ptr,
                                               LLVMPointerType(self.i8_type(), 0),
                                               c_str!(""));
        self.intrinsics.call_memcpy(ctx.builder, pointer, pointer_untyped, size);
        position.index = LLVMBuildAdd(ctx.builder, position.index, size, c_str!(""));
        Ok(buffer)
    }

    unsafe fn gen_serialize_helper(&mut self,
                           ctx: &mut FunctionContext,
                           position: &mut SerializePosition,
                           value: LLVMValueRef,
                           ty: &Type,
                           buffer: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        use codegen::llvm2::vector::VectorExt;
        use ast::Type::*;
        match *ty {
            Scalar(_) => {
                self.gen_put_value(ctx, value, buffer, position)
            }
            Struct(_) if !ty.has_pointer() => {
                self.gen_put_value(ctx, value, buffer, position)
            }
            Vector(ref elem) if !elem.has_pointer() => {
                let zero = self.i64(0);
                let vec_size = self.gen_size(ctx, ty, value)?;
                let vec_ptr = self.gen_at(ctx, ty, value, zero)?;
                // Write the 8-byte length followed by the data buffer.
                let buffer = self.gen_put_value(ctx, vec_size, buffer, position)?;
                self.gen_put_values(ctx, vec_ptr, vec_size, buffer, position)
            }
            Struct(ref tys) => {
                let mut buffer = buffer;
                for (i, ty) in tys.iter().enumerate() {
                    let value_pointer = LLVMBuildStructGEP(ctx.builder, value, i as u32, c_str!(""));
                    let struct_elem = self.load(ctx.builder, value_pointer)?;
                    buffer = self.gen_serialize_helper(ctx, position, struct_elem, ty, buffer)?; 
                }
                Ok(buffer)
            }
            Vector(ref elem) => {
                // Iterate over the vector and serialize each one individually...
                unimplemented!()
            }
            Dict(ref key, ref value) if !key.has_pointer() && !value.has_pointer() => {
                unimplemented!()
            }
            Dict(ref key, ref value) => {
                unimplemented!()
            }
            Unknown | Simd(_) | Function(_,_) | Builder(_, _) => unreachable!(),
        }
    }
}
