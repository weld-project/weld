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

use super::{LlvmGenerator, CodeGenExt, FunctionContext};

lazy_static! {
    /// The serialized type, which is a vec[i8].
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
            let zero = self.i64(0);
            let buffer = self.gen_new(ctx, &SER_TY, zero)?;
            let child_ty = ctx.sir_function.symbol_type(child)?;
            let child = ctx.get_value(child)?;
            let serialized = self.gen_serialize_helper(ctx,
                                                       &mut SerializePosition::new(zero),
                                                       child,
                                                       child_ty,
                                                       buffer)?;
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
            let output = ctx.get_value(output)?;
            let buffer = self.load(ctx.builder, ctx.get_value(child)?)?;
            let zero = self.i64(0);
            self.gen_deserialize_helper(ctx,
                                        &mut SerializePosition::new(zero),
                                        output,
                                        output_ty,
                                        buffer)?;
            // This function writes directly into the output, so a store afterward is not
            // necessary.
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
trait SerHelper {
    /// Copy a value into the serialization buffer.
    ///
    /// This function assumes that the value being put contains no nested pointers, and is also
    /// optimized for "small" values
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
    /// The function updates `index` to point to the last byte in the buffer if necessary. The
    /// passed value should be a pointer.
    unsafe fn gen_serialize_helper(&mut self,
                           ctx: &mut FunctionContext,
                           position: &mut SerializePosition,
                           value: LLVMValueRef,
                           ty: &Type,
                           buffer: LLVMValueRef) -> WeldResult<LLVMValueRef>;
}

impl SerHelper for LlvmGenerator {
    unsafe fn gen_put_value(&mut self,
                           ctx: &mut FunctionContext,
                           value: LLVMValueRef,
                           buffer: LLVMValueRef,
                           position: &mut SerializePosition) -> WeldResult<LLVMValueRef> {

        use codegen::llvm2::vector::VectorExt;
        let size = self.size_of(LLVMTypeOf(value));

        // Grow the vector to the required capacity.
        let required_size = LLVMBuildAdd(ctx.builder, position.index, size, c_str!(""));
        let buffer = self.gen_extend(ctx, &SER_TY, buffer, required_size)?;

        let ty = LLVMTypeOf(value);
        let pointer_ty = LLVMPointerType(ty, 0);
        let pointer = self.gen_at(ctx, &SER_TY, buffer, position.index)?;

        // Write the value.
        let pointer_typed = LLVMBuildBitCast(ctx.builder, pointer, pointer_ty, c_str!(""));
        LLVMBuildStore(ctx.builder, value, pointer_typed);

        // Update the position.
        position.index = required_size;
        Ok(buffer)
    }

    unsafe fn gen_put_values(&mut self,
                           ctx: &mut FunctionContext,
                           ptr: LLVMValueRef,
                           size: LLVMValueRef,
                           buffer: LLVMValueRef,
                           position: &mut SerializePosition) -> WeldResult<LLVMValueRef> {
        use codegen::llvm2::vector::VectorExt;

        let elem_size = self.size_of(LLVMGetElementType(LLVMTypeOf(ptr)));
        let size = LLVMBuildMul(ctx.builder, size, elem_size, c_str!(""));
        let required_size = LLVMBuildAdd(ctx.builder, position.index, size, c_str!(""));
        let buffer = self.gen_extend(ctx, &SER_TY, buffer, required_size)?;
        let pointer = self.gen_at(ctx, &SER_TY, buffer, position.index)?;

        // Write the value.
        let pointer_untyped = LLVMBuildBitCast(ctx.builder,
                                               ptr,
                                               LLVMPointerType(self.i8_type(), 0),
                                               c_str!(""));
        self.intrinsics.call_memcpy(ctx.builder, pointer, pointer_untyped, size);
        position.index = required_size;
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
                let value = self.load(ctx.builder, value)?;
                self.gen_put_value(ctx, value, buffer, position)
            }
            Struct(_) if !ty.has_pointer() => {
                // Use a memcpy intrinsic instead of a load and store for structs.
                let count = self.i64(1);
                self.gen_put_values(ctx, value, count, buffer, position)
            }
            Vector(ref elem) if !elem.has_pointer() => {
                let zero = self.i64(0);
                let value = self.load(ctx.builder, value)?;
                let vec_size = self.gen_size(ctx, ty, value)?;
                let vec_ptr = self.gen_at(ctx, ty, value, zero)?;
                // Write the 8-byte length followed by the data buffer.
                let buffer = self.gen_put_value(ctx, vec_size, buffer, position)?;
                self.gen_put_values(ctx, vec_ptr, vec_size, buffer, position)
            }
            Struct(ref tys) => {
                info!("Serializing a struct with pointers");
                let mut buffer = buffer;
                for (i, ty) in tys.iter().enumerate() {
                    let value_pointer = LLVMBuildStructGEP(ctx.builder, value, i as u32, c_str!(""));
                    buffer = self.gen_serialize_helper(ctx, position, value_pointer, ty, buffer)?;
                }
                Ok(buffer)
            }
            Vector(ref elem) => {
                use self::llvm_sys::LLVMIntPredicate::LLVMIntSGT;
                // This will, sadly, lead to some hard to read LLVM. Logically, these blocks are
                // inserted into the current SIR basic block. The loop created here will always
                // finish at `ser.end` and the builder is always guaranteed to be positioned at the
                // end of ser.end.
                let start_block = LLVMAppendBasicBlockInContext(self.context, ctx.llvm_function, c_str!("ser.start"));
                let loop_block = LLVMAppendBasicBlockInContext(self.context, ctx.llvm_function, c_str!("ser.loop"));
                let end_block = LLVMAppendBasicBlockInContext(self.context, ctx.llvm_function, c_str!("ser.end"));

                LLVMBuildBr(ctx.builder, start_block);
                LLVMPositionBuilderAtEnd(ctx.builder, start_block);

                let value = self.load(ctx.builder, value)?;

                let size = self.gen_size(ctx, ty, value)?;
                let start_buffer = self.gen_put_value(ctx, size, buffer, position)?;
                let zero = self.i64(0);
                let compare = LLVMBuildICmp(ctx.builder, LLVMIntSGT, size, zero, c_str!(""));
                LLVMBuildCondBr(ctx.builder, compare, loop_block, end_block);

                // Save reference to position so we can PHI from it later.
                //
                // The PHI stuff is a little confusing because it conflates the recursion in
                // setting position with the control flow of the generated code. Here's a summary
                // of what's happening:
                //
                // ser.start:
                //      write size with gen_put_value, which updates position.index. <- start_position is set
                //      to the value of position here, which we will call %START in *generated
                //      code*.
                //
                //  ser.loop
                //      phi_position = phi [ser.start, %START], [ser.loop, %UPDATED]
                //      position.index is set to phi_position.
                //      recursive call to serialize helper updates position.index to a variable in
                //      the generated code we will call %UPDATED (note that phi_position is
                //      actually updated after calling serializer_helper for this reason).
                //
                //  ser.end
                //      When we generate this basic block, position.index = %UPDATED.
                //      phi_position = phi [ser.start, %START], [ser.loop, %UPDATED] <- call the
                //      phi_position here %FINAL in generated code.
                //
                //      Now to get the correct position for subsequence calls, we want the value
                //      %FINAL in position.index. Hence, the position.index = phi_position.
                //
                //  The deserialization code works similarly.
                let start_position = position.index;

                // Looping block.
                LLVMPositionBuilderAtEnd(ctx.builder, loop_block);
                let i = LLVMBuildPhi(ctx.builder, self.i64_type(), c_str!(""));
                let phi_buffer = LLVMBuildPhi(ctx.builder, LLVMTypeOf(buffer), c_str!(""));
                let phi_position = LLVMBuildPhi(ctx.builder, self.i64_type(), c_str!(""));
                position.index = phi_position;

                let value_pointer = self.gen_at(ctx, ty, value, i)?;
                let updated_buffer = self.gen_serialize_helper(ctx, position, value_pointer, elem, phi_buffer)?;
                let updated_position = position.index;
                let updated_i = LLVMBuildNSWAdd(ctx.builder, i, self.i64(1), c_str!(""));
                let compare = LLVMBuildICmp(ctx.builder, LLVMIntSGT, size, updated_i, c_str!(""));
                LLVMBuildCondBr(ctx.builder, compare, loop_block, end_block);

                let mut blocks = [start_block, loop_block];

                // Set up the PHI nodes.
                let mut values = [self.i64(0), updated_i];
                LLVMAddIncoming(i, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);
                let mut values = [start_buffer, updated_buffer];
                LLVMAddIncoming(phi_buffer, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);
                let mut values = [start_position, updated_position];
                LLVMAddIncoming(phi_position, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);

                // End block.
                LLVMPositionBuilderAtEnd(ctx.builder, end_block);
                let buffer = LLVMBuildPhi(ctx.builder, LLVMTypeOf(buffer), c_str!(""));
                let mut values = [start_buffer, updated_buffer];
                LLVMAddIncoming(buffer, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);

                let phi_position = LLVMBuildPhi(ctx.builder, self.i64_type(), c_str!(""));
                let mut values = [start_position, updated_position];
                LLVMAddIncoming(phi_position, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);
                // This "backtracks" position if necessary: without the phi, it assumes that the
                // loop fired.
                position.index = phi_position;
                Ok(buffer)
            }
            Dict(ref key, ref value) if !key.has_pointer() && !value.has_pointer() => {
                unimplemented!()
            }
            Dict(_, _) => {
                unimplemented!()
            }
            Unknown | Simd(_) | Function(_,_) | Builder(_, _) => unreachable!(),
        }
    }
}


/// Helper for deserialization.
trait DeHelper {
    /// Return a typed value from the serialization buffer and update offset.
    unsafe fn gen_get_value(&mut self,
                           ctx: &mut FunctionContext,
                           ty: LLVMTypeRef,
                           buffer: LLVMValueRef,
                           position: &mut SerializePosition) -> WeldResult<LLVMValueRef>;

    /// Write a series of values from the serialization buffer and update offset.
    ///
    /// Returns the pointer.
    unsafe fn gen_get_values(&mut self,
                           ctx: &mut FunctionContext,
                           ptr: LLVMValueRef,
                           size: LLVMValueRef,
                           buffer: LLVMValueRef,
                           position: &mut SerializePosition) -> WeldResult<()>;

    /// A recursive function for deserializing a value.
    ///
    /// This function recursively deserializes `value`, assuming `value` has type `ty`.
    unsafe fn gen_deserialize_helper(&mut self,
                           ctx: &mut FunctionContext,
                           position: &mut SerializePosition,
                           output: LLVMValueRef,
                           ty: &Type,
                           buffer: LLVMValueRef) -> WeldResult<()>;
}

impl DeHelper for LlvmGenerator {
    /// Read a value into a register from the serialization buffer.
    unsafe fn gen_get_value(&mut self,
                           ctx: &mut FunctionContext,
                           ty: LLVMTypeRef,
                           buffer: LLVMValueRef,
                           position: &mut SerializePosition) -> WeldResult<LLVMValueRef> {
        use codegen::llvm2::vector::VectorExt;

        let size = self.size_of(ty);
        let pointer = self.gen_at(ctx, &SER_TY, buffer, position.index)?;

        // Write the value.
        let pointer_typed = LLVMBuildBitCast(ctx.builder, pointer, LLVMPointerType(ty, 0), c_str!(""));
        let value = self.load(ctx.builder, pointer_typed)?;

        // Update the position.
        position.index = LLVMBuildAdd(ctx.builder, position.index, size, c_str!(""));
        Ok(value)
    }

    /// Write a series of values from the serialization buffer and update offset.
    ///
    /// The passed pointer should have enough allocated space to hold the deserialized values; this
    /// method does not perform allocation.
    unsafe fn gen_get_values(&mut self,
                           ctx: &mut FunctionContext,
                           ptr: LLVMValueRef,
                           size: LLVMValueRef,
                           buffer: LLVMValueRef,
                           position: &mut SerializePosition) -> WeldResult<()> {
        use codegen::llvm2::vector::VectorExt;
        let elem_size = self.size_of(LLVMGetElementType(LLVMTypeOf(ptr)));
        let size = LLVMBuildNSWMul(ctx.builder, size, elem_size, c_str!(""));

        let pointer = self.gen_at(ctx, &SER_TY, buffer, position.index)?;

        // Write the value.
        let pointer_untyped = LLVMBuildBitCast(ctx.builder,
                                               ptr,
                                               LLVMPointerType(self.i8_type(), 0),
                                               c_str!(""));
        self.intrinsics.call_memcpy(ctx.builder, pointer_untyped, pointer, size);
        position.index = LLVMBuildAdd(ctx.builder, position.index, size, c_str!(""));
        Ok(())
    }

    unsafe fn gen_deserialize_helper(&mut self,
                           ctx: &mut FunctionContext,
                           position: &mut SerializePosition,
                           output: LLVMValueRef,
                           ty: &Type,
                           buffer: LLVMValueRef) -> WeldResult<()> {
        use codegen::llvm2::vector::{POINTER_INDEX, VectorExt};
        use ast::Type::*;
        match *ty {
            Scalar(_) => {
                let value = self.gen_get_value(ctx, LLVMGetElementType(LLVMTypeOf(output)), buffer, position)?;
                LLVMBuildStore(ctx.builder, value, output);
                Ok(())
            }
            Struct(_) if !ty.has_pointer() => {
                let one = self.i64(1);
                self.gen_get_values(ctx, output, one, buffer, position)?;
                Ok(())
            }
            Vector(ref elem) if !elem.has_pointer() => {
                let size_type = self.i64_type();
                let size = self.gen_get_value(ctx, size_type, buffer, position)?;
                let vector = self.gen_new(ctx, ty, size)?;
                let data_pointer = LLVMBuildExtractValue(ctx.builder, vector, POINTER_INDEX, c_str!(""));
                self.gen_get_values(ctx, data_pointer, size, buffer, position)?;
                LLVMBuildStore(ctx.builder, vector, output);
                Ok(())
            }
            Struct(ref tys) => {
                for (i, ty) in tys.iter().enumerate() {
                    let value_pointer = LLVMBuildStructGEP(ctx.builder, output, i as u32, c_str!(""));
                    self.gen_deserialize_helper(ctx, position, value_pointer, ty, buffer)?;
                }
                Ok(())
            }
            Vector(ref elem) => {
                use self::llvm_sys::LLVMIntPredicate::LLVMIntSGT;
                // Similar to the serialization version.
                let start_block = LLVMAppendBasicBlockInContext(self.context, ctx.llvm_function, c_str!("de.start"));
                let loop_block = LLVMAppendBasicBlockInContext(self.context, ctx.llvm_function, c_str!("de.loop"));
                let end_block = LLVMAppendBasicBlockInContext(self.context, ctx.llvm_function, c_str!("de.end"));

                LLVMBuildBr(ctx.builder, start_block);
                LLVMPositionBuilderAtEnd(ctx.builder, start_block);

                let size_type = self.i64_type();
                let size = self.gen_get_value(ctx, size_type, buffer, position)?;
                let vector = self.gen_new(ctx, ty, size)?;

                let start_position = position.index;

                let zero = self.i64(0);
                let compare = LLVMBuildICmp(ctx.builder, LLVMIntSGT, size, zero, c_str!(""));
                LLVMBuildCondBr(ctx.builder, compare, loop_block, end_block);

                // Looping block.
                LLVMPositionBuilderAtEnd(ctx.builder, loop_block);
                // phi for the loop induction variable.
                let i = LLVMBuildPhi(ctx.builder, self.i64_type(), c_str!(""));
                // phi for the position at which to read the buffer.
                let phi_position = LLVMBuildPhi(ctx.builder, self.i64_type(), c_str!(""));
                position.index = phi_position;

                let value_pointer = self.gen_at(ctx, ty, vector, i)?;
                self.gen_deserialize_helper(ctx, position, value_pointer, elem, buffer)?;
                let updated_position = position.index;

                let updated_i = LLVMBuildNSWAdd(ctx.builder, i, self.i64(1), c_str!(""));
                let compare = LLVMBuildICmp(ctx.builder, LLVMIntSGT, size, updated_i, c_str!(""));
                LLVMBuildCondBr(ctx.builder, compare, loop_block, end_block);

                let mut blocks = [start_block, loop_block];

                // Set up the PHI nodes.
                let mut values = [self.i64(0), updated_i];
                LLVMAddIncoming(i, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);
                let mut values = [start_position, updated_position];
                LLVMAddIncoming(phi_position, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);

                // End block.
                LLVMPositionBuilderAtEnd(ctx.builder, end_block);
                let phi_position = LLVMBuildPhi(ctx.builder, self.i64_type(), c_str!(""));
                let mut values = [start_position, updated_position];
                let mut blocks = [start_block, loop_block];
                LLVMAddIncoming(phi_position, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);
                LLVMBuildStore(ctx.builder, vector, output);
                position.index = phi_position;
                Ok(())
            }
            Dict(ref key, ref value) if !key.has_pointer() && !value.has_pointer() => {
                unimplemented!()
            }
            Dict(_, _) => {
                unimplemented!()
            }
            Unknown | Simd(_) | Function(_,_) | Builder(_, _) => unreachable!(),
        }
    }
}
