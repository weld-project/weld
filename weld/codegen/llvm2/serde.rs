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

use std::ffi::CStr;

use ast::*;
use error::*;
use sir::*;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;
use self::llvm_sys::LLVMLinkage;

use codegen::llvm2::vector::VectorExt;

use super::{LlvmGenerator, HasPointer, CodeGenExt, FunctionContext};

lazy_static! {
    /// The serialized type, which is a vec[i8].
    static ref SER_TY: Type = Type::Vector(Box::new(Type::Scalar(ScalarKind::I8)));
}

/// Trait for generating serialization and deserialization code.
pub trait SerDeGen {
    /// Generates code to serialize a value.
    ///
    /// Code is generated at the provided function context.
    unsafe fn gen_serialize(&mut self, ctx: &mut FunctionContext, statement: &Statement) -> WeldResult<()>;
    /// Generates code to deserialize a value.
    ///
    /// Code is generated at the provided function context.
    unsafe fn gen_deserialize(&mut self, ctx: &mut FunctionContext, statement: &Statement) -> WeldResult<()>;
}

impl SerDeGen for LlvmGenerator {
    unsafe fn gen_serialize(&mut self,
                            ctx: &mut FunctionContext,
                            statement: &Statement) -> WeldResult<()> {
        use sir::StatementKind::Serialize;
        use codegen::llvm2::vector::VectorExt;
        if let Serialize(ref child) = statement.kind {
            let zero = self.i64(0);
            let buffer = self.gen_new(ctx.builder, &SER_TY, zero, ctx.get_run())?;
            let child_ty = ctx.sir_function.symbol_type(child)?;
            let child = ctx.get_value(child)?;
            let serialized = self.gen_serialize_helper(ctx.llvm_function,
                                                       ctx.builder,
                                                       &mut SerializePosition::new(zero),
                                                       child,
                                                       child_ty,
                                                       buffer,
                                                       ctx.get_run())?;
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
            self.gen_deserialize_helper(ctx.llvm_function,
                                        ctx.builder,
                                        &mut SerializePosition::new(zero),
                                        output,
                                        output_ty,
                                        buffer,
                                        ctx.get_run())?;
            // This function writes directly into the output, so a store afterward is not
            // necessary.
            Ok(())
        } else {
            unreachable!()
        }
    }
}

/// A wrapper to track an offset into the serialization buffer.
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
    /// This function assumes that the value being put contains no nested pointers (and that the
    /// value is itself not a pointer), and is also optimized for "small" values, so the value is
    /// stored directly into the buffer.
    unsafe fn gen_put_value(&mut self,
                           builder: LLVMBuilderRef,
                           value: LLVMValueRef,
                           buffer: LLVMValueRef,
                           run: LLVMValueRef,
                           position: &mut SerializePosition) -> WeldResult<LLVMValueRef>;

    /// Copy a typed buffer of values into the serialization buffer using `memcpy`.
    ///
    /// The buffer should have `size` objects (i.e., the total size of the buffer pointed to by
    /// `ptr` should be sizeof(typeof(ptr)) * size), and the objects should not contain any nested
    /// pointers.
    unsafe fn gen_put_values(&mut self,
                           builder: LLVMBuilderRef,
                           ptr: LLVMValueRef,
                           size: LLVMValueRef,
                           buffer: LLVMValueRef,
                           run: LLVMValueRef,
                           position: &mut SerializePosition) -> WeldResult<LLVMValueRef>;

    /// A recursive function for serializing a value.
    ///
    /// The serialized value is written into buffer, with the next byte being written at
    /// `position.index`.  The function updates `position.index` to point to the last byte in the
    /// buffer. The passed value should be a pointer.
    unsafe fn gen_serialize_helper(&mut self,
                           llvm_function: LLVMValueRef,
                           builder: LLVMBuilderRef,
                           position: &mut SerializePosition,
                           value: LLVMValueRef,
                           ty: &Type,
                           buffer: LLVMValueRef,
                           run: LLVMValueRef) -> WeldResult<LLVMValueRef>;

    /// Builds a serialization routine wrapped in a function.
    ///
    /// This is used for data structures (currently, just the dictionary) that call into a
    /// serialize function. The signature of the generated function is (SER_TY, T*) -> void.
    unsafe fn gen_serialize_fn(&mut self, ty: &Type) -> WeldResult<LLVMValueRef>;
}

impl SerHelper for LlvmGenerator {
    unsafe fn gen_put_value(&mut self,
                           builder: LLVMBuilderRef,
                           value: LLVMValueRef,
                           buffer: LLVMValueRef,
                           run: LLVMValueRef,
                           position: &mut SerializePosition) -> WeldResult<LLVMValueRef> {
        let size = self.size_of(LLVMTypeOf(value));
        // Grow the vector to the required capacity.
        let required_size = LLVMBuildAdd(builder, position.index, size, c_str!(""));
        let buffer = self.gen_extend(builder, &SER_TY, buffer, required_size, run)?;

        let ty = LLVMTypeOf(value);
        let pointer_ty = LLVMPointerType(ty, 0);
        let pointer = self.gen_at(builder, &SER_TY, buffer, position.index)?;

        // Write the value.
        let pointer_typed = LLVMBuildBitCast(builder, pointer, pointer_ty, c_str!(""));
        LLVMBuildStore(builder, value, pointer_typed);

        // Update the position.
        position.index = required_size;
        Ok(buffer)
    }

    unsafe fn gen_put_values(&mut self,
                           builder: LLVMBuilderRef,
                           ptr: LLVMValueRef,
                           size: LLVMValueRef,
                           buffer: LLVMValueRef,
                           run: LLVMValueRef,
                           position: &mut SerializePosition) -> WeldResult<LLVMValueRef> {
        let elem_size = self.size_of(LLVMGetElementType(LLVMTypeOf(ptr)));
        let size = LLVMBuildMul(builder, size, elem_size, c_str!(""));
        let required_size = LLVMBuildAdd(builder, position.index, size, c_str!(""));
        let buffer = self.gen_extend(builder, &SER_TY, buffer, required_size, run)?;
        let pointer = self.gen_at(builder, &SER_TY, buffer, position.index)?;

        // Write the value.
        let pointer_untyped = LLVMBuildBitCast(builder,
                                               ptr,
                                               LLVMPointerType(self.i8_type(), 0),
                                               c_str!(""));
        self.intrinsics.call_memcpy(builder, pointer, pointer_untyped, size);
        position.index = required_size;
        Ok(buffer)
    }

    unsafe fn gen_serialize_fn(&mut self, ty: &Type) -> WeldResult<LLVMValueRef> {
        let llvm_ty = self.llvm_type(ty)?;
        let mut arg_tys = [
            LLVMPointerType(self.llvm_type(&SER_TY)?, 0),
            LLVMPointerType(llvm_ty, 0),
            self.run_handle_type()
        ];
        let ret_ty = self.void_type();
        let c_prefix = LLVMPrintTypeToString(llvm_ty);
        let prefix = CStr::from_ptr(c_prefix);
        let prefix = prefix.to_str().unwrap();
        let name = format!("{}.serialize", prefix);

        // Free the allocated string.
        LLVMDisposeMessage(c_prefix);

        // The serialization function has external visibility so data structures linked dynamically
        // with the runtime can access it.
        let (function, builder, _) = self.define_function_with_visibility(ret_ty,
                                                                          &mut arg_tys,
                                                                          LLVMLinkage::LLVMExternalLinkage,
                                                                          name);
        let buffer_pointer = LLVMGetParam(function, 0);
        let value_pointer = LLVMGetParam(function, 1);
        let run = LLVMGetParam(function, 2);

        let buffer = self.load(builder, buffer_pointer)?;
        let offset = self.gen_size(builder, &SER_TY, buffer)?;
        let ref mut position = SerializePosition::new(offset);
        let updated_buffer = self.gen_serialize_helper(function,
                                                       builder,
                                                       position,
                                                       value_pointer,
                                                       ty,
                                                       buffer,
                                                       run)?;
        LLVMBuildStore(builder, updated_buffer, buffer_pointer);
        LLVMBuildRetVoid(builder);

        LLVMDisposeBuilder(builder);
        Ok(function)
    }

    unsafe fn gen_serialize_helper(&mut self,
                           llvm_function: LLVMValueRef,
                           builder: LLVMBuilderRef,
                           position: &mut SerializePosition,
                           value: LLVMValueRef,
                           ty: &Type,
                           buffer: LLVMValueRef,
                           run: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        use codegen::llvm2::vector::VectorExt;
        use ast::Type::*;
        match *ty {
            Scalar(_) => {
                let value = self.load(builder, value)?;
                self.gen_put_value(builder, value, buffer, run, position)
            }
            Struct(_) if !ty.has_pointer() => {
                // Use a memcpy intrinsic instead of a load and store for structs.
                let count = self.i64(1);
                self.gen_put_values(builder, value, count, buffer, run, position)
            }
            Vector(ref elem) if !elem.has_pointer() => {
                // If a vector has no pointer, write the length and memcpy the buffer.
                let zero = self.i64(0);
                let value = self.load(builder, value)?;
                let vec_size = self.gen_size(builder, ty, value)?;
                let vec_ptr = self.gen_at(builder, ty, value, zero)?;
                // Write the 8-byte length followed by the data buffer.
                let buffer = self.gen_put_value(builder, vec_size, buffer, run, position)?;
                self.gen_put_values(builder, vec_ptr, vec_size, buffer, run, position)
            }
            Struct(ref tys) => {
                let mut buffer = buffer;
                for (i, ty) in tys.iter().enumerate() {
                    let value_pointer = LLVMBuildStructGEP(builder, value, i as u32, c_str!(""));
                    buffer = self.gen_serialize_helper(llvm_function, builder, position, value_pointer, ty, buffer, run)?;
                }
                Ok(buffer)
            }
            Vector(ref elem) => {
                // If a vector has pointers, we need to loop through each elemend and flatten it.
                use self::llvm_sys::LLVMIntPredicate::LLVMIntSGT;
                // This will, sadly, lead to some hard to read LLVM. Logically, these blocks are
                // inserted into the current SIR basic block. The loop created here will always
                // finish at `ser.end` and the builder is always guaranteed to be positioned at the
                // end of ser.end.
                let start_block = LLVMAppendBasicBlockInContext(self.context, llvm_function, c_str!("ser.vec.start"));
                let loop_block = LLVMAppendBasicBlockInContext(self.context, llvm_function, c_str!("ser.vec.loop"));
                let end_block = LLVMAppendBasicBlockInContext(self.context, llvm_function, c_str!("ser.vec.end"));

                LLVMBuildBr(builder, start_block);
                LLVMPositionBuilderAtEnd(builder, start_block);

                let value = self.load(builder, value)?;

                let size = self.gen_size(builder, ty, value)?;
                let start_buffer = self.gen_put_value(builder, size, buffer, run, position)?;
                let zero = self.i64(0);
                let compare = LLVMBuildICmp(builder, LLVMIntSGT, size, zero, c_str!(""));
                LLVMBuildCondBr(builder, compare, loop_block, end_block);

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
                LLVMPositionBuilderAtEnd(builder, loop_block);
                let i = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
                let phi_buffer = LLVMBuildPhi(builder, LLVMTypeOf(buffer), c_str!(""));
                let phi_position = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
                position.index = phi_position;

                let value_pointer = self.gen_at(builder, ty, value, i)?;
                // Serialize the element here.
                let updated_buffer = self.gen_serialize_helper(llvm_function, builder, position, value_pointer, elem, phi_buffer, run)?;
                let updated_position = position.index;
                let updated_i = LLVMBuildNSWAdd(builder, i, self.i64(1), c_str!(""));
                let compare = LLVMBuildICmp(builder, LLVMIntSGT, size, updated_i, c_str!(""));
                LLVMBuildCondBr(builder, compare, loop_block, end_block);

                let mut blocks = [start_block, loop_block];

                // Set up the PHI nodes.
                let mut values = [self.i64(0), updated_i];
                LLVMAddIncoming(i, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);
                let mut values = [start_buffer, updated_buffer];
                LLVMAddIncoming(phi_buffer, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);
                let mut values = [start_position, updated_position];
                LLVMAddIncoming(phi_position, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);

                // End block.
                LLVMPositionBuilderAtEnd(builder, end_block);
                let buffer = LLVMBuildPhi(builder, LLVMTypeOf(buffer), c_str!(""));
                let mut values = [start_buffer, updated_buffer];
                LLVMAddIncoming(buffer, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);

                let phi_position = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
                let mut values = [start_position, updated_position];
                LLVMAddIncoming(phi_position, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);
                // This "backtracks" position if necessary: without the phi, it assumes that the
                // loop fired.
                position.index = phi_position;
                Ok(buffer)
            }
            Dict(ref key, ref val) if !key.has_pointer() && !val.has_pointer() => {
                // Dictionaries have a special codepath for keys and values without pointers. The
                // keys and values are memcpy'd individually into the buffer.
                let dictionary = self.load(builder, value)?;
                let null_pointer = self.null_ptr(self.i8_type());
                let no_pointer_flag = self.bool(false);
                let result = {
                    let mut methods = self.dictionaries.get_mut(ty).unwrap();
                    methods.gen_serialize(builder,
                                          &self.dict_intrinsics,
                                          run,
                                          dictionary,
                                          buffer,
                                          no_pointer_flag,
                                          null_pointer,
                                          null_pointer)?
                };
                position.index  = self.gen_size(builder, &SER_TY, result)?;
                Ok(result)
            }
            Dict(ref key, ref val) => {
                // For dictionaries with pointers, the dictionary implementation calls into a
                // function to serialize each key and value. We thus generate an externalized key
                // and value serialization function and pass it to the dictionary's serializer.
                let dictionary = self.load(builder, value)?;
                let pointer_flag = self.bool(true);
                // Deserialization functions have the signature (SER_TY, T*) -> void
                // Generate the serialize function for the key.
                let key_ser_fn = self.gen_serialize_fn(key)?;
                let val_ser_fn = self.gen_serialize_fn(val)?;
                let key_ser_fn = LLVMBuildBitCast(builder, key_ser_fn, self.void_pointer_type(), c_str!(""));
                let val_ser_fn = LLVMBuildBitCast(builder, val_ser_fn, self.void_pointer_type(), c_str!(""));

                let result = {
                    let mut methods = self.dictionaries.get_mut(ty).unwrap();
                    methods.gen_serialize(builder,
                                          &self.dict_intrinsics,
                                          run,
                                          dictionary,
                                          buffer,
                                          pointer_flag,
                                          key_ser_fn,
                                          val_ser_fn)?
                };
                position.index  = self.gen_size(builder, &SER_TY, result)?;
                Ok(result)
            }
            Unknown | Simd(_) | Function(_,_) | Builder(_, _) => unreachable!(),
        }
    }
}


/// Helper for deserialization.
trait DeHelper {
    /// Return a typed value from the serialization buffer and update offset.
    ///
    /// The typed value is stored in a register.
    unsafe fn gen_get_value(&mut self,
                           builder: LLVMBuilderRef,
                           ty: LLVMTypeRef,
                           buffer: LLVMValueRef,
                           position: &mut SerializePosition) -> WeldResult<LLVMValueRef>;

    /// Write a series of values to `ptr` from the serialization buffer and update offset.
    ///
    /// Returns the pointer.
    unsafe fn gen_get_values(&mut self,
                           builder: LLVMBuilderRef,
                           ptr: LLVMValueRef,
                           size: LLVMValueRef,
                           buffer: LLVMValueRef,
                           position: &mut SerializePosition) -> WeldResult<()>;

    /// A recursive function for deserializing a value.
    ///
    /// This function recursively deserializes `value`, assuming `value` has type `ty`.
    unsafe fn gen_deserialize_helper(&mut self,
                           llvm_function: LLVMValueRef,
                           builder: LLVMBuilderRef,
                           position: &mut SerializePosition,
                           output: LLVMValueRef,
                           ty: &Type,
                           buffer: LLVMValueRef,
                           run: LLVMValueRef) -> WeldResult<()>;
}

impl DeHelper for LlvmGenerator {
    /// Read a value into a register from the serialization buffer.
    unsafe fn gen_get_value(&mut self,
                           builder: LLVMBuilderRef,
                           ty: LLVMTypeRef,
                           buffer: LLVMValueRef,
                           position: &mut SerializePosition) -> WeldResult<LLVMValueRef> {
        use codegen::llvm2::vector::VectorExt;

        let size = self.size_of(ty);
        let pointer = self.gen_at(builder, &SER_TY, buffer, position.index)?;

        // Write the value.
        let pointer_typed = LLVMBuildBitCast(builder, pointer, LLVMPointerType(ty, 0), c_str!(""));
        let value = self.load(builder, pointer_typed)?;

        // Update the position.
        position.index = LLVMBuildAdd(builder, position.index, size, c_str!(""));
        Ok(value)
    }

    /// Write a series of values from the serialization buffer and update offset.
    ///
    /// The passed pointer should have enough allocated space to hold the deserialized values; this
    /// method does not perform allocation.
    unsafe fn gen_get_values(&mut self,
                           builder: LLVMBuilderRef,
                           ptr: LLVMValueRef,
                           size: LLVMValueRef,
                           buffer: LLVMValueRef,
                           position: &mut SerializePosition) -> WeldResult<()> {
        use codegen::llvm2::vector::VectorExt;
        let elem_size = self.size_of(LLVMGetElementType(LLVMTypeOf(ptr)));
        let size = LLVMBuildNSWMul(builder, size, elem_size, c_str!(""));

        let pointer = self.gen_at(builder, &SER_TY, buffer, position.index)?;

        // Write the value.
        let pointer_untyped = LLVMBuildBitCast(builder,
                                               ptr,
                                               LLVMPointerType(self.i8_type(), 0),
                                               c_str!(""));
        self.intrinsics.call_memcpy(builder, pointer_untyped, pointer, size);
        position.index = LLVMBuildAdd(builder, position.index, size, c_str!(""));
        Ok(())
    }

    unsafe fn gen_deserialize_helper(&mut self,
                           llvm_function: LLVMValueRef,
                           builder: LLVMBuilderRef,
                           position: &mut SerializePosition,
                           output: LLVMValueRef,
                           ty: &Type,
                           buffer: LLVMValueRef,
                           run: LLVMValueRef) -> WeldResult<()> {
        use codegen::llvm2::vector::VectorExt;
        use ast::Type::*;
        match *ty {
            Scalar(_) => {
                let value = self.gen_get_value(builder, LLVMGetElementType(LLVMTypeOf(output)), buffer, position)?;
                LLVMBuildStore(builder, value, output);
                Ok(())
            }
            Struct(_) if !ty.has_pointer() => {
                let one = self.i64(1);
                self.gen_get_values(builder, output, one, buffer, position)?;
                Ok(())
            }
            Vector(ref elem) if !elem.has_pointer() => {
                let size_type = self.i64_type();
                let size = self.gen_get_value(builder, size_type, buffer, position)?;
                let vector = self.gen_new(builder, ty, size, run)?;
                let zero = self.i64(0);
                let data_pointer = self.gen_at(builder, ty, vector, zero)?;
                self.gen_get_values(builder, data_pointer, size, buffer, position)?;
                LLVMBuildStore(builder, vector, output);
                Ok(())
            }
            Struct(ref tys) => {
                for (i, ty) in tys.iter().enumerate() {
                    let value_pointer = LLVMBuildStructGEP(builder, output, i as u32, c_str!(""));
                    self.gen_deserialize_helper(llvm_function, builder, position, value_pointer, ty, buffer, run)?;
                }
                Ok(())
            }
            Vector(ref elem) => {
                use self::llvm_sys::LLVMIntPredicate::LLVMIntSGT;
                // Similar to the serialization version.
                let start_block = LLVMAppendBasicBlockInContext(self.context, llvm_function, c_str!("de.vec.start"));
                let loop_block = LLVMAppendBasicBlockInContext(self.context, llvm_function, c_str!("de.vec.loop"));
                let end_block = LLVMAppendBasicBlockInContext(self.context, llvm_function, c_str!("de.vec.end"));

                LLVMBuildBr(builder, start_block);
                LLVMPositionBuilderAtEnd(builder, start_block);

                let size_type = self.i64_type();
                let size = self.gen_get_value(builder, size_type, buffer, position)?;
                let vector = self.gen_new(builder, ty, size, run)?;

                let start_position = position.index;

                let zero = self.i64(0);
                let compare = LLVMBuildICmp(builder, LLVMIntSGT, size, zero, c_str!(""));
                LLVMBuildCondBr(builder, compare, loop_block, end_block);

                // Looping block.
                LLVMPositionBuilderAtEnd(builder, loop_block);
                // phi for the loop induction variable.
                let i = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
                // phi for the position at which to read the buffer.
                let phi_position = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
                position.index = phi_position;

                let value_pointer = self.gen_at(builder, ty, vector, i)?;
                self.gen_deserialize_helper(llvm_function, builder, position, value_pointer, elem, buffer, run)?;
                let updated_position = position.index;

                let updated_i = LLVMBuildNSWAdd(builder, i, self.i64(1), c_str!(""));
                let compare = LLVMBuildICmp(builder, LLVMIntSGT, size, updated_i, c_str!(""));
                LLVMBuildCondBr(builder, compare, loop_block, end_block);

                let mut blocks = [start_block, loop_block];

                // Set up the PHI nodes.
                let mut values = [self.i64(0), updated_i];
                LLVMAddIncoming(i, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);
                let mut values = [start_position, updated_position];
                LLVMAddIncoming(phi_position, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);

                // End block.
                LLVMPositionBuilderAtEnd(builder, end_block);
                let phi_position = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
                let mut values = [start_position, updated_position];
                let mut blocks = [start_block, loop_block];
                LLVMAddIncoming(phi_position, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);
                LLVMBuildStore(builder, vector, output);
                position.index = phi_position;
                Ok(())
            }
            Dict(ref key_ty, ref value_ty) => {
                // Codepath for dictionary deserialization is the same with and without pointers.
                // Dictionaries are encoded as a length followed by a list of key/value pairs. We
                // loop over each key/value pair and add it to the dictionary.
                //
                // NOTE: This requires re-hashing: we could look into encoding dictionaries without
                // having to do this.
                use self::llvm_sys::LLVMIntPredicate::LLVMIntSGT;
                use codegen::llvm2::hash::GenHash;

                let size_type = self.i64_type();
                let size = self.gen_get_value(builder, size_type, buffer, position)?;
                // Computes the next power-of-2.
                let capacity = self.next_pow2(builder, size);
                let dictionary = {
                    let mut methods = self.dictionaries.get_mut(ty).unwrap();
                    methods.gen_new(builder, &self.dict_intrinsics, run, capacity)?
                };

                // Build a loop that iterates over the key-value pairs.
                //
                // The loop logic here is again similar to the serialization of vectors with
                // pointers/deserialization of vectors without pointers.
                let start_block = LLVMAppendBasicBlockInContext(self.context, llvm_function, c_str!("de.dict.start"));
                let loop_block = LLVMAppendBasicBlockInContext(self.context, llvm_function, c_str!("de.dict.loop"));
                let end_block = LLVMAppendBasicBlockInContext(self.context, llvm_function, c_str!("de.dict.end"));

                LLVMBuildBr(builder, start_block);
                LLVMPositionBuilderAtEnd(builder, start_block);

                let start_position = position.index;

                let zero = self.i64(0);
                let compare = LLVMBuildICmp(builder, LLVMIntSGT, size, zero, c_str!(""));
                LLVMBuildCondBr(builder, compare, loop_block, end_block);

                // Looping block.
                LLVMPositionBuilderAtEnd(builder, loop_block);
                // phi for the loop induction variable.
                let i = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
                // phi for the position at which to read the buffer.
                let phi_position = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
                position.index = phi_position;

                // XXX this is hacky, we add an alloca at the top of the function.
                let entry_block = LLVMGetFirstBasicBlock(llvm_function);
                let first_inst = LLVMGetFirstInstruction(entry_block);
                let alloca_builder = LLVMCreateBuilderInContext(self.context);
                LLVMPositionBuilderBefore(alloca_builder, first_inst);
                let key_pointer = LLVMBuildAlloca(alloca_builder,
                                                  self.llvm_type(key_ty)?,
                                                  c_str!(""));
                LLVMDisposeBuilder(alloca_builder);

                // Deserialize the key here.
                self.gen_deserialize_helper(llvm_function, builder, position, key_pointer, key_ty, buffer, run)?;

                let hash = self.gen_hash(key_ty, builder, key_pointer, None)?;
                let value_pointer = {
                    let mut methods = self.dictionaries.get_mut(ty).unwrap();
                    methods.gen_get_slot(builder,
                                    &self.dict_intrinsics,
                                    run,
                                    dictionary,
                                    key_pointer,
                                    hash)?
                };

                // Deserialize the value directly into the dictionary slot.
                self.gen_deserialize_helper(llvm_function, builder, position, value_pointer, value_ty, buffer, run)?;
                let updated_position = position.index;

                let updated_i = LLVMBuildNSWAdd(builder, i, self.i64(1), c_str!(""));
                let compare = LLVMBuildICmp(builder, LLVMIntSGT, size, updated_i, c_str!(""));
                LLVMBuildCondBr(builder, compare, loop_block, end_block);

                let mut blocks = [start_block, loop_block];

                // Set up the PHI nodes.
                let mut values = [self.i64(0), updated_i];
                LLVMAddIncoming(i, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);
                let mut values = [start_position, updated_position];
                LLVMAddIncoming(phi_position, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);

                // End block.
                LLVMPositionBuilderAtEnd(builder, end_block);
                let phi_position = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
                let mut values = [start_position, updated_position];
                let mut blocks = [start_block, loop_block];
                LLVMAddIncoming(phi_position, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);
                LLVMBuildStore(builder, dictionary, output);
                position.index = phi_position;
                Ok(())
            }
            Unknown | Simd(_) | Function(_,_) | Builder(_, _) => unreachable!(),
        }
    }
}
