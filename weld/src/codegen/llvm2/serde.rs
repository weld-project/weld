//! Implements serialization and deserialization of Weld types.
//!
//! This module provides code generation for the following SIR statements:
//!
//! * `Serialize`
//! * `Deserialize`

use llvm_sys;

use std::ffi::CStr;

use crate::ast::Type::*;
use crate::ast::*;
use crate::codegen::llvm2::vector::VectorExt;
use crate::error::*;
use crate::sir::*;

use self::llvm_sys::core::*;
use self::llvm_sys::prelude::*;

use super::{CodeGenExt, FunctionContext, HasPointer, LlvmGenerator};

lazy_static! {
    /// The serialized type, which is a vec[u8].
    static ref SER_TY: Type = Type::Vector(Box::new(Type::Scalar(ScalarKind::U8)));
    /// The type returned by the serialization function.
    static ref SER_RET_TY: Type = Type::Struct(vec![SER_TY.clone(), Scalar(ScalarKind::I64)]);
}

/// Trait for generating serialization and deserialization code.
pub trait SerDeGen {
    /// Generates code to serialize a value.
    ///
    /// Code is generated at the provided function context. The code generator may add a helper
    /// function to serialize the type if one does not exist.
    unsafe fn gen_serialize(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()>;

    /// Generates code to deserialize a value.
    ///
    /// Code is generated at the provided function context. The code generator may add a helper
    /// function to serialize the type if one does not exist.
    unsafe fn gen_deserialize(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()>;
}

impl SerDeGen for LlvmGenerator {
    unsafe fn gen_serialize(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()> {
        use crate::sir::StatementKind::Serialize;
        if let Serialize(ref child) = statement.kind {
            let zero = self.i64(0);
            let buffer = self.gen_new(ctx.builder, &SER_TY, zero, ctx.get_run())?;
            let child_ty = ctx.sir_function.symbol_type(child)?;
            let child = ctx.get_value(child)?;
            let (serialized, _) = self.gen_serialize_helper(
                ctx.builder,
                zero,
                child,
                child_ty,
                buffer,
                ctx.get_run(),
            )?;
            let output = statement.output.as_ref().unwrap();
            LLVMBuildStore(ctx.builder, serialized, ctx.get_value(output)?);
            Ok(())
        } else {
            unreachable!()
        }
    }

    unsafe fn gen_deserialize(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()> {
        use crate::sir::StatementKind::Deserialize;
        if let Deserialize(ref child) = statement.kind {
            let output = statement.output.as_ref().unwrap();
            let output_ty = ctx.sir_function.symbol_type(output)?;
            let output = ctx.get_value(output)?;
            let buffer = self.load(ctx.builder, ctx.get_value(child)?)?;
            let zero = self.i64(0);
            // TODO we could add a check to ensure that the final position = the length of the
            // input buffer, and throw some inconsistency error if the two do not match.
            let _ = self.gen_deserialize_helper(
                ctx.builder,
                zero,
                output,
                output_ty,
                buffer,
                ctx.get_run(),
            )?;
            // This function writes directly into the output, so a store afterward is not
            // necessary.
            Ok(())
        } else {
            unreachable!()
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
    ///
    /// Returns the updated buffer and the end offset.
    unsafe fn gen_put_value(
        &mut self,
        builder: LLVMBuilderRef,
        value: LLVMValueRef,
        buffer: LLVMValueRef,
        run: LLVMValueRef,
        position: LLVMValueRef,
    ) -> WeldResult<(LLVMValueRef, LLVMValueRef)>;

    /// Copy a typed buffer of values into the serialization buffer using `memcpy`.
    ///
    /// The buffer should have `size` objects (i.e., the total size of the buffer pointed to by
    /// `ptr` should be sizeof(typeof(ptr)) * size), and the objects should not contain any nested
    /// pointers.
    ///
    /// Returns the updated buffer and the end offset.
    unsafe fn gen_put_values(
        &mut self,
        builder: LLVMBuilderRef,
        ptr: LLVMValueRef,
        size: LLVMValueRef,
        buffer: LLVMValueRef,
        run: LLVMValueRef,
        position: LLVMValueRef,
    ) -> WeldResult<(LLVMValueRef, LLVMValueRef)>;

    /// A recursive function for serializing a value.
    ///
    /// The serialized value is written into buffer, with the next byte being written at
    /// `position`. The passed value should be a pointer.
    ///
    /// The function returns the updated buffer and the position to write into it next.
    ///
    /// ## Notes
    ///
    /// This function uses `gen_serialize_fn` to generate functions to serialize each (sub)type. It
    /// then calls the top-level function to serialize the passed value.
    unsafe fn gen_serialize_helper(
        &mut self,
        builder: LLVMBuilderRef,
        position: LLVMValueRef,
        value: LLVMValueRef,
        ty: &Type,
        buffer: LLVMValueRef,
        run: LLVMValueRef,
    ) -> WeldResult<(LLVMValueRef, LLVMValueRef)>;

    /// Builds a serialization routine wrapped in a function.
    ///
    /// The generated function has the following signature:
    /// (SER_TY, i64, *value, RunHandle) -> { SER_TY, i64 }
    unsafe fn gen_serialize_fn(&mut self, ty: &Type) -> WeldResult<LLVMValueRef>;
}

impl SerHelper for LlvmGenerator {
    unsafe fn gen_put_value(
        &mut self,
        builder: LLVMBuilderRef,
        value: LLVMValueRef,
        buffer: LLVMValueRef,
        run: LLVMValueRef,
        position: LLVMValueRef,
    ) -> WeldResult<(LLVMValueRef, LLVMValueRef)> {
        let size = self.size_of(LLVMTypeOf(value));
        // Grow the vector to the required capacity.
        let required_size = LLVMBuildAdd(builder, position, size, c_str!(""));
        let buffer = self.gen_extend(builder, &SER_TY, buffer, required_size, run)?;

        let ty = LLVMTypeOf(value);
        let pointer_ty = LLVMPointerType(ty, 0);
        let pointer = self.gen_at(builder, &SER_TY, buffer, position)?;

        // Write the value.
        let pointer_typed = LLVMBuildBitCast(builder, pointer, pointer_ty, c_str!(""));
        LLVMBuildStore(builder, value, pointer_typed);

        Ok((buffer, required_size))
    }

    unsafe fn gen_put_values(
        &mut self,
        builder: LLVMBuilderRef,
        ptr: LLVMValueRef,
        size: LLVMValueRef,
        buffer: LLVMValueRef,
        run: LLVMValueRef,
        position: LLVMValueRef,
    ) -> WeldResult<(LLVMValueRef, LLVMValueRef)> {
        let elem_size = self.size_of(LLVMGetElementType(LLVMTypeOf(ptr)));
        let size = LLVMBuildMul(builder, size, elem_size, c_str!(""));
        let required_size = LLVMBuildAdd(builder, position, size, c_str!(""));
        let buffer = self.gen_extend(builder, &SER_TY, buffer, required_size, run)?;
        let pointer = self.gen_at(builder, &SER_TY, buffer, position)?;

        // Write the value.
        let pointer_untyped =
            LLVMBuildBitCast(builder, ptr, LLVMPointerType(self.i8_type(), 0), c_str!(""));
        self.intrinsics
            .call_memcpy(builder, pointer, pointer_untyped, size);
        Ok((buffer, required_size))
    }

    unsafe fn gen_serialize_fn(&mut self, ty: &Type) -> WeldResult<LLVMValueRef> {
        if !self.serialize_fns.contains_key(ty) {
            let llvm_ty = self.llvm_type(ty)?;
            let buffer_ty = self.llvm_type(&SER_TY)?;

            // Buffer, position, value*, run
            let mut arg_tys = [
                buffer_ty,
                self.i64_type(),
                LLVMPointerType(llvm_ty, 0),
                self.run_handle_type(),
            ];
            let ret_ty = self.llvm_type(&SER_RET_TY)?;

            let c_prefix = LLVMPrintTypeToString(llvm_ty);
            let prefix = CStr::from_ptr(c_prefix);
            let prefix = prefix.to_str().unwrap();
            let name = format!("{}.serialize", prefix);

            // Free the allocated string.
            LLVMDisposeMessage(c_prefix);

            let (function, builder, entry_block) = self.define_function(ret_ty, &mut arg_tys, name);

            // TODO Set alwaysinline

            let buffer = LLVMGetParam(function, 0);
            let position = LLVMGetParam(function, 1);
            let value = LLVMGetParam(function, 2);
            let run = LLVMGetParam(function, 3);

            let (updated_buffer, updated_position) = match *ty {
                Scalar(_) => {
                    let value = self.load(builder, value)?;
                    self.gen_put_value(builder, value, buffer, run, position)?
                }
                Struct(_) if !ty.has_pointer() => {
                    // Use a memcpy intrinsic instead of a load and store for structs.
                    let count = self.i64(1);
                    self.gen_put_values(builder, value, count, buffer, run, position)?
                }
                Vector(ref elem) if !elem.has_pointer() => {
                    // If a vector has no pointer, write the length and memcpy the buffer.
                    let zero = self.i64(0);
                    let value = self.load(builder, value)?;
                    let vec_size = self.gen_size(builder, ty, value)?;
                    let vec_ptr = self.gen_at(builder, ty, value, zero)?;
                    // Write the 8-byte length followed by the data buffer.
                    let (buffer, position) =
                        self.gen_put_value(builder, vec_size, buffer, run, position)?;
                    self.gen_put_values(builder, vec_ptr, vec_size, buffer, run, position)?
                }
                Struct(ref tys) => {
                    let mut buffer = buffer;
                    let mut position = position;
                    for (i, ty) in tys.iter().enumerate() {
                        let value_pointer =
                            LLVMBuildStructGEP(builder, value, i as u32, c_str!(""));
                        let (buffer_tmp, position_tmp) = self.gen_serialize_helper(
                            builder,
                            position,
                            value_pointer,
                            ty,
                            buffer,
                            run,
                        )?;
                        buffer = buffer_tmp;
                        position = position_tmp;
                    }
                    (buffer, position)
                }
                Vector(ref elem) => {
                    // If a vector has pointers, we need to loop through each elemend and flatten it.
                    use self::llvm_sys::LLVMIntPredicate::LLVMIntSGT;
                    // This will, sadly, lead to some hard to read LLVM. Logically, these blocks are
                    // inserted into the current SIR basic block. The loop created here will always
                    // finish at `ser.end` and the builder is always guaranteed to be positioned at the
                    // end of ser.end.
                    let start_block = LLVMAppendBasicBlockInContext(
                        self.context,
                        function,
                        c_str!("ser.vec.start"),
                    );
                    let loop_block = LLVMAppendBasicBlockInContext(
                        self.context,
                        function,
                        c_str!("ser.vec.loop"),
                    );
                    let end_block = LLVMAppendBasicBlockInContext(
                        self.context,
                        function,
                        c_str!("ser.vec.end"),
                    );

                    LLVMBuildBr(builder, start_block);
                    LLVMPositionBuilderAtEnd(builder, start_block);

                    let value = self.load(builder, value)?;

                    let size = self.gen_size(builder, ty, value)?;
                    let (start_buffer, start_position) =
                        self.gen_put_value(builder, size, buffer, run, position)?;
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
                    //      write size with gen_put_value, which updates position. <- start_position is set
                    //      to the value of position here, which we will call %START in *generated
                    //      code*.
                    //
                    //  ser.loop
                    //      phi_position = phi [ser.start, %START], [ser.loop, %UPDATED]
                    //      position is set to phi_position.
                    //      recursive call to serialize helper updates position to a variable in
                    //      the generated code we will call %UPDATED (note that phi_position is
                    //      actually updated after calling serializer_helper for this reason).
                    //
                    //  ser.end
                    //      When we generate this basic block, position = %UPDATED.
                    //      phi_position = phi [ser.start, %START], [ser.loop, %UPDATED] <- call the
                    //      phi_position here %FINAL in generated code.
                    //
                    //      Now to get the correct position for subsequence calls, we want the value
                    //      %FINAL in position. Hence, the position = phi_position.
                    //
                    //  The deserialization code works similarly.

                    // Looping block.
                    LLVMPositionBuilderAtEnd(builder, loop_block);
                    let i = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
                    let phi_buffer = LLVMBuildPhi(builder, LLVMTypeOf(buffer), c_str!(""));
                    let phi_position = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));

                    let value_pointer = self.gen_at(builder, ty, value, i)?;
                    // Serialize the element here.
                    let (updated_buffer, updated_position) = self.gen_serialize_helper(
                        builder,
                        phi_position,
                        value_pointer,
                        elem,
                        phi_buffer,
                        run,
                    )?;

                    let updated_i = LLVMBuildNSWAdd(builder, i, self.i64(1), c_str!(""));
                    let compare = LLVMBuildICmp(builder, LLVMIntSGT, size, updated_i, c_str!(""));
                    LLVMBuildCondBr(builder, compare, loop_block, end_block);

                    let mut blocks = [start_block, loop_block];

                    // Set up the PHI nodes.
                    let mut values = [self.i64(0), updated_i];
                    LLVMAddIncoming(
                        i,
                        values.as_mut_ptr(),
                        blocks.as_mut_ptr(),
                        values.len() as u32,
                    );
                    let mut values = [start_buffer, updated_buffer];
                    LLVMAddIncoming(
                        phi_buffer,
                        values.as_mut_ptr(),
                        blocks.as_mut_ptr(),
                        values.len() as u32,
                    );
                    let mut values = [start_position, updated_position];
                    LLVMAddIncoming(
                        phi_position,
                        values.as_mut_ptr(),
                        blocks.as_mut_ptr(),
                        values.len() as u32,
                    );

                    // End block.
                    LLVMPositionBuilderAtEnd(builder, end_block);
                    let buffer = LLVMBuildPhi(builder, LLVMTypeOf(buffer), c_str!(""));
                    let mut values = [start_buffer, updated_buffer];
                    LLVMAddIncoming(
                        buffer,
                        values.as_mut_ptr(),
                        blocks.as_mut_ptr(),
                        values.len() as u32,
                    );

                    let phi_position = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
                    let mut values = [start_position, updated_position];
                    LLVMAddIncoming(
                        phi_position,
                        values.as_mut_ptr(),
                        blocks.as_mut_ptr(),
                        values.len() as u32,
                    );
                    // This "backtracks" position if necessary: without the phi, it assumes that the
                    // loop fired.
                    (buffer, phi_position)
                }
                Dict(ref key, ref val) => {
                    let dictionary = self.load(builder, value)?;
                    let key_ser_fn = self.gen_serialize_fn(key)?;
                    let val_ser_fn = self.gen_serialize_fn(val)?;
                    let methods = self.dictionaries.get_mut(ty).unwrap();
                    let buffer_vector = self.vectors.get_mut(&Scalar(ScalarKind::U8)).unwrap();
                    methods.gen_serialize(
                        builder,
                        function,
                        entry_block,
                        &mut self.intrinsics,
                        buffer_vector,
                        (buffer, position, dictionary, run),
                        key_ser_fn,
                        val_ser_fn,
                    )?
                }
                Unknown | Alias(_, _) | Simd(_) | Function(_, _) | Builder(_, _) => unreachable!(),
            };

            let ret =
                LLVMBuildInsertValue(builder, LLVMGetUndef(ret_ty), updated_buffer, 0, c_str!(""));
            let ret = LLVMBuildInsertValue(builder, ret, updated_position, 1, c_str!(""));
            LLVMBuildRet(builder, ret);

            LLVMDisposeBuilder(builder);
            self.serialize_fns.insert(ty.clone(), function);
        }
        Ok(self.serialize_fns.get(ty).cloned().unwrap())
    }

    unsafe fn gen_serialize_helper(
        &mut self,
        builder: LLVMBuilderRef,
        position: LLVMValueRef,
        value: LLVMValueRef,
        ty: &Type,
        buffer: LLVMValueRef,
        run: LLVMValueRef,
    ) -> WeldResult<(LLVMValueRef, LLVMValueRef)> {
        let function = self.gen_serialize_fn(ty)?;

        // Call the function.
        let mut args = [buffer, position, value, run];
        let ret_val = LLVMBuildCall(
            builder,
            function,
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        );
        let buffer = LLVMBuildExtractValue(builder, ret_val, 0, c_str!(""));
        let position = LLVMBuildExtractValue(builder, ret_val, 1, c_str!(""));
        Ok((buffer, position))
    }
}

/// Helper for deserialization.
trait DeHelper {
    /// Return a typed value from the serialization buffer and the new offset.
    ///
    /// The typed value is stored in a register.
    unsafe fn gen_get_value(
        &mut self,
        builder: LLVMBuilderRef,
        ty: LLVMTypeRef,
        buffer: LLVMValueRef,
        position: LLVMValueRef,
    ) -> WeldResult<(LLVMValueRef, LLVMValueRef)>;

    /// Write a series of values to `ptr` from the serialization buffer.
    ///
    /// Returns the new offset.
    unsafe fn gen_get_values(
        &mut self,
        builder: LLVMBuilderRef,
        ptr: LLVMValueRef,
        size: LLVMValueRef,
        buffer: LLVMValueRef,
        position: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef>;

    /// A recursive function for deserializing a value.
    ///
    /// This function recursively deserializes `value`, assuming `value` has type `ty`.
    unsafe fn gen_deserialize_helper(
        &mut self,
        builder: LLVMBuilderRef,
        position: LLVMValueRef,
        output: LLVMValueRef,
        ty: &Type,
        buffer: LLVMValueRef,
        run: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef>;
}

impl DeHelper for LlvmGenerator {
    /// Read a value into a register from the serialization buffer.
    unsafe fn gen_get_value(
        &mut self,
        builder: LLVMBuilderRef,
        ty: LLVMTypeRef,
        buffer: LLVMValueRef,
        position: LLVMValueRef,
    ) -> WeldResult<(LLVMValueRef, LLVMValueRef)> {
        let size = self.size_of(ty);
        let pointer = self.gen_at(builder, &SER_TY, buffer, position)?;

        // Write the value.
        let pointer_typed = LLVMBuildBitCast(builder, pointer, LLVMPointerType(ty, 0), c_str!(""));
        let value = self.load(builder, pointer_typed)?;

        // Update the position.
        let new_position = LLVMBuildAdd(builder, position, size, c_str!(""));
        Ok((value, new_position))
    }

    /// Write a series of values from the serialization buffer and update offset.
    ///
    /// The passed pointer should have enough allocated space to hold the deserialized values; this
    /// method does not perform allocation.
    unsafe fn gen_get_values(
        &mut self,
        builder: LLVMBuilderRef,
        ptr: LLVMValueRef,
        size: LLVMValueRef,
        buffer: LLVMValueRef,
        position: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef> {
        let elem_size = self.size_of(LLVMGetElementType(LLVMTypeOf(ptr)));
        let size = LLVMBuildNSWMul(builder, size, elem_size, c_str!(""));

        let pointer = self.gen_at(builder, &SER_TY, buffer, position)?;

        // Write the value.
        let pointer_untyped =
            LLVMBuildBitCast(builder, ptr, LLVMPointerType(self.i8_type(), 0), c_str!(""));
        self.intrinsics
            .call_memcpy(builder, pointer_untyped, pointer, size);
        let new_position = LLVMBuildAdd(builder, position, size, c_str!(""));
        Ok(new_position)
    }

    unsafe fn gen_deserialize_helper(
        &mut self,
        builder: LLVMBuilderRef,
        position: LLVMValueRef,
        output: LLVMValueRef,
        ty: &Type,
        buffer: LLVMValueRef,
        run: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef> {
        if !self.deserialize_fns.contains_key(ty) {
            let llvm_ty = self.llvm_type(ty)?;
            let buffer_ty = self.llvm_type(&SER_TY)?;
            // Buffer, position, output*, run
            let mut arg_tys = [
                buffer_ty,
                self.i64_type(),
                LLVMTypeOf(output),
                self.run_handle_type(),
            ];
            // Return the position.
            let ret_ty = self.i64_type();

            let c_prefix = LLVMPrintTypeToString(llvm_ty);
            let prefix = CStr::from_ptr(c_prefix);
            let prefix = prefix.to_str().unwrap();
            let name = format!("{}.deserialize", prefix);

            // Free the allocated string.
            LLVMDisposeMessage(c_prefix);

            let (function, builder, entry_block) = self.define_function(ret_ty, &mut arg_tys, name);

            // TODO Set alwaysinline

            let buffer = LLVMGetParam(function, 0);
            let position = LLVMGetParam(function, 1);
            let output = LLVMGetParam(function, 2);
            let run = LLVMGetParam(function, 3);

            let updated_position = match *ty {
                Scalar(_) => {
                    let (value, position) = self.gen_get_value(
                        builder,
                        LLVMGetElementType(LLVMTypeOf(output)),
                        buffer,
                        position,
                    )?;
                    LLVMBuildStore(builder, value, output);
                    position
                }
                Struct(_) if !ty.has_pointer() => {
                    // Copy a single value of the given type.
                    let one = self.i64(1);
                    self.gen_get_values(builder, output, one, buffer, position)?
                }
                Vector(ref elem) if !elem.has_pointer() => {
                    let size_type = self.i64_type();
                    let (size, position) =
                        self.gen_get_value(builder, size_type, buffer, position)?;
                    let vector = self.gen_new(builder, ty, size, run)?;
                    let zero = self.i64(0);
                    let data_pointer = self.gen_at(builder, ty, vector, zero)?;
                    let position =
                        self.gen_get_values(builder, data_pointer, size, buffer, position)?;
                    LLVMBuildStore(builder, vector, output);
                    position
                }
                Struct(ref tys) => {
                    let mut position = position;
                    for (i, ty) in tys.iter().enumerate() {
                        let value_pointer =
                            LLVMBuildStructGEP(builder, output, i as u32, c_str!(""));
                        position = self.gen_deserialize_helper(
                            builder,
                            position,
                            value_pointer,
                            ty,
                            buffer,
                            run,
                        )?;
                    }
                    position
                }
                Vector(ref elem) => {
                    use self::llvm_sys::LLVMIntPredicate::LLVMIntSGT;
                    // Similar to the serialization version.
                    let start_block = LLVMAppendBasicBlockInContext(
                        self.context,
                        function,
                        c_str!("de.vec.start"),
                    );
                    let loop_block = LLVMAppendBasicBlockInContext(
                        self.context,
                        function,
                        c_str!("de.vec.loop"),
                    );
                    let end_block =
                        LLVMAppendBasicBlockInContext(self.context, function, c_str!("de.vec.end"));

                    LLVMBuildBr(builder, start_block);
                    LLVMPositionBuilderAtEnd(builder, start_block);

                    let size_type = self.i64_type();
                    let (size, start_position) =
                        self.gen_get_value(builder, size_type, buffer, position)?;
                    let vector = self.gen_new(builder, ty, size, run)?;

                    let zero = self.i64(0);
                    let compare = LLVMBuildICmp(builder, LLVMIntSGT, size, zero, c_str!(""));
                    LLVMBuildCondBr(builder, compare, loop_block, end_block);

                    // Looping block.
                    LLVMPositionBuilderAtEnd(builder, loop_block);
                    // phi for the loop induction variable.
                    let i = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
                    // phi for the position at which to read the buffer.
                    let phi_position = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));

                    let value_pointer = self.gen_at(builder, ty, vector, i)?;
                    let updated_position = self.gen_deserialize_helper(
                        builder,
                        phi_position,
                        value_pointer,
                        elem,
                        buffer,
                        run,
                    )?;

                    let updated_i = LLVMBuildNSWAdd(builder, i, self.i64(1), c_str!(""));
                    let compare = LLVMBuildICmp(builder, LLVMIntSGT, size, updated_i, c_str!(""));
                    LLVMBuildCondBr(builder, compare, loop_block, end_block);

                    let mut blocks = [start_block, loop_block];

                    // Set up the PHI nodes.
                    let mut values = [self.i64(0), updated_i];
                    LLVMAddIncoming(
                        i,
                        values.as_mut_ptr(),
                        blocks.as_mut_ptr(),
                        values.len() as u32,
                    );
                    let mut values = [start_position, updated_position];
                    LLVMAddIncoming(
                        phi_position,
                        values.as_mut_ptr(),
                        blocks.as_mut_ptr(),
                        values.len() as u32,
                    );

                    // End block.
                    LLVMPositionBuilderAtEnd(builder, end_block);
                    let phi_position = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
                    let mut values = [start_position, updated_position];
                    let mut blocks = [start_block, loop_block];
                    LLVMAddIncoming(
                        phi_position,
                        values.as_mut_ptr(),
                        blocks.as_mut_ptr(),
                        values.len() as u32,
                    );
                    LLVMBuildStore(builder, vector, output);
                    phi_position
                }
                Dict(ref key_ty, ref value_ty) => {
                    // Codepath for dictionary deserialization is the same with and without pointers.
                    // Dictionaries are encoded as a length followed by a list of key/value pairs. We
                    // loop over each key/value pair and add it to the dictionary.
                    //
                    // NOTE: This requires re-hashing: we could look into encoding dictionaries without
                    // having to do this.
                    use self::llvm_sys::LLVMIntPredicate::LLVMIntSGT;
                    use crate::codegen::llvm2::hash::GenHash;

                    let size_type = self.i64_type();
                    let (size, start_position) =
                        self.gen_get_value(builder, size_type, buffer, position)?;
                    // Computes the next power-of-2.
                    let capacity = self.next_pow2(builder, size);
                    let dictionary = {
                        let methods = self.dictionaries.get_mut(ty).unwrap();
                        methods.gen_new(builder, &mut self.intrinsics, capacity, run)?
                    };

                    // Build a loop that iterates over the key-value pairs.
                    //
                    // The loop logic here is again similar to the serialization of vectors with
                    // pointers/deserialization of vectors without pointers.
                    let start_block = LLVMAppendBasicBlockInContext(
                        self.context,
                        function,
                        c_str!("de.dict.start"),
                    );
                    let loop_block = LLVMAppendBasicBlockInContext(
                        self.context,
                        function,
                        c_str!("de.dict.loop"),
                    );
                    let end_block = LLVMAppendBasicBlockInContext(
                        self.context,
                        function,
                        c_str!("de.dict.end"),
                    );

                    LLVMBuildBr(builder, start_block);
                    LLVMPositionBuilderAtEnd(builder, start_block);

                    let zero = self.i64(0);
                    let compare = LLVMBuildICmp(builder, LLVMIntSGT, size, zero, c_str!(""));
                    LLVMBuildCondBr(builder, compare, loop_block, end_block);

                    // Looping block.
                    LLVMPositionBuilderAtEnd(builder, loop_block);
                    // phi for the loop induction variable.
                    let i = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
                    // phi for the position at which to read the buffer.
                    let phi_position = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));

                    let first_inst = LLVMGetFirstInstruction(entry_block);
                    let alloca_builder = LLVMCreateBuilderInContext(self.context);
                    LLVMPositionBuilderBefore(alloca_builder, first_inst);
                    let key_pointer =
                        LLVMBuildAlloca(alloca_builder, self.llvm_type(key_ty)?, c_str!(""));
                    LLVMDisposeBuilder(alloca_builder);

                    // Deserialize the key here.
                    let mut updated_position = self.gen_deserialize_helper(
                        builder,
                        phi_position,
                        key_pointer,
                        key_ty,
                        buffer,
                        run,
                    )?;

                    let hash = self.gen_hash(key_ty, builder, key_pointer, None)?;

                    let value_pointer = {
                        let value_llvm_ty = self.llvm_type(value_ty)?;
                        let zero = self.zero(value_llvm_ty);
                        let methods = self.dictionaries.get_mut(ty).unwrap();
                        let slot = methods.gen_upsert(
                            builder,
                            &mut self.intrinsics,
                            dictionary,
                            key_pointer,
                            hash,
                            zero,
                            run,
                        )?;
                        methods.slot_ty.value(builder, slot)
                    };

                    // Deserialize the value directly into the dictionary slot.
                    updated_position = self.gen_deserialize_helper(
                        builder,
                        updated_position,
                        value_pointer,
                        value_ty,
                        buffer,
                        run,
                    )?;

                    let updated_i = LLVMBuildNSWAdd(builder, i, self.i64(1), c_str!(""));
                    let compare = LLVMBuildICmp(builder, LLVMIntSGT, size, updated_i, c_str!(""));
                    LLVMBuildCondBr(builder, compare, loop_block, end_block);

                    let mut blocks = [start_block, loop_block];

                    // Set up the PHI nodes.
                    let mut values = [self.i64(0), updated_i];
                    LLVMAddIncoming(
                        i,
                        values.as_mut_ptr(),
                        blocks.as_mut_ptr(),
                        values.len() as u32,
                    );
                    let mut values = [start_position, updated_position];
                    LLVMAddIncoming(
                        phi_position,
                        values.as_mut_ptr(),
                        blocks.as_mut_ptr(),
                        values.len() as u32,
                    );

                    // End block.
                    LLVMPositionBuilderAtEnd(builder, end_block);
                    let phi_position = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
                    let mut values = [start_position, updated_position];
                    let mut blocks = [start_block, loop_block];
                    LLVMAddIncoming(
                        phi_position,
                        values.as_mut_ptr(),
                        blocks.as_mut_ptr(),
                        values.len() as u32,
                    );
                    LLVMBuildStore(builder, dictionary, output);
                    phi_position
                }
                Unknown | Alias(_, _) | Simd(_) | Function(_, _) | Builder(_, _) => unreachable!(),
            };

            LLVMBuildRet(builder, updated_position);
            LLVMDisposeBuilder(builder);
            self.deserialize_fns.insert(ty.clone(), function);
        }

        // Call the function.
        let function = self.deserialize_fns.get(ty).cloned().unwrap();
        let mut args = [buffer, position, output, run];
        Ok(LLVMBuildCall(
            builder,
            function,
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        ))
    }
}
