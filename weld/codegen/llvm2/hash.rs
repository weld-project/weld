//! Implements hashing for Weld types.
//!
//! The main trait this module exposes is `GenHash`, which provides an interface for generating a
//! hash function for the specified Weld type. Currently, this module only supports hashing with
//! CRC32 using Intel's SSE 4.2 CRC32 intrinsics. Eventually, the module is slated to use a Weld
//! configuration to choose among a collection of hash functions.

extern crate llvm_sys;

use std::ffi::CStr;

use ast::*;
use ast::Type::*;
use ast::ScalarKind::*;

use error::*;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;
use self::llvm_sys::LLVMIntPredicate::{LLVMIntNE, LLVMIntEQ};

use super::llvm_exts::*;
use super::llvm_exts::LLVMExtAttribute::*;
use super::vector::*;

use super::target::*;

use super::HasPointer;
use super::CodeGenExt;
use super::LlvmGenerator;

pub const CRC32_SEED: u32 = 0xffffffff;

/// Trait for generating hashing code.
pub trait GenHash {
    /// Generates a hash function for a type and calls it.
    /// 
    /// A hash function has the type `(T*, i32) -> i32`, where the arguments are a pointer to the
    /// value to hash and a seed, and the return value is the hash. This function generates a hash
    /// function if one does not exist for the specified Weld type, and then calls it.
    ///
    /// If a `seed` value is not provided, the default seed for the configured hash function is
    /// used.
    unsafe fn gen_hash(&mut self,
                       ty: &Type,
                       builder: LLVMBuilderRef,
                       value_pointer: LLVMValueRef,
                       seed: Option<LLVMValueRef>) -> WeldResult<LLVMValueRef>;
}


impl GenHash for LlvmGenerator {
    unsafe fn gen_hash(&mut self,
                          ty: &Type,
                          builder: LLVMBuilderRef,
                          value_pointer: LLVMValueRef,
                          seed: Option<LLVMValueRef>) -> WeldResult<LLVMValueRef> {

        if !self.hash_fns.contains_key(ty) {
            let llvm_ty = self.llvm_type(ty)?;
            let mut arg_tys = [LLVMPointerType(llvm_ty, 0), self.hash_type()];

            let ret_ty = self.hash_type();

            let c_prefix = LLVMPrintTypeToString(llvm_ty);
            let prefix = CStr::from_ptr(c_prefix);
            let prefix = prefix.to_str().unwrap();
            let name = format!("{}.hash", prefix);
            // Free the allocated string.
            LLVMDisposeMessage(c_prefix);

            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            // Always inline calls that do not generate loops.
            if !ty.has_pointer() {
                LLVMExtAddAttrsOnFunction(self.context, function, &[AlwaysInline]);
            } else {
                LLVMExtAddAttrsOnFunction(self.context, function, &[InlineHint]);
            }

            let param = LLVMGetParam(function, 0);
            let seed = LLVMGetParam(function, 1);

            // We use a CRC32 hash function using the SSE4.2 intrinsic.
            //
            // TODO This can be a dyn Trait so we can support multiple hash functions. For now,
            // since everything we care about will have SSE 4.2, this is sufficient.
            let hash = if self.target.features.x86_supports(X86Feature::SSE4_2) {
                // x86 CRC intrinsics.
                let crc64 = "llvm.x86.sse42.crc32.64.64";
                let crc32 = "llvm.x86.sse42.crc32.32.32";
                let crc16 = "llvm.x86.sse42.crc32.32.16";
                let crc8 = "llvm.x86.sse42.crc32.32.8";

                let u64_ty = self.u64_type();
                let u32_ty = self.u32_type();
                let u16_ty = self.u16_type();
                let u8_ty = self.u8_type();

                let _ = self.intrinsics.add(crc64, u64_ty, &mut [u64_ty, u64_ty]);
                let _ = self.intrinsics.add(crc32, u32_ty, &mut [u32_ty, u32_ty]);
                let _ = self.intrinsics.add(crc16, u32_ty, &mut [u32_ty, u16_ty]);
                let _ = self.intrinsics.add(crc8, u32_ty, &mut [u32_ty, u8_ty]);

                // Use the CRC-32 set of hash functions. These functions are exposed as target-specific
                // x86 intrinsics.
                let ref funcs = HashFuncs {
                    hash64: self.intrinsics.get(crc64).unwrap(),
                    hash32: self.intrinsics.get(crc32).unwrap(),
                    hash16: self.intrinsics.get(crc16).unwrap(),
                    hash8: self.intrinsics.get(crc8).unwrap(),
                };

                // Generate the hash code. We use a method similar to
                // https://github.com/apache/impala/blob/master/be/src/codegen/llvm-codegen.cc
                // (see the LlvmCodeGen::GetHashFunction function).
                self.hash(function, builder, ty, funcs, seed, param)?
            } else {
                // TODO implement a default hashing scheme such as MurMur3.
                unimplemented!()
            };

            LLVMBuildRet(builder, hash);
            LLVMDisposeBuilder(builder);

            self.hash_fns.insert(ty.clone(), function);
        }

        // Call the function.
        let function = self.hash_fns.get(ty).cloned().unwrap();
        let mut args = [value_pointer, seed.unwrap_or(self.u32(CRC32_SEED))];
        return Ok(LLVMBuildCall(builder,
                                function,
                                args.as_mut_ptr(),
                                args.len() as u32,
                                c_str!("")))
    }
}

/// A wrapper for holding hash functions for different native widths.
struct HashFuncs {
    /// 64-bit hash function.
    ///
    /// This function combines a 64-bit hash value with a 64-bit bitstring to produce a new
    /// 64-bit hash value.
    hash64: LLVMValueRef,
    /// 32-bit hash function.
    ///
    /// This function combines a 32-bit hash value with a 32-bit bitstring to produce a new
    /// 32-bit hash value.
    hash32: LLVMValueRef,
    /// 16-bit hash function.
    ///
    /// This function combines a 32-bit hash value with a 16-bit bitstring to produce a new
    /// 32-bit hash value.
    hash16: LLVMValueRef,
    /// 8-bit hash function.
    ///
    /// This function combines a 32-bit hash value with an 8-bit bitstring to produce a new
    /// 32-bit hash value.
    hash8: LLVMValueRef,
}

/// An internal trait for hashing values.
///
/// This trait takes a set of hash functions `funcs` that are used to hash values of different
/// bit-widths. The `hash` and `hash_scalar` functions are thus used as a generic implementation
/// for many different kinds of hash functions.
trait Hash {
    /// Hash a typed value given some hash functions.
    ///
    /// The initial hash value is seeded with `seed`. `param` should be a pointer that this
    /// function will load before hashing.
    unsafe fn hash(&mut self,
                   function: LLVMValueRef,
                   builder: LLVMBuilderRef,
                   ty: &Type,
                   funcs: &HashFuncs,
                   seed: LLVMValueRef,
                   param: LLVMValueRef) -> WeldResult<LLVMValueRef>;

    /// Hash a loaded scalar value.
    ///
    /// The value is hashed into the existing value `hash`, and the updated value is returned.
    unsafe fn hash_scalar(&mut self,
                          builder: LLVMBuilderRef,
                          kind: ScalarKind,
                          funcs: &HashFuncs,
                          hash: LLVMValueRef,
                          value: LLVMValueRef) -> LLVMValueRef;

    /// Build a loop to hash a value.
    ///
    /// The constructed loop uses `kind` to determine how many bytes to hash per loop iteration.
    /// For example, if kind == `I64`, then each loop iteration will load a 64-bit value and hash
    /// it.  The built loop will not hash more than `size_in_bytes` bytes (but may hash fewer bytes
    /// if the `size_in_bytes` is not a multiple of `kind.bits()`).
    ///
    /// This function returns 1) the updated hash value and 2) the number of bytes actually
    /// consumed. The number of bytes consumed will be `kind.bits() % size_in_bytes`.
    unsafe fn hash_loop(&mut self,
                        function: LLVMValueRef,
                        builder: LLVMBuilderRef,
                        kind: ScalarKind,
                        funcs: &HashFuncs,
                        size_in_bytes: LLVMValueRef,
                        base_pointer: LLVMValueRef,
                        seed: LLVMValueRef) -> WeldResult<(LLVMValueRef, LLVMValueRef)>;
}

impl Hash for LlvmGenerator {
    /// Hash a scalar.
    unsafe fn hash_scalar(&mut self,
                          builder: LLVMBuilderRef,
                          kind: ScalarKind,
                          funcs: &HashFuncs,
                          hash: LLVMValueRef,
                          value: LLVMValueRef) -> LLVMValueRef {

        if self.conf.trace_run {
            use std::ffi::CString;
            let string = CString::new(format!("Hashing bitwidth {}", kind)).unwrap();
            let rht = self.run_handle_type();
            let rht = LLVMConstPointerNull(rht);
            let _ = self.gen_print(builder, rht, string).unwrap();
        }

        match kind {
            I64 | U64 | F64 => {
                // Extend the 32-bit hash so we can pass it to the hashing function.
                let mut hash = LLVMBuildZExt(builder, hash, self.u64_type(), c_str!(""));
                let value = LLVMBuildBitCast(builder, value, self.u64_type(), c_str!(""));
                let mut args = [hash, value];
                hash = LLVMBuildCall(builder, funcs.hash64, args.as_mut_ptr(), args.len() as u32, c_str!(""));
                LLVMBuildTrunc(builder, hash, self.hash_type(), c_str!(""))
            }
            I32 | U32 | F32 => {
                let value = LLVMBuildBitCast(builder, value, self.u32_type(), c_str!(""));
                let mut args = [hash, value];
                LLVMBuildCall(builder, funcs.hash32, args.as_mut_ptr(), args.len() as u32, c_str!(""))
            }
            I16 | U16 => {
                let value = LLVMBuildBitCast(builder, value, self.u16_type(), c_str!(""));
                let mut args = [hash, value];
                LLVMBuildCall(builder, funcs.hash16, args.as_mut_ptr(), args.len() as u32, c_str!(""))
            }
            I8 | U8 => {
                let value = LLVMBuildBitCast(builder, value, self.u8_type(), c_str!(""));
                let mut args = [hash, value];
                LLVMBuildCall(builder, funcs.hash8, args.as_mut_ptr(), args.len() as u32, c_str!(""))
            }
            Bool => {
                let value = LLVMBuildZExt(builder, value, self.u8_type(), c_str!(""));
                let mut args = [hash, value];
                LLVMBuildCall(builder, funcs.hash8, args.as_mut_ptr(), args.len() as u32, c_str!(""))
            }
        }
    }

    /// Build a loop to hash a value.
    unsafe fn hash_loop(&mut self,
                        function: LLVMValueRef,
                        builder: LLVMBuilderRef,
                        kind: ScalarKind,
                        funcs: &HashFuncs,
                        size_in_bytes: LLVMValueRef,
                        base_pointer: LLVMValueRef,
                        seed: LLVMValueRef) -> WeldResult<(LLVMValueRef, LLVMValueRef)> {

        // Get the type that we hash.
        let int_ty = self.llvm_type(&Type::Scalar(kind))?;
        let bitwidth = LLVMGetIntTypeWidth(int_ty) as i64;

        let start_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));
        let loop_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));
        let end_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));

        LLVMBuildBr(builder, start_block);
        LLVMPositionBuilderAtEnd(builder, start_block);

        // Number of words in the loop = size_in_bytes / (bytes in int_ty).
        let words = LLVMBuildUDiv(builder, size_in_bytes, self.i64(bitwidth / 8), c_str!(""));

        // Cast the pointer as an T* (where T is a int with the given bitwidth).
        let word_pointer = LLVMBuildBitCast(builder, base_pointer, LLVMPointerType(int_ty, 0), c_str!(""));

        // Build the loop that hashes over word-sized blocks.
        let check = LLVMBuildICmp(builder, LLVMIntNE, words, self.i64(0), c_str!(""));
        LLVMBuildCondBr(builder, check, loop_block, end_block);
        LLVMPositionBuilderAtEnd(builder, loop_block);

        // Index variable - this traverses the buffer in sizeof(word) offsets.
        let phi_i = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
        // Hash value so far.
        let phi_hash = LLVMBuildPhi(builder, LLVMTypeOf(seed), c_str!(""));

        let mut indices = [phi_i];
        let offset_pointer = LLVMBuildInBoundsGEP(builder,
                                                  word_pointer,
                                                  indices.as_mut_ptr(),
                                                  indices.len() as u32,
                                                  c_str!(""));
        let loaded = self.load(builder, offset_pointer)?;
        let updated_hash = self.hash_scalar(builder, kind, funcs, phi_hash, loaded);

        let updated_i = LLVMBuildNSWAdd(builder, phi_i, self.i64(1), c_str!(""));
        let check2 = LLVMBuildICmp(builder, LLVMIntEQ, updated_i, words, c_str!(""));
        LLVMBuildCondBr(builder, check2, end_block, loop_block);

        let mut blocks = [start_block, loop_block];
        let mut values = [self.i64(0), updated_i];
        LLVMAddIncoming(phi_i, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);

        let mut values = [seed, updated_hash];
        LLVMAddIncoming(phi_hash, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);

        // Finish block - set result and compute number of consumed bits.
        LLVMPositionBuilderAtEnd(builder, end_block);

        let result = LLVMBuildPhi(builder, LLVMTypeOf(seed), c_str!(""));
        let mut blocks = [start_block, loop_block];
        let mut values = [seed, updated_hash];
        LLVMAddIncoming(result, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);

        // Return the number of bytes consumed and hash built so far.
        let consumed = LLVMBuildNUWMul(builder, words, self.i64(bitwidth / 8), c_str!(""));
        Ok((result, consumed))
    }

    /// Hash a typed value given some hash functions.
    unsafe fn hash(&mut self,
                   function: LLVMValueRef,
                   builder: LLVMBuilderRef,
                   ty: &Type,
                   funcs: &HashFuncs,
                   seed: LLVMValueRef,
                   param: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        let hash = match *ty {
            Scalar(kind) => {
                let loaded = self.load(builder, param)?;
                self.hash_scalar(builder, kind, funcs, seed, loaded)
            }
            Simd(_) => {
                unimplemented!()
            }
            /*  // NOTE: For now, since we use padded structs, these optimizations are not safe.
            Struct(_) if !ty.has_pointer() => {
                // The static size of the value in bits. We decompose this into a series of
                // calls to `hash_scalar`, calling the widest possible version.
                let llvm_ty = self.llvm_type(ty)?;
                let size = self.size_of_bits(llvm_ty);
                let mut remaining = size;
                let mut hash = seed;

                let bytebuf = LLVMBuildBitCast(builder, param, LLVMPointerType(self.i8_type(), 0), c_str!(""));

                while remaining > 0 {
                    // Compute the base pointer to load next.
                    let mut offset = [self.i32(((size - remaining) / 8) as i32)];
                    let pointer = LLVMBuildInBoundsGEP(builder,
                                                       bytebuf,
                                                       offset.as_mut_ptr(),
                                                       offset.len() as u32,
                                                       c_str!(""));
                    hash = if remaining >= 64 {
                        let pointer = LLVMBuildBitCast(builder, pointer, LLVMPointerType(self.i64_type(), 0), c_str!(""));
                        let loaded = self.load(builder, pointer)?;
                        remaining -= 64;
                        self.hash_scalar(builder, I64, funcs, hash, loaded)
                    } else if remaining >= 32 {
                        let pointer = LLVMBuildBitCast(builder, pointer, LLVMPointerType(self.i32_type(), 0), c_str!(""));
                        let loaded = self.load(builder, pointer)?;
                        remaining -= 32;
                        self.hash_scalar(builder, I32, funcs, hash, loaded)
                    } else if remaining >= 16 {
                        let pointer = LLVMBuildBitCast(builder, pointer, LLVMPointerType(self.i16_type(), 0), c_str!(""));
                        let loaded = self.load(builder, pointer)?;
                        remaining -= 16;
                        self.hash_scalar(builder, I16, funcs, hash, loaded)
                    } else if remaining == 8 {
                        // Already an i8*
                        let loaded = self.load(builder, pointer)?;
                        remaining -= 8;
                        self.hash_scalar(builder, I8, funcs, hash, loaded)
                    } else {
                        unreachable!()
                    };
                }
                assert_eq!(remaining, 0);
                hash
            }
            */
            Struct(ref elems) => {
                // We don't want to hash pointers, so just hash each struct element
                // individually.
                let mut hash = seed;
                for (i, ty) in elems.iter().enumerate() {
                    let pointer = LLVMBuildStructGEP(builder, param, i as u32, c_str!(""));
                    hash = self.gen_hash(ty, builder, pointer, Some(hash))?;
                }
                hash
            }
            // NOTE: Ideally, we would play this trick for any type without pointers. Padding gets
            // in our way for now, however.
            Vector(ref elem) if elem.is_scalar() => {
                // Vectors are hashed in a manner similar to structs: we use the widest available
                // hash function in a loop until all bytes are hashed. The generated code looks
                // like this (where offsets are in *bits*):
                //
                // while offset > len(vec) - 64:
                //      hash(vec[offset..offset+64])
                //      offset += 64
                // while offset > len(vec) - 32:
                //      hash(vec[offset..offset+32])
                //      offset += 32
                // while offset > len(vec) - 16:
                //      hash(vec[offset..offset+16])
                //      offset += 16
                // while offset > len(vec) - 8:
                //      offset += 8
                let elem_llvm_ty = self.llvm_type(elem)?;
                let vector = self.load(builder, param)?;
                let size = self.gen_size(builder, ty, vector)?;

                let zero = self.i64(0);
                let base_pointer = self.gen_at(builder, ty, vector, zero)?;

                // The loops we will generate. If the element type is a multiple of one of the
                // widths, we do not need to generate loops for widths smaller than the multiple.
                let mut kinds = vec![I64];
                let elem_size_bits = self.size_of_bits(elem_llvm_ty);
                if elem_size_bits % 64 != 0 {
                    kinds.push(I32);
                }
                if elem_size_bits % 32 != 0 {
                    kinds.push(I16);
                }
                if elem_size_bits % 16 != 0 {
                    kinds.push(I8);
                }

                // Total number of bytes remaining in the vector (size * sizeof(elem))
                let mut length_in_bytes = LLVMBuildNUWMul(builder,
                                                          size,
                                                          self.size_of(elem_llvm_ty),
                                                          c_str!(""));
                // The base pointer in bytes.
                let mut base_pointer = LLVMBuildBitCast(builder,
                                                        base_pointer,
                                                        LLVMPointerType(self.i8_type(), 0),
                                                        c_str!(""));
                // Holds the updated hash value
                let mut hash = seed;
                // Holds the number of bytes consumed so far (equivalently, the offset from the
                // base pointer)
                let mut consumed;

                // Generate each loop in turn.
                for kind in kinds.into_iter() {
                    // Generates the actual loop: hash holds the new hash value,
                    let (new_hash, new_consumed) = self.hash_loop(function,
                                                                  builder,
                                                                  kind,
                                                                  funcs,
                                                                  length_in_bytes,
                                                                  base_pointer,
                                                                  hash)?;

                    hash = new_hash;
                    consumed = new_consumed;

                    // Update the base pointer to point to the first un-consumed byte.
                    let mut indices = [consumed];
                    base_pointer = LLVMBuildInBoundsGEP(builder,
                                                            base_pointer,
                                                            indices.as_mut_ptr(),
                                                            indices.len() as u32,
                                                            c_str!(""));

                    // Update the remaining bytes to (PreviousRemainingBytes - BytesConsumed)
                    length_in_bytes = LLVMBuildNUWSub(builder, length_in_bytes, consumed, c_str!(""));
                }
                hash
            }
            Vector(ref elem) => {
                // Since each element has a pointer, we loop over the vector and call each
                // element's hash function.
                let start_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));
                let loop_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));
                let end_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));

                LLVMBuildBr(builder, start_block);
                LLVMPositionBuilderAtEnd(builder, start_block);

                let vector = self.load(builder, param)?;
                let size = self.gen_size(builder, ty, vector)?;

                // Build a loop that hashes each vector element.
                let check = LLVMBuildICmp(builder, LLVMIntNE, size, self.i64(0), c_str!(""));
                LLVMBuildCondBr(builder, check, loop_block, end_block);
                LLVMPositionBuilderAtEnd(builder, loop_block);

                // Induction variable.
                let phi_i = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
                // Hash value so far.
                let phi_hash = LLVMBuildPhi(builder, LLVMTypeOf(seed), c_str!(""));

                let pointer = self.gen_at(builder, ty, vector, phi_i)?;
                let updated_hash = self.gen_hash(elem, builder, pointer, Some(phi_hash))?;

                let updated_i = LLVMBuildNSWAdd(builder, phi_i, self.i64(1), c_str!(""));
                let check2 = LLVMBuildICmp(builder, LLVMIntEQ, updated_i, size, c_str!(""));
                LLVMBuildCondBr(builder, check2, end_block, loop_block);

                let mut blocks = [start_block, loop_block];
                let mut values = [self.i64(0), updated_i];
                LLVMAddIncoming(phi_i, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);

                let mut values = [seed, updated_hash];
                LLVMAddIncoming(phi_hash, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);

                // Finish block - set result and compute number of consumed bits.
                LLVMPositionBuilderAtEnd(builder, end_block);

                let result = LLVMBuildPhi(builder, LLVMTypeOf(seed), c_str!(""));
                let mut blocks = [start_block, loop_block];
                let mut values = [seed, updated_hash];
                LLVMAddIncoming(result, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);
                result
            }
            Dict(_,_) | Builder(_,_) | Function(_,_) | Unknown | Alias(_, _) => {
                return compile_err!("Unhashable type {}", ty);
            }
        };
        Ok(hash)
    }
}

