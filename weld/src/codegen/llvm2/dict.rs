//! A dictionary in Weld.
//!
//! # Overview
//!
//! Weld's dictionary is an open addressed, linearly probed hash table. The hash code is 32-bits
//! wide and can be any value. The dictionary methods take a hash code as input, so the hash value
//! should be generated before calling the dictionary.
//!
//! Dictionaries provide a number of key methods, whose semantics are described below.
//!
//! ### Lookup
//!
//!

use llvm_sys;

use std::ffi::CString;

use crate::error::*;

use self::llvm_sys::core::*;
use self::llvm_sys::prelude::*;
use self::llvm_sys::LLVMIntPredicate::*;
use self::llvm_sys::LLVMTypeKind;

use crate::codegen::llvm2::intrinsic::Intrinsics;
use crate::codegen::llvm2::llvm_exts::LLVMExtAttribute::*;
use crate::codegen::llvm2::llvm_exts::*;

// Need vector type for ToVec and Serialize.
use crate::codegen::llvm2::vector;
use crate::codegen::llvm2::vector::Vector;

use super::CodeGenExt;

/// Slot index of dictionary key.
const KEY_INDEX: u32 = 0;
/// Slot index of dictionary value.
const VALUE_INDEX: u32 = 1;
/// Slot index storing the 32-bit hash.
const HASH_INDEX: u32 = 2;
/// Slot index of whether a slot is filled (a single byte value).
///
/// The filled byte also indicates the *capacity* of a grouping vector in a dictionary, in a power
/// of two. The capacity of the vector is (1 << FILLED) - 1. If the filled byte is 0 (i.e., the
/// slot is not filled, the vector is uninitialized.
const FILLED_INDEX: u32 = 3;

// Dictionary Layout: { slot_array*, capacity, size }

/// Data index.
const SLOT_ARR_INDEX: u32 = 0;
/// Capacity index.
const CAP_INDEX: u32 = 1;
/// Size index.
const SIZE_INDEX: u32 = 2;

/// Maximum load factor of the dictionary (out of 10).
const MAX_LOAD_FACTOR: i64 = 7;
/// Initial capacity of the dictionary. Must be a power-of-two.
///
/// After the initial capacity is resized, the load factor should be less than the maximum allowed
/// load factor.
pub const INITIAL_CAPACITY: i64 = 16;

/// The default capacity of a grouping vector.
const DEFAULT_GROUP_CAPACITY: i64 = 8;
/// The default value of the filled byte for grouping dictionaries, after initialization.
///
/// (0x1 << DEFAULT_GROUP_FILLED) == DEFAULT_GROUP_CAPACITY.
const DEFAULT_GROUP_FILLED: i8 = 3;

/// A dictionary data structure and its associated methods.
///
/// This struct defines methods generated for a particular dictionary type. A dictionary type is
/// defined by its key and value types.
pub struct Dict {
    pub name: String,
    pub dict_ty: LLVMTypeRef,
    pub slot_ty: SlotType,
    key_comparator: LLVMValueRef,
    dict_inner_ty: LLVMTypeRef,
    context: LLVMContextRef,
    module: LLVMModuleRef,
    slot_for_key: Option<LLVMValueRef>, // DONE
    new: Option<LLVMValueRef>,          // DONE
    lookup: Option<LLVMValueRef>,       // DONE
    opt_lookup: Option<LLVMValueRef>,   // DONE
    upsert: Option<LLVMValueRef>,       // DONE
    resize: Option<LLVMValueRef>,       // DONE
    key_exists: Option<LLVMValueRef>,   // DONE
    to_vec: Option<LLVMValueRef>,       // DONE

    // For grouping
    merge_grouped: Option<LLVMValueRef>, // DONE
}

/// Extensions for grouping dictionaries (i.e., the GroupMerger).
///
/// The grouping dictionary supports grouping values into vectors. The same underlying Dict type is
/// used for a grouping dictionary, except a special merge function is used to add values to the
/// dictionaries.
///
/// It is *incorrect behavior* to use the `upsert` method on a grouping dictionary -- values should
/// be inserted using the methods in this trait instead.
pub trait GroupingDict {
    /// Merge `value` into the group for `key` with the given `hash`.
    ///
    /// This method takes a `Vector`, which holds methods for the type `vec[V]`.
    unsafe fn gen_merge_grouped(
        &mut self,
        builder: LLVMBuilderRef,
        intrinsics: &mut Intrinsics,
        group_vector: &mut Vector,
        dict: LLVMValueRef,
        key: LLVMValueRef,
        hash: LLVMValueRef,
        value: LLVMValueRef,
        run: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef>;
}

impl CodeGenExt for Dict {
    fn module(&self) -> LLVMModuleRef {
        self.module
    }

    fn context(&self) -> LLVMContextRef {
        self.context
    }
}

/// A dictionary slot.
///
/// This struct provides accessor methods local to a single dictionary slot.
pub struct SlotType {
    /// The type of the slot.
    slot_ty: LLVMTypeRef,
    /// The key type.
    key_ty: LLVMTypeRef,
    /// The value type.
    val_ty: LLVMTypeRef,
    module: LLVMModuleRef,
    context: LLVMContextRef,
}

impl CodeGenExt for SlotType {
    fn module(&self) -> LLVMModuleRef {
        self.module
    }

    fn context(&self) -> LLVMContextRef {
        self.context
    }
}

impl SlotType {
    unsafe fn new<T: AsRef<str>>(
        name: T,
        key_ty: LLVMTypeRef,
        val_ty: LLVMTypeRef,
        context: LLVMContextRef,
        module: LLVMModuleRef,
    ) -> SlotType {
        // Create a name for the dictionary.
        let c_name = CString::new(name.as_ref()).unwrap();

        // A slot is a a struct with { key, value, hash, filled }
        let mut layout = [
            key_ty,
            val_ty,
            LLVMInt32TypeInContext(context),
            LLVMInt8TypeInContext(context),
        ];

        // For consistency.
        debug_assert!(LLVMGetTypeKind(layout[KEY_INDEX as usize]) == LLVMGetTypeKind(key_ty));
        debug_assert!(LLVMGetTypeKind(layout[VALUE_INDEX as usize]) == LLVMGetTypeKind(val_ty));
        debug_assert!(
            LLVMGetTypeKind(layout[HASH_INDEX as usize]) == LLVMTypeKind::LLVMIntegerTypeKind
        );
        debug_assert!(
            LLVMGetTypeKind(layout[FILLED_INDEX as usize]) == LLVMTypeKind::LLVMIntegerTypeKind
        );

        let slot_ty = LLVMStructCreateNamed(context, c_name.as_ptr());
        LLVMStructSetBody(slot_ty, layout.as_mut_ptr(), layout.len() as u32, 0);

        SlotType {
            slot_ty,
            key_ty,
            val_ty,
            context,
            module,
        }
    }

    /// Initialize a slot by marking it as filled and setting the key/value.
    ///
    /// The slot should be a pointer, and the key and value should be loaded (i.e., should have
    /// type `key_ty` and `val_ty`).
    unsafe fn init(
        &mut self,
        builder: LLVMBuilderRef,
        slot: LLVMValueRef,
        key: LLVMValueRef,
        hash: LLVMValueRef,
        value: LLVMValueRef,
    ) {
        let key_pointer = self.key(builder, slot);
        LLVMBuildStore(builder, key, key_pointer);
        let value_pointer = self.value(builder, slot);
        LLVMBuildStore(builder, value, value_pointer);
        let hash_pointer = self.hash(builder, slot);
        LLVMBuildStore(builder, hash, hash_pointer);
        let filled_pointer = LLVMBuildStructGEP(builder, slot, FILLED_INDEX, c_str!(""));
        LLVMBuildStore(builder, self.i8(1), filled_pointer);
    }

    /// Return the key pointer for the provided slot.
    ///
    /// The slot should be a pointer.
    pub unsafe fn key(&mut self, builder: LLVMBuilderRef, value: LLVMValueRef) -> LLVMValueRef {
        LLVMBuildStructGEP(builder, value, KEY_INDEX, c_str!("slot.key"))
    }

    /// Return the value pointer for the provide slot.
    ///
    /// The slot should be a pointer.
    pub unsafe fn value(&mut self, builder: LLVMBuilderRef, value: LLVMValueRef) -> LLVMValueRef {
        LLVMBuildStructGEP(builder, value, VALUE_INDEX, c_str!("slot.value"))
    }

    /// Return the hash pointer for the provide slot.
    ///
    /// The slot should be a pointer.
    pub unsafe fn hash(&mut self, builder: LLVMBuilderRef, value: LLVMValueRef) -> LLVMValueRef {
        LLVMBuildStructGEP(builder, value, HASH_INDEX, c_str!("slot.hash"))
    }

    /// Return whether the provided slot is filled as an `i1`.
    ///
    /// The slot is filled if the byte value of the filled field is non-zero.
    ///
    /// The slot should be a pointer.
    pub unsafe fn filled(&mut self, builder: LLVMBuilderRef, value: LLVMValueRef) -> LLVMValueRef {
        let filled_pointer = LLVMBuildStructGEP(builder, value, FILLED_INDEX, c_str!(""));
        let filled = self.load(builder, filled_pointer).unwrap();
        LLVMBuildICmp(
            builder,
            LLVMIntNE,
            filled,
            self.i8(0),
            c_str!("slot.filled"),
        )
    }

    /// Return the filled value (used for grouping dictionaries).
    ///
    /// The slot should be a pointer.
    pub unsafe fn filled_value(
        &mut self,
        builder: LLVMBuilderRef,
        value: LLVMValueRef,
    ) -> LLVMValueRef {
        let filled_pointer = LLVMBuildStructGEP(builder, value, FILLED_INDEX, c_str!(""));
        LLVMBuildLoad(builder, filled_pointer, c_str!("slot.filledValue"))
    }

    /// Set the filled value to a value between 1 and 255.
    ///
    /// The filled value represents the capacity of a grouping vector in a grouping dictionary. A
    /// nonzero filled value indicates an initialized slot.
    ///
    /// NOTE: It is invalid to "unfill" a slot by changing it from a nonzero value to a zero value.
    /// We could add a debug assertion here to check that this never happens.
    pub unsafe fn set_filled(
        &mut self,
        builder: LLVMBuilderRef,
        slot: LLVMValueRef,
        value: LLVMValueRef,
    ) {
        let filled_pointer = LLVMBuildStructGEP(builder, slot, FILLED_INDEX, c_str!(""));
        LLVMBuildStore(builder, value, filled_pointer);
    }
}

impl Dict {
    /// Define a new dictionary type.
    ///
    /// The definition requires a key and a value type.
    pub unsafe fn define<T: AsRef<str>>(
        name: T,
        key_ty: LLVMTypeRef,
        key_comparator: LLVMValueRef,
        val_ty: LLVMTypeRef,
        context: LLVMContextRef,
        module: LLVMModuleRef,
    ) -> Dict {
        let c_name = CString::new(name.as_ref()).unwrap();
        let slot_ty = SlotType::new(
            format!("{}.slot", name.as_ref()),
            key_ty,
            val_ty,
            context,
            module,
        );

        // A dictionary holds an array of slots along with a capacity and size.
        let mut layout = [
            LLVMPointerType(slot_ty.slot_ty, 0),
            LLVMInt64TypeInContext(context),
            LLVMInt64TypeInContext(context),
        ];

        debug_assert!(
            LLVMGetTypeKind(layout[SLOT_ARR_INDEX as usize]) == LLVMTypeKind::LLVMPointerTypeKind
        );
        debug_assert!(
            LLVMGetTypeKind(layout[CAP_INDEX as usize]) == LLVMTypeKind::LLVMIntegerTypeKind
        );
        debug_assert!(
            LLVMGetTypeKind(layout[SIZE_INDEX as usize]) == LLVMTypeKind::LLVMIntegerTypeKind
        );

        let dict_inner_ty = LLVMStructCreateNamed(context, c_name.as_ptr());
        LLVMStructSetBody(dict_inner_ty, layout.as_mut_ptr(), layout.len() as u32, 0);

        let name = c_name.into_string().unwrap();

        Dict {
            name,
            dict_ty: LLVMPointerType(dict_inner_ty, 0),
            dict_inner_ty,
            key_comparator,
            slot_ty,
            context,
            module,
            slot_for_key: None,
            new: None,
            lookup: None,
            opt_lookup: None,
            upsert: None,
            resize: None,
            key_exists: None,
            to_vec: None,
            merge_grouped: None,
        }
    }

    unsafe fn slot_array(&mut self, builder: LLVMBuilderRef, value: LLVMValueRef) -> LLVMValueRef {
        let slot_array_ptr = LLVMBuildStructGEP(builder, value, SLOT_ARR_INDEX, c_str!(""));
        LLVMBuildLoad(builder, slot_array_ptr, c_str!("dict.slotarray"))
    }

    unsafe fn capacity(&mut self, builder: LLVMBuilderRef, value: LLVMValueRef) -> LLVMValueRef {
        let capacity_ptr = LLVMBuildStructGEP(builder, value, CAP_INDEX, c_str!(""));
        LLVMBuildLoad(builder, capacity_ptr, c_str!("dict.capacity"))
    }

    unsafe fn size(&mut self, builder: LLVMBuilderRef, value: LLVMValueRef) -> LLVMValueRef {
        let size_ptr = LLVMBuildStructGEP(builder, value, SIZE_INDEX, c_str!(""));
        LLVMBuildLoad(builder, size_ptr, c_str!("dict.size"))
    }

    /// Returns the probe index for an index value by computing the mod.
    ///
    /// For a power-of-2 hash table, this is `value & (capacity - 1)`.
    unsafe fn probe_index(
        &mut self,
        builder: LLVMBuilderRef,
        value: LLVMValueRef,
        capacity: LLVMValueRef,
    ) -> LLVMValueRef {
        let tmp = LLVMBuildSub(builder, capacity, self.i64(1), c_str!(""));
        LLVMBuildAnd(builder, value, tmp, c_str!(""))
    }

    /// Returns whether two keys are equal using their equality function.
    unsafe fn compare_keys(
        &mut self,
        builder: LLVMBuilderRef,
        slot_key: LLVMValueRef,
        key: LLVMValueRef,
    ) -> LLVMValueRef {
        let mut args = [slot_key, key];
        LLVMBuildCall(
            builder,
            self.key_comparator,
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        )
    }

    /// Returns the slot pointer at a given index into the slot array.
    unsafe fn slot_at_index(
        &mut self,
        builder: LLVMBuilderRef,
        slot_array: LLVMValueRef,
        index: LLVMValueRef,
    ) -> LLVMValueRef {
        LLVMBuildGEP(builder, slot_array, [index].as_mut_ptr(), 1, c_str!(""))
    }

    /// Returns an `i1` determining whether the dictionary should be resized.
    unsafe fn should_resize(
        &mut self,
        builder: LLVMBuilderRef,
        size: LLVMValueRef,
        capacity: LLVMValueRef,
    ) -> LLVMValueRef {
        let size = LLVMBuildNSWMul(builder, size, self.i64(10), c_str!(""));
        let capacity = LLVMBuildNSWMul(builder, capacity, self.i64(MAX_LOAD_FACTOR), c_str!(""));
        LLVMBuildICmp(builder, LLVMIntSGE, size, capacity, c_str!(""))
    }

    /// Returns a new dictionary with the given capacity.
    unsafe fn gen_new_dict_with_capacity(
        &mut self,
        builder: LLVMBuilderRef,
        intrinsics: &mut Intrinsics,
        capacity: LLVMValueRef,
        run: LLVMValueRef,
    ) -> LLVMValueRef {
        let alloc_size = LLVMBuildNSWMul(
            builder,
            capacity,
            self.size_of(self.slot_ty.slot_ty),
            c_str!(""),
        );
        let bytes = intrinsics.call_weld_run_malloc(builder, run, alloc_size, None);
        let _ = intrinsics.call_memset_zero(builder, bytes, alloc_size);
        let slot_array = LLVMBuildBitCast(
            builder,
            bytes,
            LLVMPointerType(self.slot_ty.slot_ty, 0),
            c_str!(""),
        );

        let mut dict = LLVMGetUndef(self.dict_inner_ty);
        dict = LLVMBuildInsertValue(builder, dict, slot_array, SLOT_ARR_INDEX, c_str!(""));
        dict = LLVMBuildInsertValue(builder, dict, capacity, CAP_INDEX, c_str!(""));
        dict = LLVMBuildInsertValue(builder, dict, self.i64(0), SIZE_INDEX, c_str!(""));
        dict
    }

    /// Resize the dictionary if necessary.
    ///
    /// Returns an `i1` indicating whether a resize occurred. If the dictionary was resized, slots
    /// acquired from the dictionary should be re-acquired.
    unsafe fn gen_resize(
        &mut self,
        builder: LLVMBuilderRef,
        intrinsics: &mut Intrinsics,
        dict: LLVMValueRef,
        run: LLVMValueRef,
    ) -> LLVMValueRef {
        if self.resize.is_none() {
            let mut arg_tys = [self.dict_ty, self.run_handle_type()];
            let ret_ty = self.i1_type();
            let name = format!("{}.resize", self.name);

            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias], 0);
            LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, NoCapture, NonNull], 1);

            let dict = LLVMGetParam(function, 0);
            let run = LLVMGetParam(function, 1);

            // Type of the alloca is same as `self.dict_ty`
            let new_dictionary = LLVMBuildAlloca(builder, self.dict_inner_ty, c_str!(""));

            let old_size = self.size(builder, dict);
            let old_capacity = self.capacity(builder, dict);
            let old_slot_array = self.slot_array(builder, dict);

            // Resize
            let resize_block = LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));

            // Copy loop.
            let copy_top_block =
                LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));
            let copy_cpy_block =
                LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));

            // Probe for empty slot.
            let probe_top_block =
                LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));
            let probe_chk_block =
                LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));
            let probe_bot_block =
                LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));

            // End of copy loop.
            let copy_bot_block =
                LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));
            let copy_fin_block =
                LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));

            // Exit
            let exit_block = LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));

            let should_resize = self.should_resize(builder, old_size, old_capacity);
            LLVMBuildCondBr(builder, should_resize, resize_block, exit_block);

            LLVMPositionBuilderAtEnd(builder, resize_block);

            // Resize the dictionary -- double it capacity.
            let new_capacity = LLVMBuildNSWMul(builder, old_capacity, self.i64(2), c_str!(""));
            let new_dictionary_inner =
                self.gen_new_dict_with_capacity(builder, intrinsics, new_capacity, run);
            LLVMBuildStore(builder, new_dictionary_inner, new_dictionary);

            let new_slot_array = self.slot_array(builder, new_dictionary);
            LLVMBuildBr(builder, copy_top_block);

            // Loop to copy slots:
            //
            // top:
            //  i = phi [ resizeblock, 0 ], [ bottom, i2 ]
            //  slot = slotatindex(i)
            //  br slot.filled ? copyblock : bottom
            // cpy:
            //  newslot = getempty(new_dictionary, slot.hash) XXX geherated with gen_probe_loop
            //  memcpy(slot, newslot, sizeof(slot))
            //  br bot
            // bot:
            //  i2 = i + 1
            //  br i2 == capacity : fin ? top
            // fin:
            //  free(dict.slot_array)
            //  memcpy(dict, new_dictionary, sizeof(dict))
            //  br exit
            // exit:
            //  ...

            // Top Block.
            LLVMPositionBuilderAtEnd(builder, copy_top_block);
            let index = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
            let old_slot = self.slot_at_index(builder, old_slot_array, index);
            let filled = self.slot_ty.filled(builder, old_slot);
            LLVMBuildCondBr(builder, filled, copy_cpy_block, copy_bot_block);

            // Copy Block.
            LLVMPositionBuilderAtEnd(builder, copy_cpy_block);
            let hash_pointer = self.slot_ty.hash(builder, old_slot);
            let old_slot_hash = self.load(builder, hash_pointer).unwrap();
            let new_slot = self.gen_probe_loop(
                builder,
                new_slot_array,
                new_capacity,
                old_slot_hash,
                (
                    copy_cpy_block,
                    probe_top_block,
                    probe_chk_block,
                    probe_bot_block,
                ),
                None,
            );

            let new_slot_bytes =
                LLVMBuildBitCast(builder, new_slot, self.void_pointer_type(), c_str!(""));
            let old_slot_bytes =
                LLVMBuildBitCast(builder, old_slot, self.void_pointer_type(), c_str!(""));
            let _ = intrinsics.call_memcpy(
                builder,
                new_slot_bytes,
                old_slot_bytes,
                self.size_of(self.slot_ty.slot_ty),
            );

            // Update the size.
            let size_pointer = LLVMBuildStructGEP(builder, new_dictionary, SIZE_INDEX, c_str!(""));
            let size = self.load(builder, size_pointer).unwrap();
            let new_size = LLVMBuildNSWAdd(builder, size, self.i64(1), c_str!(""));
            LLVMBuildStore(builder, new_size, size_pointer);
            LLVMBuildBr(builder, copy_bot_block);

            // Bottom Block.
            LLVMPositionBuilderAtEnd(builder, copy_bot_block);
            let update_index = LLVMBuildNSWAdd(builder, index, self.i64(1), c_str!(""));
            let finished =
                LLVMBuildICmp(builder, LLVMIntEQ, update_index, old_capacity, c_str!(""));
            LLVMBuildCondBr(builder, finished, copy_fin_block, copy_top_block);

            // Set the PHI value for the index.
            let mut blocks = [resize_block, copy_bot_block];
            let mut values = [self.i64(0), update_index];
            LLVMAddIncoming(
                index,
                values.as_mut_ptr(),
                blocks.as_mut_ptr(),
                values.len() as u32,
            );

            // Finish Block.
            LLVMPositionBuilderAtEnd(builder, copy_fin_block);
            // 1. Free Old Dictionary buffer.
            let old_slotarray_bytes = LLVMBuildBitCast(
                builder,
                old_slot_array,
                self.void_pointer_type(),
                c_str!(""),
            );
            let _ = intrinsics.call_weld_run_free(builder, run, old_slotarray_bytes);

            // 2. Move the new dictionary state to the old dictionary pointer.
            let new_dict_bytes = LLVMBuildBitCast(
                builder,
                new_dictionary,
                self.void_pointer_type(),
                c_str!(""),
            );
            let old_dict_bytes =
                LLVMBuildBitCast(builder, dict, self.void_pointer_type(), c_str!(""));
            let _ = intrinsics.call_memcpy(
                builder,
                old_dict_bytes,
                new_dict_bytes,
                self.size_of(self.dict_inner_ty),
            );
            LLVMBuildBr(builder, exit_block);

            // Finished at last! Hallelujah. Who would've thought resizing a dictionary is so hard.
            LLVMPositionBuilderAtEnd(builder, exit_block);
            LLVMBuildRet(builder, should_resize);

            self.resize = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [dict, run];
        LLVMBuildCall(
            builder,
            self.resize.unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        )
    }

    /// Generate a probe loop. The probe loop uses the provided hash to search the hash table using
    /// linear probing. If a key is provided, either the matching slot or an empty slot is
    /// returned. If a key is not provided, the first empty slot starting at the hash's index is
    /// returned.
    ///
    /// This method takes four basic blocks for loop control flow: the basic block the builder is
    /// currently positioned in, a top block, a check block to check for the key, and an exit
    /// block.
    ///
    /// When the function returns, the builder is guaranteed to be positioned at the beginning of the
    /// exit block. The function returns the LLVMValue representing the slot found with the probe.
    unsafe fn gen_probe_loop(
        &mut self,
        builder: LLVMBuilderRef,
        slot_array: LLVMValueRef,
        capacity: LLVMValueRef,
        hash: LLVMValueRef,
        blocks: (
            LLVMBasicBlockRef,
            LLVMBasicBlockRef,
            LLVMBasicBlockRef,
            LLVMBasicBlockRef,
        ),
        key: Option<LLVMValueRef>,
    ) -> LLVMValueRef {
        // Generated Code:
        //
        //   start = probe_index(hash, cap);
        //   jump top
        // top:
        //   i = phi [ start, top ] [ i2, check_key ]
        //   s = getslot(i)
        //   br s.filled ? check_key : return
        // check_key:
        //   tmp = i + 1
        //   i2 = probe_index(tmp)
        //   br s.key == key ? return : top | br top, depending on `check_key`
        // return:
        //   return s
        // }

        let (start_block, top_block, check_block, end_block) = blocks;

        let hash = LLVMBuildZExt(builder, hash, self.i64_type(), c_str!(""));
        let start_index = self.probe_index(builder, hash, capacity);
        LLVMBuildBr(builder, top_block);

        LLVMPositionBuilderAtEnd(builder, top_block);
        let index = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
        let slot = self.slot_at_index(builder, slot_array, index);

        let filled = self.slot_ty.filled(builder, slot);
        LLVMBuildCondBr(builder, filled, check_block, end_block);

        LLVMPositionBuilderAtEnd(builder, check_block);
        let update_index = LLVMBuildNSWAdd(builder, index, self.i64(1), c_str!(""));
        let update_index = self.probe_index(builder, update_index, capacity);
        if let Some(key) = key {
            let slot_key = self.slot_ty.key(builder, slot);
            let keys_eq = self.compare_keys(builder, slot_key, key);
            LLVMBuildCondBr(builder, keys_eq, end_block, top_block);
        } else {
            LLVMBuildBr(builder, top_block);
        }

        let mut blocks = [start_block, check_block];
        let mut values = [start_index, update_index];
        LLVMAddIncoming(
            index,
            values.as_mut_ptr(),
            blocks.as_mut_ptr(),
            values.len() as u32,
        );

        LLVMPositionBuilderAtEnd(builder, end_block);
        slot
    }

    /// Return the slot for a given key. The returned slot may be uninitialized if the key does not
    /// exist in the dictionary.
    ///
    /// The current algorithm assumes a power-of-two hash table and performs standard linear
    /// probing.
    unsafe fn gen_slot_for_key(
        &mut self,
        builder: LLVMBuilderRef,
        slot_array: LLVMValueRef,
        capacity: LLVMValueRef,
        hash: LLVMValueRef,
        key: LLVMValueRef,
    ) -> LLVMValueRef {
        if self.slot_for_key.is_none() {
            let mut arg_tys = [
                LLVMPointerType(self.slot_ty.slot_ty, 0),
                self.i64_type(),
                self.i32_type(),
                LLVMPointerType(self.slot_ty.key_ty, 0),
            ];
            let ret_ty = LLVMPointerType(self.slot_ty.slot_ty, 0);
            let name = format!("{}.slot_for_key", self.name);

            let (function, builder, entry_block) = self.define_function(ret_ty, &mut arg_tys, name);

            LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, NoCapture, ReadOnly], 0);
            LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, NoCapture, ReadOnly], 3);

            let slot_array = LLVMGetParam(function, 0);
            let capacity = LLVMGetParam(function, 1);
            let hash = LLVMGetParam(function, 2);
            let key = LLVMGetParam(function, 3);

            let top_block = LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));
            let check_key_block =
                LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));
            let return_block = LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));

            let slot = self.gen_probe_loop(
                builder,
                slot_array,
                capacity,
                hash,
                (entry_block, top_block, check_key_block, return_block),
                Some(key),
            );

            LLVMBuildRet(builder, slot);

            self.slot_for_key = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [slot_array, capacity, hash, key];
        LLVMBuildCall(
            builder,
            self.slot_for_key.unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        )
    }
}

/// Public API.
impl Dict {
    /// Create a new dictionary.
    ///
    /// Dictionaries are hidden behind pointers so their ABI type is always a `void*`.
    pub unsafe fn gen_new(
        &mut self,
        builder: LLVMBuilderRef,
        intrinsics: &mut Intrinsics,
        capacity: LLVMValueRef,
        run: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef> {
        if self.new.is_none() {
            let mut arg_tys = [self.i64_type(), self.run_handle_type()];
            let ret_ty = self.dict_ty;
            let name = format!("{}.new", self.name);

            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            LLVMExtAddAttrsOnReturn(self.context, function, &[NoAlias]);
            LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, NoCapture, NonNull], 1);

            let capacity = LLVMGetParam(function, 0);
            let run = LLVMGetParam(function, 1);

            // Use a minimum initial capacity.
            let capacity_too_small = LLVMBuildICmp(
                builder,
                LLVMIntSGT,
                self.i64(INITIAL_CAPACITY),
                capacity,
                c_str!(""),
            );
            let capacity = LLVMBuildSelect(
                builder,
                capacity_too_small,
                self.i64(INITIAL_CAPACITY),
                capacity,
                c_str!(""),
            );
            let dict_inner = self.gen_new_dict_with_capacity(builder, intrinsics, capacity, run);

            // Wrap the dictionary in a pointer - the external view of a dictionary is always a
            // pointer so its easier to change the internal layout.
            let alloc_size = self.size_of(self.dict_inner_ty);
            let bytes = intrinsics.call_weld_run_malloc(builder, run, alloc_size, None);
            let dict_pointer = LLVMBuildBitCast(builder, bytes, self.dict_ty, c_str!(""));
            LLVMBuildStore(builder, dict_inner, dict_pointer);
            LLVMBuildRet(builder, dict_pointer);

            LLVMDisposeBuilder(builder);
            self.new = Some(function);
        }

        let mut args = [capacity, run];
        Ok(LLVMBuildCall(
            builder,
            self.new.unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        ))
    }

    /// Returns the pointer to the slot for a key.
    ///
    /// If the key does not exist, a slot is initialized for the key, and a default value is
    /// inserted into the slot. The slot is then returned. The insertion may cause the dictionary
    /// to be resized.
    pub unsafe fn gen_upsert(
        &mut self,
        builder: LLVMBuilderRef,
        intrinsics: &mut Intrinsics,
        dict: LLVMValueRef,
        key: LLVMValueRef,
        hash: LLVMValueRef,
        default: LLVMValueRef,
        run: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef> {
        if self.upsert.is_none() {
            let mut arg_tys = [
                self.dict_ty,
                LLVMPointerType(self.slot_ty.key_ty, 0),
                self.hash_type(),
                self.slot_ty.val_ty,
                self.run_handle_type(),
            ];

            // Generated Code:
            //
            // slot_array = slot.slot_array
            // capacity = slot.capacity
            // slot = slot_for_key(slot, capacity, hash, key)
            //
            // br slot.filled ? return : set_default
            //
            // set_default:
            // key = load keypointer
            // slot.key = key           |
            // slot.hash = hash         |-- slot.init
            // slot.filled = filled     |
            // dict.size = size + 1
            // resized = dict.resize(dict)
            // br resized ? reacquire : upsert
            //
            // reacquire:
            // resized_slot = slot_for_key(slot, capacity, hash, key)
            // br upsert
            //
            // upsert:
            // upsert_slot = phi [ slot, setdefault ] [ resized_slot, require ]
            // upsert_slot.value = default
            // br return
            //
            // return:
            // return_slot = phi [ slot, entry ] [ upsert_slot, upsert ]

            let ret_ty = LLVMPointerType(self.slot_ty.slot_ty, 0);

            let name = format!("{}.upsert", self.name);
            let (function, builder, entry_block) = self.define_function(ret_ty, &mut arg_tys, name);

            LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, NoCapture, NonNull], 0);
            LLVMExtAddAttrsOnParameter(
                self.context,
                function,
                &[NoAlias, NoCapture, NonNull, ReadOnly],
                1,
            );
            LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, NoCapture, NonNull], 4);

            let set_default_block =
                LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));
            let reacquire_slot_block =
                LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));
            let upsert_block = LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));
            let return_block = LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));

            let dict = LLVMGetParam(function, 0);
            let key = LLVMGetParam(function, 1);
            let hash = LLVMGetParam(function, 2);
            let default = LLVMGetParam(function, 3);
            let run = LLVMGetParam(function, 4);

            let slot_array = self.slot_array(builder, dict);
            let capacity = self.capacity(builder, dict);
            let slot = self.gen_slot_for_key(builder, slot_array, capacity, hash, key);

            let filled = self.slot_ty.filled(builder, slot);
            LLVMBuildCondBr(builder, filled, return_block, set_default_block);

            LLVMPositionBuilderAtEnd(builder, set_default_block);
            let key_loaded = self.load(builder, key).unwrap();

            self.slot_ty.init(builder, slot, key_loaded, hash, default);

            let size_pointer = LLVMBuildStructGEP(builder, dict, SIZE_INDEX, c_str!(""));
            let size = self.load(builder, size_pointer).unwrap();
            let new_size = LLVMBuildNSWAdd(builder, size, self.i64(1), c_str!(""));
            LLVMBuildStore(builder, new_size, size_pointer);

            // Check for resize.
            let resized = self.gen_resize(builder, intrinsics, dict, run);
            LLVMBuildCondBr(builder, resized, reacquire_slot_block, upsert_block);

            LLVMPositionBuilderAtEnd(builder, reacquire_slot_block);
            // Builder was resized - we need to reacquire the slot.
            let resized_slot_array = self.slot_array(builder, dict);
            let resized_capacity = self.capacity(builder, dict);
            let resized_slot =
                self.gen_slot_for_key(builder, resized_slot_array, resized_capacity, hash, key);

            LLVMBuildBr(builder, upsert_block);
            LLVMPositionBuilderAtEnd(builder, upsert_block);

            let slot_pointer_ty = LLVMPointerType(self.slot_ty.slot_ty, 0);

            let upsert_slot = LLVMBuildPhi(builder, slot_pointer_ty, c_str!(""));
            let value_pointer = self.slot_ty.value(builder, upsert_slot);
            LLVMBuildStore(builder, default, value_pointer);
            LLVMBuildBr(builder, return_block);

            LLVMPositionBuilderAtEnd(builder, return_block);
            let return_slot = LLVMBuildPhi(builder, ret_ty, c_str!(""));
            LLVMBuildRet(builder, return_slot);

            // Set the PHI values.
            let mut blocks = [set_default_block, reacquire_slot_block];
            let mut values = [slot, resized_slot];
            LLVMAddIncoming(
                upsert_slot,
                values.as_mut_ptr(),
                blocks.as_mut_ptr(),
                values.len() as u32,
            );

            let mut blocks = [entry_block, upsert_block];
            let mut values = [slot, upsert_slot];
            LLVMAddIncoming(
                return_slot,
                values.as_mut_ptr(),
                blocks.as_mut_ptr(),
                values.len() as u32,
            );

            self.upsert = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [dict, key, hash, default, run];
        Ok(LLVMBuildCall(
            builder,
            self.upsert.unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        ))
    }

    /// Returns the slot for a key.
    ///
    /// If the key is not in the hash table, returns an uninitialized slot. It is *invalid
    /// behavior* to modify an uninitialized slothe caller should observe the `filled` value of
    /// the slot to see whether it is initialized.
    pub unsafe fn gen_opt_lookup(
        &mut self,
        builder: LLVMBuilderRef,
        dict: LLVMValueRef,
        key: LLVMValueRef,
        hash: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef> {
        if self.opt_lookup.is_none() {
            let mut arg_tys = [
                self.dict_ty,
                LLVMPointerType(self.slot_ty.key_ty, 0),
                self.hash_type(),
            ];
            let ret_ty = LLVMPointerType(self.slot_ty.slot_ty, 0);

            let name = format!("{}.optlookup", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, ReadOnly], 0);
            LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, ReadOnly], 1);

            let dict = LLVMGetParam(function, 0);
            let key = LLVMGetParam(function, 1);
            let hash = LLVMGetParam(function, 2);

            let slot_array = self.slot_array(builder, dict);
            let capacity = self.capacity(builder, dict);
            let slot = self.gen_slot_for_key(builder, slot_array, capacity, hash, key);
            LLVMBuildRet(builder, slot);

            self.opt_lookup = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [dict, key, hash];
        Ok(LLVMBuildCall(
            builder,
            self.opt_lookup.unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        ))
    }

    /// Returns the pointer to the slot for a key.
    ///
    /// If the key does not exist, throws an KeyNotFoundError.
    ///
    /// It is *invalid* to change a filled slot to be unfilled (i.e., setting a non-zero filled
    /// value to 0).
    pub unsafe fn gen_lookup(
        &mut self,
        builder: LLVMBuilderRef,
        intrinsics: &mut Intrinsics,
        dict: LLVMValueRef,
        key: LLVMValueRef,
        hash: LLVMValueRef,
        run: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef> {
        use crate::runtime::WeldRuntimeErrno::KeyNotFoundError;

        if self.lookup.is_none() {
            let mut arg_tys = [
                self.dict_ty,
                LLVMPointerType(self.slot_ty.key_ty, 0),
                self.hash_type(),
                self.run_handle_type(),
            ];
            let ret_ty = LLVMPointerType(self.slot_ty.slot_ty, 0);

            let name = format!("{}.lookup", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, ReadOnly], 0);
            LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, ReadOnly], 1);
            LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, ReadOnly], 3);

            let crash_block =
                LLVMAppendBasicBlockInContext(self.context(), function, c_str!("keynotfound"));
            let return_block =
                LLVMAppendBasicBlockInContext(self.context(), function, c_str!("return"));

            let dict = LLVMGetParam(function, 0);
            let key = LLVMGetParam(function, 1);
            let hash = LLVMGetParam(function, 2);
            let run = LLVMGetParam(function, 3);

            let slot_array = self.slot_array(builder, dict);
            let capacity = self.capacity(builder, dict);
            let slot = self.gen_slot_for_key(builder, slot_array, capacity, hash, key);

            let filled = self.slot_ty.filled(builder, slot);
            LLVMBuildCondBr(builder, filled, return_block, crash_block);

            // Crash if the key is not found.
            LLVMPositionBuilderAtEnd(builder, crash_block);
            let error = self.i64(KeyNotFoundError as i64);
            intrinsics.call_weld_run_set_errno(builder, run, error, None);
            LLVMBuildUnreachable(builder);

            LLVMPositionBuilderAtEnd(builder, return_block);
            LLVMBuildRet(builder, slot);

            self.lookup = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [dict, key, hash, run];
        Ok(LLVMBuildCall(
            builder,
            self.lookup.unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        ))
    }

    /// Returns whether a key exists.
    ///
    /// TODO This expression may become deprecated if Lookup returns a boolean to indicate whether
    /// a value is contained within a dictionary.
    pub unsafe fn gen_key_exists(
        &mut self,
        builder: LLVMBuilderRef,
        dict: LLVMValueRef,
        key: LLVMValueRef,
        hash: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef> {
        if self.key_exists.is_none() {
            let mut arg_tys = [
                self.dict_ty,
                LLVMPointerType(self.slot_ty.key_ty, 0),
                self.hash_type(),
            ];

            let ret_ty = self.bool_type(); // LLVMPointerType(self.slot_ty.slot_ty, 0);

            let name = format!("{}.keyexists", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, ReadOnly], 0);
            LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, ReadOnly], 1);

            let dict = LLVMGetParam(function, 0);
            let key = LLVMGetParam(function, 1);
            let hash = LLVMGetParam(function, 2);

            let slot_array = self.slot_array(builder, dict);
            let capacity = self.capacity(builder, dict);
            let slot = self.gen_slot_for_key(builder, slot_array, capacity, hash, key);

            let filled = self.slot_ty.filled(builder, slot);
            let filled = self.i1_to_bool(builder, filled);
            LLVMBuildRet(builder, filled);

            self.key_exists = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [dict, key, hash];
        Ok(LLVMBuildCall(
            builder,
            self.key_exists.unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        ))
    }

    /// Returns the number of keys in the dictionary.
    pub unsafe fn gen_size(
        &mut self,
        builder: LLVMBuilderRef,
        dict: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef> {
        Ok(self.size(builder, dict))
    }

    /// Converts this dictionary to a vector of key/value pairs.
    pub unsafe fn gen_to_vec(
        &mut self,
        builder: LLVMBuilderRef,
        intrinsics: &mut Intrinsics,
        kv_vector: &mut Vector,
        dict: LLVMValueRef,
        run: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef> {
        if self.to_vec.is_none() {
            let mut arg_tys = [self.dict_ty, self.run_handle_type()];
            let ret_ty = kv_vector.vector_ty;

            let name = format!("{}.tovec", self.name);
            let (function, builder, entry_block) = self.define_function(ret_ty, &mut arg_tys, name);

            LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, ReadOnly], 0);
            LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, ReadOnly], 1);

            let dict = LLVMGetParam(function, 0);
            let run = LLVMGetParam(function, 1);

            // Generated Code:
            //
            // size = dict.size
            // size_nonzero = size > 0
            // br size_nonzero, makekvvec, return0
            //
            // makekvvec:
            // vec = vec.new(size)
            // capacity = dict.capacity
            // br top
            //
            // top:
            // i = [makkvvec, 0], [other, i2]
            // j = [makkvvec, 0], [other, j3]
            // slot = getslot(i)
            // br slot.filled ? tokv : bot
            //
            // tokv:
            // vec_ptr = gep vec, j
            // key_ptr = gep vecptr, 0
            // store slot.key into key_ptr
            // val_ptr = gep vecptr, 1
            // store slot.val into val_ptr
            // j2 = j+1
            // br bot
            //
            // bot:
            // j3 = phi [tokv, j2], [top, j]
            // i++
            // finished = i == capacity
            // br finished ? top : done
            //
            // done:
            // ret = phi [ bot, vec ], [ entry, zeroinitializer]
            //

            // A constant zero-vector.
            let mut zero_vector = LLVMGetUndef(kv_vector.vector_ty);
            zero_vector = LLVMConstInsertValue(
                zero_vector,
                self.null_ptr(kv_vector.elem_ty),
                [vector::POINTER_INDEX].as_mut_ptr(),
                1,
            );
            zero_vector = LLVMConstInsertValue(
                zero_vector,
                self.i64(0),
                [vector::SIZE_INDEX].as_mut_ptr(),
                1,
            );

            let after_nullcheck_block =
                LLVMAppendBasicBlockInContext(self.context(), function, c_str!("after.nullcheck"));
            let start_convert_block =
                LLVMAppendBasicBlockInContext(self.context(), function, c_str!("start.convert"));
            let top_block =
                LLVMAppendBasicBlockInContext(self.context(), function, c_str!("loop.top"));
            let copy_kv_block =
                LLVMAppendBasicBlockInContext(self.context(), function, c_str!("copy.kv"));
            let bot_block =
                LLVMAppendBasicBlockInContext(self.context(), function, c_str!("loop.bot"));
            let return_block =
                LLVMAppendBasicBlockInContext(self.context(), function, c_str!("return"));

            // Hack: For uninitalized dictionaries (which *must* be null pointers), return a
            // zero-vector.
            let is_null = LLVMBuildICmp(
                builder,
                LLVMIntEQ,
                dict,
                self.null_ptr(self.dict_inner_ty),
                c_str!("isNull"),
            );
            LLVMBuildCondBr(builder, is_null, return_block, after_nullcheck_block);

            LLVMPositionBuilderAtEnd(builder, after_nullcheck_block);
            let size = self.size(builder, dict);

            let size_nonzero = LLVMBuildICmp(builder, LLVMIntSGT, size, self.i64(0), c_str!(""));
            LLVMBuildCondBr(builder, size_nonzero, start_convert_block, return_block);

            LLVMPositionBuilderAtEnd(builder, start_convert_block);
            let capacity = self.capacity(builder, dict);
            let slot_array = self.slot_array(builder, dict);
            let vec = kv_vector.gen_new(builder, intrinsics, run, size)?;
            LLVMBuildBr(builder, top_block);

            LLVMPositionBuilderAtEnd(builder, top_block);
            // Index into the dictionary slot array.
            let slot_arr_index = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
            // Index into the vector.
            let kv_vec_index = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));

            let slot = self.slot_at_index(builder, slot_array, slot_arr_index);
            let filled = self.slot_ty.filled(builder, slot);
            LLVMBuildCondBr(builder, filled, copy_kv_block, bot_block);

            // Copy block - copy the key/value into the vector.
            LLVMPositionBuilderAtEnd(builder, copy_kv_block);
            // Pointer to KV struct at the correct index.
            let vector_ptr = kv_vector.gen_at(builder, vec, kv_vec_index)?;

            // Copy the key.
            let vec_key_ptr = LLVMBuildStructGEP(builder, vector_ptr, 0, c_str!(""));
            let slot_key_ptr = self.slot_ty.key(builder, slot);
            let slot_key = self.load(builder, slot_key_ptr).unwrap();
            LLVMBuildStore(builder, slot_key, vec_key_ptr);

            // Copy the value.
            let vec_val_ptr = LLVMBuildStructGEP(builder, vector_ptr, 1, c_str!(""));
            let slot_val_ptr = self.slot_ty.value(builder, slot);
            let slot_val = self.load(builder, slot_val_ptr).unwrap();
            LLVMBuildStore(builder, slot_val, vec_val_ptr);

            let inc_kv_vec_index = LLVMBuildNSWAdd(builder, kv_vec_index, self.i64(1), c_str!(""));
            LLVMBuildBr(builder, bot_block);

            // Bottom of loop -- increment induction variables and loop or exit.
            LLVMPositionBuilderAtEnd(builder, bot_block);
            let new_kv_vec_index = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
            let new_slot_arr_index =
                LLVMBuildNSWAdd(builder, slot_arr_index, self.i64(1), c_str!(""));
            let finished =
                LLVMBuildICmp(builder, LLVMIntEQ, new_slot_arr_index, capacity, c_str!(""));
            LLVMBuildCondBr(builder, finished, return_block, top_block);

            // Return Block.
            LLVMPositionBuilderAtEnd(builder, return_block);
            let ret = LLVMBuildPhi(builder, kv_vector.vector_ty, c_str!(""));
            LLVMBuildRet(builder, ret);

            // Set the PHI value for the return value.
            let mut blocks = [entry_block, after_nullcheck_block, bot_block];
            let mut values = [zero_vector, zero_vector, vec];
            LLVMAddIncoming(
                ret,
                values.as_mut_ptr(),
                blocks.as_mut_ptr(),
                values.len() as u32,
            );

            // Set the PHI value for the slot array induction variable.
            let mut blocks = [start_convert_block, bot_block];
            let mut values = [self.i64(0), new_slot_arr_index];
            LLVMAddIncoming(
                slot_arr_index,
                values.as_mut_ptr(),
                blocks.as_mut_ptr(),
                values.len() as u32,
            );

            // Set the PHI value for the offset into the KV vector.
            let mut blocks = [start_convert_block, bot_block];
            let mut values = [self.i64(0), new_kv_vec_index];
            LLVMAddIncoming(
                kv_vec_index,
                values.as_mut_ptr(),
                blocks.as_mut_ptr(),
                values.len() as u32,
            );

            // Set the PHI value for the intermediate KV vector offset in bot_block.
            let mut blocks = [top_block, copy_kv_block];
            let mut values = [kv_vec_index, inc_kv_vec_index];
            LLVMAddIncoming(
                new_kv_vec_index,
                values.as_mut_ptr(),
                blocks.as_mut_ptr(),
                values.len() as u32,
            );

            self.to_vec = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [dict, run];
        Ok(LLVMBuildCall(
            builder,
            self.to_vec.unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        ))
    }

    /// Generates the serialize function for dictionaries.
    ///
    /// The serialize function takes four arguments: the serializeation buffer, the  position in
    /// the buffer, the value to serialize (i.e., the dictionary), and the run handle.
    ///
    /// This function returns the updated serialization vector.
    ///
    /// `key_ser` and `val_ser` are the serialization functions for the key and value,
    /// respectively.
    ///
    /// # Return Value
    ///
    /// Returns the updated buffer and updated position.
    pub unsafe fn gen_serialize(
        &mut self,
        builder: LLVMBuilderRef,
        function: LLVMValueRef,
        entry_block: LLVMBasicBlockRef,
        intrinsics: &mut Intrinsics,
        buffer_vector: &mut Vector,
        arguments: (LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef),
        key_ser: LLVMValueRef,
        val_ser: LLVMValueRef,
    ) -> WeldResult<(LLVMValueRef, LLVMValueRef)> {
        // Function is already defined by caller - we just need to generate code for it.

        let (buffer, position, dict, run) = arguments;

        let start_convert_block =
            LLVMAppendBasicBlockInContext(self.context(), function, c_str!("start.ser"));
        let top_block =
            LLVMAppendBasicBlockInContext(self.context(), function, c_str!("ser.loop.top"));
        let copy_kv_block =
            LLVMAppendBasicBlockInContext(self.context(), function, c_str!("copy.kv"));
        let bot_block =
            LLVMAppendBasicBlockInContext(self.context(), function, c_str!("ser.loop.bot"));
        let return_block =
            LLVMAppendBasicBlockInContext(self.context(), function, c_str!("return"));

        // Write the 8-byte size into the buffer.
        let dict_size = self.size(builder, dict);

        let dict_size_ty = LLVMTypeOf(dict_size);
        let bytes_to_write = self.size_of(dict_size_ty);

        let required_size = LLVMBuildAdd(builder, position, bytes_to_write, c_str!("capWithSize"));
        let pre_loop_buffer =
            buffer_vector.gen_extend(builder, intrinsics, run, buffer, required_size)?;

        let pointer_ty = LLVMPointerType(dict_size_ty, 0);
        let pointer = buffer_vector.gen_at(builder, pre_loop_buffer, position)?;
        let pointer_typed = LLVMBuildBitCast(builder, pointer, pointer_ty, c_str!(""));
        LLVMBuildStore(builder, dict_size, pointer_typed);

        let pre_loop_position = required_size;

        let size_nonzero = LLVMBuildICmp(
            builder,
            LLVMIntSGT,
            dict_size,
            self.i64(0),
            c_str!("sizeNonZero"),
        );
        LLVMBuildCondBr(builder, size_nonzero, start_convert_block, return_block);

        LLVMPositionBuilderAtEnd(builder, start_convert_block);
        let capacity = self.capacity(builder, dict);
        let slot_array = self.slot_array(builder, dict);
        LLVMBuildBr(builder, top_block);

        LLVMPositionBuilderAtEnd(builder, top_block);
        // Index into the dictionary slot array.
        let slot_arr_index = LLVMBuildPhi(builder, self.i64_type(), c_str!("slotArrIndex"));

        // The buffer and serialization position.
        let ser_buffer = LLVMBuildPhi(builder, buffer_vector.vector_ty, c_str!("serBuf"));
        let ser_position = LLVMBuildPhi(builder, self.i64_type(), c_str!("serPos"));

        let slot = self.slot_at_index(builder, slot_array, slot_arr_index);
        let filled = self.slot_ty.filled(builder, slot);
        LLVMBuildCondBr(builder, filled, copy_kv_block, bot_block);

        // Copy block - copy the key/value into the vector.
        LLVMPositionBuilderAtEnd(builder, copy_kv_block);

        // Serialize the key and value.
        let slot_key_ptr = self.slot_ty.key(builder, slot);
        let mut args = [ser_buffer, ser_position, slot_key_ptr, run];

        // The serializeation function returns a { buffer, position } struct.
        let buffer_and_pos = LLVMBuildCall(
            builder,
            key_ser,
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        );

        let updated_buffer =
            LLVMBuildExtractValue(builder, buffer_and_pos, 0, c_str!("updatedBuf"));
        let updated_position =
            LLVMBuildExtractValue(builder, buffer_and_pos, 1, c_str!("updatedPos"));

        let slot_val_ptr = self.slot_ty.value(builder, slot);
        let mut args = [updated_buffer, updated_position, slot_val_ptr, run];
        let buffer_and_pos = LLVMBuildCall(
            builder,
            val_ser,
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        );

        let updated_buffer =
            LLVMBuildExtractValue(builder, buffer_and_pos, 0, c_str!("updatedBuf"));
        let updated_position =
            LLVMBuildExtractValue(builder, buffer_and_pos, 1, c_str!("updatedPos"));

        LLVMBuildBr(builder, bot_block);

        // Bottom of loop -- increment induction variables and loop or exit.
        LLVMPositionBuilderAtEnd(builder, bot_block);
        let new_ser_buffer = LLVMBuildPhi(builder, buffer_vector.vector_ty, c_str!("newBuf"));
        let new_ser_position = LLVMBuildPhi(builder, self.i64_type(), c_str!("newPos"));

        let new_slot_arr_index =
            LLVMBuildNSWAdd(builder, slot_arr_index, self.i64(1), c_str!("newArrIdx"));
        let finished = LLVMBuildICmp(
            builder,
            LLVMIntEQ,
            new_slot_arr_index,
            capacity,
            c_str!("finished"),
        );
        LLVMBuildCondBr(builder, finished, return_block, top_block);

        // Return Block.
        LLVMPositionBuilderAtEnd(builder, return_block);
        let ret_buffer = LLVMBuildPhi(builder, buffer_vector.vector_ty, c_str!("retBuf"));
        let ret_position = LLVMBuildPhi(builder, self.i64_type(), c_str!("retPos"));

        // Set the PHI value for the return value.
        let mut blocks = [entry_block, bot_block];
        let mut values = [pre_loop_buffer, new_ser_buffer];
        LLVMAddIncoming(
            ret_buffer,
            values.as_mut_ptr(),
            blocks.as_mut_ptr(),
            values.len() as u32,
        );
        let mut values = [pre_loop_position, new_ser_position];
        LLVMAddIncoming(
            ret_position,
            values.as_mut_ptr(),
            blocks.as_mut_ptr(),
            values.len() as u32,
        );

        // Set the PHI value for the slot array induction variable.
        let mut blocks = [start_convert_block, bot_block];
        let mut values = [self.i64(0), new_slot_arr_index];
        LLVMAddIncoming(
            slot_arr_index,
            values.as_mut_ptr(),
            blocks.as_mut_ptr(),
            values.len() as u32,
        );

        // Set the PHI value for the offset into the KV vector.
        let mut blocks = [start_convert_block, bot_block];
        let mut values = [pre_loop_position, new_ser_position];
        LLVMAddIncoming(
            ser_position,
            values.as_mut_ptr(),
            blocks.as_mut_ptr(),
            values.len() as u32,
        );

        // Set the PHI value for the intermediate KV vector offset in bot_block.
        let mut blocks = [top_block, copy_kv_block];
        let mut values = [ser_position, updated_position];
        LLVMAddIncoming(
            new_ser_position,
            values.as_mut_ptr(),
            blocks.as_mut_ptr(),
            values.len() as u32,
        );

        // Set the PHI value for the buffer.
        let mut blocks = [start_convert_block, bot_block];
        let mut values = [pre_loop_buffer, new_ser_buffer];
        LLVMAddIncoming(
            ser_buffer,
            values.as_mut_ptr(),
            blocks.as_mut_ptr(),
            values.len() as u32,
        );

        // Set the PHI value for the intermediate buffer in bot_block.
        let mut blocks = [top_block, copy_kv_block];
        let mut values = [ser_buffer, updated_buffer];
        LLVMAddIncoming(
            new_ser_buffer,
            values.as_mut_ptr(),
            blocks.as_mut_ptr(),
            values.len() as u32,
        );

        // Return the values: the calling serialization code will package them into a struct and
        // return them.
        Ok((ret_buffer, ret_position))
    }
}

impl GroupingDict for Dict {
    /// Merge `value` into the group for `key` with the given `hash`.
    ///
    /// This method takes a `Vector`, which holds methods for the type `vec[V]`.
    unsafe fn gen_merge_grouped(
        &mut self,
        builder: LLVMBuilderRef,
        intrinsics: &mut Intrinsics,
        group_vector: &mut Vector,
        dict: LLVMValueRef,
        key: LLVMValueRef,
        hash: LLVMValueRef,
        value: LLVMValueRef,
        run: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef> {
        if self.merge_grouped.is_none() {
            let mut arg_tys = [
                self.dict_ty,
                LLVMPointerType(self.slot_ty.key_ty, 0),
                self.hash_type(),
                group_vector.elem_ty,
                self.run_handle_type(),
            ];
            let ret_ty = self.void_type();

            let name = format!("{}.merge_grouped", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let dict = LLVMGetParam(function, 0);
            let key = LLVMGetParam(function, 1);
            let hash = LLVMGetParam(function, 2);
            let value = LLVMGetParam(function, 3);
            let run = LLVMGetParam(function, 4);

            LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias], 0);
            LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias], 1);
            LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias], 4);

            let init_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!("init"));
            let checkcap_block =
                LLVMAppendBasicBlockInContext(self.context, function, c_str!("checkcapacity"));
            let resize_block =
                LLVMAppendBasicBlockInContext(self.context, function, c_str!("resize"));
            let merge_block =
                LLVMAppendBasicBlockInContext(self.context, function, c_str!("merge"));

            let mut zero_vector = LLVMGetUndef(group_vector.vector_ty);
            zero_vector = LLVMConstInsertValue(
                zero_vector,
                self.null_ptr(group_vector.elem_ty),
                [vector::POINTER_INDEX].as_mut_ptr(),
                1,
            );
            zero_vector = LLVMConstInsertValue(
                zero_vector,
                self.i64(0),
                [vector::SIZE_INDEX].as_mut_ptr(),
                1,
            );

            // Upsert the zero vector. This will create a slot if one does not exist already. We
            // can then initialize the grouping vector if necessary, and append the new value to
            // the grouping vector.
            //
            // Generated Code:
            //
            //  slot = upsert(dict, key, hash, zeroinitializer, run)
            //  filled = slot.filled
            //  if filled == 1 ? init : checkcapacity
            // init:
            //  vec = vector.new()
            //  store vec into slot.value
            //  filled = log2(DEFAULT_CAPACITY)
            //  br merge
            //
            // checkcapacity:
            //  capacity = (1 << filled)
            //  vector = slot.value
            //  if capacity <= vector.size ? resize : merge
            //
            // resize:
            //  vector.extend
            //  filled ++
            //  br merge
            //
            // merge:
            //  size = vector.size
            //  ptr = vector.at(size)
            //  store value at ptr

            let slot = self.gen_upsert(builder, intrinsics, dict, key, hash, zero_vector, run)?;

            let value_pointer = self.slot_ty.value(builder, slot);

            // Array's size pointer. This points into the slot's value.
            let size_pointer = LLVMBuildStructGEP(
                builder,
                value_pointer,
                vector::SIZE_INDEX,
                c_str!("sizePtr"),
            );
            let filled_value = self.slot_ty.filled_value(builder, slot);

            // If capacity == 1, the slot was just initialized via the upsert. We need to
            // initialize the vector.
            let was_initialized = LLVMBuildICmp(
                builder,
                LLVMIntEQ,
                filled_value,
                self.i8(1),
                c_str!("initialized"),
            );
            LLVMBuildCondBr(builder, was_initialized, init_block, checkcap_block);

            // Initalize the vector with the default capacity. The vector capacity must be a
            // power-of-2 so we can represent it using the filled byte.
            LLVMPositionBuilderAtEnd(builder, init_block);
            let new_vector =
                group_vector.gen_new(builder, intrinsics, run, self.i64(DEFAULT_GROUP_CAPACITY))?;
            LLVMBuildStore(builder, new_vector, value_pointer);

            // We must set the size to 0, since we manually track the size within the vector. The
            // capacity is tracked via the filled byte.
            LLVMBuildStore(builder, self.i64(0), size_pointer);

            // Store the capacity as a power-of-2.
            let default_filled = self.i8(DEFAULT_GROUP_FILLED);
            self.slot_ty.set_filled(builder, slot, default_filled);
            LLVMBuildBr(builder, merge_block);

            // If the vector was already there, make sure we can fit the new element.
            LLVMPositionBuilderAtEnd(builder, checkcap_block);
            let ext_filled_value =
                LLVMBuildZExt(builder, filled_value, self.u64_type(), c_str!(""));
            let capacity = LLVMBuildShl(builder, self.i64(1), ext_filled_value, c_str!("capacity"));
            let cur_vector = self.load(builder, value_pointer).unwrap();
            let size = group_vector.gen_size(builder, cur_vector)?;
            let needs_resize =
                LLVMBuildICmp(builder, LLVMIntEQ, size, capacity, c_str!("shouldResize"));
            LLVMBuildCondBr(builder, needs_resize, resize_block, merge_block);

            // Extend the vector and update the slot.
            LLVMPositionBuilderAtEnd(builder, resize_block);
            let new_capacity =
                LLVMBuildNUWMul(builder, capacity, self.i64(2), c_str!("newCapacity"));
            let resized_vector =
                group_vector.gen_extend(builder, intrinsics, run, cur_vector, new_capacity)?;

            // Since extend sets the size, we need to "reset" it back to the previous size.
            LLVMBuildStore(builder, resized_vector, value_pointer);
            LLVMBuildStore(builder, size, size_pointer);

            // Set the new capacity: since we double the size of the vector, we just incremented
            // the filled value by 1.
            let new_filled_value =
                LLVMBuildAdd(builder, filled_value, self.i8(1), c_str!("newFilled"));
            self.slot_ty.set_filled(builder, slot, new_filled_value);
            LLVMBuildBr(builder, merge_block);

            // Merge the value. The capacity of the vector is guaranteed to accomdate the merge
            // value now.
            LLVMPositionBuilderAtEnd(builder, merge_block);
            let array_pointer = LLVMBuildStructGEP(
                builder,
                value_pointer,
                vector::POINTER_INDEX,
                c_str!("arrayPtr"),
            );
            let array_pointer = self.load(builder, array_pointer).unwrap();

            // Merge the value.
            let offset = self.load(builder, size_pointer).unwrap();
            let merge_pointer = LLVMBuildGEP(
                builder,
                array_pointer,
                [offset].as_mut_ptr(),
                1,
                c_str!("mergePtr"),
            );
            LLVMBuildStore(builder, value, merge_pointer);

            // Update the size.
            let inc_size = LLVMBuildNSWAdd(builder, offset, self.i64(1), c_str!("newSize"));
            LLVMBuildStore(builder, inc_size, size_pointer);
            LLVMBuildRetVoid(builder);

            self.merge_grouped = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [dict, key, hash, value, run];
        Ok(LLVMBuildCall(
            builder,
            self.merge_grouped.unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        ))
    }
}
