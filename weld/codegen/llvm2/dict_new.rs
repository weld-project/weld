//! A dictionary in Weld.

extern crate lazy_static;
extern crate llvm_sys;
extern crate libc;

use std::ffi::CString;

use error::*;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;
use self::llvm_sys::LLVMIntPredicate::*;
use self::llvm_sys::LLVMTypeKind;

use codegen::llvm2::llvm_exts::LLVMExtAttribute::*;
use codegen::llvm2::llvm_exts::*;


use super::CodeGenExt;

/// Slot index of dictionary key.
const KEY_INDEX: u32 = 0;
/// Slot index of dictionary value.
const VALUE_INDEX: u32 = KEY_INDEX + 1;
/// Slot index of whether a slot is filled.
const FILLED_INDEX: u32 = VALUE_INDEX + 1;

// Dictionary Layout: { slot_array*, capacity, size }

/// Data index.
const SLOT_ARR_INDEX: u32 = 0;
/// Capacity index.
const CAP_INDEX: u32 = SLOT_ARR_INDEX + 1;
/// Size index.
const SIZE_INDEX: u32 = CAP_INDEX + 1;

pub struct Dict {
    pub name: String,
    pub dict_ty: LLVMTypeRef,
    dict_inner_ty: LLVMTypeRef,
    pub key_ty: LLVMTypeRef,
    pub val_ty: LLVMTypeRef,
    key_comparator: LLVMValueRef,
    slot_ty: SlotType,
    context: LLVMContextRef,
    module: LLVMModuleRef,
    slot_for_key: Option<LLVMValueRef>,
    new: Option<LLVMValueRef>,
    lookup: Option<LLVMValueRef>,
    upsert: Option<LLVMValueRef>,
    resize: Option<LLVMValueRef>,
    key_exists: Option<LLVMValueRef>,
    size: Option<LLVMValueRef>,
    to_vec: Option<LLVMValueRef>,
    free: Option<LLVMValueRef>,
    serialize: Option<LLVMValueRef>,
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
    unsafe fn new<T: AsRef<str>>(name: T,
                  key_ty: LLVMTypeRef,
                  val_ty: LLVMTypeRef,
                  context: LLVMContextRef,
                  module: LLVMModuleRef) -> SlotType {

        // Create a name for the dictionary.
        let c_name = CString::new(name.as_ref()).unwrap();

        // A slot is a a struct with { key, value, filled }
        let mut layout = [
            key_ty,
            val_ty,
            LLVMInt8TypeInContext(context)
        ];

        // For consistency.
        debug_assert!(LLVMGetTypeKind(layout[KEY_INDEX as usize]) == LLVMGetTypeKind(key_ty));
        debug_assert!(LLVMGetTypeKind(layout[VALUE_INDEX as usize]) == LLVMGetTypeKind(val_ty));
        debug_assert!(LLVMGetTypeKind(layout[FILLED_INDEX as usize]) == LLVMTypeKind::LLVMIntegerTypeKind);

        let slot_ty = LLVMStructCreateNamed(context, c_name.as_ptr());
        LLVMStructSetBody(slot_ty, layout.as_mut_ptr(), layout.len() as u32, 0);

        SlotType {
            slot_ty: slot_ty,
            key_ty: key_ty,
            val_ty: val_ty,
            context: context,
            module: module,
        }
    }

    /// Return the key pointer for the provided slot.
    ///
    /// The slot should be a pointer.
    pub unsafe fn key(&mut self, builder: LLVMBuilderRef, value: LLVMValueRef) -> LLVMValueRef {
        LLVMBuildStructGEP(builder, value, KEY_INDEX, c_str!(""))
    }

    /// Return the value pointer for the provide slot.
    ///
    /// The slot should be a pointer.
    pub unsafe fn value(&mut self, builder: LLVMBuilderRef, value: LLVMValueRef) -> LLVMValueRef {
        LLVMBuildStructGEP(builder, value, VALUE_INDEX, c_str!(""))
    }

    /// Return whether the provided slot is filled as an `i1`.
    ///
    /// The slot should be a pointer.
    pub unsafe fn filled(&mut self, builder: LLVMBuilderRef, value: LLVMValueRef) -> LLVMValueRef {
        let filled_pointer = LLVMBuildStructGEP(builder, value, FILLED_INDEX, c_str!(""));
        let filled = self.load(builder, filled_pointer).unwrap();
        LLVMBuildICmp(builder, LLVMIntNE, filled, self.i8(0), c_str!(""))
    }
}

impl Dict {
    /// Define a new dictionary type.
    ///
    /// The definition requires a key and a value type. 
    pub unsafe fn define<T: AsRef<str>>(name: T,
                                        key_ty: LLVMTypeRef,
                                        key_comparator: LLVMValueRef,
                                        val_ty: LLVMTypeRef,
                                        context: LLVMContextRef,
                                        module: LLVMModuleRef) -> Dict {

        let c_name = CString::new(name.as_ref()).unwrap();
        let slot_ty = SlotType::new(format!("{}.slot", name.as_ref()), key_ty, val_ty, context, module);

        // A dictionary holds an array of slots along with a capacity and size.
        let mut layout = [
            LLVMPointerType(slot_ty.slot_ty, 0),
            LLVMInt64TypeInContext(context),
            LLVMInt64TypeInContext(context)
        ];

        debug_assert!(LLVMGetTypeKind(layout[SLOT_ARR_INDEX as usize]) == LLVMTypeKind::LLVMPointerTypeKind);
        debug_assert!(LLVMGetTypeKind(layout[CAP_INDEX as usize]) == LLVMTypeKind::LLVMIntegerTypeKind);
        debug_assert!(LLVMGetTypeKind(layout[SIZE_INDEX as usize]) == LLVMTypeKind::LLVMIntegerTypeKind);

        let dict_inner_ty = LLVMStructCreateNamed(context, c_name.as_ptr());
        LLVMStructSetBody(dict_inner_ty, layout.as_mut_ptr(), layout.len() as u32, 0);

        let name = c_name.into_string().unwrap();

        Dict {
            name: name,
            dict_ty: LLVMPointerType(dict_inner_ty, 0),
            dict_inner_ty: dict_inner_ty,
            key_ty: key_ty,
            key_comparator: key_comparator,
            val_ty: val_ty,
            slot_ty: slot_ty,
            context: context,
            module: module,
            slot_for_key: None,
            new: None,
            lookup: None,
            upsert: None,
            resize: None,
            key_exists: None,
            size: None,
            to_vec: None,
            free: None,
            serialize: None,
        }
    }

    unsafe fn slot_array(&mut self, builder: LLVMBuilderRef, value: LLVMValueRef) -> LLVMValueRef {
        self.load(builder, LLVMBuildStructGEP(builder, value, SLOT_ARR_INDEX, c_str!(""))).unwrap()
    }

    unsafe fn capacity(&mut self, builder: LLVMBuilderRef, value: LLVMValueRef) -> LLVMValueRef {
        self.load(builder, LLVMBuildStructGEP(builder, value, CAP_INDEX, c_str!(""))).unwrap()
    }

    unsafe fn size(&mut self, builder: LLVMBuilderRef, value: LLVMValueRef) -> LLVMValueRef {
        self.load(builder, LLVMBuildStructGEP(builder, value, SIZE_INDEX, c_str!(""))).unwrap()
    }

    /// Returns the probe start index for a hash.
    ///
    /// For a power-of-2 hash table, this is `hash & (capacity - 1)`.
    unsafe fn index_for_hash(&mut self,
                             builder: LLVMBuilderRef,
                             hash: LLVMValueRef,
                             capacity: LLVMValueRef) -> LLVMValueRef {
        let hash = LLVMBuildZExt(builder, hash, self.i64_type(), c_str!(""));
        let val = LLVMBuildSub(builder, capacity, self.i64(1), c_str!(""));
        LLVMBuildAnd(builder, hash, val, c_str!(""))
    }

    /// Returns whether two keys are equal using their equality function.
    unsafe fn compare_keys(&mut self,
                           builder: LLVMBuilderRef,
                           slot_key: LLVMValueRef,
                           key: LLVMValueRef) -> LLVMValueRef {
        let mut args = [slot_key, key];
        LLVMBuildCall(builder, self.key_comparator,
                      args.as_mut_ptr(), args.len() as u32, c_str!(""))
    }

    /// Returns the slot pointer at a given index into the slot array.
    unsafe fn slot_at_index(&mut self,
                            builder: LLVMBuilderRef,
                            slot_array: LLVMValueRef,
                            index: LLVMValueRef) -> LLVMValueRef {
        LLVMBuildGEP(builder, slot_array, [index].as_mut_ptr(), 1, c_str!(""))
    }

    /// Returns an `i1` determining whether the dictionary should be resized. 
    unsafe fn should_resize(&mut self,
                            builder: LLVMBuilderRef,
                            size: LLVMValueRef,
                            capacity: LLVMValueRef) -> LLVMValueRef {
        let size = LLVMBuildNSWMul(builder, size, self.i64(10), c_str!(""));
        let capacity = LLVMBuildNSWMul(builder, capacity, self.i64(MAX_LOAD_FACTOR), c_str!(""));
        LLVMBuildICmp(builder, LLVMIntSGE, size, capacity)
    }

    unsafe fn gen_new_dict_with_capacity(&mut self,
                                         builder: LLVMBuilderRef,
                                         intrinsics: &Intrisics,
                                         capacity: LLVMValueRef,
                                         run: LLVMValueRef) -> LLVMValueRef {

        let alloc_size = LLVMBuildNSWMul(builder, capacity, self.size_of(self.slot_ty.slot_ty));
        let bytes = intrinsics.call_weld_run_malloc(builder, run, alloc_size, None);
        let slot_array = LLVMBuildBitCast(builder, bytes, LLVMPointerType(self.slot_ty.slot_ty, 0));

        let mut dict = LLVMBuildInsertValue(
            builder, LLVMGetUndef(self.dict_inner_ty), slot_array, SLOT_ARR_INDEX, c_str!(""));
        dict = LLVMBuildInsertValue(builder, dict, self.i64(capacity), CAP_INDEX, c_str!(""));
        LLVMBuildInsertValue(builder, dict, self.i64(0), SIZE_INDEX, c_str!(""))
    }

    /// Resize the dictionary if necessary.
    ///
    /// Returns an `i1` indicating whether a resize occurred. If the dictionary was resized, slots
    /// acquired from the dictionary should be re-acquired.
    unsafe fn gen_resize(&mut self,
                         builder: LLVMBuilderRef,
                         intrinsics: &Intrinsics,
                         dict: LLVMValueRef,
                         run: LLVMValueRef) -> LLVMValueRef {

        if self.resize.is_none() {
            let mut arg_tys = [
                self.dict_ty,
                self.run_handle_type()
            ];

            let ret_ty = self.i1_type(),
            let name = format!("{}.resize", self.name);

            let (function, builder, entry_block) = self.define_function(ret_ty, &mut arg_tys, name);

            let dict = LLVMGetParam(function, 0);
            let run = LLVMGetParam(function, 1);

            let new_dictionary = LLVMBuildAlloca(builder, self.dict_ty, c_str!(""));

            let old_size = self.size(builder, dict);
            let old_capacity = self.capacity(builder, dict);
            let old_slot_array = self.slot_array(builder, dict);

            // Resize
            let resize_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));

            // Copy loop.
            let copy_top_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));
            let copy_cpy_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));

            // Probe for empty slot.
            let probe_top_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));
            let probe_chk_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));
            let probe_bot_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));

            // End of copy loop.
            let copy_bot_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));
            let copy_fin_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));

            // Exit
            let exit_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));

            let should_resize = self.should_resize(builder, old_size, old_capacity);
            LLVMBuildCondBr(builder, should_resize, resize_block, exit_block);

            LLVMPositionBuilderAtEnd(builder, resize_block);

            // Resize the dictionary.
            let new_capacity = LLVMBuildNSWMul(builder, old_capacity, self.i64(2), c_str!(""));
            let new_dictionary_inner = self.gen_new_dict_with_capacity(builder, intrinsics, new_capacity, run);
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
            let old_slot = self.slot_at_index(builder, old_slot_array);
            let filled = self.slot_ty.filled(old_slot);
            LLVMBuildCondBr(builder, filled, copy_cpy_block, copy_bot_block);
            
            // Copy Block.
            LLVMPositionBuilderAtEnd(builder, copy_cpy_block);
            let new_slot = self.gen_probe_loop(builder,
                                           new_slot_array,
                                           new_capacity,
                                           slot_hash,
                                           (probe_top_block, probe_chk_block, probe_bot_block),
                                           None);
            let new_slot_bytes = LLVMBuildBitcast(builder, new_slot, self.void_pointer_type(), c_str!(""));
            let old_slot_bytes = LLVMBuildBitcast(builder, old_slot, self.void_pointer_type(), c_str!(""));
            let _ = intrinsics.call_memcpy(builder, new_slot_bytes, old_slot_bytes, self.size_of(self.slot_ty.slot_ty));
            LLVMBuildBr(builder, copy_bot_block);

            // Bottom Block.
            LLVMPositionBuilderAtEnd(builder, copy_bot_block);
            let update_index = LLVMBuildNSWAdd(builder, index, self.i64(1), c_str!(""));
            let finished = LLVMBuildICmp(builder, LLVMIntEQ, update_index, capacity, c_str!(""));
            LLVMBuildCondBr(builder, finished, copy_fin_block, copy_top_block);

            // Set the PHI value for the index.
            let mut blocks = [resize_block, copy_bot_block];
            let mut values = [self.i64(0), update_index];
            LLVMAddIncoming(index, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);

            // Finish Block.
            LLVMPositionBuilderAtEnd(builder, copy_fin_block);
            // 1. Free Old Dictionary buffer.
            let old_slotarray_bytes = LLVMBuildBitcast(builder, old_slot_array, self.void_pointer_type(), c_str!(""));
            let _ = intrinsics.call_weld_run_free(builder, run, old_slotarray_bytes);
            // 2. Move New Dictionary to Old Dictionary pointer.
            let new_dict_bytes = LLVMBuildBitcast(builder, new_dictionary, self.void_pointer_type(), c_str!(""));
            let old_dict_bytes = LLVMBuildBitcast(builder, dict, self.void_pointer_type(), c_str!(""));
            let _ = intrinsics.call_memcpy(builder, dict, new_dict_bytes, self.size_of(self.dict_inner_ty));
            LLVMBuildBr(builder, exit_block);

            // Finished at last! Hallelujah. Who would've thought resizing a dictionary is so hard.
            LLVMPositionBuilderAtEnd(builder, exit_block);
            let resized = LLVMBuildPhi(builder, self.i1_type(), c_str!(""));
            LLVMBuildRet(builder, resized);

            self.resize = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [dict, run];
        LLVMBuildCall(builder,
                         self.slot_for_key.unwrap(),
                         args.as_mut_ptr(),
                         args.len() as u32, c_str!(""))

    }

    /// Generate a probe loop. The probe loop uses the provided hash to search the hash table using
    /// linear probing. If a key is provided, either the matching slot or an empty slot is
    /// returned. If a key is not provided, the first empty slot starting at the hash's index is
    /// returned.
    ///
    /// This method takes three basic blocks for loop control flow: a top block, a check block to
    /// check for the key, and an exit block.
    ///
    /// When the function returns, the builder is guaranteed to be positioned at the beginning of the
    /// exit block. The function returns the LLVMValue representing the slot found with the probe.
    unsafe fn gen_probe_loop(&mut self,
                             builder: LLVMBuilderRef,
                             slot_array: LLVMValueRef,
                             capacity: LLVMValueRef,
                             hash: LLVMValueRef,
                             blocks: (LLVMValueRef, LLVMValueRef, LLVMValueRef),
                             key: Option<LLVMValueRef>) -> LLVMValueRef {

        // Generated Code:
        //
        //   start = getstart(hash, cap);
        //   jump top
        // top:
        //   i = phi [ start, top ] [ i2, check_key ] 
        //   s = getslot(i)
        //   br s.filled ? check_key : return
        // check_key:                                   
        //   i2 = i + 1                                 
        //   br s.key == key ? return : top | br top, depending on `check_key`
        // return:
        //   return s
        // }

        let (top_block, check_block, end_block) = blocks;

        let start_index = self.index_for_hash(builder, hash, capacity);
        LLVMBuildBr(builder, top_block);

        LLVMPositionBuilderAtEnd(builder, top_block);
        let index = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
        let slot = self.slot_at_index(builder, slot_array, index);

        let filled = self.slot_ty.filled(builder, slot);
        LLVMBuildCondBr(builder, filled, check_block, end_block);

        LLVMPositionBuilderAtEnd(builder, check_block);

        let update_index = LLVMBuildNSWAdd(builder, index, self.i64(1), c_str!(""));
        if let Some(key) = key {
            let slot_key = self.slot_ty.key(builder, slot);
            let keys_eq = self.compare_keys(builder, slot_key, key);
            LLVMBuildCondBr(builder, keys_eq, end_block, top_block);
        } else {
            LLVMBuildBr(builder, top_block);
        }

        LLVMPositionBuilderAtEnd(builder, end_block);
        slot

    }

    /// Return the slot for a given key. The returned slot may be uninitialized if the key does not
    /// exist in the dictionary.
    ///
    /// The current algorithm assumes a power-of-two hash table and performs standard linear
    /// probing.
    unsafe fn gen_slot_for_key(&mut self,
                               builder: LLVMBuilderRef,
                               slot_array: LLVMValueRef,
                               capacity: LLVMValueRef,
                               hash: LLVMValueRef,
                               key: LLVMValueRef) -> LLVMValueRef {

        if self.slot_for_key.is_none() {
            let mut arg_tys = [
                LLVMPointerType(self.slot_ty.slot_ty, 0),
                self.i64_type(),
                self.i32_type(),
                LLVMPointerType(self.key_ty, 0)
            ];
            let ret_ty = LLVMPointerType(self.slot_ty.slot_ty, 0);
            let name = format!("{}.slot_for_key", self.name);

            // TODO Attributes!!

            let (function, builder, entry_block) = self.define_function(ret_ty, &mut arg_tys, name);

            let slot_array = LLVMGetParam(function, 0);
            let capacity = LLVMGetParam(function, 1);
            let hash = LLVMGetParam(function, 2);
            let key = LLVMGetParam(function, 3);

            let top_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));
            let check_key_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));
            let return_block = LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));

            let slot = self.gen_probe_loop(builder,
                                           slot_array,
                                           capacity,
                                           hash,
                                           (top_block, check_key_block, return_block),
                                           Some(key));

            LLVMBuildRet(builder, slot);

            // Set the PHI value.
            let mut blocks = [entry_block, check_key_block];
            let mut values = [start_index, update_index];
            LLVMAddIncoming(index, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);

            self.slot_for_key = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [slot_array, capacity, hash, key];
        LLVMBuildCall(builder,
                         self.slot_for_key.unwrap(),
                         args.as_mut_ptr(),
                         args.len() as u32, c_str!(""))
    }

    /// Returns the pointer to the slot for a key.
    ///
    /// If the key does not exist, the first uninitialized slot is returned.
    pub unsafe fn gen_upsert(&mut self,
                         builder: LLVMBuilderRef,
                         intrinsics: &Intrinsics,
                         dict: LLVMValueRef,
                         key: LLVMValueRef,
                         hash: LLVMValueRef,
                         default: LLVMValueRef,
                         run: LLVMValueRef) -> WeldResult<LLVMValueRef> {

        if self.lookup.is_none() {
            let mut arg_tys = [
                self.dict_ty,
                LLVMPointerType(self.key_ty, 0),
                self.hash_type(),
                self.val_ty,
                self.run_handle_type(),
            ];
            let ret_ty = LLVMPointerType(self.slot_ty.slot_ty, 0);

            let name = format!("{}.upsert", self.name);
            let (function, builder, entry_block) = self.define_function(ret_ty, &mut arg_tys, name);

            let set_default_block = LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));
            let reacquire_slot_block = LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));
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
            LLVMBuildCondBr(filled, return_block, set_default_block);

            LLVMPositionBuilderAtEnd(builder, set_default_block);

            self.slot_ty.init(builder, slot, key, default);
            let size_pointer = LLVMBuildStructGEP(builder, value, SIZE_INDEX, c_str!(""));
            let size = self.load(size_pointer).unwrap();
            let new_size = LLVMBuildNSWAdd(builder, size, self.i64(1), c_str!(""));
            LLVMBuildStore(builder, size, size_pointer);

            // Check for resize.
            let resized = self.gen_resize(builder, intrinsics, dict, run);
            LLVMBuildCondBr(resized, reacquire_slot_block, return_block);

            LLVMPositionBuilderAtEnd(builder, reacquire_slot_block);
            // Builder was resized - we need to reacquire the slot.
            let resized_slot = self.gen_slot_for_key(builder, slot_array, capacity, hash, key);

            LLVMPositionBuilderAtEnd(builder, return_block);

            let return_slot = LLVMBuildPhi(builder, ret_ty, c_str!(""));
            LLVMBuildRet(builder, return_slot);

            // Set the PHI value.
            let mut blocks = [entry_block, reacquire_slot_block];
            let mut values = [slot, resized_slot];
            LLVMAddIncoming(return_slot, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);

            self.upsert = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [dict, key, hash, default, run];
        return Ok(LLVMBuildCall(builder,
                                self.upsert.unwrap(),
                                args.as_mut_ptr(),
                                args.len() as u32,
                                c_str!("")))
    }

    /// Returns the pointer to the slot for a key.
    ///
    /// If the key does not exist, the first uninitialized slot is returned.
    pub unsafe fn gen_lookup(&mut self,
                         builder: LLVMBuilderRef,
                         dict: LLVMValueRef,
                         key: LLVMValueRef,
                         hash: LLVMValueRef) -> WeldResult<LLVMValueRef> {

        if self.lookup.is_none() {
            let mut arg_tys = [
                self.dict_ty,
                LLVMPointerType(self.key_ty, 0),
                self.hash_type(),
            ];
            let ret_ty = LLVMPointerType(self.slot_ty.slot_ty, 0);

            let name = format!("{}.lookup", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let dict = LLVMGetParam(function, 0);
            let key = LLVMGetParam(function, 1);
            let hash = LLVMGetParam(function, 2);

            let slot_array = self.slot_array(builder, dict);
            let capacity = self.capacity(builder, dict);
            let slot = self.gen_slot_for_key(builder, slot_array, capacity, hash, key);

            LLVMBuildRet(builder, slot);

            self.lookup = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [dict, key, hash];
        return Ok(LLVMBuildCall(builder,
                                self.lookup.unwrap(),
                                args.as_mut_ptr(),
                                args.len() as u32,
                                c_str!("")))
    }
}

