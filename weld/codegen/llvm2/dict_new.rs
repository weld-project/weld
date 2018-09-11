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

// Dictionary Layout: { data*, capacity, size }

/// Data index.
const DATA_INDEX: u32 = 0;
/// Capacity index.
const CAP_INDEX: u32 = DATA_INDEX + 1;
/// Size index.
const SIZE_INDEX: u32 = CAP_INDEX + 1;

pub struct Dict {
    pub name: String,
    pub dict_ty: LLVMTypeRef,
    pub key_ty: LLVMTypeRef,
    pub val_ty: LLVMTypeRef,
    key_comparator: LLVMValueRef,
    slot_ty: SlotType,
    context: LLVMContextRef,
    module: LLVMModuleRef,
    slot_for_key: Option<LLVMValueRef>,
    new: Option<LLVMValueRef>,
    get: Option<LLVMValueRef>,
    upsert: Option<LLVMValueRef>,
    get_slot: Option<LLVMValueRef>,
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
struct SlotType {
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
    unsafe fn key(&mut self, builder: LLVMBuilderRef, value: LLVMValueRef) -> LLVMValueRef {
        LLVMBuildStructGEP(builder, value, KEY_INDEX, c_str!(""))
    }

    /// Return the value pointer for the provide slot.
    ///
    /// The slot should be a pointer.
    unsafe fn value(&mut self, builder: LLVMBuilderRef, value: LLVMValueRef) -> LLVMValueRef {
        LLVMBuildStructGEP(builder, value, VALUE_INDEX, c_str!(""))
    }

    /// Return whether the provided slot is filled as an `i1`.
    ///
    /// The slot should be a pointer.
    unsafe fn filled(&mut self, builder: LLVMBuilderRef, value: LLVMValueRef) -> LLVMValueRef {
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

        debug_assert!(LLVMGetTypeKind(layout[DATA_INDEX as usize]) == LLVMTypeKind::LLVMPointerTypeKind);
        debug_assert!(LLVMGetTypeKind(layout[CAP_INDEX as usize]) == LLVMTypeKind::LLVMIntegerTypeKind);
        debug_assert!(LLVMGetTypeKind(layout[SIZE_INDEX as usize]) == LLVMTypeKind::LLVMIntegerTypeKind);

        let dict_ty = LLVMStructCreateNamed(context, c_name.as_ptr());
        LLVMStructSetBody(dict_ty, layout.as_mut_ptr(), layout.len() as u32, 0);

        let name = c_name.into_string().unwrap();

        Dict {
            name: name,
            dict_ty: dict_ty,
            key_ty: key_ty,
            key_comparator: key_comparator,
            val_ty: val_ty,
            slot_ty: slot_ty,
            context: context,
            module: module,
            slot_for_key: None,
            new: None,
            get: None,
            upsert: None,
            get_slot: None,
            key_exists: None,
            size: None,
            to_vec: None,
            free: None,
            serialize: None,
        }
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

    unsafe fn slot_at_index(&mut self,
                            builder: LLVMBuilderRef,
                            data: LLVMValueRef,
                            index: LLVMValueRef) -> LLVMValueRef {
        LLVMBuildGEP(builder, data, [index].as_mut_ptr(), 1, c_str!(""))
    }

    /// Return the slot for a given key.
    unsafe fn gen_slot_for_key(&mut self,
                               builder: LLVMBuilderRef,
                               data: LLVMValueRef,
                               capacity: LLVMValueRef,
                               hash: LLVMValueRef,
                               key: LLVMValueRef) -> LLVMValueRef {

        // This function assumes that the last slot is guaranteed to be empty.
        //
        // Generated Code:
        //
        // define probe(dict.slot *data, i64 capacity, i32 hash, key *key) -> slot {
        //   start = getstart(hash, cap);
        //   jump top
        // top:
        //   i = phi [ start, top ] [ i2, check_key ] 
        //   s = getslot(i)
        //   br s.filled ? check_key : return
        // check_key:
        //   i2 = i + 1
        //   br s.key == key ? return : top
        // return:
        //   return s
        // }

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

            let top_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));
            let check_key_block = LLVMAppendBasicBlockInContext(self.context(), function,  c_str!(""));
            let return_block = LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));

            let start_index = self.index_for_hash(builder, hash, capacity);
            LLVMBuildBr(builder, top_block);

            LLVMPositionBuilderAtEnd(builder, top_block);
            let index = LLVMBuildPhi(builder, self.i64_type(), c_str!(""));
            let slot = self.slot_at_index(builder, data, index);

            let filled = self.slot_ty.filled(builder, slot);
            LLVMBuildCondBr(builder, filled, check_key_block, return_block);

            // Check keys.
            LLVMPositionBuilderAtEnd(builder, check_key_block);

            let update_index = LLVMBuildNSWAdd(builder, index, self.i64(1), c_str!(""));
            let slot_key = self.slot_ty.key(builder, slot);
            let keys_eq = self.compare_keys(builder, slot_key, key);
            LLVMBuildCondBr(builder, keys_eq, return_block, top_block);

            // Return the slot.
            LLVMPositionBuilderAtEnd(builder, return_block);
            LLVMBuildRet(builder, slot);

            // Set the PHI value.
            let mut blocks = [entry_block, check_key_block];
            let mut values = [start_index, update_index];
            LLVMAddIncoming(index, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);

            self.slot_for_key = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [data, capacity, hash, key];
        LLVMBuildCall(builder,
                         self.slot_for_key.unwrap(),
                         args.as_mut_ptr(),
                         args.len() as u32, c_str!(""))
    }

    /*
    pub unsafe fn gen_upsert(&mut self,
                             builder: LLVMBuilderRef,
                             run: LLVMValueRef,
                             dict: LLVMValueRef,
                             key: LLVMValueRef,
                             hash: LLVMValueRef,
                             val: LLVMValueRef) -> WeldResult<LLVMValueRef> {

        // Type checking.
        debug_assert!(LLVMGetTypeKind(dict) == LLVMTypeKind::LLVMPointerTypeKind);
        debug_assert!(LLVMGetTypeKind(key) == LLVMTypeKind::LLVMPointerTypeKind);
        debug_assert!(LLVMGetTypeKind(val) == LLVMTypeKind::LLVMPointerTypeKind);
        debug_assert!(LLVMGetTypeKind(ihash) == LLVMTypeKind::LLVMIntegerTypeKind);

        SlotType s = probe(capacity, hash, key);
        if s.filled {
            return s.value
        } else {
            s.filled = true;
            s.key = key;
            s.value = value;
            if (resize()) {
                s = probe(capacity, hash, key);
            } 
            return s.value
        }
    }

    pub unsafe fn gen_lookup(&mut self,
                         builder: LLVMValueRef,
                         run: LLVMValueRef,
                         key: LLVMValueRef,
                         hash: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        SlotType s = probe(capacity, hash, key);
        return s;
    }
    */
}

