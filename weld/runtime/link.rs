//! C Functions called by the Weld runtime.
//!
//! These functions should not be called by users of Weld - they are exposed because LLVM cannot
//! resolve them unless they are marked with `pub`.

use libc::{c_void, int64_t, int32_t};

// Redefine here as an FFI-safe pointer.
type WeldRuntimeContextRef = *mut c_void;
type VoidPtr = *mut c_void;
type KeyComparator = extern "C" fn(VoidPtr, VoidPtr) -> int32_t;

#[link(name="dictst", kind="static")]
extern "C" {

    // Dictionary
    
    #[no_mangle]
    pub fn weld_st_dict_new(run: WeldRuntimeContextRef,
                            keysz: int32_t,
                            valsz: int32_t,
                            eq: KeyComparator,
                            capacity: int64_t) -> VoidPtr;
    #[no_mangle]
    pub fn weld_st_dict_get(run: WeldRuntimeContextRef, 
                            dict: VoidPtr, 
                            key: VoidPtr, 
                            hash: int32_t) -> VoidPtr;
    #[no_mangle]
    pub fn weld_st_dict_get_slot(run: WeldRuntimeContextRef, 
                            dict: VoidPtr, 
                            key: VoidPtr, 
                            hash: int32_t) -> VoidPtr;
    #[no_mangle]
    pub fn weld_st_dict_upsert(run: WeldRuntimeContextRef,
                               dict: VoidPtr,
                               key: VoidPtr,
                               hash: int32_t,
                               val: VoidPtr) -> VoidPtr;
    #[no_mangle]
    pub fn weld_st_dict_keyexists(run: WeldRuntimeContextRef, 
                            dict: VoidPtr, 
                            key: VoidPtr, 
                            hash: int32_t) -> int32_t;
    #[no_mangle]
    pub fn weld_st_dict_size(run: WeldRuntimeContextRef, 
                            dict: VoidPtr) -> int64_t; 
    #[no_mangle]
    pub fn weld_st_dict_free(run: WeldRuntimeContextRef, 
                            dict: VoidPtr); 
    #[no_mangle]
    pub fn weld_st_dict_tovec(run: WeldRuntimeContextRef, 
                            dict: VoidPtr,
                            value_offset: int32_t,
                            struct_size: int32_t,
                            out_pointer: VoidPtr); 
    #[no_mangle]
    pub fn weld_st_dict_serialize(run: WeldRuntimeContextRef, 
                            dict: VoidPtr,
                            buf: VoidPtr,
                            has_pointer: int32_t,
                            key_ser: VoidPtr,
                            val_ser: VoidPtr); 

    // GroupMerger
 
    #[no_mangle]
    pub fn weld_st_gb_new(run: WeldRuntimeContextRef,
                            keysz: int32_t,
                            valsz: int32_t,
                            eq: KeyComparator,
                            capacity: int64_t) -> VoidPtr;
    #[no_mangle]
    pub fn weld_st_gb_merge(run: WeldRuntimeContextRef, 
                            dict: VoidPtr, 
                            key: VoidPtr, 
                            hash: int32_t,
                            val: VoidPtr);
    #[no_mangle]
    pub fn weld_st_gb_result(run: WeldRuntimeContextRef, 
                            dict: VoidPtr) -> VoidPtr; 
}
