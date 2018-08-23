//! A wrapper for dictionaries in Weld.
//!
//! This module provides a wrapper interface for methods and utilities on dictionary types. Other
//! modules use it for dictionary-related functionality or operators over dictionaries.
//!
//! Many of the methods here are marked as `alwaysinline`, so method calls on dictionaries usually
//! have no overhead.
//!
//! The dictionary is currently implemented as a statically linked C++ library. This means that
//! this file mostly contains wrappers for calling into that implementation. It should thus be
//! modified with care, and should ensure that the following properties always hold:
//!
//! 1. The definition of each function in `Intrinsics::populate` must *exactly* match the
//!    corresponding C++ `"extern" C` definition. The LLVM compiler will not check to see whether
//!    the argument lists and return types match: it only checks the symbol name. If there is a
//!    mismatch here, it can lead to some hairy bugs.
//! 2. The layout of the dictionary slot in `Dict::define` must be consistent with the C++
//!    definition. In particular, the slot is defined as a header followed by a key and value here,
//!    and the header is redefined in C++: the headers must thus be consistent.

extern crate lazy_static;
extern crate llvm_sys;
extern crate libc;

use libc::{c_char};

use std::ffi::CString;

use error::*;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;

use codegen::llvm2::llvm_exts::LLVMExtAttribute::*;
use codegen::llvm2::llvm_exts::*;

use super::CodeGenExt;

pub const HASH_INDEX: u32 = 0;
pub const STATE_INDEX: u32 = 1;
pub const KEY_INDEX: u32 = 2;
pub const VALUE_INDEX: u32 = 3;

pub struct Dict {
    pub name: String,
    pub dict_ty: LLVMTypeRef,
    pub key_ty: LLVMTypeRef,
    pub val_ty: LLVMTypeRef,
    pub key_comparator: LLVMValueRef,
    entry_ty: LLVMTypeRef,
    slot_ty: LLVMTypeRef,
    context: LLVMContextRef,
    module: LLVMModuleRef,
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

/// Exeternal intrinsic API for dictionaries.
///
/// This struct defines intrinsics in its context and modules and provides functions for calling
/// them.
pub struct Intrinsics {
    /// Allocates a new dictionary.
    ///
    /// Arguments: Run Handle, Key Size, Value Size, Comparator Function, Capacity.
    /// Returns: Newly allocated dictionary
    ///
    /// Sizes and capacities are expressed in *bytes*.
    new: Option<LLVMValueRef>,
    /// Gets a value for a given key.
    ///
    /// Produces an error if the key does not exist.
    ///
    /// Arguments: Run Handle, Dictionary, Key, Hash
    /// Returns: Slot for Key
    get: Option<LLVMValueRef>,
    /// Gets a value for a given slot.
    ///
    /// If the slot does not exist, one is allocated for the key and the default value is inserted
    /// before the slot is returned.
    ///
    /// Arguments: Run Handle, Dictionary, Key, Hash, Default Value
    /// Returns: Slot for Key
    upsert: Option<LLVMValueRef>,
    /// Get a pointer to a slot.
    ///
    /// This method returns the slot for a key, and allocates one *without initializing it* if the
    /// slot does not exist.
    get_slot: Option<LLVMValueRef>,
    /// Returns whether a key exists in a dictonary.
    ///
    /// Arguments: Run Handle, Dictionary, Key, Hash
    /// Returns: Boolean indicator
    key_exists: Option<LLVMValueRef>,
    /// Returns the number of keys in the dictionary.
    ///
    /// Arguments: Run Handle, Dictionary
    /// Returns: Size
    size: Option<LLVMValueRef>,
    /// Returns a vec[{K,V}] from the dictionary.
    ///
    /// Arguments: Run Handle, Dictionary
    /// Returns: vector of Key/Value structs. The vector is already in the weld format.
    to_vec: Option<LLVMValueRef>,
    /// Free a dictionary.
    free: Option<LLVMValueRef>,
    /// Serialize a dictionary.
    serialize: Option<LLVMValueRef>,
    context: LLVMContextRef,
    module: LLVMModuleRef,
}

impl CodeGenExt for Intrinsics {
    fn context(&self) -> LLVMContextRef {
        self.context
    }

    fn module(&self) -> LLVMModuleRef {
        self.module
    }
}

impl Intrinsics {
    /// Generate intrinsics for a dictionary.
    ///
    /// This function also splices in the dictionary code into the module.
    pub unsafe fn new(context: LLVMContextRef, module: LLVMModuleRef) -> Intrinsics {
        let mut intrinsics = Intrinsics {
            new: None,
            get: None,
            upsert: None,
            get_slot: None,
            key_exists: None,
            size: None,
            to_vec: None,
            free: None,
            serialize: None,
            context: context,
            module: module,
        };
        intrinsics.populate();
        intrinsics
    }


    /// Generate a call to the `new` intrinsic.
    pub unsafe fn call_new(&self, builder: LLVMBuilderRef,
                           run: LLVMValueRef,
                           key_size: LLVMValueRef,
                           val_size: LLVMValueRef,
                           comparator: LLVMValueRef,
                           capacity: LLVMValueRef,
                           name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run, key_size, val_size, comparator, capacity];
        LLVMBuildCall(builder, self.new.unwrap(), args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Generate a call to the `get` intrinsic.
    pub unsafe fn call_get(&self,
                           builder: LLVMBuilderRef,
                           run: LLVMValueRef,
                           dictionary: LLVMValueRef,
                           key: LLVMValueRef,
                           hash: LLVMValueRef,
                           name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run, dictionary, key, hash];
        LLVMBuildCall(builder, self.get.unwrap(), args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Generate a call to the `upsert` intrinsic.
    pub unsafe fn call_upsert(&self,
                              builder: LLVMBuilderRef,
                              run: LLVMValueRef,
                              dictionary: LLVMValueRef,
                              key: LLVMValueRef,
                              hash: LLVMValueRef,
                              default: LLVMValueRef,
                              name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run, dictionary, key, hash, default];
        LLVMBuildCall(builder, self.upsert.unwrap(), args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Generate a call to the `get_slot` intrinsic.
    pub unsafe fn call_get_slot(&self,
                              builder: LLVMBuilderRef,
                              run: LLVMValueRef,
                              dictionary: LLVMValueRef,
                              key: LLVMValueRef,
                              hash: LLVMValueRef,
                              name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run, dictionary, key, hash];
        LLVMBuildCall(builder, self.get_slot.unwrap(), args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Generate a call to the `key_exists` intrinsic.
    pub unsafe fn call_key_exists(&self,
                              builder: LLVMBuilderRef,
                              run: LLVMValueRef,
                              dictionary: LLVMValueRef,
                              key: LLVMValueRef,
                              hash: LLVMValueRef,
                              name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run, dictionary, key, hash];
        LLVMBuildCall(builder, self.key_exists.unwrap(), args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Generate a call to the `size` intrinsic.
    pub unsafe fn call_size(&self,
                              builder: LLVMBuilderRef,
                              run: LLVMValueRef,
                              dictionary: LLVMValueRef,
                              name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run, dictionary];
        LLVMBuildCall(builder, self.size.unwrap(), args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Generate a call to the `to_vec` intrinsic.
    pub unsafe fn call_to_vec(&self,
                              builder: LLVMBuilderRef,
                              run: LLVMValueRef,
                              dictionary: LLVMValueRef,
                              offset: LLVMValueRef,
                              struct_size: LLVMValueRef,
                              out_pointer: LLVMValueRef,
                              name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run, dictionary, offset, struct_size, out_pointer];
        LLVMBuildCall(builder, self.to_vec.unwrap(), args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Generate a call to the `serialize` intrinsic.
    pub unsafe fn call_serialize(&self,
                              builder: LLVMBuilderRef,
                              run: LLVMValueRef,
                              dictionary: LLVMValueRef,
                              buf: LLVMValueRef,
                              has_pointer: LLVMValueRef,
                              keys_ser: LLVMValueRef,
                              vals_ser: LLVMValueRef,
                              name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run, dictionary, buf, has_pointer, keys_ser, vals_ser];
        LLVMBuildCall(builder, self.serialize.unwrap(), args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Generate a call to the `free` intrinsic.
    pub unsafe fn call_free(&self,
                              builder: LLVMBuilderRef,
                              run: LLVMValueRef,
                              dictionary: LLVMValueRef,
                              name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run, dictionary];
        LLVMBuildCall(builder, self.free.unwrap(), args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
    }

    /// Add a function to the module.
    unsafe fn declare(&mut self,
                      name: &str,
                      params: &mut Vec<LLVMTypeRef>,
                      ret: LLVMTypeRef) -> LLVMValueRef {
        let name = CString::new(name).unwrap();
        let fn_type = LLVMFunctionType(ret, params.as_mut_ptr(), params.len() as u32, 0);
        LLVMAddFunction(self.module, name.as_ptr(), fn_type)
    }

    /// Populate `self` with the dictionary intrinsics.
    ///
    /// These intrinsic definitions must be consistent with the C++ equivalent, so this function
    /// should be modified with care.
    unsafe fn populate(&mut self) {

        // Common types.
        let dict_type = LLVMPointerType(self.i8_type(), 0);
        let hash_type = self.i32_type();

        let ref mut params = vec![
            self.run_handle_type(),     // run
            self.i32_type(),            // key size 
            self.i32_type(),            // val size
            self.opaque_cmp_type(), // key comparator function
            self.i64_type()             // initial capacity (power of 2)
        ];
        let ret = dict_type;
        let function = self.declare("weld_st_dict_new", params, ret);

        LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, NonNull], 0);
        LLVMExtAddAttrsOnReturn(self.context, function, &[NoAlias]);
        self.new = Some(function);

        let ref mut params = vec![
            self.run_handle_type(),     // run
            dict_type,                  // dictionary
            self.void_pointer_type(),   // key
            hash_type                   // hash
        ];
        let ret = self.void_pointer_type();
        let function = self.declare("weld_st_dict_get", params, ret);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 0);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 1);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 2);
        self.get = Some(function);

        let ref mut params = vec![
            self.run_handle_type(),     // run
            dict_type,                  // dictionary
            self.void_pointer_type(),   // key
            hash_type,                  // hash
            self.void_pointer_type()    // init value
        ];
        let ret = self.void_pointer_type();
        let function = self.declare("weld_st_dict_upsert", params, ret);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 0);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull], 1);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, NonNull, ReadOnly], 4);
        self.upsert = Some(function);

        let ref mut params = vec![
            self.run_handle_type(),     // run
            dict_type,                  // dictionary
            self.void_pointer_type(),   // key
            hash_type,                  // hash
        ];
        let ret = self.void_pointer_type();
        let function = self.declare("weld_st_dict_get_slot", params, ret);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 0);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull], 1);
        self.get_slot = Some(function);

        let ref mut params = vec![
            self.run_handle_type(),     // run
            dict_type,                  // dictionary
            self.void_pointer_type(),   // key
            hash_type                   // hash
        ];
        let ret = self.i32_type();
        let function = self.declare("weld_st_dict_keyexists", params, ret);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 0);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 1);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 2);
        self.key_exists = Some(function);

        let ref mut params = vec![
            self.run_handle_type(),     // run
            dict_type                   // dictionary
        ];
        let ret = self.i64_type();
        let function = self.declare("weld_st_dict_size", params, ret);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 0);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 1);
        self.size = Some(function);

        let ref mut params = vec![
            self.run_handle_type(),     // run
            dict_type,                  // dictionary
            self.i32_type(),            // value offset in struct
            self.i32_type(),            // total struct size
            self.void_pointer_type()    // output vec[T] pointer
        ];
        let ret = self.void_type();
        let function = self.declare("weld_st_dict_tovec", params, ret);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 0);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 1);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, NonNull, WriteOnly], 4);
        self.to_vec = Some(function);

        let ref mut params = vec![
            self.run_handle_type(),     // run
            dict_type                   // dictionary
        ];
        let ret = self.void_type();
        let function = self.declare("weld_st_dict_free", params, ret);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 0);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, NonNull], 1);
        self.free = Some(function);

        let ref mut params = vec![
            self.run_handle_type(),     // run
            dict_type,                  // dictionary
            self.void_pointer_type(),   // buffer
            self.i32_type(),            // has pointer in key or value flag
            self.void_pointer_type(),   // key serialize function (null if has_pointer == false)
            self.void_pointer_type(),   // val serialize function (null if has_pointer == false)

        ];
        let ret = self.i64_type();
        let function = self.declare("weld_st_dict_serialize", params, ret);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 0);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, NonNull, ReadOnly], 1);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull], 2);
        self.serialize = Some(function);
    }
}

impl Dict {
    /// Define a new dictionary type.
    ///
    /// The definition requires a key and a value type, as well as a generated LLVM function *with
    /// external visiblity* for comparing two keys.
    pub unsafe fn define<T: AsRef<str>>(name: T,
                                key_ty: LLVMTypeRef,
                                val_ty: LLVMTypeRef,
                                key_comparator: LLVMValueRef,
                                context: LLVMContextRef,
                                module: LLVMModuleRef) -> Dict {
        let c_name = CString::new(name.as_ref()).unwrap();
        let mut layout = [LLVMInt8TypeInContext(context)];
        let dict_ty = LLVMStructCreateNamed(context, c_name.as_ptr());
        LLVMStructSetBody(dict_ty, layout.as_mut_ptr(), layout.len() as u32, 0);

        // A dictionary is just an opaque named pointer.
        let dict_ty = LLVMPointerType(dict_ty, 0);

        let name = c_name.into_string().unwrap();

        // Create the entry type.
        //
        // The entry is laid out as followed, and *must* be consistent with the entry definition in
        // c++:
        //
        // 32-bit hash
        // 32-bit Internal state
        // <key>
        // <value>
        let mut layout = [
            LLVMInt32TypeInContext(context),
            LLVMInt32TypeInContext(context),
            key_ty,
            val_ty
        ];


        let entry_name = format!("{}.entry", &name);
        let c_entry_name = CString::new(entry_name).unwrap();
        let entry_ty = LLVMStructCreateNamed(context, c_entry_name.as_ptr());
        // Marked as packed.
        LLVMStructSetBody(entry_ty, layout.as_mut_ptr(), layout.len() as u32, 1);

        Dict {
            name: name,
            dict_ty: dict_ty,
            key_ty: key_ty,
            val_ty: val_ty,
            key_comparator: key_comparator,
            entry_ty: entry_ty,
            slot_ty: LLVMPointerType(entry_ty, 0),
            context: context,
            module: module,
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

    /// Generates the `new` method.
    pub unsafe fn gen_new(&mut self,
                          builder: LLVMBuilderRef,
                          intrinsics: &Intrinsics,
                          run: LLVMValueRef,
                          capacity: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.new.is_none() {
            let mut arg_tys = [self.i64_type(), self.run_handle_type()];
            let ret_ty = self.dict_ty;

            let name = format!("{}.new", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let capacity = LLVMGetParam(function, 0);
            let run = LLVMGetParam(function, 1);
            let key_size = LLVMConstTruncOrBitCast(self.size_of(self.key_ty), self.i32_type());
            let val_size = LLVMConstTruncOrBitCast(self.size_of(self.val_ty), self.i32_type());
            let result = intrinsics.call_new(builder, run, key_size, val_size, self.key_comparator, capacity, None);

            let result = LLVMBuildBitCast(builder, result, self.dict_ty, c_str!(""));

            LLVMBuildRet(builder, result);

            self.new = Some(function);
            LLVMDisposeBuilder(builder);
        }
        let mut args = [capacity, run];
        return Ok(LLVMBuildCall(builder, self.new.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    /// Generates the `get_slot` method.
    ///
    /// This method looks up the key in the dictionary. If a value is found, a pointer to the value
    /// is returned. Otherwise, a new slot is initialized and a pointer to its value is returned.
    /// The returned slot has an *uninitialized* value.
    pub unsafe fn gen_get_slot(&mut self,
                          builder: LLVMBuilderRef,
                          intrinsics: &Intrinsics,
                          run: LLVMValueRef,
                          dict: LLVMValueRef,
                          key: LLVMValueRef,
                          hash: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.get_slot.is_none() {
            let mut arg_tys = [
                self.dict_ty,
                LLVMPointerType(self.key_ty, 0),
                self.hash_type(),
                self.run_handle_type()
            ];
            let ret_ty = LLVMPointerType(self.val_ty, 0);

            // XXX this is a bit of a misnomer since it returns a pointer to the value, not to the
            // slot.
            let name = format!("{}.get_slot", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let dict = LLVMGetParam(function, 0);
            let key = LLVMGetParam(function, 1);
            let hash = LLVMGetParam(function, 2);
            let run = LLVMGetParam(function, 3);

            let dict = LLVMBuildBitCast(builder, dict, self.void_pointer_type(), c_str!(""));
            let key = LLVMBuildBitCast(builder, key, self.void_pointer_type(), c_str!(""));
            let slot_pointer = intrinsics.call_get_slot(builder, run, dict, key, hash, None);
            let slot_pointer = LLVMBuildBitCast(builder, slot_pointer, self.slot_ty, c_str!(""));
            let result = LLVMBuildStructGEP(builder, slot_pointer, VALUE_INDEX, c_str!(""));

            LLVMBuildRet(builder, result);

            self.get_slot = Some(function);
            LLVMDisposeBuilder(builder);
        }
        let mut args = [dict, key, hash, run];
        return Ok(LLVMBuildCall(builder, self.get_slot.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    /// Generates the `upsert` method.
    ///
    /// This method looks up the key in the dictionary. If a value is found, a pointer to the value
    /// is returned. Otherwise, `val` is inserted into the dictionary, and a pointer to the newly
    /// created slot value is returned.
    pub unsafe fn gen_upsert(&mut self,
                          builder: LLVMBuilderRef,
                          intrinsics: &Intrinsics,
                          run: LLVMValueRef,
                          dict: LLVMValueRef,
                          key: LLVMValueRef,
                          hash: LLVMValueRef,
                          val: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.upsert.is_none() {
            let mut arg_tys = [
                self.dict_ty,
                LLVMPointerType(self.key_ty, 0),
                self.hash_type(),
                self.val_ty,
                self.run_handle_type()
            ];
            let ret_ty = LLVMPointerType(self.val_ty, 0);

            let name = format!("{}.upsert", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let dict = LLVMGetParam(function, 0);
            let key = LLVMGetParam(function, 1);
            let hash = LLVMGetParam(function, 2);
            let val = LLVMGetParam(function, 3);
            let run = LLVMGetParam(function, 4);

            let val_pointer = LLVMBuildAlloca(builder, self.val_ty, c_str!(""));
            LLVMBuildStore(builder, val, val_pointer);

            let dict = LLVMBuildBitCast(builder, dict, self.void_pointer_type(), c_str!(""));
            let key = LLVMBuildBitCast(builder, key, self.void_pointer_type(), c_str!(""));
            let val = LLVMBuildBitCast(builder, val_pointer, self.void_pointer_type(), c_str!(""));
            let slot_pointer = intrinsics.call_upsert(builder, run, dict, key, hash, val, None);
            let slot_pointer = LLVMBuildBitCast(builder, slot_pointer, self.slot_ty, c_str!(""));
            let result = LLVMBuildStructGEP(builder, slot_pointer, VALUE_INDEX, c_str!(""));

            LLVMBuildRet(builder, result);

            self.upsert = Some(function);
            LLVMDisposeBuilder(builder);
        }
        let mut args = [dict, key, hash, val, run];
        return Ok(LLVMBuildCall(builder, self.upsert.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    /// Generates the `get` method, which returns a loaded value.
    ///
    /// This function can raise an error if the key is not found.
    pub unsafe fn gen_get(&mut self,
                          builder: LLVMBuilderRef,
                          intrinsics: &Intrinsics,
                          run: LLVMValueRef,
                          dict: LLVMValueRef,
                          key: LLVMValueRef,
                          hash: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.get.is_none() {
            let mut arg_tys = [
                self.dict_ty,
                LLVMPointerType(self.key_ty, 0),
                self.hash_type(),
                self.run_handle_type()
            ];
            let ret_ty = self.val_ty;

            let name = format!("{}.get", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let dict = LLVMGetParam(function, 0);
            let key = LLVMGetParam(function, 1);
            let hash = LLVMGetParam(function, 2);
            let run = LLVMGetParam(function, 3);

            let dict = LLVMBuildBitCast(builder, dict, self.void_pointer_type(), c_str!(""));
            let key = LLVMBuildBitCast(builder, key, self.void_pointer_type(), c_str!(""));
            let slot_pointer = intrinsics.call_get(builder, run, dict, key, hash, None);
            let slot_pointer = LLVMBuildBitCast(builder, slot_pointer, self.slot_ty, c_str!(""));
            let value_pointer = LLVMBuildStructGEP(builder, slot_pointer, VALUE_INDEX, c_str!(""));
            let result = self.load(builder, value_pointer)?;

            LLVMBuildRet(builder, result);

            self.get = Some(function);
            LLVMDisposeBuilder(builder);
        }
        let mut args = [dict, key, hash, run];
        return Ok(LLVMBuildCall(builder, self.get.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    /// Generates the `key_exists` method.
    pub unsafe fn gen_key_exists(&mut self,
                          builder: LLVMBuilderRef,
                          intrinsics: &Intrinsics,
                          run: LLVMValueRef,
                          dict: LLVMValueRef,
                          key: LLVMValueRef,
                          hash: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.key_exists.is_none() {
            let mut arg_tys = [
                self.dict_ty,
                LLVMPointerType(self.key_ty, 0),
                self.hash_type(),
                self.run_handle_type()
            ];
            let ret_ty = self.bool_type();

            let name = format!("{}.keyexists", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let dict = LLVMGetParam(function, 0);
            let key = LLVMGetParam(function, 1);
            let hash = LLVMGetParam(function, 2);
            let run = LLVMGetParam(function, 3);

            let dict = LLVMBuildBitCast(builder, dict, self.void_pointer_type(), c_str!(""));
            let key = LLVMBuildBitCast(builder, key, self.void_pointer_type(), c_str!(""));
            let filled = intrinsics.call_key_exists(builder, run, dict, key, hash, None);
            let result = LLVMBuildTrunc(builder, filled, ret_ty, c_str!(""));

            LLVMBuildRet(builder, result);

            self.key_exists = Some(function);
            LLVMDisposeBuilder(builder);
        }
        let mut args = [dict, key, hash, run];
        return Ok(LLVMBuildCall(builder, self.key_exists.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    /// Generates the `size` method.
    ///
    /// This method returns the number of keys in a dictionary.
    pub unsafe fn gen_size(&mut self,
                          builder: LLVMBuilderRef,
                          intrinsics: &Intrinsics,
                          run: LLVMValueRef,
                          dict: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.size.is_none() {
            let mut arg_tys = [self.dict_ty, self.run_handle_type()];
            let ret_ty = self.i64_type();

            let name = format!("{}.size", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let dict = LLVMGetParam(function, 0);
            let run = LLVMGetParam(function, 1);

            let dict = LLVMBuildBitCast(builder, dict, self.void_pointer_type(), c_str!(""));
            let result = intrinsics.call_size(builder, run, dict, None);

            LLVMBuildRet(builder, result);

            self.size = Some(function);
            LLVMDisposeBuilder(builder);
        }
        let mut args = [dict, run];
        return Ok(LLVMBuildCall(builder, self.size.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))

    }

    /// Returns the `to_vec` method.
    ///
    /// This method converts a dictionary into a vector of key/value pairs.
    pub unsafe fn gen_to_vec(&mut self,
                          builder: LLVMBuilderRef,
                          intrinsics: &Intrinsics,
                          kv_vec_ty: LLVMTypeRef,
                          kv_ty: LLVMTypeRef,
                          run: LLVMValueRef,
                          dict: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.to_vec.is_none() {
            let mut arg_tys = [self.dict_ty, self.run_handle_type()];
            let ret_ty = kv_vec_ty;

            let name = format!("{}.tovec", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            // Compute the (constant) parameters required to convert the dictionary into a
            // key/value vector.
            let mut indices = [self.i32(0), self.i32(1)];
            let offset = LLVMConstGEP(self.null_ptr(kv_ty),
            indices.as_mut_ptr(),
            indices.len() as u32);
            let offset = LLVMConstPtrToInt(offset, self.i32_type());
            let struct_size = LLVMConstTruncOrBitCast(self.size_of(kv_ty), self.i32_type());

            let dict = LLVMGetParam(function, 0);
            let run = LLVMGetParam(function, 1);

            let out_pointer = LLVMBuildAlloca(builder, kv_vec_ty, c_str!(""));

            let dict = LLVMBuildBitCast(builder, dict, self.void_pointer_type(), c_str!(""));
            let out_pointer_opaque = LLVMBuildBitCast(builder, out_pointer, self.void_pointer_type(), c_str!(""));
            // No return type for to_vec.
            let _ = intrinsics.call_to_vec(builder,
                                                run,
                                                dict,
                                                offset,
                                                struct_size,
                                                out_pointer_opaque, None);
            let result = self.load(builder, out_pointer)?;
            LLVMBuildRet(builder, result);

            self.to_vec = Some(function);
            LLVMDisposeBuilder(builder);
        }

        let mut args = [dict, run];
        return Ok(LLVMBuildCall(builder, self.to_vec.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    /// Generates the `free` method to free a dictionary.
    pub unsafe fn gen_free(&mut self,
                          builder: LLVMBuilderRef,
                          intrinsics: &Intrinsics,
                          run: LLVMValueRef,
                          dict: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.free.is_none() {
            let mut arg_tys = [self.dict_ty, self.run_handle_type()];
            let ret_ty = self.void_type();

            let name = format!("{}.free", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let dict = LLVMGetParam(function, 0);
            let run = LLVMGetParam(function, 1);

            let dict = LLVMBuildBitCast(builder, dict, self.void_pointer_type(), c_str!(""));
            let _ = intrinsics.call_free(builder, run, dict, None);

            LLVMBuildRetVoid(builder);

            self.free = Some(function);
            LLVMDisposeBuilder(builder);
        }
        let mut args = [dict, run];
        return Ok(LLVMBuildCall(builder, self.free.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    /// Generates the `serialize` method to serialize a dictionary.
    ///
    /// The type of the serialization buffer is inferred from the `buf` argument.
    pub unsafe fn gen_serialize(&mut self,
                              builder: LLVMBuilderRef,
                              intrinsics: &Intrinsics,
                              run: LLVMValueRef,
                              dictionary: LLVMValueRef,
                              buf: LLVMValueRef,
                              has_pointer: LLVMValueRef,
                              keys_ser: LLVMValueRef,
                              vals_ser: LLVMValueRef) -> WeldResult<LLVMValueRef> {

        if self.serialize.is_none() {
            let ser_ty = LLVMTypeOf(buf);
            let mut arg_tys = [
                self.dict_ty,
                ser_ty,
                self.i1_type(),
                self.void_pointer_type(),
                self.void_pointer_type(),
                self.run_handle_type()
            ];
            let ret_ty = ser_ty;

            let name = format!("{}.serialize", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let buf_pointer = LLVMBuildAlloca(builder, ser_ty, c_str!(""));

            let dict = LLVMGetParam(function, 0);
            let buf = LLVMGetParam(function, 1);
            let has_pointer = LLVMGetParam(function, 2);
            let key_ser = LLVMGetParam(function, 3);
            let val_ser = LLVMGetParam(function, 4);
            let run = LLVMGetParam(function, 5);

            // Store the serialize buffer so we can pass a pointer to it.
            LLVMBuildStore(builder, buf, buf_pointer);
            let buffer_opaque = LLVMBuildBitCast(builder, buf_pointer, self.void_pointer_type(), c_str!(""));
            let flag_as_i32 = LLVMBuildZExt(builder, has_pointer, self.i32_type(), c_str!(""));

            let dict = LLVMBuildBitCast(builder, dict, self.void_pointer_type(), c_str!(""));
            let _ = intrinsics.call_serialize(
                builder,
                run,
                dict,
                buffer_opaque,
                flag_as_i32,
                key_ser,
                val_ser,
                None);

            // The serialize buffer is updated in place.
            let result = self.load(builder, buf_pointer)?;
            LLVMBuildRet(builder, result);

            self.serialize = Some(function);
            LLVMDisposeBuilder(builder);
        }
        let mut args = [dictionary, buf, has_pointer, keys_ser, vals_ser, run];
        return Ok(LLVMBuildCall(builder, self.serialize.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }
}
