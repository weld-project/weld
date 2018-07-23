//! A wrapper for dictionaries in Weld.

// Suppress annoying warnings for now.
#![allow(unused_variables,unused_imports)]

extern crate llvm_sys;
extern crate libc;

use libc::c_char;

use std::ffi::CString;

use ast::Type;
use error::*;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;

use codegen::llvm2::llvm_exts::LLVMExtAttribute::*;
use codegen::llvm2::llvm_exts::*;

use super::LLVM_VECTOR_WIDTH;
use super::CodeGenExt;
use super::FunctionContext;
use super::LlvmGenerator;
use super::intrinsic;

pub const HASH_INDEX: u32 = 0;
pub const FILLED_INDEX: u32 = 1;
pub const LOCK_INDEX: u32 = 2;
pub const KEY_INDEX: u32 = 4;
pub const VALUE_INDEX: u32 = 5;

pub struct Dict {
    pub name: String,
    pub dict_ty: LLVMTypeRef,
    pub key_ty: LLVMTypeRef,
    pub val_ty: LLVMTypeRef,
    entry_ty: LLVMTypeRef,
    slot_ty: LLVMTypeRef,
    context: LLVMContextRef,
    module: LLVMModuleRef,
    new: Option<LLVMValueRef>,
    get: Option<LLVMValueRef>,
    upsert: Option<LLVMValueRef>,
    key_exists: Option<LLVMValueRef>,
    size: Option<LLVMValueRef>,
    to_vec: Option<LLVMValueRef>,
    free: Option<LLVMValueRef>,
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
    /// Arguments: Run Handle
    /// Returns: Newly allocated dictionary
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
    /// Returns: vector of Key/Value structs.
    to_vec: Option<LLVMValueRef>,
    /// Free a dictionary.
    free: Option<LLVMValueRef>,
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
    pub unsafe fn new(context: LLVMContextRef, module: LLVMModuleRef) -> Intrinsics {
        let mut intrinsics = Intrinsics {
            new: None,
            get: None,
            upsert: None,
            key_exists: None,
            size: None,
            to_vec: None,
            free: None,
            context: context,
            module: module,
        };
        intrinsics.populate();
        intrinsics
    }

    /// Generate a call to the `new` intrinsic.
    pub unsafe fn call_new(&self, builder: LLVMBuilderRef,
                           run: LLVMValueRef,
                           capacity: LLVMValueRef,
                           name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run, capacity];
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
                              name: Option<*const c_char>) -> LLVMValueRef {
        let mut args = [run, dictionary];
        LLVMBuildCall(builder, self.to_vec.unwrap(), args.as_mut_ptr(), args.len() as u32, name.unwrap_or(c_str!("")))
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

    /// Populate `self` with the dictionary intrinsics.
    unsafe fn populate(&mut self) {
        // Common types.
        let dict_type = LLVMPointerType(self.i8_type(), 0);
        let void_ptr = LLVMPointerType(self.i8_type(), 0);
        let hash_type = self.i32_type();

        // Parameters: Run Handle, Capacity.
        // Returns: Pointer to Dictionary
        let mut params = vec![self.run_handle_type(), self.i64_type()];
        let name = CString::new("weld_runst_dict_new").unwrap();
        let fn_type = LLVMFunctionType(dict_type, params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        // Run handle
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull], 0);
        LLVMExtAddAttrsOnReturn(self.context, function, &[NoAlias]);
        self.new = Some(function);

        // Parameters: Run Handle, Dictionary, Key, Hash
        // Returns: `i32` that will be 0 or 1 depending on whether the key exists.
        let mut params = vec![self.run_handle_type(), dict_type, void_ptr, hash_type];
        let name = CString::new("weld_runst_dict_get").unwrap();
        let fn_type = LLVMFunctionType(void_ptr, params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        // Run handle
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 0);
        // Dictionary
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 1);
        // Key
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 2);
        self.get = Some(function);

        // Parameters: Run Handle, Dictionary, Key, Hash, Default.
        // Returns: the slot for the given key.
        let mut params = vec![self.run_handle_type(), dict_type, void_ptr, hash_type, void_ptr];
        let name = CString::new("weld_runst_dict_upsert").unwrap();
        let fn_type = LLVMFunctionType(void_ptr, params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        // Run handle
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 0);
        // Dictionary
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull], 1);
        // Key
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, NonNull, ReadOnly], 2);
        // Default value
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, NonNull, ReadOnly], 4);
        self.upsert = Some(function);

        // Parameters: Run Handle, Dictionary, Key, Hash
        // Returns: `i32` that will be 0 or 1 depending on whether the key exists.
        let mut params = vec![self.run_handle_type(), dict_type, void_ptr, hash_type];
        let name = CString::new("weld_runst_dict_keyexists").unwrap();
        let fn_type = LLVMFunctionType(self.i32_type(), params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        // Run handle
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 0);
        // Dictionary
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 1);
        // Key
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 2);
        self.key_exists = Some(function);

        // Parameters: Run Handle, Dictionary
        // Returns: Size of dictionary in # of keys.
        let mut params = vec![self.run_handle_type(), dict_type];
        let name = CString::new("weld_runst_dict_size").unwrap();
        let fn_type = LLVMFunctionType(self.i64_type(), params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        // Run handle
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 0);
        // Dictionary
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 1);
        self.size = Some(function);

        // Parameters: Run Handle, Dictionary
        // Returns: A vector of key/value pairs.
        let mut params = vec![self.run_handle_type(), dict_type];
        let name = CString::new("weld_runst_dict_tovec").unwrap();
        let fn_type = LLVMFunctionType(void_ptr, params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        // Run handle
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 0);
        // Dictionary
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 1);
        LLVMExtAddAttrsOnReturn(self.context, function, &[NoAlias]);
        self.to_vec = Some(function);

        // Parameters: Run Handle, Dictionary
        // Returns: Nothing
        let mut params = vec![self.run_handle_type(), dict_type];
        let name = CString::new("weld_runst_dict_free").unwrap();
        let fn_type = LLVMFunctionType(self.void_type(), params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        // Run handle
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull, ReadOnly], 0);
        // Dictionary
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoAlias, NonNull], 1);
        self.free = Some(function);
    }
}

impl Dict {
    pub unsafe fn define<T: AsRef<str>>(name: T,
                                key_ty: LLVMTypeRef,
                                val_ty: LLVMTypeRef,
                                context: LLVMContextRef,
                                module: LLVMModuleRef) -> Dict {
        let c_name = CString::new(name.as_ref()).unwrap();
        let mut layout = [LLVMPointerType(LLVMInt8TypeInContext(context), 0)];
        let dict_ty = LLVMStructCreateNamed(context, c_name.as_ptr());
        LLVMStructSetBody(dict_ty, layout.as_mut_ptr(), layout.len() as u32, 0);

        // Create the entry type.
        let mut layout = [
            LLVMInt32TypeInContext(context),
            LLVMInt8TypeInContext(context),
            LLVMInt8TypeInContext(context),
            LLVMInt16TypeInContext(context),
            key_ty,
            val_ty
        ];

        let mut name = c_name.into_string().unwrap();
        name.push_str(".entry");
        let c_name = CString::new(name).unwrap();
        let entry_ty = LLVMStructCreateNamed(context, c_name.as_ptr());
        LLVMStructSetBody(entry_ty, layout.as_mut_ptr(), layout.len() as u32, 0);

        Dict {
            name: c_name.into_string().unwrap(),
            dict_ty: dict_ty,
            key_ty: key_ty,
            val_ty: val_ty,
            entry_ty: entry_ty,
            slot_ty: LLVMPointerType(entry_ty, 0),
            context: context,
            module: module,
            new: None,
            get: None,
            upsert: None,
            key_exists: None,
            size: None,
            to_vec: None,
            free: None,
        }
    }

    pub unsafe fn gen_new(&mut self,
                          builder: LLVMBuilderRef,
                          capacity: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.new.is_none() {
            unimplemented!()
        }
        let mut args = [capacity];
        return Ok(LLVMBuildCall(builder, self.new.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    pub unsafe fn gen_upsert(&mut self,
                          builder: LLVMBuilderRef,
                          dict: LLVMValueRef,
                          key: LLVMValueRef,
                          hash: LLVMValueRef,
                          val: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.upsert.is_none() {
            unimplemented!()
        }
        let mut args = [dict, key, hash, val];
        return Ok(LLVMBuildCall(builder, self.upsert.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    pub unsafe fn gen_get(&mut self,
                          builder: LLVMBuilderRef,
                          dict: LLVMValueRef,
                          key: LLVMValueRef,
                          hash: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.get.is_none() {
            unimplemented!()
        }
        let mut args = [dict, key, hash];
        return Ok(LLVMBuildCall(builder, self.get.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    pub unsafe fn gen_key_exists(&mut self,
                          builder: LLVMBuilderRef,
                          dict: LLVMValueRef,
                          key: LLVMValueRef,
                          hash: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.key_exists.is_none() {
            unimplemented!()
        }
        let mut args = [dict, key, hash];
        return Ok(LLVMBuildCall(builder, self.key_exists.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    pub unsafe fn gen_size(&mut self,
                          builder: LLVMBuilderRef,
                          dict: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.size.is_none() {
            unimplemented!()
        }
        let mut args = [dict];
        return Ok(LLVMBuildCall(builder, self.size.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))

    }

    pub unsafe fn gen_to_vec(&mut self,
                          builder: LLVMBuilderRef,
                          dict: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.to_vec.is_none() {
            unimplemented!()
        }
        let mut args = [dict];
        return Ok(LLVMBuildCall(builder, self.to_vec.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }

    pub unsafe fn gen_free(&mut self,
                          builder: LLVMBuilderRef,
                          dict: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.free.is_none() {
            unimplemented!()
        }
        let mut args = [dict];
        return Ok(LLVMBuildCall(builder, self.free.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }
}
