//! A wrapper for dictionaries in Weld.

// Suppress annoying warnings for now.
#![allow(unused_variables,unused_imports)]

extern crate lazy_static;
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
    pub key_comparator: LLVMValueRef,
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

    pub unsafe fn key_comparator_type(&self) -> LLVMTypeRef {
        let mut arg_tys = [self.void_pointer_type(), self.void_pointer_type()];
        let fn_type = LLVMFunctionType(self.i32_type(), arg_tys.as_mut_ptr(), arg_tys.len() as u32, 0);
        LLVMPointerType(fn_type, 0)
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

        // Parameters: Run Handle, Key Size, Value Size, KeyComparator, Capacity.
        // Returns: Pointer to Dictionary
        let mut params = vec![
            self.run_handle_type(),
            self.i64_type(),
            self.i64_type(),
            self.key_comparator_type(),
            self.i64_type()
        ];
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
        let mut layout = [LLVMPointerType(LLVMInt8TypeInContext(context), 0)];
        let dict_ty = LLVMStructCreateNamed(context, c_name.as_ptr());
        LLVMStructSetBody(dict_ty, layout.as_mut_ptr(), layout.len() as u32, 0);

        let name = c_name.into_string().unwrap();

        // Create the entry type.
        let mut layout = [
            LLVMInt32TypeInContext(context),
            LLVMInt8TypeInContext(context),
            LLVMInt8TypeInContext(context),
            LLVMInt16TypeInContext(context),
            key_ty,
            val_ty
        ];


        let entry_name = format!("{}.entry", &name);
        let c_entry_name = CString::new(entry_name).unwrap();
        let entry_ty = LLVMStructCreateNamed(context, c_entry_name.as_ptr());
        LLVMStructSetBody(entry_ty, layout.as_mut_ptr(), layout.len() as u32, 0);

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
            key_exists: None,
            size: None,
            to_vec: None,
            free: None,
        }
    }

    unsafe fn hash_type(&self) -> LLVMTypeRef {
        self.i32_type()
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
            let key_size = self.size_of(self.key_ty);
            let val_size = self.size_of(self.val_ty);
            let result = intrinsics.call_new(builder, run, key_size, val_size, self.key_comparator, capacity, None);

            // Wrap the pointer in the struct.
            let result = LLVMBuildInsertValue(builder, LLVMGetUndef(self.dict_ty), result, 0, c_str!(""));

            LLVMBuildRet(builder, result);

            self.new = Some(function);
            LLVMDisposeBuilder(builder);
        }
        let mut args = [capacity, run];
        return Ok(LLVMBuildCall(builder, self.new.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
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

            let dict = LLVMBuildExtractValue(builder, dict, 0, c_str!(""));
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

            let dict = LLVMBuildExtractValue(builder, dict, 0, c_str!(""));
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

            let dict = LLVMBuildExtractValue(builder, dict, 0, c_str!(""));
            let key = LLVMBuildBitCast(builder, key, self.void_pointer_type(), c_str!(""));
            let slot_pointer = intrinsics.call_key_exists(builder, run, dict, key, hash, None);
            let slot_pointer = LLVMBuildBitCast(builder, slot_pointer, self.slot_ty, c_str!(""));
            let value_pointer = LLVMBuildStructGEP(builder, slot_pointer, FILLED_INDEX, c_str!(""));
            let filled = self.load(builder, value_pointer)?;
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

            let dict = LLVMBuildExtractValue(builder, dict, 0, c_str!(""));
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
                          run: LLVMValueRef,
                          dict: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        if self.to_vec.is_none() {
            let mut arg_tys = [self.dict_ty, self.run_handle_type()];
            let ret_ty = kv_vec_ty;

            let name = format!("{}.tovec", self.name);
            let (function, builder, _) = self.define_function(ret_ty, &mut arg_tys, name);

            let dict = LLVMGetParam(function, 0);
            let run = LLVMGetParam(function, 1);

            let dict = LLVMBuildExtractValue(builder, dict, 0, c_str!(""));
            let pointer = intrinsics.call_to_vec(builder, run, dict, None);
            let result = LLVMBuildBitCast(builder, pointer, LLVMPointerType(kv_vec_ty, 0), c_str!(""));
            let result = self.load(builder, result)?;

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

            let dict = LLVMBuildExtractValue(builder, dict, 0, c_str!(""));
            let _ = intrinsics.call_free(builder, run, dict, None);

            LLVMBuildRetVoid(builder);

            self.free = Some(function);
            LLVMDisposeBuilder(builder);
        }
        let mut args = [dict, run];
        return Ok(LLVMBuildCall(builder, self.free.unwrap(), args.as_mut_ptr(), args.len() as u32, c_str!("")))
    }
}
