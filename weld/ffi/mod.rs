//! C-compatible foreign function interface to Weld.
//!
//! This module provides a C-compatible interface to the Weld library.

extern crate libc;

use super::*;
use libc::{c_char, c_void, int64_t};

use std::ptr;
use std::ffi::CStr;

// Re-export the FFI for the runtime.
pub use runtime::ffi::*;

/// An opauqe handle to a Weld configuration.
pub type WeldConfRef = *mut WeldConf;
/// An opauqe handle to a Weld error.
pub type WeldErrorRef = *mut WeldError;
/// An opauqe handle to a Weld module.
pub type WeldModuleRef = *mut WeldModule;
/// An opauqe handle to a Weld data value.
pub type WeldValueRef = *mut WeldValue;
/// An opauqe handle to a Weld context.
pub type WeldContextRef = *mut WeldContext;

pub use super::WeldRuntimeErrno;
pub use super::WeldLogLevel;

trait ToRustStr {
    fn to_str(&self) -> &str;

    fn to_string(&self) -> String {
        self.to_str().to_string()
    }
}

impl ToRustStr for *const c_char {
    fn to_str(&self) -> &str {
        let c_str = unsafe { CStr::from_ptr(*self) };
        c_str.to_str().unwrap()
    }
}

#[no_mangle]
/// Creates a new context.
///
/// This function is a wrapper for `WeldContext::new`.
pub unsafe extern "C" fn weld_context_new(conf: WeldConfRef) -> WeldContextRef {
    let conf = &*conf;
    if let Ok(context) = WeldContext::new(conf) {
        Box::into_raw(Box::new(context))
    } else {
        // XXX Should this take an error too?
        ptr::null_mut()
    }
}

#[no_mangle]
/// Gets the memory allocated by a Weld context.
///
/// This includes all live memory allocated in the given context.
pub unsafe extern "C" fn weld_context_memory_usage(context: WeldContextRef) -> int64_t {
    let context = &*context;
    context.memory_usage() as int64_t
}

#[no_mangle]
/// Frees a context.
///
/// All contexts created by the FFI should be freed. This includes contexts obtained from
/// `weld_context_new` and `weld_value_context`.
pub unsafe extern "C" fn weld_context_free(ptr: WeldContextRef) {
    if ptr != ptr::null_mut() {
        Box::from_raw(ptr);
    }
}

#[no_mangle]
/// Creates a new configuration.
///
/// This function is a wrapper for `WeldConf::new`.
pub extern "C" fn weld_conf_new() -> WeldConfRef {
    Box::into_raw(Box::new(WeldConf::new()))
}

#[no_mangle]
/// Free a configuration.
///
/// The passsed configuration should have been allocated using `weld_conf_new`.
pub unsafe extern "C" fn weld_conf_free(ptr: WeldConfRef) {
    if ptr != ptr::null_mut() {
        Box::from_raw(ptr);
    }
}

#[no_mangle]
/// Get a value associated with a key for a configuration.
///
/// This function is a wrapper for `WeldConf::get`.
pub unsafe extern "C" fn weld_conf_get(ptr: WeldConfRef, key: *const c_char) -> *const c_char {
    let conf = &*ptr;
    match conf.get(key.to_str()) {
        Some(k) => k.as_ptr(),
        None => ptr::null_mut(),
    }
}

#[no_mangle]
/// Set the value of `key` to `value`.
///
/// This function is a wrapper for `WeldConf::set`.
pub unsafe extern "C" fn weld_conf_set(ptr: WeldConfRef, key: *const c_char, value: *const c_char) {
    let conf = &mut *ptr;
    let key = key.to_string();
    let val = CStr::from_ptr(value).to_owned();
    conf.set(key, val);
}

#[no_mangle]
/// Returns a new Weld value.
///
/// The value returned by this method is *not* owned by the runtime, so any data this value refers
/// to must be managed by the caller. The created value will always have a NULL context.
///
/// This function is a wrapper for `WeldValue::new_from_data`.
pub extern "C" fn weld_value_new(data: *const c_void) -> WeldValueRef {
    Box::into_raw(Box::new(WeldValue::new_from_data(data)))
}

#[no_mangle]
/// Returns the Run ID of the value.
///
/// If this value was not returned by a Weld program, this function returns `-1`.
pub unsafe extern "C" fn weld_value_run(value: WeldValueRef) -> int64_t {
    let value = &*value;
    return value.run_id().unwrap_or(-1) as int64_t;
}

#[no_mangle]
/// Returns the context of a value.
///
/// Since contexts are internally reference-counted, this function increases the reference count of
/// the context. The context must be freed with `weld_context_free` to decrement the internal
/// reference count. Since Weld values owned by a context internally hold a reference to the
/// context, the value is guaranteed to be live until `weld_value_free` is called.
pub unsafe extern "C" fn weld_value_context(value: WeldValueRef) -> WeldContextRef {
    let value = &*value;
    if let Some(context) = value.context() {
        Box::into_raw(Box::new(context))
    } else {
        ptr::null_mut()
    }
}

#[no_mangle]
/// Returns the data pointer for a Weld value.
///
/// This function is a wrapper for `WeldValue::data`. If this value is owned by the runtime, the
/// returned pointer should never be freed -- instead, use `weld_value_free` to free the data.
pub unsafe extern "C" fn weld_value_data(value: WeldValueRef) -> *const c_void {
    let value = &*value;
    value.data()
}

#[no_mangle]
/// Frees a Weld value.
///
/// All Weld values must be freed using this call. Weld values which are owned by the runtime also
/// free the data they contain.  Weld values which are not owned by the runtime only free the
/// structure used to wrap the data; the actual data itself is owned by the caller.
pub unsafe extern "C" fn weld_value_free(value: WeldValueRef) {
    if value != ptr::null_mut() {
        Box::from_raw(value);
    }
}


#[no_mangle]
/// Compiles a Weld program into a runnable module.
///
/// The compilation can be configured using the passed configuration pointer. This function stores
/// an error (which indicates a compilation error or success) in `err`.
///
/// This function is a wrapper for `WeldModule::compile`.
pub unsafe extern "C" fn weld_module_compile(code: *const c_char,
                                             conf: WeldConfRef,
                                             err: WeldErrorRef) -> WeldModuleRef {
    let code = code.to_str();
    let conf = &*conf;
    let err = &mut *err;
    match WeldModule::compile(code, conf) {
        Ok(module) => {
            *err = WeldError::new_success();
            Box::into_raw(Box::new(module))
        }
        Err(compile_err) => {
            *err = compile_err;
            ptr::null_mut()
        }
    }
}

#[no_mangle]
/// Runs a compiled Weld module.
///
/// The module is run with a given configuration and argument list, and returns the result wrapped
/// as a `WeldValue`. If the run raised a runtime error, the method writes the erorr into `err`,
/// and a the method returns `null`. Otherwise, `err` indicates success.
///
/// This function is a wrapper for `WeldModule::run`.
pub unsafe extern "C" fn weld_module_run(module: WeldModuleRef,
                                         context: WeldContextRef,
                                         arg: WeldValueRef,
                                         err: WeldErrorRef) -> WeldValueRef {
    let module = &mut *module;
    let context = &mut *context;
    let arg = &*arg;
    let err = &mut *err;

    match module.run(context, arg) {
        Ok(result) => {
            *err = WeldError::new_success();
            Box::into_raw(Box::new(result))
        }
        Err(runtime_err) => {
            eprintln!("{:?}", runtime_err);
            *err = runtime_err;
            ptr::null_mut()
        }
    }
}

#[no_mangle]
/// Frees a module.
///
/// Freeing a module does not free the memory it may have allocated. Values returned by the module
/// must be freed explicitly using `weld_value_free`.
pub unsafe extern "C" fn weld_module_free(ptr: WeldModuleRef) {
    if ptr != ptr::null_mut() {
        Box::from_raw(ptr);
    }
}

#[no_mangle]
/// Creates a new Weld error object.
pub extern "C" fn weld_error_new() -> WeldErrorRef {
    Box::into_raw(Box::new(WeldError::default()))
}

#[no_mangle]
/// Returns the error code for a Weld error.
///
/// This function is a wrapper for `WeldError::code`.
pub unsafe extern "C" fn weld_error_code(err: WeldErrorRef) -> WeldRuntimeErrno {
    let err = &*err;
    err.code()
}

#[no_mangle]
/// Returns a Weld error message.
///
/// This function is a wrapper for `WeldError::message`.
pub unsafe extern "C" fn weld_error_message(err: WeldErrorRef) -> *const c_char {
    let err = &*err;
    err.message().as_ptr()
}

#[no_mangle]
/// Frees a Weld error object.
pub unsafe extern "C" fn weld_error_free(err: WeldErrorRef) {
    if err != ptr::null_mut() {
        Box::from_raw(err);
    }
}


#[no_mangle]
/// Load a dynamic library that a Weld program can access.
///
/// The dynamic library is a C dynamic library identified by its filename.
/// This function is a wrapper for `load_linked_library`.
pub unsafe extern "C" fn weld_load_library(filename: *const c_char, err: WeldErrorRef) {
    let err = &mut *err;
    let filename = filename.to_str();
    if let Err(e) = load_linked_library(filename) {
        *err = e;
    } else {
        *err = WeldError::new_success();
    }
}

#[no_mangle]
/// Enables logging to stderr in Weld with the given log level.
///
/// This function is ignored if it has already been called once, or if some other code in the
/// process has initialized logging using Rust's `log` crate.
pub extern "C" fn weld_set_log_level(level: WeldLogLevel) {
    set_log_level(level)
}
