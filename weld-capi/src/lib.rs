//! A C-compatible foreign function interface to Weld.
//!
//! This crate provides a C-compatible interface to the Weld library and is compiled as a shared
//! library and a static library.
//!
//! See the docs for the main Weld crate for more details on these functions.

use weld;

use libc::{c_char, c_void};

use std::ffi::CStr;
use std::ptr;

// Re-export the FFI for the runtime.
pub use weld::runtime::ffi::*;

// These variants should never be constructed: they aren't 0-variant structs to enforce their
// pointer alignment.
#[repr(u64)]
pub enum WeldConf {
    _A,
}
#[repr(u64)]
pub enum WeldContext {
    _A,
}
#[repr(u64)]
pub enum WeldError {
    _A,
}
#[repr(u64)]
pub enum WeldModule {
    _A,
}
#[repr(u64)]
pub enum WeldValue {
    _A,
}

/// An opaque handle to a Weld configuration.
#[allow(non_camel_case_types)]
pub type weld_conf_t = *mut WeldConf;
/// An opaque handle to a Weld context.
#[allow(non_camel_case_types)]
pub type weld_context_t = *mut WeldContext;
/// An opaque handle to a Weld error.
#[allow(non_camel_case_types)]
pub type weld_error_t = *mut WeldError;
/// An opaque handle to a Weld module.
#[allow(non_camel_case_types)]
pub type weld_module_t = *mut WeldModule;
/// An opaque handle to a Weld data value.
#[allow(non_camel_case_types)]
pub type weld_value_t = *mut WeldValue;

pub use weld::WeldLogLevel;
pub use weld::WeldRuntimeErrno;

#[allow(non_camel_case_types)]
pub type weld_log_level_t = WeldLogLevel;
#[allow(non_camel_case_types)]
pub type weld_errno_t = WeldRuntimeErrno;

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
pub unsafe extern "C" fn weld_context_new(conf: weld_conf_t) -> weld_context_t {
    let conf = conf as *mut weld::WeldConf;
    let conf = &*conf;
    if let Ok(context) = weld::WeldContext::new(conf) {
        Box::into_raw(Box::new(context)) as _
    } else {
        // XXX Should this take an error too?
        ptr::null_mut()
    }
}

#[no_mangle]
/// Gets the memory allocated by a Weld context.
///
/// This includes all live memory allocated in the given context.
pub unsafe extern "C" fn weld_context_memory_usage(context: weld_context_t) -> i64 {
    let context = context as *mut weld::WeldContext;
    let context = &*context;
    context.memory_usage()
}

#[no_mangle]
/// Frees a context.
///
/// All contexts created by the FFI should be freed. This includes contexts obtained from
/// `weld_context_new` and `weld_value_context`.
pub unsafe extern "C" fn weld_context_free(context: weld_context_t) {
    let context = context as *mut weld::WeldContext;
    if !context.is_null() {
        Box::from_raw(context);
    }
}

#[no_mangle]
/// Creates a new configuration.
///
/// This function is a wrapper for `WeldConf::new`.
pub extern "C" fn weld_conf_new() -> weld_conf_t {
    Box::into_raw(Box::new(weld::WeldConf::new())) as _
}

#[no_mangle]
/// Free a configuration.
///
/// The passsed configuration should have been allocated using `weld_conf_new`.
pub unsafe extern "C" fn weld_conf_free(conf: weld_conf_t) {
    let conf = conf as *mut weld::WeldConf;
    if !conf.is_null() {
        Box::from_raw(conf);
    }
}

#[no_mangle]
/// Get a value associated with a key for a configuration.
///
/// This function is a wrapper for `WeldConf::get`.
pub unsafe extern "C" fn weld_conf_get(conf: weld_conf_t, key: *const c_char) -> *const c_char {
    let conf = conf as *mut weld::WeldConf;
    let conf = &*conf;
    match conf.get(key.to_str()) {
        Some(k) => k.as_ptr(),
        None => ptr::null_mut(),
    }
}

#[no_mangle]
/// Set the value of `key` to `value`.
///
/// This function is a wrapper for `WeldConf::set`.
pub unsafe extern "C" fn weld_conf_set(
    conf: weld_conf_t,
    key: *const c_char,
    value: *const c_char,
) {
    let conf = conf as *mut weld::WeldConf;
    let conf = &mut *conf;
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
pub extern "C" fn weld_value_new(data: *const c_void) -> weld_value_t {
    Box::into_raw(Box::new(weld::WeldValue::new_from_data(data))) as _
}

#[no_mangle]
/// Returns the Run ID of the value.
///
/// If this value was not returned by a Weld program, this function returns `-1`.
pub unsafe extern "C" fn weld_value_run(value: weld_value_t) -> i64 {
    let value = value as *mut weld::WeldValue;
    let value = &*value;
    value.run_id().unwrap_or(-1)
}

#[no_mangle]
/// Returns the context of a value.
///
/// Since contexts are internally reference-counted, this function increases the reference count of
/// the context. The context must be freed with `weld_context_free` to decrement the internal
/// reference count. Since Weld values owned by a context internally hold a reference to the
/// context, the value is guaranteed to be live until `weld_value_free` is called.
pub unsafe extern "C" fn weld_value_context(value: weld_value_t) -> weld_context_t {
    let value = value as *mut weld::WeldValue;
    let value = &*value;
    if let Some(context) = value.context() {
        Box::into_raw(Box::new(context)) as _
    } else {
        ptr::null_mut() as _
    }
}

#[no_mangle]
/// Returns the data pointer for a Weld value.
///
/// This function is a wrapper for `WeldValue::data`. If this value is owned by the runtime, the
/// returned pointer should never be freed -- instead, use `weld_value_free` to free the data.
pub unsafe extern "C" fn weld_value_data(value: weld_value_t) -> *mut c_void {
    let value = value as *mut weld::WeldValue;
    let value = &*value;
    value.data() as _
}

#[no_mangle]
/// Frees a Weld value.
///
/// All Weld values must be freed using this call. Weld values which are owned by the runtime also
/// free the data they contain.  Weld values which are not owned by the runtime only free the
/// structure used to wrap the data; the actual data itself is owned by the caller.
pub unsafe extern "C" fn weld_value_free(value: weld_value_t) {
    let value = value as *mut weld::WeldValue;
    if !value.is_null() {
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
pub unsafe extern "C" fn weld_module_compile(
    code: *const c_char,
    conf: weld_conf_t,
    err: weld_error_t,
) -> weld_module_t {
    let code = code.to_str();
    let conf = conf as *mut weld::WeldConf;
    let conf = &*conf;
    let err = err as *mut weld::WeldError;
    let err = &mut *err;
    match weld::WeldModule::compile(code, conf) {
        Ok(module) => {
            *err = weld::WeldError::new_success();
            Box::into_raw(Box::new(module)) as _
        }
        Err(compile_err) => {
            *err = compile_err;
            ptr::null_mut() as _
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
pub unsafe extern "C" fn weld_module_run(
    module: weld_module_t,
    context: weld_context_t,
    arg: weld_value_t,
    err: weld_error_t,
) -> weld_value_t {
    let module = module as *mut weld::WeldModule;
    let module = &mut *module;
    let context = context as *mut weld::WeldContext;
    let context = &mut *context;
    let arg = arg as *mut weld::WeldValue;
    let arg = &*arg;
    let err = err as *mut weld::WeldError;
    let err = &mut *err;

    match module.run(context, arg) {
        Ok(result) => {
            *err = weld::WeldError::new_success();
            Box::into_raw(Box::new(result)) as _
        }
        Err(runtime_err) => {
            eprintln!("{:?}", runtime_err);
            *err = runtime_err;
            ptr::null_mut() as _
        }
    }
}

#[no_mangle]
/// Frees a module.
///
/// Freeing a module does not free the memory it may have allocated. Values returned by the module
/// must be freed explicitly using `weld_value_free`.
pub unsafe extern "C" fn weld_module_free(module: weld_module_t) {
    let module = module as *mut weld::WeldModule;
    if !module.is_null() {
        Box::from_raw(module);
    }
}

#[no_mangle]
/// Creates a new Weld error object.
pub extern "C" fn weld_error_new() -> weld_error_t {
    Box::into_raw(Box::new(weld::WeldError::default())) as _
}

#[no_mangle]
/// Returns the error code for a Weld error.
///
/// This function is a wrapper for `WeldError::code`.
pub unsafe extern "C" fn weld_error_code(err: weld_error_t) -> u64 {
    let err = err as *mut weld::WeldError;
    let err = &*err;
    err.code() as _
}

#[no_mangle]
/// Returns a Weld error message.
///
/// This function is a wrapper for `WeldError::message`.
pub unsafe extern "C" fn weld_error_message(err: weld_error_t) -> *const c_char {
    let err = err as *mut weld::WeldError;
    let err = &*err;
    err.message().as_ptr()
}

#[no_mangle]
/// Frees a Weld error object.
pub unsafe extern "C" fn weld_error_free(err: weld_error_t) {
    let err = err as *mut weld::WeldError;
    if !err.is_null() {
        Box::from_raw(err);
    }
}

#[no_mangle]
/// Load a dynamic library that a Weld program can access.
///
/// The dynamic library is a C dynamic library identified by its filename.
/// This function is a wrapper for `load_linked_library`.
pub unsafe extern "C" fn weld_load_library(filename: *const c_char, err: weld_error_t) {
    let err = err as *mut weld::WeldError;
    let err = &mut *err;
    let filename = filename.to_str();
    if let Err(e) = weld::load_linked_library(filename) {
        *err = e;
    } else {
        *err = weld::WeldError::new_success();
    }
}

#[no_mangle]
/// Enables logging to stderr in Weld with the given log level.
///
/// This function is ignored if it has already been called once, or if some other code in the
/// process has initialized logging using Rust's `log` crate.
pub extern "C" fn weld_set_log_level(level: u64) {
    weld::set_log_level(level.into())
}
