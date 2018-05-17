
extern crate libc;

use libc::{c_char, c_void, int64_t};

use std::ptr;
use std::ffi::CStr;

use super::*;
use super::common::WeldRuntimeErrno;

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
/// Creates a new configuration.
///
/// This function is a wrapper for `WeldConf::new`.
pub extern "C" fn weld_conf_new() -> *mut WeldConf {
    Box::into_raw(Box::new(WeldConf::new()))
}

#[no_mangle]
/// Free a configuration.
///
/// The passsed configuration should have been allocated using `weld_conf_new`.
pub unsafe extern "C" fn weld_conf_free(ptr: *mut WeldConf) {
    if ptr != ptr::null_mut() {
        Box::from_raw(ptr);
    }
}

#[no_mangle]
/// Get a value associated with a key for a configuration.
///
/// This function is a wrapper for `WeldConf::get`.
pub unsafe extern "C" fn weld_conf_get(ptr: *const WeldConf, key: *const c_char) -> *const c_char {
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
pub unsafe extern "C" fn weld_conf_set(ptr: *mut WeldConf, key: *const c_char, value: *const c_char) {
    let conf = &mut *ptr;
    let key = key.to_string();
    let val = CStr::from_ptr(value).to_owned();
    conf.set(key, val);
}

#[no_mangle]
/// Returns a new Weld value.
///
/// The value returned by this method is *not* owned by the runtime, so any data this value refers
/// to must be managed by the caller.
///
/// This function is a wrapper for `WeldValue::new_from_data`.
pub extern "C" fn weld_value_new(data: *const c_void) -> *mut WeldValue {
    Box::into_raw(Box::new(WeldValue::new_from_data(data)))
}

#[no_mangle]
/// Returns the Run ID of the value.
///
/// If this value was not returned by a Weld program, this function returns `-1`.
pub unsafe extern "C" fn weld_value_run(value: *const WeldValue) -> int64_t {
    let value = &*value;
    return value.run_id().unwrap_or(-1) as int64_t;
}

#[no_mangle]
/// Returns the data pointer for a Weld value.
///
/// This function is a wrapper for `WeldValue::data`.
pub unsafe extern "C" fn weld_value_data(value: *const WeldValue) -> *const c_void {
    let value = &*value;
    value.data()
}

#[no_mangle]
/// Frees a Weld value.
///
/// All Weld values must be freed using this call. Weld values which are owned by the runtime also
/// free the data they contain.  Weld values which are not owned by the runtime only free the
/// structure used to wrap the data; the actual data itself is owned by the caller.
pub unsafe extern "C" fn weld_value_free(value: *mut WeldValue) {
    if value != ptr::null_mut() {
        Box::from_raw(value);
    }
}

#[no_mangle]
/// Gets the memory allocated to a Weld value.
///
/// This generally includes all live memory allocated during the run that produced the value.  it.
/// This function returns `-1` if the value was not returned by a Weld program.
pub unsafe extern "C" fn weld_value_memory_usage(value: *mut WeldValue) -> int64_t {
    let value = &mut *value;
    value.memory_usage().unwrap_or(-1) as int64_t
}

#[no_mangle]
/// Compiles a Weld program into a runnable module.
///
/// The compilation can be configured using the passed configuration pointer. This function stores
/// an error (which indicates a compilation error or success) in `err`.
///
/// This function is a wrapper for `WeldModule::compile`.
pub unsafe extern "C" fn weld_module_compile(code: *const c_char,
                                             conf: *const WeldConf,
                                             err: *mut WeldError)
                                             -> *mut WeldModule {
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
pub unsafe extern "C" fn weld_module_run(module: *mut WeldModule,
                                         conf: *const WeldConf,
                                         arg: *const WeldValue,
                                         err: *mut WeldError)
                                         -> *mut WeldValue {
    let module = &mut *module;
    let conf = &*conf;
    let arg = &*arg;
    let err = &mut *err;

    match module.run(conf, arg) {
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
pub unsafe extern "C" fn weld_module_free(ptr: *mut WeldModule) {
    if ptr != ptr::null_mut() {
        Box::from_raw(ptr);
    }
}

#[no_mangle]
/// Creates a new Weld error object.
pub extern "C" fn weld_error_new() -> *mut WeldError {
    Box::into_raw(Box::new(WeldError::default()))
}

#[no_mangle]
/// Returns the error code for a Weld error.
///
/// This function is a wrapper for `WeldError::code`.
pub unsafe extern "C" fn weld_error_code(err: *const WeldError) -> WeldRuntimeErrno {
    let err = &*err;
    err.code()
}

#[no_mangle]
/// Returns a Weld error message.
///
/// This function is a wrapper for `WeldError::message`.
pub unsafe extern "C" fn weld_error_message(err: *const WeldError) -> *const c_char {
    let err = &*err;
    err.message().as_ptr()
}

#[no_mangle]
/// Frees a Weld error object.
pub unsafe extern "C" fn weld_error_free(err: *mut WeldError) {
    if err != ptr::null_mut() {
        Box::from_raw(err);
    }
}


#[no_mangle]
/// Load a dynamic library that a Weld program can access.
///
/// The dynamic library is a C dynamic library identified by its filename.
/// This function is a wrapper for `load_linked_library`.
pub unsafe extern "C" fn weld_load_library(filename: *const c_char, err: *mut WeldError) {
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
