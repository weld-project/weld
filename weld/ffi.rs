
extern crate libc;

use libc::{c_char, c_void, int64_t};

use std::ptr;
use std::ffi::CStr;

use super::api::*;
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
        // TODO make this nicer?
        c_str.to_str().unwrap()
    }
}

#[no_mangle]
/// Return a new Weld configuration.
pub extern "C" fn weld_conf_new() -> *mut WeldConf {
    Box::into_raw(Box::new(WeldConf::new()))
}

#[no_mangle]
/// Free a Weld configuration.
pub unsafe extern "C" fn weld_conf_free(ptr: *mut WeldConf) {
    Box::from_raw(ptr);
}

#[no_mangle]
/// Get a value for a given configuration.
pub unsafe extern "C" fn weld_conf_get(ptr: *const WeldConf, key: *const c_char) -> *const c_char {
    let conf = &*ptr;
    match conf.get(key.to_str()) {
        Some(k) => k.as_ptr(),
        None => ptr::null_mut(),
    }
}

#[no_mangle]
/// Set the value of `key` to `value`.
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
pub extern "C" fn weld_value_new(data: *const c_void) -> *mut WeldValue {
    Box::into_raw(Box::new(WeldValue::new_from_data(data)))
}

#[no_mangle]
/// Returns the Run ID of the value if the value is owned by the Weld runtime, or -1 if the caller
/// owns the value.
pub unsafe extern "C" fn weld_value_run(value: *const WeldValue) -> int64_t {
    let value = &*value;
    return value.run_id().unwrap_or(-1) as int64_t;
}

#[no_mangle]
/// Returns a pointer to the data wrapped by the given Weld value.
pub unsafe extern "C" fn weld_value_data(value: *const WeldValue) -> *const c_void {
    let value = &*value;
    value.data()
}

#[no_mangle]
/// Frees a Weld value.
///
/// All Weld values must be freed using this call.
/// Weld values which are owned by the runtime also free the data they contain.
/// Weld values which are not owned by the runtime only free the structure used
/// to wrap the data; the actual data itself is owned by the caller.
pub unsafe extern "C" fn weld_value_free(value: *mut WeldValue) {
    let value = &mut *value;
    Box::from_raw(value);
}

#[no_mangle]
/// Gets the memory allocated to a Weld value, which generally includes all memory allocated during
/// the execution of the module that created it (Weld does not currently free intermediate values
/// while running a module). Returns -1 if the value is not part of a valid Weld run.
pub unsafe extern "C" fn weld_value_memory_usage(value: *mut WeldValue) -> int64_t {
    let value = &mut *value;
    value.memory_usage().unwrap_or(-1) as int64_t
}

#[no_mangle]
/// Given some Weld code and a configuration, returns a runnable Weld module.
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
/// Runs a module.
///
/// The module may write a value into the provided error pointer.
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
    Box::from_raw(ptr);
}

#[no_mangle]
/// Return a new Weld error object.
pub extern "C" fn weld_error_new() -> *mut WeldError {
    Box::into_raw(Box::new(WeldError::default()))
}

#[no_mangle]
/// Returns an error code for a Weld error object.
pub unsafe extern "C" fn weld_error_code(err: *const WeldError) -> WeldRuntimeErrno {
    let err = &*err;
    err.code()
}

#[no_mangle]
/// Returns a Weld error message.
pub unsafe extern "C" fn weld_error_message(err: *const WeldError) -> *const c_char {
    let err = &*err;
    err.message().as_ptr()
}

#[no_mangle]
/// Frees a Weld error object.
pub unsafe extern "C" fn weld_error_free(err: *mut WeldError) {
    Box::from_raw(err);
}
