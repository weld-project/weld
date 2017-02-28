// Disable dead code macros in "build" mode but keep them on in "test" builds so that we don't
// get spurious warnings for functions that are currently only used in tests. This is useful for
// development but can be removed later.
#![cfg_attr(not(test), allow(dead_code))]

#[macro_use]
extern crate lazy_static;
extern crate regex;
extern crate easy_ll;
extern crate libc;

extern crate weld_common;

use std::collections::HashMap;
use std::error::Error;
use libc::{c_char, c_void};
use std::ffi::{CStr, CString};

use weld_common::WeldRuntimeErrno;

/// Utility macro to create an Err result with a WeldError from a format string.
macro_rules! weld_err {
    ( $($arg:tt)* ) => ({
        ::std::result::Result::Err($crate::error::WeldError::new(format!($($arg)*)))
    })
}

pub mod ast;
pub mod code_builder;
pub mod error;
pub mod llvm;
pub mod macro_processor;
pub mod parser;
pub mod partial_types;
pub mod pretty_print;
pub mod program;
pub mod sir;
pub mod tokenizer;
pub mod transforms;
pub mod type_inference;
pub mod conf;
pub mod util;

/// A clean alias for a compiled LLVM module.
pub type WeldModule = easy_ll::CompiledModule;

/// An error passed as an opaque pointer using the runtime API.
pub struct WeldError {
    errno: WeldRuntimeErrno,
    message: CString,
}

impl WeldError {
    /// Returns a new WeldError where the message is just the errno name.
    fn new(errno: WeldRuntimeErrno) -> WeldError {
        WeldError {
            errno: errno,
            message: CString::new(errno.to_string()).unwrap(),
        }
    }
}

/// A value passed as an opaque pointer using the runtime API. `WeldValue` encapsulates some raw
/// data along with a `run_id` which designates the run a value is associated with. If the `run_id`
/// is `None` the data is owned by a caller outside the runtime.
pub struct WeldValue {
    data: *const c_void,
    run_id: Option<i64>,
}

/// A configuration passed into the runtime to set compilation or execution parameters. This
/// struct has two methods other than the ones for allocation: `get` and `set`. `get` returns
/// the current string value of a configuration parameter, while `set` associates a string
/// parameter with a value.
pub struct WeldConf {
    dict: HashMap<CString, CString>,
}

impl WeldConf {
    fn new() -> WeldConf {
        WeldConf { dict: HashMap::new() }
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
    assert!(!ptr.is_null());
    Box::from_raw(ptr);
}

#[no_mangle]
/// Get a value for a given configuration.
pub unsafe extern "C" fn weld_conf_get(ptr: *const WeldConf, key: *const c_char) -> *const c_char {
    assert!(!ptr.is_null());
    let conf = &*ptr;

    let key = CStr::from_ptr(key);
    let key = key.to_owned();
    match conf.dict.get(&key) {
        Some(k) => k.as_ptr() as *const c_char,
        None => std::ptr::null_mut() as *const c_char,
    }
}

#[no_mangle]
/// Set the value of `key` to `value`.
pub unsafe extern "C" fn weld_conf_set(ptr: *mut WeldConf,
                                       key: *const c_char,
                                       value: *const c_char) {
    assert!(!ptr.is_null());
    let mut conf = &mut *ptr;

    let key = CStr::from_ptr(key);
    let key = key.to_owned();
    let val = CStr::from_ptr(value);
    let val = val.to_owned();

    conf.dict.insert(key, val);
}

#[no_mangle]
/// Returns a new Weld value.
///
/// The value returned by this method is *not* owned by the runtime, so any data this value refers
/// to must be managed by the caller.
pub extern "C" fn weld_value_new(data: *const c_void) -> *mut WeldValue {
    Box::into_raw(Box::new(WeldValue {
        data: data,
        run_id: None,
    }))
}

#[no_mangle]
/// Returns the Run ID of the value if the value is owned by the Weld runtime, or -1 if the caller
/// owns the value.
pub unsafe extern "C" fn weld_value_run(obj: *const WeldValue) -> i64 {
    assert!(!obj.is_null());
    let obj = &*obj;
    if let Some(rid) = obj.run_id {
        rid as i64
    } else {
        -1
    }
}

#[no_mangle]
/// Returns a pointer to the data wrapped by the given Weld value.
pub unsafe extern "C" fn weld_value_data(obj: *const WeldValue) -> *const c_void {
    assert!(!obj.is_null());
    let obj = &*obj;
    obj.data
}

#[no_mangle]
/// Frees a Weld value.
///
/// All Weld values must be freed using this call.
/// Weld values which are owned by the runtime also free the data they contain.
/// Weld values which are not owned by the runtime only free the structure used
/// to wrap the data; the actual data itself is owned by the caller.
pub unsafe extern "C" fn weld_value_free(obj: *mut WeldValue) {
    if obj.is_null() {
        return;
    }
    let value = &mut *obj;
    if let Some(run_id) = value.run_id {
        // TODO(shoumik): Cache this? No need to compile it each time.
        let module = llvm::generate_runtime_interface_module().unwrap();
        let arg = run_id;
        let _ = module.run(arg);
    }
    Box::from_raw(obj);
}

#[no_mangle]
/// Given some Weld code and a configuration, returns a runnable Weld module.
pub unsafe extern "C" fn weld_module_compile(code: *const c_char,
                                             _: *const WeldConf,
                                             err: *mut *mut WeldError)
                                             -> *mut WeldModule {
    assert!(!code.is_null());

    let code = CStr::from_ptr(code);
    let code = code.to_str().unwrap().trim();

    *err = Box::into_raw(Box::new(WeldError::new(WeldRuntimeErrno::Success)));
    let err = &mut **err;

    if let Err(e) = util::load_runtime_library() {
        println!("{}", e);
    }

    let module = llvm::compile_program(&parser::parse_program(code).unwrap());

    if let Err(ref e) = module {
        err.errno = WeldRuntimeErrno::CompileError;
        err.message = CString::new(e.description().to_string()).unwrap();
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(module.unwrap()))
}

#[no_mangle]
/// Runs a module.
///
/// The module may write a value into the provided error pointer.
pub unsafe extern "C" fn weld_module_run(module: *mut WeldModule,
                                         conf: *const WeldConf,
                                         arg: *const WeldValue,
                                         err: *mut *mut WeldError)
                                         -> *mut WeldValue {
    assert!(!module.is_null());
    assert!(!conf.is_null());
    assert!(!arg.is_null());

    let module = &mut *module;
    let arg = &*arg;
    let conf = &*conf;

    *err = Box::into_raw(Box::new(WeldError::new(WeldRuntimeErrno::Success)));
    let err = &mut **err;

    let mem_limit = conf::parse_memory_limit(conf.dict
        .get(&CString::new(conf::MEMORY_LIMIT_KEY).unwrap())
        .unwrap_or(&CString::new("").unwrap())
        .clone());

    let threads = conf::parse_threads(conf.dict
        .get(&CString::new(conf::THREADS_KEY).unwrap())
        .unwrap_or(&CString::new("").unwrap())
        .clone());

    let input = Box::new(llvm::WeldInputArgs {
        input: arg.data as i64,
        nworkers: threads as i32,
        mem_limit: mem_limit as i64,
    });
    let ptr = Box::into_raw(input) as i64;
    let result_raw = module.run(ptr) as *const llvm::WeldOutputArgs;
    let result = (*result_raw).clone();

    let ret = Box::into_raw(Box::new(WeldValue {
        data: result.output as *const c_void,
        run_id: Some(result.run_id),
    }));

    if result.errno != WeldRuntimeErrno::Success {
        weld_value_free(ret);
        *err = WeldError::new(result.errno);
        std::ptr::null_mut()
    } else {
        ret
    }
}

#[no_mangle]
/// Frees a module.
///
/// Freeing a module does not free the memory it may have allocated. Values returned by the module
/// must be freed explicitly using `weld_value_free`.
pub unsafe extern "C" fn weld_module_free(ptr: *mut easy_ll::CompiledModule) {
    if ptr.is_null() {
        return;
    }
    Box::from_raw(ptr);
}

#[no_mangle]
/// Returns an error code for a Weld error object.
pub unsafe extern "C" fn weld_error_code(err: *mut WeldError) -> WeldRuntimeErrno {
    // TODO(shoumik): Handle this better.
    if err.is_null() {
        return WeldRuntimeErrno::Success;
    }
    let err = &mut *err;
    err.errno
}

#[no_mangle]
/// Returns a Weld error message.
pub unsafe extern "C" fn weld_error_message(err: *mut WeldError) -> *const c_char {
    if err.is_null() {
        return std::ptr::null();
    }
    let err = &mut *err;
    err.message.as_ptr() as *const c_char
}

#[no_mangle]
/// Frees a Weld error object.
pub unsafe extern "C" fn weld_error_free(err: *mut WeldError) {
    if err.is_null() {
        return;
    }
    Box::from_raw(err);
}

#[cfg(test)]
mod tests;
