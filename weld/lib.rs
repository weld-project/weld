// Disable dead code macros in "build" mode but keep them on in "test" builds so that we don't
// get spurious warnings for functions that are currently only used in tests. This is useful for
// development but can be removed later.
#![cfg_attr(not(test), allow(dead_code))]

#[macro_use]
extern crate lazy_static;
extern crate regex;
extern crate easy_ll;
extern crate libc;

use std::collections::HashMap;
use std::collections::HashSet;
use std::error::Error;
use libc::{c_char, c_void};
use std::ffi::{CString, CStr};
use std::mem;

/// Utility macro to create an Err result with a WeldError from a format string.
macro_rules! weld_err {
    ( $($arg:tt)* ) => ({
        ::std::result::Result::Err($crate::error::WeldError::new(format!($($arg)*)))
    })
}

// TODO: Not all of these should be public
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
pub mod util;

use std::sync::Mutex;

lazy_static! {
    // Maps a run to a list of pointers that must be freed for a given run of a module.
    // The key is a globally unique run ID.
    static ref ALLOCATIONS: Mutex<HashMap<i64, HashSet<u64>>> = Mutex::new(HashMap::new());
}

// C Functions used by the Weld runtime.
extern "C" {
    pub fn malloc(size: libc::size_t) -> *mut c_void;
    pub fn realloc(ptr: *mut c_void, size: libc::size_t) -> *mut c_void;
    pub fn free(ptr: *mut c_void);
}

pub struct WeldError {
    code: i32,
    message: CString,
}

pub struct WeldValue {
    data: *const c_void,
    run_id: Option<i64>,
}

impl WeldError {
    fn new(code: i32, message: &str) -> WeldError {
        WeldError {
            code: code,
            message: CString::new(message).unwrap(),
        }
    }
}

#[no_mangle]
pub extern "C" fn weld_value_new(data: *const c_void) -> *mut WeldValue {
    Box::into_raw(Box::new(WeldValue {
        data: data,
        run_id: None,
    }))
}

#[no_mangle]
pub extern "C" fn weld_value_owned(obj: *const WeldValue) -> i32 {
    let obj = unsafe {
        assert!(!obj.is_null());
        &*obj
    };
    obj.run_id.is_some() as i32
}

#[no_mangle]
pub extern "C" fn weld_value_data(obj: *const WeldValue) -> *const c_void {
    let obj = unsafe {
        assert!(!obj.is_null());
        &*obj
    };
    obj.data
}

#[no_mangle]
pub extern "C" fn weld_value_free(obj: *mut WeldValue) {
    let value = unsafe {
        if obj.is_null() {
            return;
        } else {
            &mut *obj
        }
    };

    if let Some(run_id) = value.run_id {
        // Free all the memory associated with the run.
        weld_run_free(run_id);
    }
    mem::drop(obj);
}

#[no_mangle]
pub extern "C" fn weld_module_compile(code: *const c_char,
                                      conf: *const c_char,
                                      err: *mut *mut WeldError)
                                      -> *mut easy_ll::CompiledModule {
    let code = unsafe {
        assert!(!code.is_null());
        CStr::from_ptr(code)
    };
    let code = code.to_str().unwrap().trim();

    let conf = unsafe {
        assert!(!conf.is_null());
        CStr::from_ptr(conf)
    };
    let conf = conf.to_str().unwrap();

    let err = unsafe {
        *err = Box::into_raw(Box::new(WeldError::new(0, "Success")));
        &mut **err
    };

    let module = llvm::compile_program(&parser::parse_program(code).unwrap());
    if let Err(ref e) = module {
        err.code = 1;
        err.message = CString::new(e.description()).unwrap();
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(module.unwrap()))
}

#[no_mangle]
pub extern "C" fn weld_module_run(module: *mut easy_ll::CompiledModule,
                                  arg: *const WeldValue,
                                  err: *mut *mut WeldError)
                                  -> *mut WeldValue {
    let module = unsafe {
        assert!(!module.is_null());
        &mut *module
    };

    let arg = unsafe {
        assert!(!arg.is_null());
        &*arg
    };

    // TODO(shoumik): Error reporting - at least basic crashes
    // TODO(shoumik): Set the number of threads correctly
    let result = module.run(arg.data as i64, 1) as *const c_void;
    Box::into_raw(Box::new(WeldValue {
        data: result,
        run_id: Some(0),
    }))
}

#[no_mangle]
pub extern "C" fn weld_module_free(ptr: *mut easy_ll::CompiledModule) {
    if ptr.is_null() {
        return;
    }
    unsafe { Box::from_raw(ptr) };
}

#[no_mangle]
pub extern "C" fn weld_error_code(err: *mut WeldError) -> i32 {
    let err = unsafe {
        if err.is_null() {
            return 1;
        }
        &mut *err
    };
    err.code as i32
}

#[no_mangle]
pub extern "C" fn weld_error_message(err: *mut WeldError) -> *const c_char {
    let err = unsafe {
        if err.is_null() {
            return std::ptr::null();
        }
        &mut *err
    };
    err.message.as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn weld_error_free(err: *mut WeldError) {
    if err.is_null() {
        return;
    }
    unsafe { Box::from_raw(err) };
}

#[no_mangle]
/// Used by modules to allocate data.
pub extern "C" fn weld_rt_malloc(run_id: i64, size: libc::int64_t) -> *mut c_void {
    let mut guarded = ALLOCATIONS.lock().unwrap();
    let ptr = unsafe { malloc(size as usize) };
    if !guarded.contains_key(&0) {
        // TODO(shoumik): Track runs.
        guarded.insert(0, HashSet::new());
    }
    // TODO(shoumik): Track runs.
    let entries = guarded.entry(0).or_insert(HashSet::new());
    entries.insert(ptr as u64);
    ptr
}

#[no_mangle]
/// Used by modules to reallocate data.
pub extern "C" fn weld_rt_realloc(data: *mut c_void, size: libc::int64_t) -> *mut c_void {
    let mut guarded = ALLOCATIONS.lock().unwrap();
    let ptr = unsafe { realloc(data, size as usize) };
    // TODO(shoumik): Track runs.
    if !guarded.contains_key(&0) {
        panic!("Unseen run {}", 0);
    }
    let entries = guarded.entry(0).or_insert(HashSet::new());
    entries.remove(&(data as u64));
    entries.insert(ptr as u64);
    ptr
}

#[no_mangle]
/// This can be used to free data allocated by the Weld runtime.
pub extern "C" fn weld_rt_free(data: *mut c_void) {
    let mut guarded = ALLOCATIONS.lock().unwrap();
    unsafe { free(data) };
    // TODO(shoumik): Track runs
    let entries = guarded.entry(0).or_insert(HashSet::new());
    entries.remove(&(data as u64));

}

/// Used by tests to free memory after a run.
pub fn weld_run_free(run_id: i64) {
    let mut guarded = ALLOCATIONS.lock().unwrap();
    // TODO(shoumik): Track runs
    if !guarded.contains_key(&0) {
        return;
    }
    {
        let entries = guarded.entry(0).or_insert(HashSet::new());
        for entry in entries.iter() {
            unsafe { free(*entry as *mut c_void) };
        }
    }
    // TODO(shoumik): Track runs
    guarded.remove(&0);
}

#[cfg(test)]
mod tests;
