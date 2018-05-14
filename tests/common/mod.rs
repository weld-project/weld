//! Utilities and helper functions for integration tests.
#![allow(dead_code)]

extern crate libc;
extern crate weld;

use weld::common::WeldRuntimeErrno;
use weld::*;

use std::convert::AsRef;

use self::libc::{c_char, c_void};
use std::ffi::{CStr, CString};

#[derive(Clone, Debug)]
#[repr(C)]
pub struct WeldVec<T> {
    pub data: *const T,
    pub len: i64,
}

impl<T> WeldVec<T> {
    /// Return a new WeldVec from a pointer and a length.
    ///
    /// Consider using `WeldVec::from` instead, which automatically derives the length.
    pub fn new(ptr: *const T, len: i64) -> WeldVec<T> {
        WeldVec {
            data: ptr,
            len: len,
        }
    }
}

impl<'a, T, U> From<&'a U> for WeldVec<T>
where
    U: AsRef<[T]>,
{
    fn from(s: &'a U) -> WeldVec<T> {
        WeldVec::new(s.as_ref().as_ptr(), s.as_ref().len() as i64)
    }
}

impl<T> PartialEq for WeldVec<T>
where
    T: PartialEq + Clone,
{
    fn eq(&self, other: &WeldVec<T>) -> bool {
        if self.len != other.len {
            return false;
        }
        for i in 0..self.len {
            let v1 = unsafe { (*self.data.offset(i as isize)).clone() };
            let v2 = unsafe { (*other.data.offset(i as isize)).clone() };
            if v1 != v2 {
                return false;
            }
        }
        true
    }
}

#[derive(Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Pair<K, V> {
    pub ele1: K,
    pub ele2: V,
}

impl<K, V> Pair<K, V> {
    pub fn new(a: K, b: V) -> Pair<K, V> {
        Pair { ele1: a, ele2: b }
    }
}

/// Returns a default configuration which uses a single thread.
pub fn default_conf() -> *mut WeldConf {
    conf(1)
}

/// Returns a configuration which uses several threads.
pub fn many_threads_conf() -> *mut WeldConf {
    conf(4)
}

/// Runs `code` with the given `conf` and input data pointer `ptr`, expecting
/// a succeessful result to be returned. Panics if an error is thrown by the runtime.
pub fn compile_and_run<T>(code: &str, conf: *mut WeldConf, ptr: &T) -> *mut WeldValue {
    match unsafe { _compile_and_run(code, conf, ptr) } {
        Ok(val) => val,
        Err(err) => panic!(format!("Compile failed: {:?}", unsafe {
            CStr::from_ptr(weld_error_message(err))
        })),
    }
}

/// Runs `code` with the given `conf` and input data pointer `ptr`, expecting
/// a runtime error to be thrown. Panics if no error is thrown.
pub fn compile_and_run_error<T>(code: &str, conf: *mut WeldConf, ptr: &T) -> *mut WeldError {
    match unsafe { _compile_and_run(code, conf, ptr) } {
        Ok(_) => panic!("Expected an error but got a value"),
        Err(e) => e,
    }
}

/// Frees a value and its corresponding module.
pub unsafe fn free_value_and_module(value: *mut WeldValue) {
    let module = weld_value_module(value);
    weld_value_free(value);
    weld_module_free(module);
}

fn conf(threads: i32) -> *mut WeldConf {
    let threads = format!("{}", threads);
    let conf = weld_conf_new();
    let key = CString::new("weld.threads").unwrap().into_raw() as *const c_char;
    let value = CString::new(threads).unwrap().into_raw() as *const c_char;
    unsafe { weld_conf_set(conf, key, value) };
    conf
}

/// Compiles and runs some code on a configuration and input data pointer. If the run is
/// successful, returns the resulting value. If the run fails (via a runtime error), returns an
/// error. Both the value and error must be freed by the caller. The  `conf` passed to this
/// function is freed.
unsafe fn _compile_and_run<T>(
    code: &str,
    conf: *mut WeldConf,
    ptr: &T,
) -> Result<*mut WeldValue, *mut WeldError> {
    let code = CString::new(code).unwrap();
    let input_value = weld_value_new(ptr as *const _ as *const c_void);
    let err = weld_error_new();
    let module = weld_module_compile(code.into_raw() as *const c_char, conf, err);

    if weld_error_code(err) != WeldRuntimeErrno::Success {
        weld_module_free(module);
        weld_value_free(input_value);
        weld_conf_free(conf);
        return Err(err);
    }
    weld_error_free(err);

    let err = weld_error_new();
    let ret_value = weld_module_run(module, conf, input_value, err);
    if weld_error_code(err) != WeldRuntimeErrno::Success {
        weld_conf_free(conf);
        weld_value_free(input_value);
        weld_value_free(ret_value);
        weld_module_free(module);
        return Err(err);
    }

    weld_error_free(err);
    weld_value_free(input_value);
    weld_conf_free(conf);

    return Ok(ret_value);
}
