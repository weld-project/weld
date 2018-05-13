//! Utilities and helper functions for integration tests.

extern crate weld;
extern crate libc;
extern crate fnv;

use std::env;
use std::str;
use std::slice;
use std::thread;
use std::cmp;
use std::collections::hash_map::Entry;

use weld::common::WeldRuntimeErrno;

use weld::WeldConf;
use weld::WeldValue;
use weld::WeldError;
use weld::{weld_value_new, weld_value_data, weld_value_module, weld_value_free};
use weld::{weld_module_compile, weld_module_run, weld_module_free};
use weld::{weld_error_new, weld_error_code, weld_error_message, weld_error_free};
use weld::{weld_conf_new, weld_conf_set, weld_conf_free};

use std::f64::consts::PI;
use std::ffi::{CStr, CString};
use libc::{c_char, c_void};

/// Compares a and b, and returns true if their difference is less than 0.000...1 (cmp_decimals)
pub fn approx_equal(a: f64, b: f64, cmp_decimals: u32) -> bool {
    if a == b {
        return true;
    }
    let thresh = 0.1 / ((10i32.pow(cmp_decimals)) as f64);
    let diff = (a - b).abs();
    diff <= thresh
}


/// Compares a and b, and returns true if their difference is less than 0.000...1 (cmp_decimals)
pub fn approx_equal_f32(a: f32, b: f32, cmp_decimals: u32) -> bool {
    if a == b {
        return true;
    }
    let thresh = 0.1 / ((10i32.pow(cmp_decimals)) as f32);
    let diff = (a - b).abs();
    diff <= thresh
}

/// An in memory representation of a Weld vector.
#[derive(Clone, Debug)]
#[allow(dead_code)]
#[repr(C)]
pub struct WeldVec<T> {
    pub data: *const T,
    pub len: i64,
}

impl<T> WeldVec<T> {
    pub fn new(ptr: *const T, len: i64) -> WeldVec<T> {
        WeldVec {
            data: ptr,
            len: len
        }
    }
}

impl<T> PartialEq for WeldVec<T> where T: PartialEq + Clone {
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
#[allow(dead_code)]
#[repr(C)]
pub struct Pair<K, V> {
    pub ele1: K,
    pub ele2: V,
}

impl<K, V> Pair<K, V> {
    pub fn new(a: K, b: V) -> Pair<K,V> {
        Pair {
            ele1: a,
            ele2: b,
        }
    }
}

/// Returns a default configuration which uses a single thread.
pub fn default_conf() -> *mut WeldConf {
    let conf = weld_conf_new();
    let key = CString::new("weld.threads").unwrap().into_raw() as *const c_char;
    let value = CString::new("1").unwrap().into_raw() as *const c_char;
    unsafe { weld_conf_set(conf, key, value) };
    conf
}

/// Returns a configuration which uses several threads.
pub fn many_threads_conf() -> *mut WeldConf {
    let conf = weld_conf_new();
    let key = CString::new("weld.threads").unwrap().into_raw() as *const c_char;
    let value = CString::new("4").unwrap().into_raw() as *const c_char;
    unsafe { weld_conf_set(conf, key, value) };
    conf
}

/// Compiles and runs some code on a configuration and input data pointer. If the run is
/// successful, returns the resulting value. If the run fails (via a runtime error), returns an
/// error. Both the value and error must be freed by the caller. The  `conf` passed to this
/// function is freed.
unsafe fn _compile_and_run<T>(code: &str,
                              conf: *mut WeldConf,
                              ptr: &T)
                              -> Result<*mut WeldValue, *mut WeldError> {

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

pub unsafe fn free_value_and_module(value: *mut WeldValue) {
    let module = weld_value_module(value);
    weld_value_free(value);
    weld_module_free(module);
}

/// Runs `code` with the given `conf` and input data pointer `ptr`, expecting
/// a runtime error to be thrown. Panics if no error is thrown.
pub fn compile_and_run_error<T>(code: &str, conf: *mut WeldConf, ptr: &T) -> *mut WeldError {
    match unsafe { _compile_and_run(code, conf, ptr) } {
        Ok(_) => panic!("Expected an error but got a value"),
        Err(e) => e,
    }
}

/// Runs `code` with the given `conf` and input data pointer `ptr`, expecting
/// a succeessful result to be returned. Panics if an error is thrown by the runtime.
pub fn compile_and_run<T>(code: &str, conf: *mut WeldConf, ptr: &T) -> *mut WeldValue {
    match unsafe { _compile_and_run(code, conf, ptr) } {
        Ok(val) => val,
        Err(err) => {
            panic!(format!("Compile failed: {:?}",
                           unsafe { CStr::from_ptr(weld_error_message(err)) }))
        }
    }
}
