//! Utilities and helper functions for integration tests.
#![allow(dead_code)]

extern crate libc;
extern crate weld;

use weld::*;
use std::convert::AsRef;

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

pub fn default_conf() -> WeldConf {
    conf(1)
}

pub fn many_threads_conf() -> WeldConf {
    conf(4)
}

fn conf(threads: i32) -> WeldConf {
    let threads = format!("{}", threads);
    let mut conf = WeldConf::new();
    conf.set("weld.threads", threads);
    conf
}

/// Compiles and runs some code on a configuration and input data pointer. If the run is
/// successful, returns the resulting value. If the run fails (via a runtime error), returns an
/// error. Both the value and error must be freed by the caller. The  `conf` passed to this
/// function is freed.
unsafe fn _compile_and_run<T>(code: &str, conf: &WeldConf, ptr: &T) -> WeldResult<WeldValue> {
    let ref input_value = WeldValue::new_from_data(ptr as *const _ as Data);
    let mut module = match WeldModule::compile(code, conf) {
        Ok(module) => module,
        Err(err) => {
            return Err(err);
        }
    };
    module.run(conf, input_value)
}

/// Runs `code` with the given `conf` and input data pointer `ptr`, expecting
/// a succeessful result to be returned. Panics if an error is thrown by the runtime.
pub fn compile_and_run<T>(code: &str, conf: &WeldConf, ptr: &T) -> WeldValue {
    unsafe { _compile_and_run(code, conf, ptr).expect("Run failed!") }
}


/// Runs `code` with the given `conf` and input data pointer `ptr`, expecting
/// a succeessful result to be returned. Panics if an error is thrown by the runtime.
pub fn compile_and_run_error<T>(code: &str, conf: &WeldConf, ptr: &T) -> WeldError {
    unsafe { _compile_and_run(code, conf, ptr).expect_err("Expected error!") }
}
