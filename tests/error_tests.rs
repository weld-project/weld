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

mod common;

use common::*;

#[test]
fn iters_outofbounds_error_test() {
    let code = "|x:vec[i32]| result(for(iter(x,0L,20000L,1L), appender, |b,i,e| merge(b,e+1)))";
    let conf = many_threads_conf();

    let input_vec = [4; 1000 as usize];
    let ref input_data = WeldVec {
        data: &input_vec as *const i32,
        len: input_vec.len() as i64,
    };

    let err_value = compile_and_run_error(code, conf, input_data);
    assert_eq!(unsafe { weld_error_code(err_value) },
               WeldRuntimeErrno::BadIteratorLength);
    unsafe { weld_error_free(err_value) };
}

#[test]
fn outofmemory_error_test() {
    let code = "|x:vec[i32]| result(for(x, vecmerger[i32,+](x), |b,i,e| merge(b,{i,e+1})))";
    let conf = default_conf();

    // Set the memory to something small.
    let key = CString::new("weld.memory.limit").unwrap().into_raw() as *const c_char;
    let value = CString::new("50000").unwrap().into_raw() as *const c_char;
    unsafe { weld_conf_set(conf, key, value) };

    let x = vec![4; 50000 / 4 as usize];
    let ref input_data = WeldVec {
        data: x.as_ptr() as *const i32,
        len: x.len() as i64,
    };

    let err_value = compile_and_run_error(code, conf, input_data);
    assert_eq!(unsafe { weld_error_code(err_value) },
               WeldRuntimeErrno::OutOfMemory);
    unsafe { weld_error_free(err_value) };
}
