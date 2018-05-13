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
fn simple_for_vecmerger_loop() {
    let code = "|x:vec[i32]| result(for(x, vecmerger[i32,+](x), |b,i,e| b))";
    let conf = default_conf();

    let input_vec = [1, 1, 1, 1, 1];
    let ref input_data = WeldVec {
        data: &input_vec as *const i32,
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<i32> };
    let result = unsafe { (*data).clone() };
    assert_eq!(result.len, input_vec.len() as i64);
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, input_vec[i as usize]);
    }
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn simple_for_vecmerger_binops() {
    let code = "|x:vec[i64]| {
        result(for(x, vecmerger[i64,+](x), |b,i,e| merge(b, {i,e*7L}))),
        result(for(x, vecmerger[i64,*](x), |b,i,e| merge(b, {i, i}))),
        result(for(x, vecmerger[i64,min](x), |b,i,e| merge(b, {i, i}))),
        result(for(x, vecmerger[i64,max](x), |b,i,e| merge(b, {i, i})))
        }";
    let conf = default_conf();

    #[derive(Clone)]
    #[allow(dead_code)]
    struct Output {
        sum: WeldVec<i64>,
        prod: WeldVec<i64>,
        min: WeldVec<i64>,
        max: WeldVec<i64>,
    }

    let input_vec: Vec<i64> = vec![3, 3, 3, 3, 3];
    let ref input_data = WeldVec {
        data: input_vec.as_ptr() as *const i64,
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const Output };
    let result = unsafe { (*data).clone() };
    for i in 0..(input_vec.len() as isize) {
        assert_eq!(unsafe { *result.sum.data.offset(i) },
                   input_vec[i as usize] + input_vec[i as usize] * 7);
        assert_eq!(unsafe { *result.prod.data.offset(i) }, input_vec[i as usize] * (i as i64));
        assert_eq!(unsafe { *result.min.data.offset(i) }, cmp::min(input_vec[i as usize], i as i64));
        assert_eq!(unsafe { *result.max.data.offset(i) }, cmp::max(input_vec[i as usize], i as i64));
    }
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn parallel_for_vecmerger_loop() {
    let code = "|x:vec[i32]| result(@(grain_size: 100)for(x, vecmerger[i32,+](x), |b,i,e| merge(b, {i,e*7})))";
    let conf = many_threads_conf();

    let input_vec = [1; 4096];
    let ref input_data = WeldVec {
        data: &input_vec as *const i32,
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<i32> };
    let result = unsafe { (*data).clone() };

    assert_eq!(result.len, input_vec.len() as i64);
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) },
                   input_vec[i as usize] + input_vec[i as usize] * 7);
    }

    unsafe { free_value_and_module(ret_value) };
}

