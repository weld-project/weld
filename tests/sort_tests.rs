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
fn string_sort() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<WeldVec<u8>>
    }
    let bs = vec!['T' as u8, 'R' as u8, 'U' as u8, 'E' as u8];
    let cs = vec!['P' as u8, 'A' as u8, 'R' as u8];
    let ds = vec!['F' as u8, 'A' as u8, 'L' as u8, 'S' as u8, 'E' as u8];
    let sorted = vec![ds.clone(), cs.clone(), bs.clone()];
    let bs_vec = WeldVec {
        data: bs.as_ptr() as *const u8,
        len: bs.len() as i64,
    };
    let cs_vec = WeldVec {
        data: cs.as_ptr() as *const u8,
        len: cs.len() as i64,
    };
    let ds_vec = WeldVec {
        data: ds.as_ptr() as *const u8,
        len: ds.len() as i64,
    };
    let strs = vec![bs_vec, cs_vec, ds_vec];

    let ref input_data = Args {
        x: WeldVec {
            data: strs.as_ptr() as *const WeldVec<u8>,
            len: strs.len() as i64,
        }
    };

    let code = "|e0: vec[vec[u8]]| sort(e0, |i:vec[u8]| i)";

    let conf = default_conf();
    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<WeldVec<u8>> };
    let result = unsafe { (*data).clone() };

    for i in 0..(result.len as isize) {
        let ivec = unsafe { (*result.data.offset(i)).clone() };
        for j in 0..(ivec.len as isize) {
            let val = unsafe { (*ivec.data.offset(j)) };
                assert_eq!(val, sorted[i as usize][j as usize])
        }
    }

    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn if_sort() {
    let ys = vec![2, 3, 1, 4, 5];
    let ref input_data = WeldVec {
        data: ys.as_ptr() as *const i32,
        len: ys.len() as i64,
    };

    let code = "|ys:vec[i32]| sort(ys, |x:i32| if(x != 5, x + 1, 0))";
    let conf = default_conf();
    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<i32> };
    let result = unsafe { (*data).clone() };

    let expected = [5, 1, 2, 3, 4];
    assert_eq!(result.len, expected.len() as i64);

    for i in 0..(expected.len() as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, expected[i as usize])
    }

    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn simple_sort() {
    let ys = vec![2, 3, 1, 4, 5];
    let ref input_data = WeldVec {
        data: ys.as_ptr() as *const i32,
        len: ys.len() as i64,
    };

    let code = "|ys:vec[i32]| sort(ys, |x:i32| x + 1)";
    let conf = default_conf();
    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<i32> };
    let result = unsafe { (*data).clone() };

    let expected = [1, 2, 3, 4, 5];
    assert_eq!(result.len, expected.len() as i64);

    for i in 0..(expected.len() as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, expected[i as usize])
    }

    unsafe { free_value_and_module(ret_value) };

    let ys = vec![2.0, 3.0, 1.0, 5.001, 5.0001];
    let ref input_data = WeldVec {
        data: ys.as_ptr() as *const f64,
        len: ys.len() as i64,
    };

    let code = "|ys:vec[f64]| sort(sort(ys, |x:f64| x), |x:f64| x)";
    let conf = default_conf();
    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<f64> };
    let result = unsafe { (*data).clone() };

    let expected = [1.0, 2.0, 3.0, 5.0001, 5.001];
    assert_eq!(result.len, expected.len() as i64);
    for i in 0..(expected.len() as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, expected[i as usize])
    }

    let code = "|ys:vec[f64]| sort(ys, |x:f64| 1.0 / exp(-1.0*x))";
    let conf = default_conf();
    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<f64> };
    let result = unsafe { (*data).clone() };

    assert_eq!(result.len, expected.len() as i64);
    for i in 0..(expected.len() as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, expected[i as usize])
    }

    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn complex_sort() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }
    let ys = vec![5, 4, 3, 2, 1];
    let xs = vec![1, 2, 3, 4, 5];
    let ref input_data = Args {
        x: WeldVec {
            data: ys.as_ptr() as *const i32,
            len: ys.len() as i64,
        },
        y: WeldVec {
            data: xs.as_ptr() as *const i32,
            len: xs.len() as i64,
        }
    };

    let code = "|ys:vec[i32], xs:vec[i32]|
                  sort(
                    result(
                      for(
                        zip(xs,ys),
                        appender[{i32,i32}],
                        |b,i,e| merge(b, e)
                      )
                    ),
                    |x:{i32, i32}| x.$0
                )";
    let conf = default_conf();
    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<Pair<i32, i32>> };
    let result = unsafe { (*data).clone() };

    let expected = [[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]];
    assert_eq!(result.len, expected.len() as i64);

    for i in 0..(expected.len() as isize) {
        assert_eq!(unsafe { (*result.data.offset(i)).ele1 }, expected[i as usize][0]);
        assert_eq!(unsafe { (*result.data.offset(i)).ele2 }, expected[i as usize][1]);
    }

    unsafe { free_value_and_module(ret_value) };
}
