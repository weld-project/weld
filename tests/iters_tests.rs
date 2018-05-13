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
fn range_iter_1() {
    let end = 1000;
    let code = format!("|a: i64| result(for(rangeiter(1L, {}L + 1L, 1L), merger[i64,+], |b,i,e| merge(b, a+e)))", end);

    #[allow(dead_code)]
    struct Args {
        a: i64,
    };
    let conf = default_conf();
    let ref input_data = Args { a: 0 };

    let ret_value = compile_and_run(&code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i64 };
    let result = unsafe { (*data).clone() };
    let output = end * (end + 1) / 2;
    assert_eq!(result, output);
    unsafe { free_value_and_module(ret_value) };
}

fn range_iter_zipped_helper(parallel: bool) {
    let grain_size = if parallel { 100 } else  { 4096 };
    let conf = if parallel { many_threads_conf() } else { default_conf() };

    let end = 1000;
    let code = format!("|v: vec[i64]| result(
        @(grain_size: {grain_size})for(zip(v, rangeiter(1L, {end}L + 1L, 1L)), merger[i64,+], |b,i,e| merge(b, e.$0 + e.$1)
    ))", grain_size=grain_size, end=end);

    #[allow(dead_code)]
    struct Args {
        v: WeldVec<i64>,
    };
    let input_vec = vec![1 as i64; end as usize];
    let ref input_data = Args {
        v: WeldVec {
            data: input_vec.as_ptr() as *const i64,
            len: input_vec.len() as i64,
        },
    };

    let ret_value = compile_and_run(&code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i64 };
    let result = unsafe { (*data).clone() };
    let output = end * (end + 1) / 2 + end;
    assert_eq!(result, output);
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn range_iter_2() {
    range_iter_zipped_helper(false)
}

#[test]
fn range_iter_parallel() {
    range_iter_zipped_helper(true)
}

#[test]
fn iters_for_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]| result(for(zip(iter(x,0L,4L,2L), y), appender, |b,i,e|
                merge(b,e.$0+e.$1)))";
    let conf = default_conf();

    let x = [1, 2, 3, 4];
    let y = [5, 6];
    let ref input_data = Args {
        x: WeldVec {
            data: &x as *const i32,
            len: x.len() as i64,
        },
        y: WeldVec {
            data: &y as *const i32,
            len: y.len() as i64,
        },
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<i32> };
    let result = unsafe { (*data).clone() };

    let output = [6, 9];
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
    }

    unsafe { free_value_and_module(ret_value) };
}

/// Helper function for nditer - in order to simulate the behaviour of numpy's non-contiguous
/// multi-dimensional arrays using counter, strides.
/// returns idx = dot(counter, strides)
fn get_idx(counter: [i64;3], strides: [i64;3]) -> i64 {
    let mut sum:i64 = 0;
    for i in 0..3 {
        sum += counter[i]*strides[i];
    }
    return sum;
}
/// increments counter as in numpy / and nditer implementation. For e.g.,
/// let shapes :[i64; 3] = [2, 3, 4];
/// Now counter would start from (0,0,0).
/// Each index would go upto shapes, and then reset to 0. 
/// eg. (0,0,0), (0,0,1), (0,0,2), (0,0,3), (0,1,0) etc.
fn update_counter(mut counter: [i64; 3], shapes: [i64; 3]) -> [i64; 3] {
    let v = vec![2, 1, 0];
    for i in v {
        counter[i] += 1;
        if counter[i] == shapes[i] {
            counter[i] = 0;
        } else {
            return counter;
        }
    }
    return counter;
}

/// Tests that nditer correctly iterates over each element of a non-contiguous array and applies
/// the op to it. Note: In order to simulate non-contiguous arrays, we use counter/shapes/strides
/// to mimic the behaviour of numpy. This has also been tested against numpy, and appears to work
/// fine.
#[test]
fn nditer_basic_op_test() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<f64>,
        shapes: WeldVec<i64>,
        strides: WeldVec<i64>,
    }
    let code = "|x:vec[f64], shapes:vec[i64], strides:vec[i64]| result(for(nditer(x,0L,24L,1L,shapes,strides), appender, |b,i,e|
                merge(b,log(e))))";

    let conf = default_conf();
    let mut x :[f64; 100] = [0.0; 100];
    for i in 0..100 {
        x[i] = i as f64;
    }
    /* Number of elements to go forward in each index to get to next element. 
     * These are arbitrarily chosen here for testing purposes so get_idx can simulate the behaviour
     * nditer should be doing (idx = dot(counter, strides).
     */
    let strides :[i64; 3] = [5, 2, 3];
    // nditer with this shape will contain: 2*3*4 = 24 elements.
    let shapes :[i64; 3] = [2, 3, 4];
    let mut counter :[i64; 3] = [0, 0, 0];

    let ref input_data = Args {
        x: WeldVec {
            data: &x as *const f64,
            len: x.len() as i64,
        },
        shapes: WeldVec {
            data: &shapes as *const i64,
            len: shapes.len() as i64,
        },
        strides: WeldVec {
            data: &strides as *const i64,
            len: strides.len() as i64,
        },
    };
    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<f64> };
    let result = unsafe { (*data).clone() };
    for i in 0..(result.len as isize) {
        /* next idx for the original array, x, based on how numpy would behave with the given
         * shapes/strides */
        let idx = get_idx(counter, strides);
        assert_eq!(unsafe { *result.data.offset(i) }, x[idx as usize].ln());
        /* update counter according to the numpy above */
        counter = update_counter(counter, shapes);
    }
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn nditer_zip() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i64>,
        y: WeldVec<i64>,
        shapes: WeldVec<i64>,
        strides: WeldVec<i64>,
    }
    let code = "|x:vec[i64], y:vec[i64], shapes:vec[i64], strides:vec[i64]| result(for(zip(nditer(x,0L,24L,1L, shapes, strides), nditer(y,0L,24L,1L,shapes,strides)),  \
    appender, |b,i,e| merge(b,e.$0+e.$1)))";

    let conf = default_conf();
    let mut x :[i64; 100] = [5; 100];
    let mut y :[i64; 100] = [0; 100];
    for i in 0..100 {
        x[i] = i as i64 + 5;
    }
    for i in 0..100 {
        y[i] = i as i64;
    }

    let strides :[i64; 3] = [5, 2, 2];
    let shapes :[i64; 3] = [2, 3, 4];
    let mut counter :[i64; 3] = [0, 0, 0];

    let ref input_data = Args {
        x: WeldVec {
            data: &x as *const i64,
            len: x.len() as i64,
        },
        y: WeldVec {
            data: &y as *const i64,
            len: y.len() as i64,
        },
        shapes: WeldVec {
            data: &shapes as *const i64,
            len: shapes.len() as i64,
        },
        strides: WeldVec {
            data: &strides as *const i64,
            len: strides.len() as i64,
        },
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<i64> };
    let result = unsafe { (*data).clone() };

    for i in 0..(result.len as isize) {
        let idx = get_idx(counter, strides);
        assert_eq!(unsafe { *result.data.offset(i) }, x[idx as usize] + y[idx as usize]);
        counter = update_counter(counter, shapes);
    }
    unsafe { free_value_and_module(ret_value) };
}
