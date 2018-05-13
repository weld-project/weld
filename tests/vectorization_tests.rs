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
fn simple_for_vectorizable_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
    }

    let code = "|x:vec[i32]| result(for(simditer(x), merger[i32,+], |b,i,e:simd[i32]| let a = broadcast(1); let a2 = a +
                    broadcast(1); merge(b, e+a2)))";
    let conf = default_conf();

    let size = 1000;
    let input_vec = vec![1 as i32; size as usize];
    let ref input_data = Args {
        x: WeldVec {
            data: input_vec.as_ptr() as *const i32,
            len: input_vec.len() as i64,
        },
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };
    let output = size * 3;
    assert_eq!(result, output);
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn fringed_for_vectorizable_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
    }

	let code = "|x:vec[i32]|
	let b1 = for(
		simditer(x),
		merger[i32,+],
		|b,i,e:simd[i32]| let a = broadcast(1); let a2 = a + broadcast(1); merge(b, e+a2));
	result(for(fringeiter(x),
		b1,
		|b,i,e| let a = 1; let a2 = a + 1; merge(b, e+a2)
	))";

    let conf = default_conf();

    let size = 1002;
    let input_vec = vec![1 as i32; size as usize];
    let ref input_data = Args {
        x: WeldVec {
            data: input_vec.as_ptr() as *const i32,
            len: input_vec.len() as i64,
        },
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };
    let output = size * 3;
    assert_eq!(result, output);
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn fringed_for_vectorizable_loop_with_par() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
    }

	let code = "|x:vec[i32]|
	let b1 = for(
		simditer(x),
		merger[i32,+],
		|b,i,e:simd[i32]| let a = broadcast(1); let a2 = a + broadcast(1); merge(b, e+a2));
	result(for(fringeiter(x),
		b1,
		|b,i,e| let a = 1; let a2 = a + 1; merge(b, e+a2)
	))";

    let conf = many_threads_conf();

    // Large size to invoke parallel runtime + some fringing.
    let size = 10 * 1000 * 1000 + 123;
    let input_vec = vec![1 as i32; size as usize];
    let ref input_data = Args {
        x: WeldVec {
            data: input_vec.as_ptr() as *const i32,
            len: input_vec.len() as i64,
        },
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };
    let output = size * 3;
    assert_eq!(result, output);
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn for_predicated_vectorizable_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
    }

    let code = "|x:vec[i32]|
	result(for(
			simditer(x:vec[i32]),
			merger[i32,+],
			|b:merger[i32,+],i:i64,e:simd[i32]|
			  merge(b:merger[i32,+],select(
				(e:simd[i32]>broadcast(0)),
				e:simd[i32],
				broadcast(0):simd[i32]
			  ))
	))
	";
    let conf = default_conf();

    let size = 1000;
    let input_vec = vec![1 as i32; size as usize];
    let ref input_data = Args {
        x: WeldVec {
            data: input_vec.as_ptr() as *const i32,
            len: input_vec.len() as i64,
        },
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };
    let output = size;
    assert_eq!(result, output);
    unsafe { free_value_and_module(ret_value) };
}

fn check_result_and_free(ret_value: *mut WeldValue, expected: i32) {
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };
    assert_eq!(result, expected);
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn predicate_if_iff_annotated() {
    #[allow(dead_code)]
    struct Args {
        v: WeldVec<i32>
    }

    let input_vec = vec![-1, 2, 3, 4, 5];
    let ref input_data = Args {
        v: WeldVec {
            data: input_vec.as_ptr(),
            len: input_vec.len() as i64,
        }
    };

    let expected = 14;

    /* annotation true */
    let code = "|v:vec[i32]| result(for(v, merger[i32,+], |b,i,e| @(predicate:true)if(e>0, merge(b,e), b)))";
    let conf = default_conf();
    let ret_value = compile_and_run(code, conf, input_data);
    check_result_and_free(ret_value, expected);

    /* annotation false */
    let code = "|v:vec[i32]| result(for(v, merger[i32,+], |b,i,e| @(predicate:false)if(e>0, merge(b,e), b)))";
    let conf = default_conf();
    let ret_value = compile_and_run(code, conf, input_data);
    check_result_and_free(ret_value, expected);

    /* annotation missing */
    let code = "|v:vec[i32]| result(for(v, merger[i32,+], |b,i,e| if(e>0, merge(b,e), b)))";
    let conf = default_conf();
    let ret_value = compile_and_run(code, conf, input_data);
    check_result_and_free(ret_value, expected);
}


