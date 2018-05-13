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
fn map_comparison() {
    let code = "|e0: vec[i32]| map(e0, |a: i32| a == i32(100))";
    let conf = default_conf();

    let input_vec = [100, 200, 0, 100];
    let ref input_data = WeldVec {
        data: &input_vec as *const i32,
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<bool> };
    let result = unsafe { (*data).clone() };
    assert_eq!(result.len as usize, input_vec.len());
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) },
                   input_vec[i as usize] == 100)
    }

    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn eq_between_vectors() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }
    let conf = default_conf();

    let code = "|e0: vec[i32], e1: vec[i32]| e0 == e1";
    let input_vec1 = [1, 2, 3, 4, 5];
    let input_vec2 = [1, 2, 3, 4, 5];
    let ref input_data = Args {
        x: WeldVec {
            data: &input_vec1 as *const i32,
            len: input_vec1.len() as i64,
        },
        y: WeldVec {
            data: &input_vec2 as *const i32,
            len: input_vec2.len() as i64,
        },
    };
    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const bool };
    let result = unsafe { *data };
    assert_eq!(result, true);
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn eq_between_diff_length_vectors() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }
    let conf = default_conf();

    let code = "|e0: vec[i32], e1: vec[i32]| e0 == e1";
    let input_vec1 = [1, 2, 3, 4, 5];
    let input_vec2 = [1, 2, 3, 4, 5, 6, 7];
    let ref input_data = Args {
        x: WeldVec {
            data: &input_vec1 as *const i32,
            len: input_vec1.len() as i64,
        },
        y: WeldVec {
            data: &input_vec2 as *const i32,
            len: input_vec2.len() as i64,
        },
    };
    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const bool };
    let result = unsafe { *data };
    assert_eq!(result, false);
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn ne_between_vectors() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }
    let conf = default_conf();

    let code = "|e0: vec[i32], e1: vec[i32]| e0 != e1";
    let input_vec1 = [1, 2, 3, 4, 5];
    let input_vec2 = [3, 2, 3, 4, 5];
    let ref input_data = Args {
        x: WeldVec {
            data: &input_vec1 as *const i32,
            len: input_vec1.len() as i64,
        },
        y: WeldVec {
            data: &input_vec2 as *const i32,
            len: input_vec2.len() as i64,
        },
    };
    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const bool };
    let result = unsafe { *data };
    assert_eq!(result, true);
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn lt_between_vectors() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }
    let conf = default_conf();

    let code = "|e0: vec[i32], e1: vec[i32]| e0 < e1";
    let input_vec1 = [1, 2, 3, 4, 5];
    let input_vec2 = [2, 3, 4, 5, 6];
    let ref input_data = Args {
        x: WeldVec {
            data: &input_vec1 as *const i32,
            len: input_vec1.len() as i64,
        },
        y: WeldVec {
            data: &input_vec2 as *const i32,
            len: input_vec2.len() as i64,
        },
    };
    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const bool };
    let result = unsafe { *data };
    assert_eq!(result, true);
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn le_between_vectors() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }
    let conf = default_conf();

    let code = "|e0: vec[i32], e1: vec[i32]| e0 <= e1";
    let input_vec1 = [-1, 0, 3, 4, 5];
    let input_vec2 = [-1, -3, 4, 5, 6];
    let ref input_data = Args {
        x: WeldVec {
            data: &input_vec1 as *const i32,
            len: input_vec1.len() as i64,
        },
        y: WeldVec {
            data: &input_vec2 as *const i32,
            len: input_vec2.len() as i64,
        },
    };
    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const bool };
    let result = unsafe { *data };
    assert_eq!(result, false);
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn le_between_unsigned_vectors() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }
    let conf = default_conf();

    // Note that we pass our integers in as unsigneds in Weld
    let code = "|e0: vec[u32], e1: vec[u32]| e0 <= e1";
    let input_vec1 = [-1, 0, 3, 4, 5];
    let input_vec2 = [-1, -3, 4, 5, 6];
    let ref input_data = Args {
        x: WeldVec {
            data: &input_vec1 as *const i32,
            len: input_vec1.len() as i64,
        },
        y: WeldVec {
            data: &input_vec2 as *const i32,
            len: input_vec2.len() as i64,
        },
    };
    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const bool };
    let result = unsafe { *data };
    assert_eq!(result, true);
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn eq_between_u8_vectors() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<u8>,
        y: WeldVec<u8>,
    }
    let conf = default_conf();

    let code = "|e0: vec[u8], e1: vec[u8]| e0 == e1";
    let input_vec1 = [1u8, 2u8, 3u8, 4u8, 5u8];
    let input_vec2 = [1u8, 2u8, 3u8, 4u8, 5u8];
    let ref input_data = Args {
        x: WeldVec {
            data: &input_vec1 as *const u8,
            len: input_vec1.len() as i64,
        },
        y: WeldVec {
            data: &input_vec2 as *const u8,
            len: input_vec2.len() as i64,
        },
    };
    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const bool };
    let result = unsafe { *data };
    assert_eq!(result, true);
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn eq_between_different_length_u8_vectors() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<u8>,
        y: WeldVec<u8>,
    }
    let conf = default_conf();

    let code = "|e0: vec[u8], e1: vec[u8]| e0 == e1";
    let input_vec1 = [1u8, 2u8, 3u8, 4u8, 5u8];
    let input_vec2 = [1u8, 2u8, 3u8, 4u8, 5u8, 6u8];
    let ref input_data = Args {
        x: WeldVec {
            data: &input_vec1 as *const u8,
            len: input_vec1.len() as i64,
        },
        y: WeldVec {
            data: &input_vec2 as *const u8,
            len: input_vec2.len() as i64,
        },
    };
    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const bool };
    let result = unsafe { *data };
    assert_eq!(result, false);
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn le_between_u8_vectors() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<u8>,
        y: WeldVec<u8>,
    }
    let conf = default_conf();

    let code = "|e0: vec[u8], e1: vec[u8]| e0 <= e1";
    let input_vec1 = [1u8, 2u8, 3u8, 4u8, 5u8];
    let input_vec2 = [1u8, 2u8, 3u8, 255u8, 5u8];
    let ref input_data = Args {
        x: WeldVec {
            data: &input_vec1 as *const u8,
            len: input_vec1.len() as i64,
        },
        y: WeldVec {
            data: &input_vec2 as *const u8,
            len: input_vec2.len() as i64,
        },
    };
    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const bool };
    let result = unsafe { *data };
    assert_eq!(result, true);
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn simple_vector_lookup() {
    let code = "|x:vec[i32]| lookup(x, 3L)";
    let conf = default_conf();

    let input_vec = [1, 2, 3, 4, 5];
    let ref input_data = WeldVec {
        data: &input_vec as *const i32,
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };
    assert_eq!(result, input_vec[3]);

    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn simple_vector_slice() {
    let code = "|x:vec[i32]| slice(x, 1L, 3L)";
    let conf = default_conf();

    let input_vec = [1, 2, 3, 4, 5];
    let ref input_data = WeldVec {
        data: &input_vec as *const i32,
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<i32> };
    let result = unsafe { (*data).clone() };
    let output = [2, 3, 4];

    assert_eq!(output.len() as isize, result.len as isize);
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
    }

    unsafe { free_value_and_module(ret_value) };

    // Test slicing out of bounds case
    let conf = default_conf();

    let input_vec = [1, 2];
    let ref input_data = WeldVec {
        data: &input_vec as *const i32,
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<i32> };
    let result = unsafe { (*data).clone() };
    let output = [2];

    assert_eq!(output.len() as isize, result.len as isize);
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
    }

    unsafe { free_value_and_module(ret_value) };
}
