use std::env;
use std::str;
use std::slice;
use std::thread;
use std::cmp;

extern crate weld;
extern crate libc;

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
fn approx_equal(a: f64, b: f64, cmp_decimals: u32) -> bool {
    if a == b {
        return true;
    }
    let thresh = 0.1 / ((10i32.pow(cmp_decimals)) as f64);
    let diff = (a - b).abs();
    diff <= thresh
}


/// Compares a and b, and returns true if their difference is less than 0.000...1 (cmp_decimals)
fn approx_equal_f32(a: f32, b: f32, cmp_decimals: u32) -> bool {
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
struct WeldVec<T> {
    data: *const T,
    len: i64,
}

impl<T> WeldVec<T> {
    fn new(ptr: *const T, len: i64) -> WeldVec<T> {
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
struct Pair<K, V> {
    ele1: K,
    ele2: V,
}

impl<K, V> Pair<K, V> {
    fn new(a: K, b: V) -> Pair<K,V> {
        Pair {
            ele1: a,
            ele2: b,
        }
    }
}

/// Returns a default configuration which uses a single thread.
fn default_conf() -> *mut WeldConf {
    let conf = weld_conf_new();
    let key = CString::new("weld.threads").unwrap().into_raw() as *const c_char;
    let value = CString::new("1").unwrap().into_raw() as *const c_char;
    unsafe { weld_conf_set(conf, key, value) };
    conf
}

/// Returns a configuration which uses several threads.
fn many_threads_conf() -> *mut WeldConf {
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

unsafe fn free_value_and_module(value: *mut WeldValue) {
    let module = weld_value_module(value);
    weld_value_free(value);
    weld_module_free(module);
}

/// Runs `code` with the given `conf` and input data pointer `ptr`, expecting
/// a runtime error to be thrown. Panics if no error is thrown.
fn compile_and_run_error<T>(code: &str, conf: *mut WeldConf, ptr: &T) -> *mut WeldError {
    match unsafe { _compile_and_run(code, conf, ptr) } {
        Ok(_) => panic!("Expected an error but got a value"),
        Err(e) => e,
    }
}

/// Runs `code` with the given `conf` and input data pointer `ptr`, expecting
/// a succeessful result to be returned. Panics if an error is thrown by the runtime.
fn compile_and_run<T>(code: &str, conf: *mut WeldConf, ptr: &T) -> *mut WeldValue {
    match unsafe { _compile_and_run(code, conf, ptr) } {
        Ok(val) => val,
        Err(err) => {
            panic!(format!("Compile failed: {:?}",
                           unsafe { CStr::from_ptr(weld_error_message(err)) }))
        }
    }
}

fn basic_program() {
    let code = "|| 40 + 2";
    let conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };
    assert_eq!(result, 42);

    unsafe { free_value_and_module(ret_value) };
}

fn basic_string() {
    let code = "|| \"test str\"";
    let conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<u8> };
    let result = unsafe { (*data).clone() };
    assert_eq!(result.len, 8);
    unsafe { assert_eq!(str::from_utf8(slice::from_raw_parts(result.data, result.len as usize)).unwrap(), "test str"); }

    unsafe { free_value_and_module(ret_value) };
}

fn float_literals() {
    let values = vec![0.0, PI, -PI, 1.2e20, -1.2e-20];
    for v in values {
        // Try parsing the value as a double
        let code = format!("|| {:e}", v);
        let conf = default_conf();
        let ref input_data = 0;
        let ret_value = compile_and_run(&code, conf, input_data);
        let data = unsafe { weld_value_data(ret_value) as *const f64 };
        let result = unsafe { *data };
        assert_eq!(result, v);
        unsafe { weld_value_free(ret_value) };

        // Try parsing the value as a float
        let code = format!("|| {:e}f", v);
        let conf = default_conf();
        let ref input_data = 0;
        let ret_value = compile_and_run(&code, conf, input_data);
        let data = unsafe { weld_value_data(ret_value) as *const f32 };
        let result = unsafe { *data };
        assert_eq!(result, v as f32);
        unsafe { weld_value_free(ret_value) };
    }
}

fn negation() {
    let code = "|| -(1)";
    let conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };
    assert_eq!(result, -1 as i32);

    unsafe { free_value_and_module(ret_value) };
}

fn negation_double() {
    let code = "|| -(1.0)";
    let conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const f64 };
    let result = unsafe { *data };
    assert_eq!(result, -1.0 as f64);

    unsafe { free_value_and_module(ret_value) };
}

fn negated_arithmetic() {
    // In our language, - has the lowest precedence so the paraens around -3 are mandatory.
    let code = "|| 1+2*-3-4/-2";
    let conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };
    assert_eq!(result, -3 as i32);

    unsafe { free_value_and_module(ret_value) };
}

fn bool_eq() {
    let code = "|| [(2 < 3) != (2 > 2), true == false]";

    let conf = default_conf();

    let ref input_data: i32 = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<bool> };
    let result = unsafe { (*data).clone() };

    assert_eq!(result.len, 2);

    let bool1 = unsafe { (*result.data.offset(0)).clone() };
    let bool2 = unsafe { (*result.data.offset(1)).clone() };
    assert_eq!(bool1, true);
    assert_eq!(bool2, false);

    unsafe { free_value_and_module(ret_value) };
}

// TODO(shoumik): - Failing on Linux. Will fix in a in PR
// For the C UDF integration test.
// #[no_mangle]
// pub unsafe extern "C" fn add_five(x: *const i64, result: *mut i64) {
// result = *x + 5;
// }
//
// fn c_udf() {
// let code = "|x:i64| cudf[add_five,i64](x)";
// let conf = default_conf();
//
// let ref mut input_data: i64 = 0;
// To prevent it from compiling out.
// unsafe { add_five(input_data, input_data) };
//
// let ret_value = compile_and_run(code, conf, input_data);
// let data = unsafe { weld_value_data(ret_value) as *const i64 };
// let result = unsafe { *data };
// assert_eq!(result, 10);
//
// unsafe { free_value_and_module(ret_value) };
// }
//

fn f64_cast() {
    let code = "|| f64(40 + 2)";
    let conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const f64 };
    let result = unsafe { *data };
    assert_eq!(result, 42.0);

    unsafe { free_value_and_module(ret_value) };
}

fn i32_cast() {
    let code = "|| i32(0.251 * 4.0)";
    let conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };
    assert_eq!(result, 1);

    unsafe { free_value_and_module(ret_value) };
}

fn multiple_casts() {
    let code = "|| i16(i8(u8(i64(256+255))))";
    let conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i16 };
    let result = unsafe { *data };
    assert_eq!(result, -1i16);

    unsafe { free_value_and_module(ret_value) };

    let code = "|| u32(u64(u16(i16(u8(i64(-1))))))";
    let conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const u32 };
    let result = unsafe { *data };
    assert_eq!(result, 255);

    unsafe { free_value_and_module(ret_value) };
}

fn program_with_args() {
    let code = "|x:i32| 40 + x";
    let conf = default_conf();

    let ref input_data: i32 = 2;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };
    assert_eq!(result, 42);

    unsafe { free_value_and_module(ret_value) };
}

/// Tests literal data structures such as vectors and structs.
fn struct_vector_literals() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Triple {
        a: i32,
        b: i32,
        c: i32,
    }

    let code = "|x:i32| [{x,x,x},{x,x,x}]";
    let conf = default_conf();

    let ref input_data: i32 = 2;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<Triple> };
    let result = unsafe { (*data).clone() };

    assert_eq!(result.len, 2);

    let triple = unsafe { (*result.data.offset(0)).clone() };
    assert_eq!(triple.a, 2);
    assert_eq!(triple.b, 2);
    assert_eq!(triple.c, 2);
    let triple = unsafe { (*result.data.offset(1)).clone() };
    assert_eq!(triple.a, 2);
    assert_eq!(triple.b, 2);
    assert_eq!(triple.c, 2);

    unsafe { free_value_and_module(ret_value) };
}

fn let_statement() {
    let code = "|x:i32| let y = 40 + x; y + 2";
    let conf = default_conf();

    let ref input_data: i32 = 2;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };
    assert_eq!(result, 44);

    unsafe { free_value_and_module(ret_value) };
}

fn if_statement() {
    let code = "|| if(true, 3, 4)";
    let conf = default_conf();

    let ref input_data: i32 = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };
    assert_eq!(result, 3);

    unsafe { free_value_and_module(ret_value) };
}

fn float_min() {
    let code = "|| min(3.1, 4.2)";
    let conf = default_conf();

    let ref input_data: f64 = 0.0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const f64 };
    let result = unsafe { *data };
    assert!(approx_equal(result, 3.1, 5));

    unsafe { free_value_and_module(ret_value) };
}

fn maxmin() {
    let code = "|| max(3, min(2, 4))";
    let conf = default_conf();

    let ref input_data: i32 = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };
    assert_eq!(result, 3);

    unsafe { free_value_and_module(ret_value) };
}

fn comparison() {
    let code = "|x:i32| if(x>10, x, 10)";
    let conf = default_conf();

    let ref mut input_data: i32 = 2;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };
    assert_eq!(result, 10);

    unsafe { free_value_and_module(ret_value) };

    let conf = default_conf();
    *input_data = 20;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };
    assert_eq!(result, 20);

    unsafe { free_value_and_module(ret_value) };
}

fn reused_variable() {
    // `a` is reused in different scopes
    let code = "|| let a=2; let b=map([1,2,3], |e| let a=1; e+a); lookup(b, 0L)+a";
    let conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };
    assert_eq!(result, 4);

    unsafe { free_value_and_module(ret_value) };
}

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

fn nested_if_statement_loop() {
    let code = "|p: vec[i64]|
    result(for(p,merger[i64, +], |bs, i, ns| if(ns >= 3L, if(ns < 7L, merge(bs, ns), bs), bs)))";

    let conf = default_conf();

    let input_vec: Vec<i64> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let ref input_data = WeldVec {
        data: input_vec.as_ptr() as *const i64,
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };
    let output = 18;
    assert_eq!(result, output);
    unsafe { free_value_and_module(ret_value) };
}

fn nested_if_statement_with_builders_loop() {
    let code = "|p: vec[i64]|
        let filter = result(for(p,appender[i64], |bs, i, ns|if(ns >= 3L, if(ns < 7L, merge(bs, ns), bs), bs)));
        result(for(filter, merger[i64, +], |bs2, i2, ns2| merge(bs2, ns2)))";

    let conf = default_conf();

    let input_vec: Vec<i64> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let ref input_data = WeldVec {
        data: input_vec.as_ptr() as *const i64,
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };
    let output = 18;
    assert_eq!(result, output);
    unsafe { free_value_and_module(ret_value) };
}

fn empty_appender_loop() {
    let code = "||result(for([]:vec[i32], merger[i32, +], |b, i, n| merge(b, n)))";
    let conf = default_conf();

    let ref input_data: i32 = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };
    assert_eq!(result, 0);

    unsafe { free_value_and_module(ret_value) };
}

fn simple_for_appender_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        a: i32,
    }

    let code = "|x:vec[i32], a:i32| let b=a+1; map(x, |e| e+b)";
    let conf = default_conf();

    let input_vec = [1, 2];
    let ref input_data = Args {
        x: WeldVec {
            data: &input_vec as *const i32,
            len: input_vec.len() as i64,
        },
        a: 1,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<i32> };
    let result = unsafe { (*data).clone() };
    let output = [3, 4];
    assert_eq!(result.len as usize, output.len());
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
    }

    unsafe { free_value_and_module(ret_value) };
}

fn large_unaryop_for_appender_loop() {
    let code = "|x:vec[f32]| map(x, |e| sqrt(e))";
    let conf = default_conf();

    let input_vec: Vec<f32> = vec![4.0; 1000000];
    let ref input_data: WeldVec<f32> = WeldVec {
        data: input_vec.as_ptr(),
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<f32> };
    let result = unsafe { (*data).clone() };
    assert_eq!(result.len as usize, input_vec.len());
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) as i32 }, 2)
    }
    unsafe { free_value_and_module(ret_value) };
}

fn simple_parallel_for_appender_loop() {
    let code = "|x:vec[i32]| result(@(grain_size: 100)for(x, appender[i64], |b,i,e| merge(b, i)))";
    let conf = many_threads_conf();

    let size = 10000;
    let input_vec: Vec<i32> = vec![0; size as usize];
    let ref input_data = WeldVec {
        data: input_vec.as_ptr() as *const i32,
        len: size,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<i64> };
    let result = unsafe { (*data).clone() };

    assert_eq!(result.len, size);
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, i as i64)
    }
    unsafe { free_value_and_module(ret_value) };
}

fn simple_parallel_for_multi_appender_loop() {
    let code = "|x:vec[i32]| let r = @(grain_size: 100)for(x, {appender[i64], appender[i64]}, |b,i,e|
                { merge(b.$0, i), merge(b.$1, i) }); let r2 = @(grain_size: 100)for(result(r.$0), r.$1,
                |b,i,e| merge(b, e)); result(r2)";
    let conf = many_threads_conf();

    let size = 10000;
    let input_vec: Vec<i32> = vec![0; size as usize];
    let ref input_data = WeldVec {
        data: input_vec.as_ptr() as *const i32,
        len: size,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<i64> };
    let result = unsafe { (*data).clone() };

    assert_eq!(result.len, size * 2);
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, (i as i64) % size)
    }
    unsafe { free_value_and_module(ret_value) };
}

fn complex_parallel_for_appender_loop() {
    let code = "|x:vec[i32]| let a=appender[i64]; let b=merge(a,0L); let r=@(grain_size: 100)for(x,b,|b,i,e|
                let c=merge(b,1L); let d=@(grain_size: 100)for(x,c,|b,i,e| if(i<1L, merge(b,i), b)); merge(d, 2L));
                result(merge(r,3L))";
    let conf = many_threads_conf();

    let size = 3000;
    let input_vec: Vec<i32> = vec![0; size as usize];
    let ref input_data = WeldVec {
        data: input_vec.as_ptr() as *const i32,
        len: size,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<i64> };
    let result = unsafe { (*data).clone() };

    assert_eq!(result.len, size * 3 + 2);
    assert_eq!(unsafe { *result.data.offset(0) }, 0);
    assert_eq!(unsafe { *result.data.offset((size * 3 + 1) as isize) }, 3);
    for i in 0..(size as isize) {
        assert_eq!(unsafe { *result.data.offset(i * 3 + 1) }, 1);
        assert_eq!(unsafe { *result.data.offset(i * 3 + 2) }, 0);
        assert_eq!(unsafe { *result.data.offset(i * 3 + 3) }, 2)
    }

    unsafe { free_value_and_module(ret_value) };
}

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

fn range_iter_2() {
    range_iter_zipped_helper(false)
}

fn range_iter_parallel() {
    range_iter_zipped_helper(true)
}

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

fn simple_for_merger_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        a: i32,
    }

    let code = "|x:vec[i32], a:i32| result(for(x, merger[i32,+], |b,i,e| merge(b, e+a)))";
    let conf = default_conf();

    let input_vec = vec![1, 2, 3, 4, 5];
    let ref input_data = Args {
        x: WeldVec {
            data: input_vec.as_ptr(),
            len: input_vec.len() as i64,
        },
        a: 1,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };
    let output = 20;
    assert_eq!(result, output);
    unsafe { free_value_and_module(ret_value) };
}

fn simple_zipped_for_merger_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]| result(for(zip(x,y), merger[i32,+], |b,i,e| merge(b, e.$0+e.$1)))";
    let conf = default_conf();

    let size = 2000;
    let x_data = vec![1; size as usize];
    let y_data = vec![5; size as usize];

    let ref input_data = Args {
        x: WeldVec {
            data: x_data.as_ptr(),
            len: x_data.len() as i64,
        },
        y: WeldVec {
            data: y_data.as_ptr(),
            len: y_data.len() as i64,
        },
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };
    let output = size * (x_data[0] + y_data[0]);
    assert_eq!(result, output);
    unsafe { free_value_and_module(ret_value) };
}

fn simple_for_merger_loop_product() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        a: i32,
    }

    let code = "|x:vec[i32], a:i32| result(for(x, merger[i32,*], |b,i,e| merge(b, e+a)))";
    let conf = default_conf();

    let input_vec = [1, 2, 3, 4, 5];
    let ref input_data = Args {
        x: WeldVec {
            data: &input_vec as *const i32,
            len: input_vec.len() as i64,
        },
        a: 1,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };
    let output = 720;
    assert_eq!(result, output);
    unsafe { free_value_and_module(ret_value) };
}

fn parallel_for_merger_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        a: i32,
    }

    let code = "|x:vec[i32], a:i32| result(@(grain_size: 100)for(x, merger[i32,+], |b,i,e| merge(b, e+a)))";
    let conf = many_threads_conf();

    let input_vec = [1; 4096];
    let ref input_data = Args {
        x: WeldVec {
            data: &input_vec as *const i32,
            len: input_vec.len() as i64,
        },
        a: 1,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };
    let output = (input_vec[0] + input_data.a) * (input_vec.len() as i32);
    assert_eq!(result, output);
    unsafe { free_value_and_module(ret_value) };
}

fn parallel_for_multi_merger_loop() {
    let code = "|x:vec[i32]| let r = @(grain_size: 100)for(x, {merger[i32,+], merger[i32,+]}, |b,i,e|
                { merge(b.$0, e), merge(b.$1, e) }); result(r.$0) + result(r.$1)";
    let conf = many_threads_conf();

    let input_vec = [1; 4096];
    let ref input_data = WeldVec {
        data: &input_vec as *const i32,
        len: input_vec.len() as i64
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };
    let output = input_vec[0] * 2 * (input_vec.len() as i32);
    assert_eq!(result, output);
    unsafe { free_value_and_module(ret_value) };
}

fn simple_for_merger_loop_initial_value() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        a: i32,
    }

    let code = "|x:vec[i32], a:i32| result(for(x, merger[i32,+](1000), |b,i,e| merge(b, e+a)))";
    let conf = default_conf();

    let input_vec = [1; 4096];
    let ref input_data = Args {
        x: WeldVec {
            data: &input_vec as *const i32,
            len: input_vec.len() as i64,
        },
        a: 1,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };
    let output = 1000 + (input_vec[0] + input_data.a) * (input_vec.len() as i32);
    assert_eq!(result, output);
    unsafe { free_value_and_module(ret_value) };
}

fn parallel_for_merger_loop_initial_value() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        a: i32,
    }

    let code = "|x:vec[i32], a:i32| result(@(grain_size: 100)for(x, merger[i32,+](1000), |b,i,e| merge(b, e+a)))";
    let conf = many_threads_conf();

    let input_vec = [1; 4096];
    let ref input_data = Args {
        x: WeldVec {
            data: &input_vec as *const i32,
            len: input_vec.len() as i64,
        },
        a: 1,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };
    let output = 1000 + (input_vec[0] + input_data.a) * (input_vec.len() as i32);
    assert_eq!(result, output);
    unsafe { free_value_and_module(ret_value) };
}

fn parallel_for_merger_loop_initial_value_product() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
    }

    let code = "|x:vec[i32]| result(@(grain_size: 100)for(x, merger[i32,*](1000), |b,i,e| merge(b, e)))";
    let conf = many_threads_conf();

    let input_vec = [1; 4096];
    let ref input_data = Args {
        x: WeldVec {
            data: &input_vec as *const i32,
            len: input_vec.len() as i64,
        },
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };
    let output = 1000;
    assert_eq!(result, output);
    unsafe { free_value_and_module(ret_value) };
}

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
        assert_eq!(unsafe { *result.min.data.offset(i) }, cmp::min(input_vec[i as usize], (i as i64)));
        assert_eq!(unsafe { *result.max.data.offset(i) }, cmp::max(input_vec[i as usize], (i as i64)));
    }
    unsafe { free_value_and_module(ret_value) };
}

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

fn simple_for_dictmerger_loop() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Pair {
        ele1: i32,
        ele2: i32,
    }

    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]| tovec(result(for(zip(x,y), dictmerger[i32,i32,+],
                |b,i,e| merge(b, e))))";
    let conf = default_conf();
    let keys = [1, 2, 2, 1, 3];
    let vals = [2, 3, 4, 2, 1];
    let ref input_data = Args {
        x: WeldVec {
            data: &keys as *const i32,
            len: keys.len() as i64,
        },
        y: WeldVec {
            data: &vals as *const i32,
            len: vals.len() as i64,
        },
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<Pair> };
    let result = unsafe { (*data).clone() };

    let output_keys = [1, 2, 3];
    let output_vals = [4, 7, 1];

    assert_eq!(result.len, output_keys.len() as i64);
    for i in 0..(output_keys.len() as isize) {
        let mut success = false;
        let key = unsafe { (*result.data.offset(i)).ele1 };
        let value = unsafe { (*result.data.offset(i)).ele2 };
        for j in 0..(output_keys.len()) {
            if output_keys[j] == key {
                if output_vals[j] == value {
                    success = true;
                }
            }
        }
        assert_eq!(success, true);
    }
    unsafe { free_value_and_module(ret_value) };
}

/// Similar case to parallel_for_vecmerger_loop but values and keys are structs
fn dictmerger_with_structs() {
    /// An entry after the final tovec call, which has an {i32,i32} key and {i32,f32} value.
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Entry {
        k1: i32,
        k2: i32,
        v1: i32,
        v2: f32
    }

    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]|
                    tovec(result(for(
                        zip(x,y),
                        dictmerger[{i32,i32},{i32,f32},+],
                        |b,i,e| merge(b, {{e.$0, e.$0}, {e.$1, f32(e.$1)}}))))";
    let conf = default_conf();
    let keys = [1, 2, 2, 1, 3];
    let vals = [2, 3, 4, 2, 1];
    let ref input_data = Args {
        x: WeldVec {
            data: &keys as *const i32,
            len: keys.len() as i64,
        },
        y: WeldVec {
            data: &vals as *const i32,
            len: vals.len() as i64,
        },
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<Entry> };
    let result = unsafe { (*data).clone() };

    let output_keys = [1, 2, 3];
    let output_vals = [4, 7, 1];

    assert_eq!(result.len, output_keys.len() as i64);
    for i in 0..(output_keys.len() as isize) {
        let entry = unsafe { (*result.data.offset(i)).clone() };
        // Check whether we find the entry anywhere in the expected outputs
        let mut success = false;
        for j in 0..(output_keys.len()) {
            if entry.k1 == output_keys[j] && entry.k2 == output_keys[j] &&
                    entry.v1 == output_vals[j] && entry.v2 == output_vals[j] as f32 {
                success = true;
            }
        }
        assert_eq!(success, true);
    }
    unsafe { free_value_and_module(ret_value) };
}

fn simple_groupmerger() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]| tovec(result(for(zip(x,y), groupmerger[i32,i32],
                |b,i,e| merge(b, e))))";

    let conf = default_conf();
    let keys = [1, 2, 2, 3, 3, 1];
    let vals = [2, 3, 4, 1, 0, 2];
    let ref input_data = Args {
        x: WeldVec {
            data: &keys as *const i32,
            len: keys.len() as i64,
        },
        y: WeldVec {
            data: &vals as *const i32,
            len: vals.len() as i64,
        },
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<Pair<i32, WeldVec<i32>>> };
    let result = unsafe { (*data).clone() };
    let output: Vec<(i32, Vec<i32>)> = vec![(1, vec![2, 2]), (2, vec![3, 4]), (3, vec![1, 0])];

    let mut res: Vec<(i32, Vec<i32>)> = (0..result.len)
        .into_iter()
        .map(|x| {
            let key = unsafe { (*result.data.offset(x as isize)).ele1 };
            let val = unsafe { ((*result.data.offset(x as isize)).ele2).clone() };
            let vec: Vec<i32> = (0..val.len)
                .into_iter()
                .map(|y| unsafe { *val.data.offset(y as isize) })
                .collect();
            (key, vec)
        })
        .collect();
    res.sort_by_key(|a| a.0);

    assert_eq!(res, output);
    unsafe { free_value_and_module(ret_value) };
}

fn complex_groupmerger_with_struct_key() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
        z: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32], z:vec[i32]|
                tovec(result(for(zip(x,y,z), groupmerger[{i32,i32}, i32],
                |b,i,e| merge(b, {{e.$0, e.$1}, e.$2}))))";

    let conf = default_conf();
    let keys1 = [1, 1, 2, 2, 3, 3, 3, 3];
    let keys2 = [1, 1, 2, 2, 3, 3, 4, 4];
    let vals = [2, 3, 4, 2, 1, 0, 3, 2];
    let ref input_data = Args {
        x: WeldVec {
            data: &keys1 as *const i32,
            len: keys1.len() as i64,
        },
        y: WeldVec {
            data: &keys2 as *const i32,
            len: keys2.len() as i64,
        },
        z: WeldVec {
            data: &vals as *const i32,
            len: vals.len() as i64,
        },
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data =
        unsafe { weld_value_data(ret_value) as *const WeldVec<Pair<Pair<i32, i32>, WeldVec<i32>>> };
    let result = unsafe { (*data).clone() };
    let output = vec![((1, 1), vec![2, 3]),
                      ((2, 2), vec![4, 2]),
                      ((3, 3), vec![1, 0]),
                      ((3, 4), vec![3, 2])];

    let mut res: Vec<((i32, i32), Vec<i32>)> = (0..result.len)
        .into_iter()
        .map(|x| {
            let key = unsafe { ((*result.data.offset(x as isize)).ele1).clone() };
            let val = unsafe { ((*result.data.offset(x as isize)).ele2).clone() };
            let vec: Vec<i32> = (0..val.len)
                .into_iter()
                .map(|y| unsafe { *val.data.offset(y as isize) })
                .collect();
            ((key.ele1, key.ele2), vec)
        })
        .collect();
    res.sort_by_key(|a| a.0);

    assert_eq!(res, output);
    unsafe { free_value_and_module(ret_value) };
}

fn simple_parallel_for_dictmerger_loop_helper(use_local: bool) {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Pair {
        ele1: i32,
        ele2: i32,
    }
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = format!("|x:vec[i32], y:vec[i32]| tovec(result(@(grain_size: 100)for(zip(x,y),
                dictmerger[i32,i32,+]({}L), |b,i,e| merge(b, e))))",
                if use_local { 100000000 } else { 0 });
    let conf = many_threads_conf();

    const DICT_SIZE: usize = 8192;
    let mut keys = [0; DICT_SIZE];
    let mut vals = [0; DICT_SIZE];

    for i in 0..DICT_SIZE {
        keys[i] = i as i32;
        vals[i] = i as i32;
    }
    let ref input_data = Args {
        x: WeldVec {
            data: &keys as *const i32,
            len: DICT_SIZE as i64,
        },
        y: WeldVec {
            data: &vals as *const i32,
            len: DICT_SIZE as i64,
        },
    };

    let ret_value = compile_and_run(&code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<Pair> };
    let result = unsafe { (*data).clone() };

    let output_keys = keys;
    let output_values = vals;
    for i in 0..(output_keys.len() as isize) {
        let mut success = false;
        let key = unsafe { (*result.data.offset(i)).ele1 };
        let value = unsafe { (*result.data.offset(i)).ele2 };
        for j in 0..(output_keys.len()) {
            if output_keys[j] == key {
                if output_values[j] == value {
                    success = true;
                }
            }
        }
        assert_eq!(success, true);
    }
    assert_eq!(result.len, output_keys.len() as i64);
    unsafe { free_value_and_module(ret_value) };
}

fn simple_parallel_for_dictmerger_loop_local() {
    simple_parallel_for_dictmerger_loop_helper(true);
}

fn simple_parallel_for_dictmerger_loop_global() {
    simple_parallel_for_dictmerger_loop_helper(false);
}

fn simple_dict_lookup() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]| let a = result(for(zip(x,y), dictmerger[i32,i32,+],
                |b,i,e| merge(b, e))); lookup(a, 1)";
    let conf = default_conf();

    let keys = [1, 2, 2, 1, 3];
    let vals = [2, 3, 4, 2, 1];
    let ref input_data = Args {
        x: WeldVec {
            data: &keys as *const i32,
            len: keys.len() as i64,
        },
        y: WeldVec {
            data: &vals as *const i32,
            len: vals.len() as i64,
        },
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };

    let output = 4;
    assert_eq!(output, result);
    unsafe { free_value_and_module(ret_value) };
}

fn string_dict_lookup() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32]| let v = [\"abcdefghi\", \"abcdefghi\", \"abcdefghi\"];
                let d = result(for(zip(v,x), dictmerger[vec[i8],i32,+], |b,i,e| merge(b, e)));
                lookup(d, \"abcdefghi\")";
    let conf = default_conf();

    let input_vec = [1, 1, 1];
    let ref input_data = WeldVec {
        data: &input_vec as *const i32,
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };

    let output = 3;
    assert_eq!(output, result);
    unsafe { free_value_and_module(ret_value) };
}

fn simple_dict_exists() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let keys = [1, 2, 2, 1, 3];
    let vals = [2, 3, 4, 2, 1];

    let code_true = "|x:vec[i32], y:vec[i32]| let a = result(for(zip(x,y), dictmerger[i32,i32,+],
                |b,i,e| merge(b, e))); keyexists(a, 1)";
    let code_false = "|x:vec[i32], y:vec[i32]| let a = result(for(zip(x,y),
                      dictmerger[i32,i32,+], |b,i,e| merge(b, e))); keyexists(a, 4)";
    let conf = default_conf();

    let ref input_data = Args {
        x: WeldVec {
            data: &keys as *const i32,
            len: keys.len() as i64,
        },
        y: WeldVec {
            data: &vals as *const i32,
            len: vals.len() as i64,
        },
    };

    let ret_value = compile_and_run(code_true, conf, input_data.clone());
    let data = unsafe { weld_value_data(ret_value) as *const bool };
    let result = unsafe { (*data).clone() };

    let output = true;
    assert_eq!(output, result);
    unsafe { free_value_and_module(ret_value) };

    let conf = default_conf();
    let ret_value = compile_and_run(code_false, conf, input_data.clone());
    let data = unsafe { weld_value_data(ret_value) as *const bool };
    let result = unsafe { (*data).clone() };

    let output = false;
    assert_eq!(output, result);
    unsafe { free_value_and_module(ret_value) };
}

fn simple_length() {
    let code = "|x:vec[i32]| len(x)";
    let conf = default_conf();

    let input_vec = [2, 3, 4, 2, 1];
    let ref input_data = WeldVec {
        data: &input_vec as *const i32,
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };

    let output = 5;
    assert_eq!(output, result);
    unsafe { free_value_and_module(ret_value) };
}

fn filter_length() {
    let code = "|x:vec[i32]| len(filter(x, |i| i < 4 && i > 1))";
    let conf = default_conf();

    let input_vec = [2, 3, 4, 2, 1];
    let ref input_data = WeldVec {
        data: &input_vec as *const i32,
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };

    let output = 3;
    assert_eq!(output, result);
    unsafe { free_value_and_module(ret_value) };
}

fn flat_map_length() {
    let code = "|x:vec[i32]| len(flatten(map(x, |i:i32| x)))";
    let conf = default_conf();

    let input_vec = [2, 3, 4, 2, 1];
    let ref input_data = WeldVec {
        data: &input_vec as *const i32,
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };

    let output = 25;
    assert_eq!(output, result);
    unsafe { free_value_and_module(ret_value) };
}

fn simple_log() {
    let code = "|x:f64| log(x)";
    let conf = default_conf();
    let input = 2.718281828459045;
    let ret_value = compile_and_run(code, conf, &input);
    let data = unsafe { weld_value_data(ret_value) as *const f64 };
    let result = unsafe { (*data).clone() };
    let output = 1.0f64;
    assert!(approx_equal(output, result, 5));
    unsafe { free_value_and_module(ret_value) };
}

fn log_error() {
    let code = "|x:i64| log(x)";
    let conf = default_conf();
    let input = 1;
    let err_value = compile_and_run_error(code, conf, &input);
    assert_eq!(unsafe { weld_error_code(err_value) },
               WeldRuntimeErrno::CompileError);
    unsafe { weld_error_free(err_value) };
}


fn simple_exp() {
    let code = "|x:f64| exp(x)";
    let conf = default_conf();
    let input = 1.0f64;
    let ret_value = compile_and_run(code, conf, &input);
    let data = unsafe { weld_value_data(ret_value) as *const f64 };
    let result = unsafe { (*data).clone() };
    let output = 2.718281828459045;
    assert!(approx_equal(output, result, 5));
    unsafe { free_value_and_module(ret_value) };
}

fn exp_error() {
    let code = "|x:i64| exp(x)";
    let conf = default_conf();
    let input = 1;
    let err_value = compile_and_run_error(code, conf, &input);
    assert_eq!(unsafe { weld_error_code(err_value) },
               WeldRuntimeErrno::CompileError);
    unsafe { weld_error_free(err_value) };
}

fn simple_erf() {
    let code = "|x:f64| erf(x)";
    let conf = default_conf();
    let input = 1.00;
    let ret_value = compile_and_run(code, conf, &input);
    let data = unsafe { weld_value_data(ret_value) as *const f64 };
    let result = unsafe { (*data).clone() };
    let output = 0.84270079294971478;
    assert!(approx_equal(output, result, 5));
    unsafe { free_value_and_module(ret_value) };
}


fn simple_sqrt() {
    let code = "|x:f64| sqrt(x)";
    let conf = default_conf();
    let input = 4.0;
    let ret_value = compile_and_run(code, conf, &input);
    let data = unsafe { weld_value_data(ret_value) as *const f64 };

    let result = unsafe { (*data).clone() };
    let output = 2.0f64;
    assert!(approx_equal(output, result, 5));
    unsafe { free_value_and_module(ret_value) };
}

fn simple_pow() {
    use std::f64;
    let code = "|x:f64| pow(x, 2.0)";
    let conf = default_conf();
    let input = 4.0;
    let ret_value = compile_and_run(code, conf, &input);
    let data = unsafe { weld_value_data(ret_value) as *const f64 };

    let result = unsafe { (*data).clone() };
    assert!(approx_equal(16.0, result, 5));
    unsafe { free_value_and_module(ret_value) };
}


fn simple_trig() {
    fn check_trig_f32(op: &str, input: f32, expect: f32) {
        let code = format!("|x:f32| {}(x)", op);
        let conf = default_conf();
        let ret_value = compile_and_run(&code, conf, &input);
        let data = unsafe { weld_value_data(ret_value) as *const f32 };
        let result = unsafe { (*data).clone() };
        assert!(approx_equal_f32(expect, result, 5));
        unsafe { free_value_and_module(ret_value) };
    }

    fn check_trig_f64(op: &str, input: f64, expect: f64) {
        let code = format!("|x:f64| {}(x)", op);
        let conf = default_conf();
        let ret_value = compile_and_run(&code, conf, &input);
        let data = unsafe { weld_value_data(ret_value) as *const f64 };
        let result = unsafe { (*data).clone() };
        assert!(approx_equal(expect, result, 5));
        unsafe { free_value_and_module(ret_value) };
    }

    let inp: f32 = 1.0;
    check_trig_f32("sin", inp, inp.sin());
    check_trig_f32("cos", inp, inp.cos());
    check_trig_f32("tan", inp, inp.tan());
    check_trig_f32("asin", inp, inp.asin());
    check_trig_f32("acos", inp, inp.acos());
    check_trig_f32("atan", inp, inp.atan());
    check_trig_f32("sinh", inp, inp.sinh());
    check_trig_f32("cosh", inp, inp.cosh());
    check_trig_f32("tanh", inp, inp.tanh());

    let inp: f64 = 1.0;
    check_trig_f64("sin", inp, inp.sin());
    check_trig_f64("cos", inp, inp.cos());
    check_trig_f64("tan", inp, inp.tan());
    check_trig_f64("asin", inp, inp.asin());
    check_trig_f64("acos", inp, inp.acos());
    check_trig_f64("atan", inp, inp.atan());
    check_trig_f64("sinh", inp, inp.sinh());
    check_trig_f64("cosh", inp, inp.cosh());
    check_trig_f64("tanh", inp, inp.tanh());
}

fn map_exp() {
    let code = "|x:vec[f32]| map(x, |a| exp(a))";
    let conf = default_conf();

    let input_vec = [0.0f32, 1.0f32, 2.0f32, 3.0f32];
    let ref input_data = WeldVec {
        data: &input_vec as *const f32,
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<f32> };
    let result = unsafe { (*data).clone() };

    let output = [1.0, 2.7182817, 7.389056, 20.085537];
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
    }

    unsafe { free_value_and_module(ret_value) };
}

fn simple_int_mod() {
    let code = "|x:i64| x % 3L";
    let conf = default_conf();
    let ref input_data: i64 = -10;
    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i64 };
    let result = unsafe { *data };
    assert_eq!(result, -1);
    unsafe { free_value_and_module(ret_value) };
}

fn simple_float_mod() {
    let code = "|x:f64| x % 0.04";
    let conf = default_conf();
    let ref input_data: f64 = 0.5;
    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const f64 };
    let result = unsafe { *data };
    assert!(approx_equal(result, 0.02, 5));
    unsafe { free_value_and_module(ret_value) };
}

fn if_for_loop() {
    let code = "|x:vec[i32], a:i32| if(a > 5, map(x, |e| e+1), map(x, |e| e+2))";
    let conf = default_conf();

    let input_vec = [1, 2];

    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        a: i32,
    }

    let ref input_data = Args {
        x: WeldVec {
            data: &input_vec as *const i32,
            len: input_vec.len() as i64,
        },
        a: 1,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<i32> };
    let result = unsafe { (*data).clone() };

    let output = [3, 4];
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
    }
    unsafe { free_value_and_module(ret_value) };
}

fn map_zip_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]| map(zip(x,y), |e| e.$0 + e.$1)";
    let conf = default_conf();

    let x = [1, 2, 3, 4];
    let y = [5, 6, 7, 8];
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

    let output = [6, 8, 10, 12];
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
    }

    unsafe { free_value_and_module(ret_value) };
}

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

fn iterate_non_parallel() {
    let code = "|x:i32| iterate(x, |x| {x-1, x-1>0})";
    let conf = default_conf();

    let input: i32 = 5;

    let ret_value = compile_and_run(code, conf, &input);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };

    assert_eq!(result, 0);

    unsafe { weld_value_free(ret_value) };
}

fn iterate_with_parallel_body() {
    let code = "|x:i32| let a=2; iterate({[1,2,3], 1}, |p| {{map(p.$0, |y|y*a), p.$1+1}, p.$1<x}).$0";
    let conf = default_conf();

    let input: i32 = 3;

    let ret_value = compile_and_run(code, conf, &input);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<i32> };
    let result = unsafe { (*data).clone() };

    let output = [8, 16, 24];
    assert_eq!(result.len, output.len() as i64);
    for i in 0..(output.len() as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
    }

    unsafe { weld_value_free(ret_value) };
}

fn serial_parlib_test() {
    let code = "|x:vec[i32]| result(for(x, merger[i32,+], |b,i,e| merge(b, e)))";
    let conf = default_conf();

    let size: i32 = 10000;
    let input_vec: Vec<i32> = vec![1; size as usize];

    let ref input_data = WeldVec {
        data: input_vec.as_ptr() as *const i32,
        len: size as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };

    assert_eq!(result, size as i32);
    unsafe { free_value_and_module(ret_value) };
}

fn many_mergers_test() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Output {
        a: i64,
        b: f64,
        c: i64,
        d: f64,
        e: i64,
        f: f64,
    }

    #[derive(Clone)]
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<f64>,
        y: WeldVec<f64>,
        z: WeldVec<f64>,
    }

    let code = "
|p0: vec[f64], p1: vec[f64], p2: vec[f64]|
 let merged = for(zip(p0, p1, p2),
                  {merger[i64, +], merger[f64, +], merger[i64, +], merger[f64, +], merger[i64, +], merger[f64, +]},
                  |bs1: {merger[i64, +], merger[f64, +], merger[i64, +], merger[f64, +], merger[i64, +], merger[f64, +]}, i1: i64, ns1: {f64, f64, f64}|
                   {merge(bs1.$0, 1L), merge(bs1.$1, f64(ns1.$0)), merge(bs1.$2, 1L), merge(bs1.$3, f64(ns1.$1)), merge(bs1.$4, 1L), merge(bs1.$5, f64(ns1.$2))});
 {result(merged.$0), result(merged.$1), result(merged.$2), result(merged.$3), result(merged.$4), result(merged.$5)}";

    let conf = default_conf();

    let size: i32 = 1000;
    let x = vec![1.0; size as usize];
    let y = vec![1.0; size as usize];
    let z = vec![1.0; size as usize];

    let ref input_data = Args {
        x: WeldVec {
            data: x.as_ptr() as *const f64,
            len: x.len() as i64,
        },
        y: WeldVec {
            data: y.as_ptr() as *const f64,
            len: y.len() as i64,
        },
        z: WeldVec {
            data: z.as_ptr() as *const f64,
            len: z.len() as i64,
        },
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const Output };
    let result = unsafe { (*data).clone() };

    assert_eq!(result.a, size as i64);
    assert_eq!(result.b, size as f64);
    assert_eq!(result.c, size as i64);
    assert_eq!(result.d, size as f64);
    assert_eq!(result.e, size as i64);
    assert_eq!(result.f, size as f64);
    unsafe { free_value_and_module(ret_value) };
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
#[repr(C)]
struct SerializeOutput {
    a: i32,
    b: Pair<i32,i32>,
    c: WeldVec<i32>,
    d: WeldVec<WeldVec<i32>>,
    e: Pair<i32, WeldVec<i32>>,
    f: WeldVec<Pair<i32,i32>>,
    g: WeldVec<Pair<i32,WeldVec<i32>>>,
}

impl PartialEq for SerializeOutput {
    fn eq(&self, other: &SerializeOutput) -> bool {
        let mut passed = true;
        passed &= self.a == other.a;
        passed &= self.b == other.b;
        passed &= self.c == other.c;
        passed &= self.d == other.d;
        passed &= self.e == other.e;

        fn f_into_native(v: &WeldVec<Pair<i32,i32>>) -> Vec<(i32, i32)> {
            let mut res: Vec<(i32, i32)> = (0..v.len)
                .into_iter()
                .map(|x| {
                    unsafe { ((*v.data.offset(x as isize)).ele1, (*v.data.offset(x as isize)).ele2) }
                })
            .collect();
            res.sort_by_key(|a| a.0);
            res
        }

        passed &= f_into_native(&self.f) == f_into_native(&other.f);

        // Converts field g into a native rust Vec.
        fn g_into_native(v: &WeldVec<Pair<i32,WeldVec<i32>>>) -> Vec<(i32, Vec<i32>)>{
            let mut res: Vec<(i32, Vec<i32>)> = (0..v.len)
                .into_iter()
                .map(|x| {
                    let key = unsafe { (*v.data.offset(x as isize)).ele1 };
                    let val = unsafe { ((*v.data.offset(x as isize)).ele2).clone() };
                    let vec: Vec<i32> = (0..val.len)
                        .into_iter()
                        .map(|y| unsafe { *val.data.offset(y as isize) })
                        .collect();
                    (key, vec)
                })
            .collect();

            // For dictionary outputs, we need to ignore the order.
            res.sort_by_key(|a| a.0);
            res
        }

        passed &= g_into_native(&self.g) == g_into_native(&other.g);

        passed
    }
}

fn serialize_test() {
    let code = " |v: vec[i32]|
    let dict1 = result(for(v, dictmerger[i32,i32,+], |b,i,e| merge(b, {e,e})));
    let dict2 = result(for(v, groupmerger[i32,i32], |b,i,e| merge(b, {e,e})));

    let a = deserialize[i32](serialize(lookup(v, 0L)));
    let b = deserialize[{i32,i32}](serialize({lookup(v, 0L), lookup(v, 1L)}));
    let c = deserialize[vec[i32]](serialize(v));
    let d = deserialize[vec[vec[i32]]](serialize([v, v, v]));
    let e = deserialize[{i32,vec[i32]}](serialize({lookup(v, 0L), v}));
    let f = tovec(deserialize[dict[i32,i32]](serialize(dict1)));
    let g = tovec(deserialize[dict[i32,vec[i32]]](serialize(dict2)));
    {a,b,c,d,e,f,g}";

    let conf = default_conf();

    let input_vec: Vec<i32> = (10..20).collect();
    let ref input_data = WeldVec {
        data: input_vec.as_ptr(),
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const SerializeOutput };
    let result = unsafe { (*data).clone() };

    let vv = vec![input_data.clone(), input_data.clone(), input_data.clone()];
    let dict1_vec: Vec<_> = input_vec.iter().map(|e| Pair::new(*e, *e)).collect();
    let dict2_inners: Vec<_> = input_vec.iter().map(|e| (*e, vec![*e])).collect();
    let dict2_vec: Vec<_>  = dict2_inners.iter()
        .map(|e| {
            let e1 = e.0;
            let e2 = WeldVec {
                data: e.1.as_ptr(),
                len: e.1.len() as i64,
            };
            Pair::new(e1, e2)
        })
    .collect();

    let expected = SerializeOutput {
        a: input_vec[0],
        b: Pair::new(input_vec[0], input_vec[1]),
        c: input_data.clone(),
        d: WeldVec {
            data: vv.as_ptr(),
            len: vv.len() as i64,
        },
        e: Pair::new(input_vec[0], input_data.clone()),
        f: WeldVec {
            data: dict1_vec.as_ptr(),
            len: dict1_vec.len() as i64,
        },
        g: WeldVec {
            data: dict2_vec.as_ptr(),
            len: dict2_vec.len() as i64,
        }
    };

    assert_eq!(result, expected);
    unsafe { free_value_and_module(ret_value) };
}

fn maxmin_mergers_test() {
    #[derive(Clone)]
    #[allow(dead_code)]
    // Larger types have to be first, or else the struct won't be read back correctly
    struct Output {
        i64min: i64,
        i64max: i64,
        f64min: f64,
        f64max: f64,
        f32min: f32,
        f32max: f32,
        i32min: i32,
        i32max: i32,
        i8min: i8,
        i8max: i8,
    }

    #[derive(Clone)]
    #[allow(dead_code)]
    struct Args {
        i8in: WeldVec<i8>,
        i32in: WeldVec<i32>,
        i64in: WeldVec<i64>,
        f32in: WeldVec<f32>,
        f64in: WeldVec<f64>,
    }

    let code = "
    |i8in: vec[i8], i32in: vec[i32], i64in: vec[i64], f32in: vec[f32], f64in: vec[f64]|
    let i8min = result(for(i8in, merger[i8, min], |b, i, n| merge(b, n)));
    let i8max = result(for(i8in, merger[i8, max], |b, i, n| merge(b, n)));
    let i32min = result(for(i32in, merger[i32, min], |b, i, n| merge(b, n)));
    let i32max = result(for(i32in, merger[i32, max], |b, i, n| merge(b, n)));
    let i64min = result(for(i64in, merger[i64, min], |b, i, n| merge(b, n)));
    let i64max = result(for(i64in, merger[i64, max], |b, i, n| merge(b, n)));
    let f32min = result(for(f32in, merger[f32, min], |b, i, n| merge(b, n)));
    let f32max = result(for(f32in, merger[f32, max], |b, i, n| merge(b, n)));
    let f64min = result(for(f64in, merger[f64, min], |b, i, n| merge(b, n)));
    let f64max = result(for(f64in, merger[f64, max], |b, i, n| merge(b, n)));
    {i64min, i64max, f64min, f64max, f32min, f32max, i32min, i32max, i8min, i8max}";

    let conf = default_conf();

    let i8in: Vec<i8> = vec![-2, -1, 0, 1, 2];
    let i32in: Vec<i32> = vec![-2, -1, 0, 1, 2];
    let i64in: Vec<i64> = vec![-2, -1, 0, 1, 2];
    let f32in: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let f64in: Vec<f64> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

    let ref input_data = Args {
        i8in: WeldVec {
            data: i8in.as_ptr() as *const i8,
            len: i8in.len() as i64,
        },
        i32in: WeldVec {
            data: i32in.as_ptr() as *const i32,
            len: i32in.len() as i64,
        },
        i64in: WeldVec {
            data: i64in.as_ptr() as *const i64,
            len: i64in.len() as i64,
        },
        f32in: WeldVec {
            data: f32in.as_ptr() as *const f32,
            len: f32in.len() as i64,
        },
        f64in: WeldVec {
            data: f64in.as_ptr() as *const f64,
            len: f64in.len() as i64,
        },
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const Output };
    let result = unsafe { (*data).clone() };

    assert_eq!(result.i8min, -2 as i8);
    assert_eq!(result.i32min, -2 as i32);
    assert_eq!(result.i64min, -2 as i64);
    assert_eq!(result.f32min, -2.0 as f32);
    assert_eq!(result.f64min, -2.0 as f64);
    assert_eq!(result.i8max, 2 as i8);
    assert_eq!(result.i32max, 2 as i32);
    assert_eq!(result.i64max, 2 as i64);
    assert_eq!(result.f32max, 2.0 as f32);
    assert_eq!(result.f64max, 2.0 as f64);

    unsafe { free_value_and_module(ret_value) };
}

/// A wrapper struct to allow passing pointers across threads (they aren't Send/Sync by default).
/// The default #[derive(Copy,Clone)] does not work here unless T has Copy/Clone, so we also
/// implement those traits manually.
struct UnsafePtr<T>(*mut T);
unsafe impl<T> Send for UnsafePtr<T> {}
unsafe impl<T> Sync for UnsafePtr<T> {}
impl<T> Clone for UnsafePtr<T> {
    fn clone(&self) -> UnsafePtr<T> {
        UnsafePtr(self.0)
    }
}
impl<T> Copy for UnsafePtr<T> {}

fn multithreaded_module_run() {
    let code = "|v:vec[i32]| result(for(v, appender[i32], |b,i,e| merge(b,e)))";
    let conf = UnsafePtr(default_conf());

    // Set up input data
    let len: usize = 10 * 1000 * 1000 + 1;
    let mut input_vec = vec![];
    for i in 0..len {
        input_vec.push(i as i32);
    }
    let input_data = WeldVec {
        data: input_vec.as_ptr(),
        len: input_vec.len() as i64,
    };

    unsafe {
        // Compile the module
        let code = CString::new(code).unwrap();
        let input_value = UnsafePtr(weld_value_new(&input_data as *const _ as *const c_void));
        let err = weld_error_new();
        let module = UnsafePtr(weld_module_compile(code.into_raw() as *const c_char, conf.0, err));
        assert_eq!(weld_error_code(err), WeldRuntimeErrno::Success);

        // Run several threads, each of which executes the module several times
        let mut threads = vec![];
        let num_threads = 8;
        let num_runs = 4;
        for _ in 0..num_threads {
            threads.push(thread::spawn(move || {
                for _ in 0..num_runs {
                    // Run the module
                    let err = weld_error_new();
                    let ret_value = weld_module_run(module.0, conf.0, input_value.0, err);
                    assert_eq!(weld_error_code(err), WeldRuntimeErrno::Success);

                    // Check the result
                    let ret_data = weld_value_data(ret_value) as *const WeldVec<i32>;
                    let result = (*ret_data).clone();
                    assert_eq!(result.len, len as i64);
                    for i in 0..len {
                        assert_eq!(i as i32, *result.data.offset(i as isize));
                    }
                    weld_value_free(ret_value);
                }
            }))
        }

        // Wait for everything to finish, and then clean up
        for t in threads {
            t.join().unwrap();
        }

        weld_module_free(module.0);
    }
}

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

fn check_result_and_free(ret_value: *mut WeldValue, expected: i32) {
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };
    assert_eq!(result, expected);
    unsafe { free_value_and_module(ret_value) };
}

/* Only tests for compile success -- predication test in llvm.rs */
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

fn simple_sort() {
    #[derive(Clone)]
    #[allow(dead_code)]

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

fn nested_appender_loop() {
    let size = 100;
    let r0 = vec![0; size as usize];
    let r1 = vec![1; size as usize];
    let r2 = vec![2; size as usize];
    let r3 = vec![3; size as usize];
    let r4 = vec![4; size as usize];

    // Wrap the arrays in WeldVecs.
    let wv0 = WeldVec::new(r0.as_ptr() as *const i32, r0.len() as i64);
    let wv1 = WeldVec::new(r1.as_ptr() as *const i32, r1.len() as i64);
    let wv2 = WeldVec::new(r2.as_ptr() as *const i32, r2.len() as i64);
    let wv3 = WeldVec::new(r3.as_ptr() as *const i32, r3.len() as i64);
    let wv4 = WeldVec::new(r4.as_ptr() as *const i32, r4.len() as i64);

    let input_data = [wv0, wv1, wv2, wv3, wv4];
    let ref arg = WeldVec::new(input_data.as_ptr() as *const WeldVec<WeldVec<i32>>, input_data.len() as i64);

    let expect = [r0, r1, r2, r3, r4];

    // Computes the identity.
    let code = "|e0: vec[vec[i32]]| map(e0, |x:vec[i32]| map(x, |y:i32| y))";
    let conf = default_conf();

    let ret_value = compile_and_run(code, conf, arg);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<WeldVec<i32>> };
    let result = unsafe { (*data).clone() };

    // Make sure we get the same thing back.
    assert_eq!(result.len, 5);
    for i in 0..(result.len as isize) {
        let inner = unsafe { result.data.offset(i) };
        let inner_length = unsafe { (*inner).len };
        assert_eq!(inner_length, size);
        for j in 0..(inner_length as isize) {
            assert_eq!(unsafe { *((*inner).data.offset(i)) }, expect[i as usize][j as usize]);
        }
    }
}

fn nested_for_loops() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Row {
        x: i64,
        y: i32,
    }

    let code = "|ys:vec[i64]|result(for(ys, appender[{i64, i32}], |b0, i0, y0| for(ys, b0, |b1, i1, y1| if (y1 > y0, merge(b0, {y0, i32(y1)}), b0))))";
    let conf = default_conf();

    // Input data.
    let ys = vec![1i64, 3i64, 4i64];
    let ref input_data = WeldVec {
        data: ys.as_ptr() as *const i64,
        len: ys.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<Row> };
    let result = unsafe { (*data).clone() };

    assert_eq!(result.len, 3i64);
    let row = unsafe { (*result.data.offset(0)).clone() };
    assert_eq!(row.x, 1i64);
    assert_eq!(row.y, 3);
    let row = unsafe { (*result.data.offset(1)).clone() };
    assert_eq!(row.x, 1i64);
    assert_eq!(row.y, 4);
    let row = unsafe { (*result.data.offset(2)).clone() };
    assert_eq!(row.x, 3i64);
    assert_eq!(row.y, 4);

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

fn appender_and_dictmerger_loop() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Pair {
        ele1: i32,
        ele2: i32,
    }

    #[derive(Clone)]
    #[allow(dead_code)]
    struct Output {
        append_out: WeldVec<i32>,
        dict_out: WeldVec<Pair>
    }

    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]| let rs = for(zip(x,y),{appender[i32],dictmerger[i32,i32,+]},
                |bs,i,e| {merge(bs.$0, e.$0+e.$1), merge(bs.$1, e)}); {result(rs.$0), tovec(result(rs.$1))}";
    let conf = default_conf();
    let keys = [1, 2, 2, 1, 3];
    let vals = [2, 3, 4, 2, 1];
    let ref input_data = Args {
        x: WeldVec {
            data: &keys as *const i32,
            len: keys.len() as i64,
        },
        y: WeldVec {
            data: &vals as *const i32,
            len: vals.len() as i64,
        },
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const Output };
    let result = unsafe { (*data).clone() };

    let output_appender = [3, 5, 6, 3, 4];
    let output_dict_keys = [1, 2, 3];
    let output_dict_vals = [4, 7, 1];

    assert_eq!(result.append_out.len, output_appender.len() as i64);
    for i in 0..(output_appender.len() as isize) {
        assert_eq!(unsafe { *result.append_out.data.offset(i) }, output_appender[i as usize]);
    }

    assert_eq!(result.dict_out.len, output_dict_keys.len() as i64);
    for i in 0..(output_dict_keys.len() as isize) {
        let mut success = false;
        let key = unsafe { (*result.dict_out.data.offset(i)).ele1 };
        let value = unsafe { (*result.dict_out.data.offset(i)).ele2 };
        for j in 0..(output_dict_keys.len()) {
            if output_dict_keys[j] == key {
                if output_dict_vals[j] == value {
                    success = true;
                }
            }
        }
        assert_eq!(success, true);
    }
    unsafe { free_value_and_module(ret_value) };
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let tests: Vec<(&str, fn())> =
        vec![("basic_program", basic_program),
             ("basic_string", basic_string),
             ("float_literals", float_literals),
             ("negation", negation),
             ("negation_double", negation_double),
             ("negated_arithmetic", negated_arithmetic),
             ("bool_eq", bool_eq),
             //("c_udf", c_udf),
             ("f64_cast", f64_cast),
             ("i32_cast", i32_cast),
             ("multiple_casts", multiple_casts),
             ("program_with_args", program_with_args),
             ("struct_vector_literals", struct_vector_literals),
             ("let_statement", let_statement),
             ("if_statement", if_statement),
             ("float_min", float_min),
             ("maxmin", maxmin),
             ("comparison", comparison),
             ("reused_variable", reused_variable),
             ("map_comparison", map_comparison),
             ("eq_between_vectors", eq_between_vectors),
             ("eq_between_diff_length_vectors", eq_between_diff_length_vectors),
             ("ne_between_vectors", ne_between_vectors),
             ("lt_between_vectors", lt_between_vectors),
             ("le_between_vectors", le_between_vectors),
             ("le_between_unsigned_vectors", le_between_unsigned_vectors),
             ("eq_between_u8_vectors", eq_between_u8_vectors),
             ("eq_between_different_length_u8_vectors", eq_between_different_length_u8_vectors),
             ("le_between_u8_vectors", le_between_u8_vectors),
             ("simple_vector_lookup", simple_vector_lookup),
             ("simple_vector_slice", simple_vector_slice),
             ("simple_log", simple_log),
             ("log_error", log_error),
             ("simple_exp", simple_exp),
             ("exp_error", exp_error),
             ("simple_erf", simple_erf),
             ("simple_sqrt", simple_sqrt),
             ("simple_pow", simple_pow),
             ("simple_trig", simple_trig),
             ("empty_appender_loop", empty_appender_loop),
             ("map_exp", map_exp),
             ("nested_if_statement_loop", nested_if_statement_loop),
             ("nested_if_statement_with_builders_loop", nested_if_statement_with_builders_loop),
             ("simple_for_appender_loop", simple_for_appender_loop),
             ("large_unaryop_for_appender_loop", large_unaryop_for_appender_loop),
             ("simple_parallel_for_appender_loop", simple_parallel_for_appender_loop),
             ("simple_parallel_for_multi_appender_loop", simple_parallel_for_multi_appender_loop),
             ("complex_parallel_for_appender_loop", complex_parallel_for_appender_loop),
             ("range_iter_1", range_iter_1),
             ("range_iter_2", range_iter_2),
             ("range_iter_parallel", range_iter_parallel),
             ("simple_for_vectorizable_loop", simple_for_vectorizable_loop),
             ("fringed_for_vectorizable_loop", fringed_for_vectorizable_loop),
             ("fringed_for_vectorizable_loop_with_par", fringed_for_vectorizable_loop_with_par),
             ("for_predicated_vectorizable_loop", for_predicated_vectorizable_loop),
             ("simple_for_merger_loop", simple_for_merger_loop),
             ("simple_zipped_for_merger_loop", simple_zipped_for_merger_loop),
             ("parallel_for_merger_loop", parallel_for_merger_loop),
             ("parallel_for_multi_merger_loop", parallel_for_multi_merger_loop),
             ("simple_for_merger_loop_initial_value", simple_for_merger_loop_initial_value),
             ("parallel_for_merger_loop_initial_value", parallel_for_merger_loop_initial_value),
             ("parallel_for_merger_loop_initial_value_product",
              parallel_for_merger_loop_initial_value_product),
             ("simple_for_merger_loop_product", simple_for_merger_loop_product),
             ("simple_for_vecmerger_loop", simple_for_vecmerger_loop),
             ("simple_for_vecmerger_binops", simple_for_vecmerger_binops),
             ("parallel_for_vecmerger_loop", parallel_for_vecmerger_loop),
             ("simple_for_dictmerger_loop", simple_for_dictmerger_loop),
             ("dictmerger_with_structs", dictmerger_with_structs),
             ("simple_groupmerger", simple_groupmerger),
             ("complex_groupmerger_with_struct_key", complex_groupmerger_with_struct_key),
             ("simple_parallel_for_dictmerger_loop_local", simple_parallel_for_dictmerger_loop_local),
             ("simple_parallel_for_dictmerger_loop_global", simple_parallel_for_dictmerger_loop_global),
             ("simple_dict_lookup", simple_dict_lookup),
             ("string_dict_lookup", string_dict_lookup),
             ("simple_dict_exists", simple_dict_exists),
             ("simple_length", simple_length),
             ("filter_length", filter_length),
             ("flat_map_length", flat_map_length),
             ("if_for_loop", if_for_loop),
             ("map_zip_loop", map_zip_loop),
             ("iters_for_loop", iters_for_loop),
             ("iterate_non_parallel", iterate_non_parallel),
             ("iterate_with_parallel_body", iterate_with_parallel_body),
             ("serial_parlib_test", serial_parlib_test),
             ("many_mergers_test", many_mergers_test),
             ("maxmin_mergers_test", maxmin_mergers_test),
             ("multithreaded_module_run", multithreaded_module_run),
             ("iters_outofbounds_error_test", iters_outofbounds_error_test),
             ("outofmemory_error_test", outofmemory_error_test),
             ("simple_float_mod", simple_float_mod),
             ("simple_int_mod", simple_int_mod),
             ("predicate_if_iff_annotated", predicate_if_iff_annotated),
             ("nested_for_loops", nested_for_loops),
             ("nditer_basic_op_test", nditer_basic_op_test),
             ("nditer_zip", nditer_zip),
             ("nested_appender_loop", nested_appender_loop),
             ("simple_sort", simple_sort),
             ("complex_sort", complex_sort),
             ("serialize_test", serialize_test),
             ("appender_and_dictmerger_loop", appender_and_dictmerger_loop)];

    println!("");
    println!("running tests");
    let mut passed = 0;
    for t in tests.iter() {
        if args.len() > 1 {
            if !(t.0).contains(&args[1]) {
                println!("{} ... \x1b[0;33mignored\x1b[0m", t.0);
                continue;
            }
        }
        print!("{} ... ", t.0);
        t.1();
        println!("\x1b[0;32mok\x1b[0m");
        passed += 1;
    }

    println!("");
    println!("test result: \x1b[0;32mok\x1b[0m. {} passed; 0 failed; {} ignored; 0 measured",
             passed,
             tests.len() - passed);
    println!("");
}
