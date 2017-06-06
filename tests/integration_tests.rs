use std::env;

extern crate weld;
extern crate weld_common;
extern crate libc;

use weld_common::WeldRuntimeErrno;

use weld::WeldConf;
use weld::WeldValue;
use weld::WeldError;
use weld::{weld_value_new, weld_value_data, weld_value_free};
use weld::{weld_module_compile, weld_module_run, weld_module_free};
use weld::{weld_error_new, weld_error_code, weld_error_message, weld_error_free};
use weld::{weld_conf_new, weld_conf_set, weld_conf_free};

use std::ffi::{CStr, CString};
use libc::{c_char, c_void};

/// An in memory representation of a Weld vector.
#[derive(Clone)]
#[allow(dead_code)]
#[repr(C)]
struct WeldVec<T> {
    data: *const T,
    len: i64,
}

#[derive(Clone)]
#[allow(dead_code)]
#[repr(C)]
struct Pair<K, V> {
    ele1: K,
    ele2: V,
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
        weld_conf_free(conf);
        return Err(err);
    }
    weld_error_free(err);

    let err = weld_error_new();
    let ret_value = weld_module_run(module, conf, input_value, err);
    if weld_error_code(err) != WeldRuntimeErrno::Success {
        weld_conf_free(conf);
        return Err(err);
    }

    weld_module_free(module);
    weld_error_free(err);
    weld_value_free(input_value);
    weld_conf_free(conf);

    return Ok(ret_value);
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

    unsafe { weld_value_free(ret_value) };
}

fn negation() {
    let code = "|| -1";
    let conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };
    assert_eq!(result, -1 as i32);

    unsafe { weld_value_free(ret_value) };
}

fn negation_double() {
    let code = "|| -1.0";
    let conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const f64 };
    let result = unsafe { *data };
    assert_eq!(result, -1.0 as f64);

    unsafe { weld_value_free(ret_value) };
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

    unsafe { weld_value_free(ret_value) };
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
// unsafe { weld_value_free(ret_value) };
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

    unsafe { weld_value_free(ret_value) };
}

fn i32_cast() {
    let code = "|| i32(0.251 * 4.0)";
    let conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };
    assert_eq!(result, 1);

    unsafe { weld_value_free(ret_value) };
}

fn program_with_args() {
    let code = "|x:i32| 40 + x";
    let conf = default_conf();

    let ref input_data: i32 = 2;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };
    assert_eq!(result, 42);

    unsafe { weld_value_free(ret_value) };
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

    unsafe { weld_value_free(ret_value) };
}

fn let_statement() {
    let code = "|x:i32| let y = 40 + x; y + 2";
    let conf = default_conf();

    let ref input_data: i32 = 2;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };
    assert_eq!(result, 44);

    unsafe { weld_value_free(ret_value) };
}

fn if_statement() {
    let code = "|| if(true, 3, 4)";
    let conf = default_conf();

    let ref input_data: i32 = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };
    assert_eq!(result, 3);

    unsafe { weld_value_free(ret_value) };
}

fn comparison() {
    let code = "|x:i32| if(x>10, x, 10)";
    let conf = default_conf();

    let ref mut input_data: i32 = 2;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };
    assert_eq!(result, 10);

    unsafe { weld_value_free(ret_value) };

    let conf = default_conf();
    *input_data = 20;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };
    assert_eq!(result, 20);

    unsafe { weld_value_free(ret_value) };
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

    unsafe { weld_value_free(ret_value) };
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
    unsafe { weld_value_free(ret_value) };
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
    unsafe { weld_value_free(ret_value) };
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
    unsafe { weld_value_free(ret_value) };
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
    unsafe { weld_value_free(ret_value) };
}

fn le_between_vectors() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }
    let conf = default_conf();

    let code = "|e0: vec[i32], e1: vec[i32]| e0 <= e1";
    let input_vec1 = [1, 2, 3, 4, 5];
    let input_vec2 = [0, 3, 4, 5, 6];
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
    unsafe { weld_value_free(ret_value) };
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

    unsafe { weld_value_free(ret_value) };
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

    unsafe { weld_value_free(ret_value) };

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

    unsafe { weld_value_free(ret_value) };
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

    unsafe { weld_value_free(ret_value) };
}

fn simple_parallel_for_appender_loop() {
    let code = "|x:vec[i32]| result(for(x, appender[i64], |b,i,e| merge(b, i)))";
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
    unsafe { weld_value_free(ret_value) };
}

fn complex_parallel_for_appender_loop() {
    let code = "|x:vec[i32]| let a=appender[i64]; let b=merge(a,0L); let r=for(x,b,|b,i,e| \
                let c=merge(b,1L); let d=for(x,c,|b,i,e| if(i<1L, merge(b,i), b)); merge(d, 2L)); \
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

    unsafe { weld_value_free(ret_value) };
}

fn simple_for_merger_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        a: i32,
    }

    let code = "|x:vec[i32], a:i32| result(for(x, merger[i32,+], |b,i,e| merge(b, e+a)))";
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
    let output = 20;
    assert_eq!(result, output);
    unsafe { weld_value_free(ret_value) };
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
    unsafe { weld_value_free(ret_value) };
}

fn parallel_for_merger_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        a: i32,
    }

    let code = "|x:vec[i32], a:i32| result(for(x, merger[i32,+], |b,i,e| merge(b, e+a)))";
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
    unsafe { weld_value_free(ret_value) };
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
    unsafe { weld_value_free(ret_value) };
}

fn parallel_for_merger_loop_initial_value() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        a: i32,
    }

    let code = "|x:vec[i32], a:i32| result(for(x, merger[i32,+](1000), |b,i,e| merge(b, e+a)))";
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
    unsafe { weld_value_free(ret_value) };
}

fn parallel_for_merger_loop_initial_value_product() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
    }

    let code = "|x:vec[i32]| result(for(x, merger[i32,*](1000), |b,i,e| merge(b, e)))";
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
    unsafe { weld_value_free(ret_value) };
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
    unsafe { weld_value_free(ret_value) };
}

fn simple_for_vecmerger_loop_2() {
    let code = "|x:vec[i32]| result(for(x, vecmerger[i32,+](x), |b,i,e| merge(b, {i,e*7})))";
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
        assert_eq!(unsafe { *result.data.offset(i) },
                   input_vec[i as usize] + input_vec[i as usize] * 7);
    }
    unsafe { weld_value_free(ret_value) };
}

fn parallel_for_vecmerger_loop() {
    let code = "|x:vec[i32]| result(for(x, vecmerger[i32,+](x), |b,i,e| merge(b, {i,e*7})))";
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

    unsafe { weld_value_free(ret_value) };
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

    let code = "|x:vec[i32], y:vec[i32]| tovec(result(for(zip(x,y), dictmerger[i32,i32,+], \
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
    unsafe { weld_value_free(ret_value) };
}

fn simple_groupmerger() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]| tovec(result(for(zip(x,y), groupmerger[i32,i32], \
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
    unsafe { weld_value_free(ret_value) };
}

fn complex_groupmerger_with_struct_key() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
        z: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32], z:vec[i32]| \
                tovec(result(for(zip(x,y,z), groupmerger[{i32,i32}, i32], \
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
    unsafe { weld_value_free(ret_value) };
}

fn simple_parallel_for_dictmerger_loop() {
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

    let code = "|x:vec[i32], y:vec[i32]| tovec(result(for(zip(x,y), dictmerger[i32,i32,+], \
                |b,i,e| merge(b, e))))";
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

    let ret_value = compile_and_run(code, conf, input_data);
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
    unsafe { weld_value_free(ret_value) };
}

fn simple_dict_lookup() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]| let a = result(for(zip(x,y), dictmerger[i32,i32,+], \
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
    unsafe { weld_value_free(ret_value) };
}

fn simple_dict_exists() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let keys = [1, 2, 2, 1, 3];
    let vals = [2, 3, 4, 2, 1];

    let code_true = "|x:vec[i32], y:vec[i32]| let a = result(for(zip(x,y), dictmerger[i32,i32,+], \
                |b,i,e| merge(b, e))); keyexists(a, 1)";
    let code_false = "|x:vec[i32], y:vec[i32]| let a = result(for(zip(x,y), \
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

    let conf2 = default_conf();
    let ret_value2 = compile_and_run(code_false, conf2, input_data.clone());
    let data = unsafe { weld_value_data(ret_value2) as *const bool };
    let result = unsafe { (*data).clone() };

    let output = false;
    assert_eq!(output, result);
    unsafe { weld_value_free(ret_value) };
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
    unsafe { weld_value_free(ret_value) };
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
    unsafe { weld_value_free(ret_value) };
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
    unsafe { weld_value_free(ret_value) };
}

fn simple_exp() {
    let code = "|x:f64| exp(x)";
    let conf = default_conf();
    let input = 1.0f64;
    let ret_value = compile_and_run(code, conf, &input);
    let data = unsafe { weld_value_data(ret_value) as *const f64 };
    let result = unsafe { (*data).clone() };

    let output = 2.718281828459045;
    assert_eq!(output, result);
    unsafe { weld_value_free(ret_value) };
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

    unsafe { weld_value_free(ret_value) };
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
    unsafe { weld_value_free(ret_value) };
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

    unsafe { weld_value_free(ret_value) };
}

fn iters_for_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]| result(for(zip(iter(x,0L,4L,2L), y), appender, |b,i,e| \
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
    unsafe { weld_value_free(ret_value) };
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

fn main() {
    let args: Vec<String> = env::args().collect();
    let tests: Vec<(&str, fn())> =
        vec![("basic_program", basic_program),
             ("negation", negation),
             ("negation_double", negation_double),
             ("negated_arithmetic", negated_arithmetic),
             //("c_udf", c_udf),
             ("f64_cast", f64_cast),
             ("i32_cast", i32_cast),
             ("program_with_args", program_with_args),
             ("struct_vector_literals", struct_vector_literals),
             ("let_statement", let_statement),
             ("if_statement", if_statement),
             ("comparison", comparison),
             ("map_comparison", map_comparison),
             ("eq_between_vectors", eq_between_vectors),
             ("eq_between_diff_length_vectors", eq_between_diff_length_vectors),
             ("ne_between_vectors", ne_between_vectors),
             ("lt_between_vectors", lt_between_vectors),
             ("le_between_vectors", le_between_vectors),
             ("simple_vector_lookup", simple_vector_lookup),
             ("simple_vector_slice", simple_vector_slice),
             ("simple_exp", simple_exp),
             ("exp_error", exp_error),
             ("map_exp", map_exp),
             ("simple_for_appender_loop", simple_for_appender_loop),
             ("simple_parallel_for_appender_loop", simple_parallel_for_appender_loop),
             ("complex_parallel_for_appender_loop", complex_parallel_for_appender_loop),
             ("simple_for_merger_loop", simple_for_merger_loop),
             ("parallel_for_merger_loop", parallel_for_merger_loop),
             ("simple_for_merger_loop_initial_value", simple_for_merger_loop_initial_value),
             ("parallel_for_merger_loop_initial_value", parallel_for_merger_loop_initial_value),
             ("parallel_for_merger_loop_initial_value_product",
              parallel_for_merger_loop_initial_value_product),
             ("simple_for_merger_loop_product", simple_for_merger_loop_product),
             ("simple_for_vecmerger_loop", simple_for_vecmerger_loop),
             ("simple_for_vecmerger_loop_2", simple_for_vecmerger_loop_2),
             ("parallel_for_vecmerger_loop", parallel_for_vecmerger_loop),
             ("simple_for_dictmerger_loop", simple_for_dictmerger_loop),
             ("simple_groupmerger", simple_groupmerger),
             ("complex_groupmerger_with_struct_key", complex_groupmerger_with_struct_key),
             ("simple_parallel_for_dictmerger_loop", simple_parallel_for_dictmerger_loop),
             ("simple_dict_lookup", simple_dict_lookup),
             ("simple_dict_exists", simple_dict_exists),
             ("simple_length", simple_length),
             ("filter_length", filter_length),
             ("flat_map_length", flat_map_length),
             ("if_for_loop", if_for_loop),
             ("map_zip_loop", map_zip_loop),
             ("iters_for_loop", iters_for_loop),
             ("serial_parlib_test", serial_parlib_test),
             ("iters_outofbounds_error_test", iters_outofbounds_error_test),
             ("outofmemory_error_test", outofmemory_error_test)];

    println!("");
    println!("running tests");
    let mut passed = 0;
    for t in tests.iter() {
        if args.len() > 1 {
            if !t.0.contains(args[1].as_str()) {
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
