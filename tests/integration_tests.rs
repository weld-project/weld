use std::env;

extern crate weld;
extern crate libc;

use weld::llvm::*;
use weld::parser::*;
use weld::weld_print_function_pointers;
use weld::weld_run_free;
use weld::weld_rt_get_errno;
use weld::WeldRuntimeErrno;

use weld::WeldError;
use weld::WeldValue;
use weld::{weld_value_new, weld_value_data, weld_value_free};
use weld::{weld_module_compile, weld_module_run, weld_module_free};
use weld::{weld_error_code, weld_error_message, weld_error_free};

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

/// Takes a string of Weld code and a `void *` pointer to data, and compile and run the code.
/// Panics if an error is thrown. Returns a `WeldValue` that must be freed.
fn compile_and_run<T>(code: &str, conf: &str, ptr: &T) -> *mut WeldValue {
    let code = CString::new(code).unwrap();
    let conf = CString::new(conf).unwrap();

    let input_value = weld_value_new(ptr as *const _ as *const c_void);

    let mut err = std::ptr::null_mut();
    let module = weld_module_compile(code.into_raw() as *const c_char,
                                     conf.into_raw() as *const c_char,
                                     &mut err as *mut *mut WeldError);

    if weld_error_code(err) != WeldRuntimeErrno::Success {
        panic!(format!("Compile failed: {:?}",
                       unsafe { CStr::from_ptr(weld_error_message(err)) }));
    }
    weld_error_free(err);

    let mut err = std::ptr::null_mut();
    let ret_value = weld_module_run(module, input_value, &mut err as *mut *mut WeldError);
    if weld_error_code(err) != WeldRuntimeErrno::Success {
        panic!(format!("Run failed: {:?}",
                       unsafe { CStr::from_ptr(weld_error_message(err)) }));
    }

    weld_module_free(module);
    weld_error_free(err);
    weld_value_free(input_value);

    return ret_value;
}

fn basic_program() {
    let code = "|| 40 + 2";
    let conf = "";

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = weld_value_data(ret_value) as *const i32;
    let result = unsafe { *data };
    assert_eq!(result, 42);

    weld_value_free(ret_value);
}

fn f64_cast() {
    let code = "|| f64(40 + 2)";
    let conf = "";

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = weld_value_data(ret_value) as *const f64;
    let result = unsafe { *data };
    assert_eq!(result, 42.0);

    weld_value_free(ret_value);
}

fn i32_cast() {
    let code = "|| i32(0.251 * 4.0)";
    let conf = "";

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = weld_value_data(ret_value) as *const i32;
    let result = unsafe { *data };
    assert_eq!(result, 1);

    weld_value_free(ret_value);
}

fn program_with_args() {
    let code = "|x:i32| 40 + x";
    let conf = "";

    let ref input_data: i32 = 2;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = weld_value_data(ret_value) as *const i32;
    let result = unsafe { *data };
    assert_eq!(result, 42);

    weld_value_free(ret_value);
}

fn let_statement() {
    let code = "|x:i32| let y = 40 + x; y + 2";
    let conf = "";

    let ref input_data: i32 = 2;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = weld_value_data(ret_value) as *const i32;
    let result = unsafe { *data };
    assert_eq!(result, 44);

    weld_value_free(ret_value);
}

fn if_statement() {
    let code = "|| if(true, 3, 4)";
    let conf = "";

    let ref input_data: i32 = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = weld_value_data(ret_value) as *const i32;
    let result = unsafe { *data };
    assert_eq!(result, 3);

    weld_value_free(ret_value);
}

fn comparison() {
    let code = "|x:i32| if(x>10, x, 10)";
    let conf = "";

    let ref mut input_data: i32 = 2;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = weld_value_data(ret_value) as *const i32;
    let result = unsafe { *data };
    assert_eq!(result, 10);

    weld_value_free(ret_value);

    *input_data = 20;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = weld_value_data(ret_value) as *const i32;
    let result = unsafe { *data };
    assert_eq!(result, 20);

    weld_value_free(ret_value);
}

fn simple_vector_lookup() {
    let code = "|x:vec[i32]| lookup(x, 3L)";
    let conf = "";

    let input_vec = [1, 2, 3, 4, 5];
    let ref input_data = WeldVec {
        data: &input_vec as *const i32,
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = weld_value_data(ret_value) as *const i32;
    let result = unsafe { *data };
    assert_eq!(result, input_vec[3]);

    weld_value_free(ret_value);
}

fn simple_for_appender_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        a: i32,
    }

    let code = "|x:vec[i32], a:i32| let b=a+1; map(x, |e| e+b)";
    let conf = "";

    let input_vec = [1, 2];
    let ref input_data = Args {
        x: WeldVec {
            data: &input_vec as *const i32,
            len: input_vec.len() as i64,
        },
        a: 1,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = weld_value_data(ret_value) as *const WeldVec<i32>;
    let result = unsafe { (*data).clone() };
    let output = [3, 4];
    assert_eq!(result.len as usize, output.len());
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
    }

    weld_value_free(ret_value);
}

// TODO(shoumik): Configuration parsing required for this.
fn simple_parallel_for_appender_loop() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct WeldVec {
        data: *const i32,
        len: i64,
    }
    #[derive(Clone)]
    #[allow(dead_code)]
    struct WeldVec64 {
        data: *const i64,
        len: i64,
    }
    let code = "|x:vec[i32]| result(for(x, appender[i64], |b,i,e| merge(b, i)))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let size: i32 = 10000;
    let input: Vec<i32> = vec![0; size as usize];
    let args = WeldVec {
        data: input.as_ptr() as *const i32,
        len: size as i64,
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const WeldVec as i64,
        nworkers: 4,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const WeldVec64;
    let result = unsafe { (*result_raw).clone() };
    assert_eq!(result.len as i32, size);
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, i as i64)
    }
    weld_run_free(-1);
}

// TODO(shoumik): Configuration parsing required for this.
fn complex_parallel_for_appender_loop() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct WeldVec {
        data: *const i32,
        len: i64,
    }
    #[derive(Clone)]
    #[allow(dead_code)]
    struct WeldVec64 {
        data: *const i64,
        len: i64,
    }
    let code = "|x:vec[i32]| let a=appender[i64]; let b=merge(a,0L); let r=for(x,b,|b,i,e| \
                let c=merge(b,1L); let d=for(x,c,|b,i,e| if(i<1L, merge(b,i), b)); merge(d, 2L)); \
                result(merge(r,3L))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let size: i32 = 3000;
    let input: Vec<i32> = vec![0; size as usize];
    let args = WeldVec {
        data: input.as_ptr() as *const i32,
        len: size as i64,
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const WeldVec as i64,
        nworkers: 4,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const WeldVec64;
    let result = unsafe { (*result_raw).clone() };
    assert_eq!(result.len as i32, size * 3 + 2);
    assert_eq!(unsafe { *result.data.offset(0) }, 0);
    assert_eq!(unsafe { *result.data.offset((size * 3 + 1) as isize) }, 3);
    for i in 0..(size as isize) {
        assert_eq!(unsafe { *result.data.offset(i * 3 + 1) }, 1);
        assert_eq!(unsafe { *result.data.offset(i * 3 + 2) }, 0);
        assert_eq!(unsafe { *result.data.offset(i * 3 + 3) }, 2)
    }
    weld_run_free(-1);
}

fn simple_for_merger_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        a: i32,
    }

    let code = "|x:vec[i32], a:i32| result(for(x, merger[i32,+], |b,i,e| merge(b, e+a)))";
    let conf = "";

    let input_vec = [1, 2, 3, 4, 5];
    let ref input_data = Args {
        x: WeldVec {
            data: &input_vec as *const i32,
            len: input_vec.len() as i64,
        },
        a: 1,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = weld_value_data(ret_value) as *const i32;
    let result = unsafe { (*data).clone() };
    let output = 20;
    assert_eq!(result, output);
    weld_value_free(ret_value);
}

fn simple_for_vecmerger_loop() {
    let code = "|x:vec[i32]| result(for(x, vecmerger[i32,+](x), |b,i,e| b))";
    let conf = "";

    let input_vec = [1, 1, 1, 1, 1];
    let ref input_data = WeldVec {
        data: &input_vec as *const i32,
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = weld_value_data(ret_value) as *const WeldVec<i32>;
    let result = unsafe { (*data).clone() };
    assert_eq!(result.len, input_vec.len() as i64);
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, input_vec[i as usize]);
    }
    weld_value_free(ret_value);
}

fn simple_for_vecmerger_loop_2() {
    let code = "|x:vec[i32]| result(for(x, vecmerger[i32,+](x), |b,i,e| merge(b, {i,e*7})))";
    let conf = "";

    let input_vec = [1, 1, 1, 1, 1];
    let ref input_data = WeldVec {
        data: &input_vec as *const i32,
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = weld_value_data(ret_value) as *const WeldVec<i32>;
    let result = unsafe { (*data).clone() };
    assert_eq!(result.len, input_vec.len() as i64);
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) },
                   input_vec[i as usize] + input_vec[i as usize] * 7);
    }
    weld_value_free(ret_value);
}

// TODO(parallelize)
fn parallel_for_vecmerger_loop() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }

    let code = "|x:vec[i32]| result(for(x, vecmerger[i32,+](x), |b,i,e| merge(b, {i,e*7})))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let input = [1; 4096];
    let args = Vec {
        data: &input as *const i32,
        len: input.len() as i64,
    };
    let inp = Box::new(WeldInputArgs {
        input: &args as *const Vec as i64,
        nworkers: 4,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const Vec;
    let result = unsafe { (*result_raw).clone() };
    assert_eq!(result.len, input.len() as i64);
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) },
                   input[i as usize] + input[i as usize] * 7);
    }
    weld_run_free(-1);
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
    let conf = "";
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
    let data = weld_value_data(ret_value) as *const WeldVec<Pair>;
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
    weld_value_free(ret_value);
}

// TODO(parallel)
fn simple_parallel_for_dictmerger_loop() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }
    #[allow(dead_code)]
    struct Pair {
        ele1: i32,
        ele2: i32,
    }
    #[derive(Clone)]
    #[allow(dead_code)]
    struct VecPrime {
        data: *const Pair,
        len: i64,
    }
    #[allow(dead_code)]
    struct Args {
        x: Vec,
        y: Vec,
    }

    let code = "|x:vec[i32], y:vec[i32]| tovec(result(for(zip(x,y), dictmerger[i32,i32,+], \
                |b,i,e| merge(b, e))))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    const DICT_SIZE: usize = 8192;
    let mut keys = [0; DICT_SIZE];
    let mut values = [0; DICT_SIZE];
    for i in 0..DICT_SIZE {
        keys[i] = i as i32;
        values[i] = i as i32;
    }
    let args = Args {
        x: Vec {
            data: &keys as *const i32,
            len: DICT_SIZE as i64,
        },
        y: Vec {
            data: &values as *const i32,
            len: DICT_SIZE as i64,
        },
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const Args as i64,
        nworkers: 4,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const VecPrime;
    let result = unsafe { (*result_raw).clone() };
    let output_keys = keys;
    let output_values = values;
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
    weld_run_free(-1);
}

fn simple_dict_lookup() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]| let a = result(for(zip(x,y), dictmerger[i32,i32,+], \
                |b,i,e| merge(b, e))); lookup(a, 1)";
    let conf = "";

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
    let data = weld_value_data(ret_value) as *const i32;
    let result = unsafe { (*data).clone() };

    let output = 4;
    assert_eq!(output, result);
    weld_value_free(ret_value);
}

fn simple_length() {
    let code = "|x:vec[i32]| len(x)";
    let conf = "";

    let input_vec = [2, 3, 4, 2, 1];
    let ref input_data = WeldVec {
        data: &input_vec as *const i32,
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = weld_value_data(ret_value) as *const i32;
    let result = unsafe { (*data).clone() };

    let output = 5;
    assert_eq!(output, result);
    weld_value_free(ret_value);
}

fn filter_length() {
    let code = "|x:vec[i32]| len(filter(x, |i| i < 4 && i > 1))";
    let conf = "";

    let input_vec = [2, 3, 4, 2, 1];
    let ref input_data = WeldVec {
        data: &input_vec as *const i32,
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = weld_value_data(ret_value) as *const i32;
    let result = unsafe { (*data).clone() };

    let output = 3;
    assert_eq!(output, result);
    weld_value_free(ret_value);
}

fn flat_map_length() {
    let code = "|x:vec[i32]| len(flatten(map(x, |i:i32| x)))";
    let conf = "";

    let input_vec = [2, 3, 4, 2, 1];
    let ref input_data = WeldVec {
        data: &input_vec as *const i32,
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = weld_value_data(ret_value) as *const i32;
    let result = unsafe { (*data).clone() };

    let output = 25;
    assert_eq!(output, result);
    weld_value_free(ret_value);
}

fn if_for_loop() {
    let code = "|x:vec[i32], a:i32| if(a > 5, map(x, |e| e+1), map(x, |e| e+2))";
    let conf = "";

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
    let data = weld_value_data(ret_value) as *const WeldVec<i32>;
    let result = unsafe { (*data).clone() };

    let output = [3, 4];
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
    }
    weld_value_free(ret_value);
}

fn map_zip_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]| map(zip(x,y), |e| e.$0 + e.$1)";
    let conf = "";

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
    let data = weld_value_data(ret_value) as *const WeldVec<i32>;
    let result = unsafe { (*data).clone() };

    let output = [6, 8, 10, 12];
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
    }

    weld_value_free(ret_value);
}

fn iters_for_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]| result(for(zip(iter(x,0L,4L,2L), y), appender, |b,i,e| \
                merge(b,e.$0+e.$1)))";
    let conf = "";

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
    let data = weld_value_data(ret_value) as *const WeldVec<i32>;
    let result = unsafe { (*data).clone() };

    let output = [6, 9];
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
    }

    weld_value_free(ret_value);
}

fn serial_parlib_test() {
    let code = "|x:vec[i32]| result(for(x, merger[i32,+], |b,i,e| merge(b, e)))";
    let conf = "";

    let size: i32 = 10000;
    let input_vec: Vec<i32> = vec![1; size as usize];

    let ref input_data = WeldVec {
        data: input_vec.as_ptr() as *const i32,
        len: size as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = weld_value_data(ret_value) as *const i32;
    let result = unsafe { (*data).clone() };

    assert_eq!(result, size as i32);
    weld_value_free(ret_value);
}

fn iters_outofbounds_error_test() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }
    #[allow(dead_code)]
    struct Args {
        x: Vec,
    }

    let code = "|x:vec[i32]| result(for(iter(x,0L,20000L,1L), appender, |b,i,e| merge(b,e+1)))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let x = [4; 1000 as usize];
    let args = Args {
        x: Vec {
            data: &x as *const i32,
            len: x.len() as i64,
        },
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const Args as i64,
        nworkers: 4,
        run_id: 99, // this test needs a unique run ID so we don't reset accidentally.
    });
    let ptr = Box::into_raw(inp) as i64;
    module.run(ptr) as *const i32;

    // Get the error back for the run ID we used.
    let errno = weld_rt_get_errno(99);
    assert_eq!(errno, WeldRuntimeErrno::BadIteratorLength);
    weld_run_free(99);
}

fn outofmemory_error_test() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }
    #[allow(dead_code)]
    struct Args {
        x: Vec,
    }

    // TODO(shoumik): This test (and all the other tests) can be made more robust
    // by using the API directly, since that will test how users actually use Weld.
    // What we do here uses unstable APIs and could cause tests to break in the
    // future.
    //
    // This test tests the case where the default memory limit (1GB) is exceeded.
    let code = "|x:vec[i32]| result(for(x, vecmerger[i32,+](x), |b,i,e| merge(b,{i,e+1})))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    // 1GB of data; the appender will allocate at least this much,
    // exceeding the 1GB default limit.
    let x = vec![4; 1000000000 / 4 as usize];
    let args = Args {
        x: Vec {
            data: x.as_ptr() as *const i32,
            len: x.len() as i64,
        },
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const Args as i64,
        nworkers: 4,
        run_id: 999, // this test needs a unique run ID so we don't reset accidentally.
    });
    let ptr = Box::into_raw(inp) as i64;
    module.run(ptr) as *const Vec;

    // Get the error back for the run ID we used.
    let errno = weld_rt_get_errno(999);
    assert_eq!(errno, WeldRuntimeErrno::OutOfMemory);
    weld_run_free(999);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let tests: Vec<(&str, fn())> =
        vec![("basic_program", basic_program),
             ("f64_cast", f64_cast),
             ("i32_cast", i32_cast),
             ("program_with_args", program_with_args),
             ("let_statement", let_statement),
             ("if_statement", if_statement),
             ("comparison", comparison),
             ("simple_vector_lookup", simple_vector_lookup),
             ("simple_for_appender_loop", simple_for_appender_loop),
             ("simple_parallel_for_appender_loop", simple_parallel_for_appender_loop),
             ("complex_parallel_for_appender_loop", complex_parallel_for_appender_loop),
             ("simple_for_merger_loop", simple_for_merger_loop),
             ("simple_for_vecmerger_loop", simple_for_vecmerger_loop),
             ("simple_for_vecmerger_loop_2", simple_for_vecmerger_loop_2),
             ("parallel_for_vecmerger_loop", parallel_for_vecmerger_loop),
             ("simple_for_dictmerger_loop", simple_for_dictmerger_loop),
             ("simple_parallel_for_dictmerger_loop", simple_parallel_for_dictmerger_loop),
             ("simple_dict_lookup", simple_dict_lookup),
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
        match t.0 {
            // don't run this, they exist only to make sure functions don't get optimized out
            "runtime_fns" => weld_print_function_pointers(),
            _ => {
                if args.len() > 1 {
                    if !t.0.contains(args[1].as_str()) {
                        continue;
                    }
                }
                print!("{} ... ", t.0);
                t.1();
                println!("\x1b[0;32mok\x1b[0m");
                passed += 1;
            }
        }
    }

    println!("");
    println!("test result: \x1b[0;32mok\x1b[0m. {} passed; 0 failed; {} ignored; 0 measured",
             passed,
             tests.len() - passed);
    println!("");
}
