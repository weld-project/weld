//! Various tests for the different expressions in Weld.

extern crate weld;
use weld::weld_value_data;

use std::slice;
use std::str;

use std::f64::consts::PI;

mod common;
use common::*;

#[test]
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

#[test]
fn basic_string() {
    let code = "|| \"test str\"";
    let conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<u8> };
    let result = unsafe { (*data).clone() };
    assert_eq!(result.len, 8);
    unsafe {
        assert_eq!(
            str::from_utf8(slice::from_raw_parts(result.data, result.len as usize)).unwrap(),
            "test str"
        );
    }

    unsafe { free_value_and_module(ret_value) };
}

#[test]
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
        unsafe { free_value_and_module(ret_value) };

        // Try parsing the value as a float
        let code = format!("|| {:e}f", v);
        let conf = default_conf();
        let ref input_data = 0;
        let ret_value = compile_and_run(&code, conf, input_data);
        let data = unsafe { weld_value_data(ret_value) as *const f32 };
        let result = unsafe { *data };
        assert_eq!(result, v as f32);
        unsafe { free_value_and_module(ret_value) };
    }
}

#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
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
#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
fn simple_length() {
    let code = "|x:vec[i32]| len(x)";
    let conf = default_conf();

    let input_vec = [2, 3, 4, 2, 1];
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };

    let output = 5;
    assert_eq!(output, result);
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn filter_length() {
    let code = "|x:vec[i32]| len(filter(x, |i| i < 4 && i > 1))";
    let conf = default_conf();

    let input_vec = [2, 3, 4, 2, 1];
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };

    let output = 3;
    assert_eq!(output, result);
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn flat_map_length() {
    let code = "|x:vec[i32]| len(flatten(map(x, |i:i32| x)))";
    let conf = default_conf();

    let input_vec = [2, 3, 4, 2, 1];
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };

    let output = 25;
    assert_eq!(output, result);
    unsafe { free_value_and_module(ret_value) };
}

#[test]
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
        x: WeldVec::from(&input_vec),
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

#[test]
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
        x: WeldVec::from(&x),
        y: WeldVec::from(&y),
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

#[test]
fn iterate_non_parallel() {
    let code = "|x:i32| iterate(x, |x| {x-1, x-1>0})";
    let conf = default_conf();

    let input: i32 = 5;

    let ret_value = compile_and_run(code, conf, &input);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };

    assert_eq!(result, 0);

    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn iterate_with_parallel_body() {
    let code =
        "|x:i32| let a=2; iterate({[1,2,3], 1}, |p| {{map(p.$0, |y|y*a), p.$1+1}, p.$1<x}).$0";
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

    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn serial_parlib_test() {
    let code = "|x:vec[i32]| result(for(x, merger[i32,+], |b,i,e| merge(b, e)))";
    let conf = default_conf();

    let size: i32 = 10000;
    let input_vec: Vec<i32> = vec![1; size as usize];
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };

    assert_eq!(result, size as i32);
    unsafe { free_value_and_module(ret_value) };
}
