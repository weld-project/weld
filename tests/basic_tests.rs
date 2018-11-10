//! Various tests for the different expressions in Weld.

extern crate weld;

mod common;
use common::*;

#[test]
fn basic_program() {
    let code = "|| 40 + 2";
    let ref conf = default_conf();
    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { *data };
    assert_eq!(result, 42);
}

// #[test]
fn basic_string() {
    // XXX This test is segfaulting for some reason with a regular string...
    let code = r#"|| "hello""#;
    let ref conf = default_conf();
    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<i8>;
    let result = unsafe { (*data).clone() };
    // The string as a vector includes a terminating null byte.
    assert_eq!(result.len, 6);

    unsafe {
        use std::ffi::CStr;
        let cstr = CStr::from_ptr(result.data).to_str().unwrap();
        assert_eq!(cstr, "hello");
    }
}

#[test]
fn float_literals() {
    use std::f64::consts::PI;

    let values = vec![0.0, PI, -PI, 1.2e20, -1.2e-20];
    for v in values {
        // Try parsing the value as a double
        let code = format!("|| {:e}", v);
        let ref conf = default_conf();
        let ref input_data = 0;
        let ret_value = compile_and_run(&code, conf, input_data);
        let data = ret_value.data() as *const f64;
        let result = unsafe { *data };
        assert_eq!(result, v);


        // Try parsing the value as a float
        let code = format!("|| {:e}f", v);
        let ref conf = default_conf();
        let ref input_data = 0;
        let ret_value = compile_and_run(&code, conf, input_data);
        let data = ret_value.data() as *const f32;
        let result = unsafe { *data };
        assert_eq!(result, v as f32);

    }
}

#[test]
fn not() {
    let code = "|v:bool| !v";
    let ref conf = default_conf();

    let ref input_data: u8 = 0;
    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const u8;
    let result = unsafe { *data };
    assert_eq!(result, 1);

    let ref input_data: u8 = 1;
    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const u8;
    let result = unsafe { *data };
    assert_eq!(result, 0);
}

#[test]
fn negation() {
    let code = "|| -(1)";
    let ref conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { *data };
    assert_eq!(result, -1 as i32);
}

#[test]
fn negation_double() {
    let code = "|| -(1.0)";
    let ref conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const f64;
    let result = unsafe { *data };
    assert_eq!(result, -1.0 as f64);
}

#[test]
fn negated_arithmetic() {
    // In our language, - has the lowest precedence so the paraens around -3 are mandatory.
    let code = "|| 1+2*-3-4/-2";
    let ref conf = default_conf();
    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { *data };
    assert_eq!(result, -3 as i32);
}

#[test]
fn bool_eq() {
    let code = "|| [(2 < 3) != (2 > 2), true == false]";
    let ref conf = default_conf();

    let ref input_data: i32 = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<WeldBool>;
    let result = unsafe { (*data).clone() };

    assert_eq!(result.len, 2);

    let bool1 = unsafe { (*result.data.offset(0)).clone() };
    let bool2 = unsafe { (*result.data.offset(1)).clone() };
    assert_eq!(bool1 , 1);
    assert_eq!(bool2 , 0);
}

#[test]
fn f64_cast() {
    let code = "|| f64(40 + 2)";
    let ref conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const f64;
    let result = unsafe { *data };
    assert_eq!(result, 42.0);
}

#[test]
fn i32_cast() {
    let code = "|| i32(0.251 * 4.0)";
    let ref conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { *data };
    assert_eq!(result, 1);
}

#[test]
fn multiple_casts() {
    let code = "|| i16(i8(u8(i64(256+255))))";
    let ref conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i16;
    let result = unsafe { *data };
    assert_eq!(result, -1i16);

    let code = "|| u32(u64(u16(i16(u8(i64(-1))))))";
    let ref conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const u32;
    let result = unsafe { *data };
    assert_eq!(result, 255);
}

#[test]
fn program_with_args() {
    let code = "|x:i32| 40 + x";
    let ref conf = default_conf();

    let ref input_data: i32 = 2;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { *data };
    assert_eq!(result, 42);
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
    let ref conf = default_conf();

    let ref input_data: i32 = 2;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<Triple>;
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
}

#[test]
fn let_statement() {
    let code = "|x:i32| let y = 40 + x; y + 2";
    let ref conf = default_conf();

    let ref input_data: i32 = 2;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { *data };
    assert_eq!(result, 44);
}

#[test]
fn if_statement() {
    let code = "|| if(true, 3, 4)";
    let ref conf = default_conf();

    let ref input_data: i32 = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { *data };
    assert_eq!(result, 3);
}

#[test]
fn maxmin() {
    let code = "|| max(3, min(2, 4))";
    let ref conf = default_conf();

    let ref input_data: i32 = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { *data };
    assert_eq!(result, 3);
}

#[test]
fn comparison() {
    let code = "|x:i32| if(x>10, x, 10)";
    let ref conf = default_conf();

    let ref mut input_data: i32 = 2;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { *data };
    assert_eq!(result, 10);

    let ref conf = default_conf();
    *input_data = 20;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { *data };
    assert_eq!(result, 20);
}

#[test]
fn reused_variable() {
    // `a` is reused in different scopes
    let code = "|| let a=2; let b=map([1,2,3], |e| let a=1; e+a); lookup(b, 0L)+a";
    let ref conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { *data };
    assert_eq!(result, 4);
}

#[test]
fn simple_length() {
    let code = "|x:vec[i32]| len(x)";
    let ref conf = default_conf();

    let input_vec = vec![2, 3, 4, 2, 1];
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };

    let output = 5;
    assert_eq!(output, result);
}

#[test]
fn filter_length() {
    let code = "|x:vec[i32]| len(filter(x, |i| i < 4 && i > 1))";
    let ref conf = default_conf();

    let input_vec = vec![2, 3, 4, 2, 1];
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };

    let output = 3;
    assert_eq!(output, result);
}

#[test]
fn flat_map_length() {
    let code = "|x:vec[i32]| len(flatten(map(x, |i:i32| x)))";
    let ref conf = default_conf();

    let input_vec = vec![2, 3, 4, 2, 1];
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };

    let output = 25;
    assert_eq!(output, result);
}

#[test]
fn if_for_loop() {
    let code = "|x:vec[i32], a:i32| if(a > 5, map(x, |e| e+1), map(x, |e| e+2))";
    let ref conf = default_conf();

    let input_vec = vec![1, 2];

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
    let data = ret_value.data() as *const WeldVec<i32>;
    let result = unsafe { (*data).clone() };

    let output = vec![3, 4];
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
    }
}

#[test]
fn map_zip_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]| map(zip(x,y), |e| e.$0 + e.$1)";
    let ref conf = default_conf();

    let x = vec![1, 2, 3, 4];
    let y = vec![5, 6, 7, 8];
    let ref input_data = Args {
        x: WeldVec::from(&x),
        y: WeldVec::from(&y),
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<i32>;
    let result = unsafe { (*data).clone() };

    let output = vec![6, 8, 10, 12];
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
    }
}

#[test]
fn iterate_non_parallel() {
    let code = "|x:i32| iterate(x, |x| {x-1, x-1>0})";
    let ref conf = default_conf();

    let input: i32 = 5;

    let ret_value = compile_and_run(code, conf, &input);
    let data = ret_value.data() as *const i32;
    let result = unsafe { *data };

    assert_eq!(result, 0);
}

#[test]
fn iterate_with_parallel_body() {
    let code =
        "|x:i32| let a=2; iterate({[1,2,3], 1}, |p| {{map(p.$0, |y|y*a), p.$1+1}, p.$1<x}).$0";
    let ref conf = default_conf();

    let input: i32 = 3;

    let ret_value = compile_and_run(code, conf, &input);
    let data = ret_value.data() as *const WeldVec<i32>;
    let result = unsafe { (*data).clone() };

    let output = vec![8, 16, 24];
    assert_eq!(result.len, output.len() as i64);
    for i in 0..(output.len() as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
    }
}

#[test]
fn serial_parlib_test() {
    let code = "|x:vec[i32]| result(for(x, merger[i32,+], |b,i,e| merge(b, e)))";
    let ref conf = default_conf();

    let size: i32 = 10000;
    let input_vec: Vec<i32> = vec![1; size as usize];
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };

    assert_eq!(result, size as i32);
}
