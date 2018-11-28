//! Tests for binary comparisons.

extern crate weld;

mod common;
use common::*;

#[test]
fn simple_binop() {
    let code = "|| 3.1 > 4.2";
    let ref conf = default_conf();

    let ref input_data: f64 = 0.0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const bool;
    let result = unsafe { *data };
    assert!(result == false);
}

#[test]
fn struct_cmp() {
    let code = "|| {3.0, 2.0} < {1.0, 2.0}";

    let ref conf = default_conf();

    let ref input_data: f64 = 0.0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const bool;
    let result = unsafe { *data };
    assert!(result == false);
}

#[test]
fn string_cmp() {
    let code = "|| \"abc\" > \"def\"";

    let ref conf = default_conf();

    let ref input_data: f64 = 0.0;
    
    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const bool;
    let result = unsafe { *data };

    assert!(result == false);
}

#[test]
fn vector_cmp() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x: vec[i32], y: vec[i32]| x > y";

    let ref conf = default_conf();

    let x = vec![0, 1, 2];
    let y = vec![1, 2];
    let ref input_data = Args {
        x: WeldVec::from(&x),
        y: WeldVec::from(&y),
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const bool;
    let result = unsafe { *data };

    assert!(result == false);
}

#[test]
fn substring_cmp() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code_eq = "|x: vec[i32], y: vec[i32]| x == y";
    let code_lt = "|x: vec[i32], y: vec[i32]| x < y";
    let code_gt = "|x: vec[i32], y: vec[i32]| x > y";

    let ref conf = default_conf();

    let x = vec![1, 2, 3, 4];
    let y = vec![1];
    let ref input_data = Args {
        x: WeldVec::from(&x),
        y: WeldVec::from(&y),
    };

    let ret_value = compile_and_run(code_eq, conf, input_data);
    let data = ret_value.data() as *const bool;
    let result = unsafe { *data };

    assert!(result == false);

    let ret_value = compile_and_run(code_lt, conf, input_data);
    let data = ret_value.data() as *const bool;
    let result = unsafe { *data };

    assert!(result == false);

    let ret_value = compile_and_run(code_gt, conf, input_data);
    let data = ret_value.data() as *const bool;
    let result = unsafe { *data };

    assert!(result == true);
}
