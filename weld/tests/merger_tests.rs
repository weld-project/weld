//! Tests for the Merger builder type.

mod common;
use crate::common::*;

#[test]
fn empty_merger_loop() {
    let code = "||result(for([]:vec[i32], merger[i32, +], |b, i, n| merge(b, n)))";
    let ref conf = default_conf();

    let ref input_data: i32 = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { *data };
    assert_eq!(result, 0);
}

#[test]
fn simple_for_merger_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        a: i32,
    }

    let code = "|x:vec[i32], a:i32| result(for(x, merger[i32,+], |b,i,e| merge(b, e+a)))";
    let ref conf = default_conf();

    let input_vec = vec![1, 2, 3, 4, 5];
    let ref input_data = Args {
        x: WeldVec::from(&input_vec),
        a: 1,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };
    let output = 20;
    assert_eq!(result, output);
}

#[test]
fn simple_zipped_for_merger_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]| result(for(zip(x,y), merger[i32,+], |b,i,e| merge(b, e.$0+e.$1)))";
    let ref conf = default_conf();

    let size = 2000;
    let x_data = vec![1; size as usize];
    let y_data = vec![5; size as usize];

    let ref input_data = Args {
        x: WeldVec::from(&x_data),
        y: WeldVec::from(&y_data),
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };
    let output = size * (x_data[0] + y_data[0]);
    assert_eq!(result, output);
}

#[test]
fn simple_for_merger_loop_product() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        a: i32,
    }

    let code = "|x:vec[i32], a:i32| result(for(x, merger[i32,*], |b,i,e| merge(b, e+a)))";
    let ref conf = default_conf();

    let input_vec = vec![1, 2, 3, 4, 5];
    let ref input_data = Args {
        x: WeldVec::from(&input_vec),
        a: 1,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };
    let output = 720;
    assert_eq!(result, output);
}

#[test]
fn parallel_for_merger_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        a: i32,
    }

    let code = "|x:vec[i32], a:i32| result(@(grain_size: 100)for(x, merger[i32,+], |b,i,e| merge(b, e+a)))";
    let ref conf = many_threads_conf();

    let input_vec = vec![1; 4096];
    let ref input_data = Args {
        x: WeldVec::from(&input_vec),
        a: 1,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };
    let output = (input_vec[0] + input_data.a) * (input_vec.len() as i32);
    assert_eq!(result, output);
}

#[test]
fn parallel_for_multi_merger_loop() {
    let code =
        "|x:vec[i32]| let r = @(grain_size: 100)for(x, {merger[i32,+], merger[i32,+]}, |b,i,e|
                { merge(b.$0, e), merge(b.$1, e) }); result(r.$0) + result(r.$1)";
    let ref conf = many_threads_conf();

    let input_vec = vec![1; 4096];
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };
    let output = input_vec[0] * 2 * (input_vec.len() as i32);
    assert_eq!(result, output);
}

#[test]
fn simple_for_merger_loop_initial_value() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        a: i32,
    }

    let code = "|x:vec[i32], a:i32| result(for(x, merger[i32,+](1000), |b,i,e| merge(b, e+a)))";
    let ref conf = default_conf();

    let input_vec = vec![1; 4096];
    let ref input_data = Args {
        x: WeldVec::from(&input_vec),
        a: 1,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };
    let output = 1000 + (input_vec[0] + input_data.a) * (input_vec.len() as i32);
    assert_eq!(result, output);
}

#[test]
fn parallel_for_merger_loop_initial_value() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        a: i32,
    }

    let code = "|x:vec[i32], a:i32| result(@(grain_size: 100)for(x, merger[i32,+](1000), |b,i,e| merge(b, e+a)))";
    let ref conf = many_threads_conf();

    let input_vec = vec![1; 4096];
    let ref input_data = Args {
        x: WeldVec::from(&input_vec),
        a: 1,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };
    let output = 1000 + (input_vec[0] + input_data.a) * (input_vec.len() as i32);
    assert_eq!(result, output);
}

#[test]
fn parallel_for_merger_loop_initial_value_product() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
    }

    let code =
        "|x:vec[i32]| result(@(grain_size: 100)for(x, merger[i32,*](1000), |b,i,e| merge(b, e)))";
    let ref conf = many_threads_conf();

    let input_vec = vec![1; 4096];
    let ref input_data = Args {
        x: WeldVec::from(&input_vec),
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };
    let output = 1000;
    assert_eq!(result, output);
}

#[test]
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

    let ref conf = default_conf();

    let size: i32 = 1000;
    let x = vec![1.0; size as usize];
    let y = vec![1.0; size as usize];
    let z = vec![1.0; size as usize];

    let ref input_data = Args {
        x: WeldVec::from(&x),
        y: WeldVec::from(&y),
        z: WeldVec::from(&z),
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const Output;
    let result = unsafe { (*data).clone() };

    assert_eq!(result.a, size as i64);
    assert_eq!(result.b, size as f64);
    assert_eq!(result.c, size as i64);
    assert_eq!(result.d, size as f64);
    assert_eq!(result.e, size as i64);
    assert_eq!(result.f, size as f64);
}

#[test]
fn maxmin_mergers_test() {
    #[derive(Clone)]
    #[allow(dead_code)]
    #[repr(C)]
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
    #[repr(C)]
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

    let ref mut conf = default_conf();

    let i8in: Vec<i8> = vec![-2, -1, 0, 1, 2];
    let i32in: Vec<i32> = vec![-2, -1, 0, 1, 2];
    let i64in: Vec<i64> = vec![-2, -1, 0, 1, 2];
    let f32in: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let f64in: Vec<f64> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

    let ref input_data = Args {
        i8in: WeldVec::from(&i8in),
        i32in: WeldVec::from(&i32in),
        i64in: WeldVec::from(&i64in),
        f32in: WeldVec::from(&f32in),
        f64in: WeldVec::from(&f64in),
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const Output;
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
}
