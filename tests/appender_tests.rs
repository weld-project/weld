//! Tests for the Appender builder.

extern crate weld;
use weld::weld_value_data;

mod common;
use common::*;

#[test]
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
        x: WeldVec::from(&input_vec),
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

#[test]
fn large_unaryop_for_appender_loop() {
    let code = "|x:vec[f32]| map(x, |e| sqrt(e))";
    let conf = default_conf();

    let input_vec: Vec<f32> = vec![4.0; 1000000];
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<f32> };
    let result = unsafe { (*data).clone() };
    assert_eq!(result.len as usize, input_vec.len());
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) as i32 }, 2)
    }
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn simple_parallel_for_appender_loop() {
    let code = "|x:vec[i32]| result(@(grain_size: 100)for(x, appender[i64], |b,i,e| merge(b, i)))";
    let conf = many_threads_conf();

    let size = 10000;
    let input_vec: Vec<i32> = vec![0; size as usize];
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<i64> };
    let result = unsafe { (*data).clone() };

    assert_eq!(result.len, size);
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, i as i64)
    }
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn simple_parallel_for_multi_appender_loop() {
    let code = "|x:vec[i32]| let r = @(grain_size: 100)for(x, {appender[i64], appender[i64]}, |b,i,e|
                { merge(b.$0, i), merge(b.$1, i) }); let r2 = @(grain_size: 100)for(result(r.$0), r.$1,
                |b,i,e| merge(b, e)); result(r2)";
    let conf = many_threads_conf();

    let size = 10000;
    let input_vec: Vec<i32> = vec![0; size as usize];
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<i64> };
    let result = unsafe { (*data).clone() };

    assert_eq!(result.len, size * 2);
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, (i as i64) % size)
    }
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn complex_parallel_for_appender_loop() {
    let code = "|x:vec[i32]| let a=appender[i64]; let b=merge(a,0L); let r=@(grain_size: 100)for(x,b,|b,i,e|
                let c=merge(b,1L); let d=@(grain_size: 100)for(x,c,|b,i,e| if(i<1L, merge(b,i), b)); merge(d, 2L));
                result(merge(r,3L))";
    let conf = many_threads_conf();

    let size = 3000;
    let input_vec: Vec<i32> = vec![0; size as usize];
    let ref input_data = WeldVec::from(&input_vec);

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
