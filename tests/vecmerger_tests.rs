//! Tests for the VecMerger builder type.

use std::cmp;

mod common;
use crate::common::*;

#[test]
fn simple_for_vecmerger_loop() {
    let code = "|x:vec[i32]| result(for(x, vecmerger[i32,+](x), |b,i,e| b))";
    let ref conf = default_conf();

    let input_vec = vec![1, 1, 1, 1, 1];
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<i32>;
    let result = unsafe { (*data).clone() };
    assert_eq!(result.len, input_vec.len() as i64);
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, input_vec[i as usize]);
    }
}

#[test]
fn simple_for_vecmerger_binops() {
    let code = "|x:vec[i64]| {
        result(for(x, vecmerger[i64,+](x), |b,i,e| merge(b, {i,e*7L}))),
        result(for(x, vecmerger[i64,*](x), |b,i,e| merge(b, {i, i}))),
        result(for(x, vecmerger[i64,min](x), |b,i,e| merge(b, {i, i}))),
        result(for(x, vecmerger[i64,max](x), |b,i,e| merge(b, {i, i})))
        }";
    let ref conf = default_conf();

    #[derive(Clone)]
    #[allow(dead_code)]
    struct Output {
        sum: WeldVec<i64>,
        prod: WeldVec<i64>,
        min: WeldVec<i64>,
        max: WeldVec<i64>,
    }

    let input_vec: Vec<i64> = vec![3, 3, 3, 3, 3];
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const Output;
    let result = unsafe { (*data).clone() };
    for i in 0..(input_vec.len() as isize) {
        assert_eq!(
            unsafe { *result.sum.data.offset(i) },
            input_vec[i as usize] + input_vec[i as usize] * 7
        );
        assert_eq!(
            unsafe { *result.prod.data.offset(i) },
            input_vec[i as usize] * (i as i64)
        );
        assert_eq!(
            unsafe { *result.min.data.offset(i) },
            cmp::min(input_vec[i as usize], i as i64)
        );
        assert_eq!(
            unsafe { *result.max.data.offset(i) },
            cmp::max(input_vec[i as usize], i as i64)
        );
    }
}

#[test]
fn parallel_for_vecmerger_loop() {
    let code = "|x:vec[i32]| result(@(grain_size: 100)for(x, vecmerger[i32,+](x), |b,i,e| merge(b, {i,e*7})))";
    let ref conf = many_threads_conf();

    let input_vec = vec![1; 4096];
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<i32>;
    let result = unsafe { (*data).clone() };

    assert_eq!(result.len, input_vec.len() as i64);
    for i in 0..(result.len as isize) {
        assert_eq!(
            unsafe { *result.data.offset(i) },
            input_vec[i as usize] + input_vec[i as usize] * 7
        );
    }
}
