//! Tests for the various For loop iterators.

mod common;
use crate::common::*;

#[test]
fn range_iter_1() {
    let end = 1000;
    let code = format!(
        "|a: i64| result(for(rangeiter(1L, {}L + 1L, 1L), merger[i64,+], |b,i,e| merge(b, a+e)))",
        end
    );

    #[allow(dead_code)]
    struct Args {
        a: i64,
    };
    let ref conf = default_conf();
    let ref input_data = Args { a: 0 };

    let ret_value = compile_and_run(&code, conf, input_data);
    let data = ret_value.data() as *const i64;
    let result = unsafe { (*data).clone() };
    let output = end * (end + 1) / 2;
    assert_eq!(result, output);
}

fn range_iter_zipped_helper(parallel: bool) {
    let grain_size = if parallel { 100 } else { 4096 };
    let ref conf = if parallel {
        many_threads_conf()
    } else {
        default_conf()
    };

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
        v: WeldVec::from(&input_vec),
    };

    let ret_value = compile_and_run(&code, conf, input_data);
    let data = ret_value.data() as *const i64;
    let result = unsafe { (*data).clone() };
    let output = end * (end + 1) / 2 + end;
    assert_eq!(result, output);
}

#[test]
fn range_iter_2() {
    range_iter_zipped_helper(false)
}

#[test]
fn range_iter_parallel() {
    range_iter_zipped_helper(true)
}

#[test]
fn iters_for_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]| result(for(zip(iter(x,0L,4L,2L), y), appender, |b,i,e|
                merge(b,e.$0+e.$1)))";
    let ref conf = default_conf();

    let x = vec![1, 2, 3, 4];
    let y = vec![5, 6];
    let ref input_data = Args {
        x: WeldVec::from(&x),
        y: WeldVec::from(&y),
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<i32>;
    let result = unsafe { (*data).clone() };

    let output = vec![6, 9];
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
    }
}
