//! Tests which exercise various aspects of the `For` loop.

mod common;
use crate::common::*;

#[test]
fn if_statement_with_dict_loop() {
    #[derive(Clone)]
    #[allow(dead_code)]
    #[repr(C)]
    struct Args {
        left: WeldVec<i64>,
        right: WeldVec<i64>,
    }

    // This code resembles a hash join!
    let code = "
        |left: vec[i64], right: vec[i64]|
            let bd = result(for(left,
                    groupmerger[i64,i64],
                    |b,i,e| merge(b, {e, e})
                    )
                );
            result(for(right,
              appender[{i64, i64}],
              |b: appender[{i64, i64}], i: i64, ns: i64|
                if (keyexists(bd, ns),
                  for(lookup(bd, ns),
                    b,
                    |b1: appender[{i64, i64}], i1: i64, ns1: i64|
                    merge(b1, {ns1, ns})
                  ),
                  b)
                ))";

    let ref conf = default_conf();

    let mut left = Vec::with_capacity(20);
    let mut right = Vec::with_capacity(1000);

    // 525, 550, ... 1000
    for i in 500..1000 {
        if i % 20 == 0 {
            left.push(i as i64);
        }
    }

    for i in 0..1000 {
        right.push(i as i64);
    }

    let ref input_data = Args {
        left: WeldVec::from(&left),
        right: WeldVec::from(&right),
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<Pair<i64, i64>>;
    let result = unsafe { (*data).clone() };

    // Output should be the left-side as pairs (i.e., {525,525}, {550,550},etc.)
    assert_eq!(result.len as usize, left.len());

    let mut sorted = vec![];

    for i in 0..(result.len as isize) {
        let left = unsafe { (*result.data.offset(i)).ele1 };
        let right = unsafe { (*result.data.offset(i)).ele2 };
        sorted.push((left, right));
    }

    // Compare sorted pairs with the output.
    sorted.sort();

    for (a, (b, c)) in left.into_iter().zip(sorted) {
        assert_eq!(a, b);
        assert_eq!(a, c);
    }
}

#[test]
fn nested_if_statement_loop() {
    let code = "|p: vec[i64]|
    result(for(p,merger[i64, +], |bs, i, ns| if(ns >= 3L, if(ns < 7L, merge(bs, ns), bs), bs)))";

    let ref conf = default_conf();

    let input_vec: Vec<i64> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };
    let output = 18;
    assert_eq!(result, output);
}

#[test]
fn nested_if_statement_with_builders_loop() {
    let code = "|p: vec[i64]|
        let filter = result(for(p,appender[i64], |bs, i, ns|if(ns >= 3L, if(ns < 7L, merge(bs, ns), bs), bs)));
        result(for(filter, merger[i64, +], |bs2, i2, ns2| merge(bs2, ns2)))";

    let ref conf = default_conf();

    let input_vec: Vec<i64> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };
    let output = 18;
    assert_eq!(result, output);
}

#[test]
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
    let ref arg = WeldVec::new(
        input_data.as_ptr() as *const WeldVec<WeldVec<i32>>,
        input_data.len() as i64,
    );

    let expect = [r0, r1, r2, r3, r4];

    // Computes the identity.
    let code = "|e0: vec[vec[i32]]| map(e0, |x:vec[i32]| map(x, |y:i32| y))";
    let ref conf = default_conf();

    let ret_value = compile_and_run(code, conf, arg);
    let data = ret_value.data() as *const WeldVec<WeldVec<i32>>;
    let result = unsafe { (*data).clone() };

    // Make sure we get the same thing back.
    assert_eq!(result.len, 5);
    for i in 0..(result.len as isize) {
        let inner = unsafe { result.data.offset(i) };
        let inner_length = unsafe { (*inner).len };
        assert_eq!(inner_length, size);
        for j in 0..(inner_length as isize) {
            assert_eq!(
                unsafe { *((*inner).data.offset(i)) },
                expect[i as usize][j as usize]
            );
        }
    }
}

#[test]
fn nested_for_loops() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Row {
        x: i64,
        y: i32,
    }

    let code = "|ys:vec[i64]|result(for(ys, appender[{i64, i32}], |b0, i0, y0| for(ys, b0, |b1, i1, y1| if (y1 > y0, merge(b0, {y0, i32(y1)}), b0))))";
    let ref conf = default_conf();

    // Input data.
    let ys = vec![1i64, 3i64, 4i64];
    let ref input_data = WeldVec::from(&ys);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<Row>;
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
}

#[test]
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
        dict_out: WeldVec<Pair>,
    }

    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]| let rs = for(zip(x,y),{appender[i32],dictmerger[i32,i32,+]},
                |bs,i,e| {merge(bs.$0, e.$0+e.$1), merge(bs.$1, e)}); {result(rs.$0), tovec(result(rs.$1))}";
    let ref conf = default_conf();
    let keys = [1, 2, 2, 1, 3];
    let vals = [2, 3, 4, 2, 1];
    let ref input_data = Args {
        x: WeldVec::from(&keys),
        y: WeldVec::from(&vals),
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const Output;
    let result = unsafe { (*data).clone() };

    let output_appender = [3, 5, 6, 3, 4];
    let output_dict_keys = [1, 2, 3];
    let output_dict_vals = [4, 7, 1];

    assert_eq!(result.append_out.len, output_appender.len() as i64);
    for i in 0..(output_appender.len() as isize) {
        assert_eq!(
            unsafe { *result.append_out.data.offset(i) },
            output_appender[i as usize]
        );
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
}
