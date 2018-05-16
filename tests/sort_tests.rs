//! Tests that make use of the Sort operator.

extern crate weld;

mod common;
use common::*;

#[test]
fn string_sort() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<WeldVec<u8>>,
    }
    let bs = vec!['T' as u8, 'R' as u8, 'U' as u8, 'E' as u8];
    let cs = vec!['P' as u8, 'A' as u8, 'R' as u8];
    let ds = vec!['F' as u8, 'A' as u8, 'L' as u8, 'S' as u8, 'E' as u8];
    let sorted = vec![ds.clone(), cs.clone(), bs.clone()];
    let bs_vec = WeldVec::from(&bs);
    let cs_vec = WeldVec::from(&cs);
    let ds_vec = WeldVec::from(&ds);
    let strs = vec![bs_vec, cs_vec, ds_vec];

    let ref input_data = Args {
        x: WeldVec::from(&strs),
    };

    let code = "|e0: vec[vec[u8]]| sort(e0, |i:vec[u8]| i)";

    let ref conf = default_conf();
    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<WeldVec<u8>>;
    let result = unsafe { (*data).clone() };

    for i in 0..(result.len as isize) {
        let ivec = unsafe { (*result.data.offset(i)).clone() };
        for j in 0..(ivec.len as isize) {
            let val = unsafe { (*ivec.data.offset(j)) };
            assert_eq!(val, sorted[i as usize][j as usize])
        }
    }


}

#[test]
fn if_sort() {
    let ys = vec![2, 3, 1, 4, 5];
    let ref input_data = WeldVec::from(&ys);

    let code = "|ys:vec[i32]| sort(ys, |x:i32| if(x != 5, x + 1, 0))";
    let ref conf = default_conf();
    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<i32>;
    let result = unsafe { (*data).clone() };

    let expected = [5, 1, 2, 3, 4];
    assert_eq!(result.len, expected.len() as i64);

    for i in 0..(expected.len() as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, expected[i as usize])
    }


}

#[test]
fn simple_sort() {
    let ys = vec![2, 3, 1, 4, 5];
    let ref input_data = WeldVec::from(&ys);

    let code = "|ys:vec[i32]| sort(ys, |x:i32| x + 1)";
    let ref conf = default_conf();
    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<i32>;
    let result = unsafe { (*data).clone() };

    let expected = [1, 2, 3, 4, 5];
    assert_eq!(result.len, expected.len() as i64);

    for i in 0..(expected.len() as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, expected[i as usize])
    }



    let ys = vec![2.0, 3.0, 1.0, 5.001, 5.0001];
    let ref input_data = WeldVec::from(&ys);

    let code = "|ys:vec[f64]| sort(sort(ys, |x:f64| x), |x:f64| x)";
    let ref conf = default_conf();
    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<f64>;
    let result = unsafe { (*data).clone() };

    let expected = [1.0, 2.0, 3.0, 5.0001, 5.001];
    assert_eq!(result.len, expected.len() as i64);
    for i in 0..(expected.len() as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, expected[i as usize])
    }

    let code = "|ys:vec[f64]| sort(ys, |x:f64| 1.0 / exp(-1.0*x))";
    let ref conf = default_conf();
    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<f64>;
    let result = unsafe { (*data).clone() };

    assert_eq!(result.len, expected.len() as i64);
    for i in 0..(expected.len() as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, expected[i as usize])
    }


}

#[test]
fn complex_sort() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let xs = vec![1, 2, 3, 4, 5];
    let ys = vec![5, 4, 3, 2, 1];
    let ref input_data = Args {
        x: WeldVec::from(&xs),
        y: WeldVec::from(&ys),
    };

    let code = "|xs:vec[i32], ys:vec[i32]|
                  sort(
                    result(
                      for(
                        zip(xs,ys),
                        appender[{i32,i32}],
                        |b,i,e| merge(b, e)
                      )
                    ),
                    |x:{i32, i32}| x.$0
                )";
    let ref conf = default_conf();
    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<Pair<i32,i32>>;
    let result = unsafe { (*data).clone() };

    let expected = [[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]];
    assert_eq!(result.len, expected.len() as i64);

    for i in 0..(expected.len() as isize) {
        assert_eq!(
            unsafe { (*result.data.offset(i)).ele1 },
            expected[i as usize][0]
        );
        assert_eq!(
            unsafe { (*result.data.offset(i)).ele2 },
            expected[i as usize][1]
        );
    }


}
