//! Tests for `WeldContext`.
//!
//! These tests reuse builders across Weld programs by reusing contexts.

use weld;

use weld::data::*;
use weld::*;

mod common;
use crate::common::*;

#[test]
fn reuse_appender_test() {
    let ref conf = default_conf();
    let ref mut context = WeldContext::new(conf).unwrap();

    let program1 = "|v: vec[i32]| for(v, appender[i32], |b,i,e| merge(b,e))";

    let input_vec = vec![1, 2, 3, 4, 5];
    let input_data = WeldVec::from(&input_vec);

    let ref input_value = WeldValue::new_from_data(&input_data as *const _ as Data);

    let module = WeldModule::compile(program1, conf).unwrap();
    let ret_value = unsafe { module.run(context, input_value).unwrap() };
    let data = ret_value.data() as *const Appender<i32>;
    let appender = unsafe { (*data).clone() };

    // The input for the second program.
    #[derive(Clone)]
    #[allow(dead_code)]
    #[repr(C)]
    struct Program2Args {
        vec: WeldVec<i32>,
        appender: Appender<i32>,
    }

    let program2 = "|v: vec[i32], a: appender[i32]|
        result(for(v, a, |b,i,e| merge(b,e)))";

    let ref input_data = Program2Args {
        vec: input_data,
        appender: appender,
    };

    let ref input_value = WeldValue::new_from_data(input_data as *const _ as Data);
    let module = WeldModule::compile(program2, conf).unwrap();

    // Run new module with same context.
    let ret_value = unsafe { module.run(context, input_value).unwrap() };
    let data = ret_value.data() as *const WeldVec<i32>;

    let result = unsafe { (*data).clone() };

    // Check correctness
    assert_eq!(result.len as usize, input_vec.len() * 2);
    for i in 0..result.len as isize {
        assert_eq!(
            unsafe { *result.data.offset(i) as i32 },
            input_vec[(i % input_vec.len() as isize) as usize]
        );
    }
}

#[test]
fn reuse_groupmerger_test() {
    #[derive(Clone)]
    #[allow(dead_code)]
    #[repr(C)]
    struct Program1Args {
        keys: WeldVec<i32>,
        vals: WeldVec<i32>,
    }

    let ref conf = default_conf();
    let ref mut context = WeldContext::new(conf).unwrap();

    let program1 = "|keys: vec[i32], vals: vec[i32]|
        for(zip(keys, vals), groupmerger[i32,i32], |b,i,e| merge(b,e))";

    let keys = vec![1, 2, 2, 3, 3, 1];
    let vals = vec![2, 3, 4, 1, 0, 2];
    let input_data = Program1Args {
        keys: WeldVec::from(&keys),
        vals: WeldVec::from(&vals),
    };

    let ref input_value = WeldValue::new_from_data(&input_data as *const _ as Data);

    let module = WeldModule::compile(program1, conf).unwrap();
    let ret_value = unsafe { module.run(context, input_value).unwrap() };
    let data = ret_value.data() as *const GroupMerger<i32, i32>;
    let groupmerger = unsafe { (*data).clone() };

    // The input for the second program.
    #[derive(Clone)]
    #[allow(dead_code)]
    #[repr(C)]
    struct Program2Args {
        keys: WeldVec<i32>,
        vals: WeldVec<i32>,
        builder: GroupMerger<i32, i32>,
    }

    let program2 = "|keys: vec[i32], vals: vec[i32], gm: groupmerger[i32,i32]|
        tovec(result(for(zip(keys,vals), gm, |b,i,e| merge(b,e))))";

    let ref input_data = Program2Args {
        keys: WeldVec::from(&keys),
        vals: WeldVec::from(&vals),
        builder: groupmerger,
    };

    let ref input_value = WeldValue::new_from_data(input_data as *const _ as Data);
    let module = WeldModule::compile(program2, conf).unwrap();

    // Run new module with same context.
    let ret_value = unsafe { module.run(context, input_value).unwrap() };
    let data = ret_value.data() as *const WeldVec<Pair<i32, WeldVec<i32>>>;

    let result = unsafe { (*data).clone() };

    let expect: Vec<(i32, Vec<i32>)> = vec![
        (1, vec![2, 2, 2, 2]),
        (2, vec![3, 4, 3, 4]),
        (3, vec![1, 0, 1, 0]),
    ];

    // Check correctness
    assert_eq!(result.len as usize, expect.len());

    let mut res: Vec<(i32, Vec<i32>)> = (0..result.len)
        .into_iter()
        .map(|x| {
            let key = unsafe { (*result.data.offset(x as isize)).ele1 };
            let val = unsafe { ((*result.data.offset(x as isize)).ele2).clone() };
            let vec: Vec<i32> = (0..val.len)
                .into_iter()
                .map(|y| unsafe { *val.data.offset(y as isize) })
                .collect();
            (key, vec)
        })
        .collect();
    res.sort_by_key(|a| a.0);
    assert_eq!(res, expect);
}
