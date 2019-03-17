//! Tests for serializing and deserializing Weld objects.

use std::fmt::Debug;

mod common;
use crate::common::*;

/// A helper for serialization tests.
///
/// Each Weld function should take a vec[i32] and return `T`.
fn check<T: Clone + Debug + PartialEq>(code: &str, input_vec: &Vec<i32>, expect: T) {
    let ref mut conf = default_conf();
    let ref input_data = WeldVec::from(input_vec);
    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const T;
    let result = unsafe { (*data).clone() };
    assert_eq!(expect, result);
}

#[test]
fn scalar() {
    let code = "|v: vec[i32]| deserialize[i32](serialize(lookup(v, 0L)))";
    let input_vec: Vec<i32> = (10..20).collect();
    let expect = input_vec[0];
    check(code, &input_vec, expect);
}

#[test]
fn vector_with_structs() {
    let code = "|v: vec[i32]| deserialize[{i32,i32}](serialize({lookup(v, 0L), lookup(v, 1L)}))";
    let input_vec: Vec<i32> = (10..20).collect();
    let expect = Pair::new(input_vec[0], input_vec[1]);
    check(code, &input_vec, expect);
}

#[test]
fn vector_nopointers() {
    let code = "|v: vec[i32]| deserialize[vec[i32]](serialize(v))";
    let input_vec: Vec<i32> = (10..20).collect();
    let expect = WeldVec::from(&input_vec);
    check(code, &input_vec, expect);
}

#[test]
fn nested_vectors() {
    let code = "|v: vec[i32]| deserialize[vec[vec[i32]]](serialize([v, v, v]))";
    let input_vec: Vec<i32> = (10..20).collect();

    let input_data = WeldVec::from(&input_vec);
    let ref vv = vec![input_data.clone(), input_data.clone(), input_data.clone()];
    let expect = WeldVec::from(vv);

    check(code, &input_vec, expect);
}

#[test]
fn struct_with_pointers() {
    let code = "|v:vec[i32]| deserialize[{i32,vec[i32]}](serialize({lookup(v, 0L), v}))";
    let input_vec: Vec<i32> = (10..20).collect();
    let expect = Pair::new(input_vec[0], WeldVec::from(&input_vec));
    check(code, &input_vec, expect);
}

#[test]
fn dict_nopointers() {
    let code = "|v: vec[i32]|
        let dict1 = result(for(v, dictmerger[i32,i32,+], |b,i,e| merge(b, {e,e})));
        tovec(deserialize[dict[i32,i32]](serialize(dict1)))";
    let input_vec: Vec<i32> = (10..20).collect();
    let expect_vec = input_vec
        .iter()
        .map(|e| Pair::new(*e, *e))
        .collect::<Vec<_>>();
    let expect = WeldVec::from(&expect_vec);

    let ref conf = default_conf();
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<Pair<i32, i32>>;
    let result = unsafe { (*data).clone() };

    fn to_native(v: &WeldVec<Pair<i32, i32>>) -> Vec<(i32, i32)> {
        let mut res: Vec<(i32, i32)> = (0..v.len)
            .into_iter()
            .map(|x| unsafe {
                (
                    (*v.data.offset(x as isize)).ele1,
                    (*v.data.offset(x as isize)).ele2,
                )
            })
            .collect();
        res.sort_by_key(|a| a.0);
        res
    }
    assert_eq!(to_native(&expect), to_native(&result));
}

#[test]
fn dict_pointers() {
    let code = "|v: vec[i32]|
        let dict2 = result(for(v, groupmerger[i32,i32], |b,i,e| merge(b, {e,e})));
        tovec(deserialize[dict[i32,vec[i32]]](serialize(dict2)))";
    let input_vec: Vec<i32> = (10..20).collect();

    let dict2_inners: Vec<_> = input_vec.iter().map(|e| (*e, vec![*e])).collect();
    let expect_vec = dict2_inners
        .iter()
        .map(|e| {
            let e1 = e.0;
            let e2 = WeldVec::from(&e.1);
            Pair::new(e1, e2)
        })
        .collect::<Vec<_>>();
    let expect = WeldVec::from(&expect_vec);

    let ref mut conf = default_conf();

    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<Pair<i32, WeldVec<i32>>>;
    let result = unsafe { (*data).clone() };

    fn to_native(v: &WeldVec<Pair<i32, WeldVec<i32>>>) -> Vec<(i32, Vec<i32>)> {
        let mut res: Vec<(i32, Vec<i32>)> = (0..v.len)
            .into_iter()
            .map(|x| {
                let key = unsafe { (*v.data.offset(x as isize)).ele1 };
                let val = unsafe { ((*v.data.offset(x as isize)).ele2).clone() };
                let vec: Vec<i32> = (0..val.len)
                    .into_iter()
                    .map(|y| unsafe { *val.data.offset(y as isize) })
                    .collect();
                (key, vec)
            })
            .collect();

        // For dictionary outputs, we need to ignore the order.
        res.sort_by_key(|a| a.0);
        res
    }

    assert_eq!(to_native(&expect), to_native(&result));
}

/// Tests dictionaries where the value contains nested pointers.
#[test]
fn dict_nested_pointers() {
    let code = "|v: vec[i32]|
        let dict2 = result(for(v, groupmerger[i32,vec[i32]], |b,i,e| merge(b, {e,[e,e,e,e,e]})));
        tovec(deserialize[dict[i32,vec[vec[i32]]]](serialize(dict2)))";
    let input_vec: Vec<i32> = (10..20).collect();

    let dict2_inners: Vec<(i32, Vec<Vec<i32>>)> =
        input_vec.iter().map(|e| (*e, vec![vec![*e; 5]])).collect();

    // Need this to keep the reference to the inner vectors live.
    let mut inner_refs = vec![];

    // Construct the Key/Value vector.
    let expect_vec = dict2_inners
        .iter()
        .map(|e| {
            let e1 = e.0;
            let inner_vec = e.1.iter().map(|v| WeldVec::from(v)).collect::<Vec<_>>();
            let e2 = WeldVec::from(&inner_vec);

            // Prevent the inner vector from being dropped by holding a ref to it.
            inner_refs.push(inner_vec);

            Pair::new(e1, e2)
        })
        .collect::<Vec<_>>();
    let expect = WeldVec::from(&expect_vec);

    let ref mut conf = default_conf();
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<Pair<i32, WeldVec<WeldVec<i32>>>>;
    let result = unsafe { (*data).clone() };

    fn to_native(v: &WeldVec<Pair<i32, WeldVec<WeldVec<i32>>>>) -> Vec<(i32, Vec<Vec<i32>>)> {
        let mut res: Vec<(i32, Vec<Vec<i32>>)> = (0..v.len)
            .into_iter()
            .map(|x| {
                let key = unsafe { (*v.data.offset(x as isize)).ele1 };
                let val = unsafe { ((*v.data.offset(x as isize)).ele2).clone() };
                let vec: Vec<Vec<i32>> = (0..val.len)
                    .into_iter()
                    .map(|y| unsafe {
                        let inner_vec = (*val.data.offset(y as isize)).clone();
                        (0..inner_vec.len)
                            .into_iter()
                            .map(|z| *inner_vec.data.offset(z as isize))
                            .collect()
                    })
                    .collect();
                (key, vec)
            })
            .collect();

        // For dictionary outputs, we need to ignore the order.
        res.sort_by_key(|a| a.0);
        res
    }

    assert_eq!(to_native(&expect), to_native(&result));
}
