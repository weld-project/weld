//! Tests for serializing and deserializing Weld objects.

extern crate weld;
use weld::weld_value_data;

mod common;
use common::*;

#[derive(Clone, Debug)]
#[allow(dead_code)]
#[repr(C)]
struct SerializeOutput {
    a: i32,
    b: Pair<i32, i32>,
    c: WeldVec<i32>,
    d: WeldVec<WeldVec<i32>>,
    e: Pair<i32, WeldVec<i32>>,
    f: WeldVec<Pair<i32, i32>>,
    g: WeldVec<Pair<i32, WeldVec<i32>>>,
}

impl PartialEq for SerializeOutput {
    fn eq(&self, other: &SerializeOutput) -> bool {
        let mut passed = true;
        passed &= self.a == other.a;
        passed &= self.b == other.b;
        passed &= self.c == other.c;
        passed &= self.d == other.d;
        passed &= self.e == other.e;

        // Convert field f into a native Rust Vec.
        fn f_into_native(v: &WeldVec<Pair<i32, i32>>) -> Vec<(i32, i32)> {
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

        passed &= f_into_native(&self.f) == f_into_native(&other.f);

        // Converts field g into a native Rust Vec.
        fn g_into_native(v: &WeldVec<Pair<i32, WeldVec<i32>>>) -> Vec<(i32, Vec<i32>)> {
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

        passed &= g_into_native(&self.g) == g_into_native(&other.g);

        passed
    }
}

#[test]
fn serialize_test() {
    let code = " |v: vec[i32]|
    let dict1 = result(for(v, dictmerger[i32,i32,+], |b,i,e| merge(b, {e,e})));
    let dict2 = result(for(v, groupmerger[i32,i32], |b,i,e| merge(b, {e,e})));

    let a = deserialize[i32](serialize(lookup(v, 0L)));
    let b = deserialize[{i32,i32}](serialize({lookup(v, 0L), lookup(v, 1L)}));
    let c = deserialize[vec[i32]](serialize(v));
    let d = deserialize[vec[vec[i32]]](serialize([v, v, v]));
    let e = deserialize[{i32,vec[i32]}](serialize({lookup(v, 0L), v}));
    let f = tovec(deserialize[dict[i32,i32]](serialize(dict1)));
    let g = tovec(deserialize[dict[i32,vec[i32]]](serialize(dict2)));
    {a,b,c,d,e,f,g}";

    let conf = default_conf();

    let input_vec: Vec<i32> = (10..20).collect();
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const SerializeOutput };
    let result = unsafe { (*data).clone() };

    let vv = vec![input_data.clone(), input_data.clone(), input_data.clone()];
    let dict1_vec: Vec<_> = input_vec.iter().map(|e| Pair::new(*e, *e)).collect();
    let dict2_inners: Vec<_> = input_vec.iter().map(|e| (*e, vec![*e])).collect();
    let dict2_vec: Vec<_> = dict2_inners
        .iter()
        .map(|e| {
            let e1 = e.0;
            let e2 = WeldVec::from(&e.1);
            Pair::new(e1, e2)
        })
        .collect();

    let expected = SerializeOutput {
        a: input_vec[0],
        b: Pair::new(input_vec[0], input_vec[1]),
        c: input_data.clone(),
        d: WeldVec::from(&vv),
        e: Pair::new(input_vec[0], input_data.clone()),
        f: WeldVec::from(&dict1_vec),
        g: WeldVec::from(&dict2_vec),
    };

    assert_eq!(result, expected);
    unsafe { free_value_and_module(ret_value) };
}
