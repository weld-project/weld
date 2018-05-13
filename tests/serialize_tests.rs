extern crate weld;
extern crate libc;
extern crate fnv;

use std::env;
use std::str;
use std::slice;
use std::thread;
use std::cmp;
use std::collections::hash_map::Entry;

use weld::common::WeldRuntimeErrno;

use weld::WeldConf;
use weld::WeldValue;
use weld::WeldError;
use weld::{weld_value_new, weld_value_data, weld_value_module, weld_value_free};
use weld::{weld_module_compile, weld_module_run, weld_module_free};
use weld::{weld_error_new, weld_error_code, weld_error_message, weld_error_free};
use weld::{weld_conf_new, weld_conf_set, weld_conf_free};

use std::f64::consts::PI;
use std::ffi::{CStr, CString};
use libc::{c_char, c_void};

mod common;

use common::*;

#[derive(Clone, Debug)]
#[allow(dead_code)]
#[repr(C)]
struct SerializeOutput {
    a: i32,
    b: Pair<i32,i32>,
    c: WeldVec<i32>,
    d: WeldVec<WeldVec<i32>>,
    e: Pair<i32, WeldVec<i32>>,
    f: WeldVec<Pair<i32,i32>>,
    g: WeldVec<Pair<i32,WeldVec<i32>>>,
}

impl PartialEq for SerializeOutput {
    fn eq(&self, other: &SerializeOutput) -> bool {
        let mut passed = true;
        passed &= self.a == other.a;
        passed &= self.b == other.b;
        passed &= self.c == other.c;
        passed &= self.d == other.d;
        passed &= self.e == other.e;

        fn f_into_native(v: &WeldVec<Pair<i32,i32>>) -> Vec<(i32, i32)> {
            let mut res: Vec<(i32, i32)> = (0..v.len)
                .into_iter()
                .map(|x| {
                    unsafe { ((*v.data.offset(x as isize)).ele1, (*v.data.offset(x as isize)).ele2) }
                })
            .collect();
            res.sort_by_key(|a| a.0);
            res
        }

        passed &= f_into_native(&self.f) == f_into_native(&other.f);

        // Converts field g into a native rust Vec.
        fn g_into_native(v: &WeldVec<Pair<i32,WeldVec<i32>>>) -> Vec<(i32, Vec<i32>)>{
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
    let ref input_data = WeldVec {
        data: input_vec.as_ptr(),
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const SerializeOutput };
    let result = unsafe { (*data).clone() };

    let vv = vec![input_data.clone(), input_data.clone(), input_data.clone()];
    let dict1_vec: Vec<_> = input_vec.iter().map(|e| Pair::new(*e, *e)).collect();
    let dict2_inners: Vec<_> = input_vec.iter().map(|e| (*e, vec![*e])).collect();
    let dict2_vec: Vec<_>  = dict2_inners.iter()
        .map(|e| {
            let e1 = e.0;
            let e2 = WeldVec {
                data: e.1.as_ptr(),
                len: e.1.len() as i64,
            };
            Pair::new(e1, e2)
        })
    .collect();

    let expected = SerializeOutput {
        a: input_vec[0],
        b: Pair::new(input_vec[0], input_vec[1]),
        c: input_data.clone(),
        d: WeldVec {
            data: vv.as_ptr(),
            len: vv.len() as i64,
        },
        e: Pair::new(input_vec[0], input_data.clone()),
        f: WeldVec {
            data: dict1_vec.as_ptr(),
            len: dict1_vec.len() as i64,
        },
        g: WeldVec {
            data: dict2_vec.as_ptr(),
            len: dict2_vec.len() as i64,
        }
    };

    assert_eq!(result, expected);
    unsafe { free_value_and_module(ret_value) };
}
