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

#[test]
fn simple_for_dictmerger_loop() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Pair {
        ele1: i32,
        ele2: i32,
    }

    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]| tovec(result(for(zip(x,y), dictmerger[i32,i32,+],
                |b,i,e| merge(b, e))))";
    let conf = default_conf();
    let keys = [1, 2, 2, 1, 3];
    let vals = [2, 3, 4, 2, 1];
    let ref input_data = Args {
        x: WeldVec {
            data: &keys as *const i32,
            len: keys.len() as i64,
        },
        y: WeldVec {
            data: &vals as *const i32,
            len: vals.len() as i64,
        },
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<Pair> };
    let result = unsafe { (*data).clone() };

    let output_keys = [1, 2, 3];
    let output_vals = [4, 7, 1];

    assert_eq!(result.len, output_keys.len() as i64);
    for i in 0..(output_keys.len() as isize) {
        let mut success = false;
        let key = unsafe { (*result.data.offset(i)).ele1 };
        let value = unsafe { (*result.data.offset(i)).ele2 };
        for j in 0..(output_keys.len()) {
            if output_keys[j] == key {
                if output_vals[j] == value {
                    success = true;
                }
            }
        }
        assert_eq!(success, true);
    }
    unsafe { free_value_and_module(ret_value) };
}

/// Similar case to parallel_for_vecmerger_loop but values and keys are structs
#[test]
fn dictmerger_with_structs() {
    /// An entry after the final tovec call, which has an {i32,i32} key and {i32,f32} value.
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Entry {
        k1: i32,
        k2: i32,
        v1: i32,
        v2: f32
    }

    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]|
                    tovec(result(for(
                        zip(x,y),
                        dictmerger[{i32,i32},{i32,f32},+],
                        |b,i,e| merge(b, {{e.$0, e.$0}, {e.$1, f32(e.$1)}}))))";
    let conf = default_conf();
    let keys = [1, 2, 2, 1, 3];
    let vals = [2, 3, 4, 2, 1];
    let ref input_data = Args {
        x: WeldVec {
            data: &keys as *const i32,
            len: keys.len() as i64,
        },
        y: WeldVec {
            data: &vals as *const i32,
            len: vals.len() as i64,
        },
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<Entry> };
    let result = unsafe { (*data).clone() };

    let output_keys = [1, 2, 3];
    let output_vals = [4, 7, 1];

    assert_eq!(result.len, output_keys.len() as i64);
    for i in 0..(output_keys.len() as isize) {
        let entry = unsafe { (*result.data.offset(i)).clone() };
        // Check whether we find the entry anywhere in the expected outputs
        let mut success = false;
        for j in 0..(output_keys.len()) {
            if entry.k1 == output_keys[j] && entry.k2 == output_keys[j] &&
                    entry.v1 == output_vals[j] && entry.v2 == output_vals[j] as f32 {
                success = true;
            }
        }
        assert_eq!(success, true);
    }
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn simple_groupmerger() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]| tovec(result(for(zip(x,y), groupmerger[i32,i32],
                |b,i,e| merge(b, e))))";

    let conf = default_conf();
    let keys = [1, 2, 2, 3, 3, 1];
    let vals = [2, 3, 4, 1, 0, 2];
    let ref input_data = Args {
        x: WeldVec {
            data: &keys as *const i32,
            len: keys.len() as i64,
        },
        y: WeldVec {
            data: &vals as *const i32,
            len: vals.len() as i64,
        },
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<Pair<i32, WeldVec<i32>>> };
    let result = unsafe { (*data).clone() };
    let output: Vec<(i32, Vec<i32>)> = vec![(1, vec![2, 2]), (2, vec![3, 4]), (3, vec![1, 0])];

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

    assert_eq!(res, output);
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn complex_groupmerger_with_struct_key() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
        z: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32], z:vec[i32]|
                tovec(result(for(zip(x,y,z), groupmerger[{i32,i32}, i32],
                |b,i,e| merge(b, {{e.$0, e.$1}, e.$2}))))";

    let conf = default_conf();
    let keys1 = [1, 1, 2, 2, 3, 3, 3, 3];
    let keys2 = [1, 1, 2, 2, 3, 3, 4, 4];
    let vals = [2, 3, 4, 2, 1, 0, 3, 2];
    let ref input_data = Args {
        x: WeldVec {
            data: &keys1 as *const i32,
            len: keys1.len() as i64,
        },
        y: WeldVec {
            data: &keys2 as *const i32,
            len: keys2.len() as i64,
        },
        z: WeldVec {
            data: &vals as *const i32,
            len: vals.len() as i64,
        },
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data =
        unsafe { weld_value_data(ret_value) as *const WeldVec<Pair<Pair<i32, i32>, WeldVec<i32>>> };
    let result = unsafe { (*data).clone() };
    let output = vec![((1, 1), vec![2, 3]),
                      ((2, 2), vec![4, 2]),
                      ((3, 3), vec![1, 0]),
                      ((3, 4), vec![3, 2])];

    let mut res: Vec<((i32, i32), Vec<i32>)> = (0..result.len)
        .into_iter()
        .map(|x| {
            let key = unsafe { ((*result.data.offset(x as isize)).ele1).clone() };
            let val = unsafe { ((*result.data.offset(x as isize)).ele2).clone() };
            let vec: Vec<i32> = (0..val.len)
                .into_iter()
                .map(|y| unsafe { *val.data.offset(y as isize) })
                .collect();
            ((key.ele1, key.ele2), vec)
        })
        .collect();
    res.sort_by_key(|a| a.0);

    assert_eq!(res, output);
    unsafe { free_value_and_module(ret_value) };
}

/// Tests a the dictionary by merging multiple keys key multiple times into a dictionary.
/// `use_local` specifies whether to use the local-global adaptive dictionary or the purely global
/// dictionary.
fn simple_parallel_for_dictmerger_loop_helper(use_local: bool) {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Pair {
        ele1: i32,
        ele2: i32,
    }
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = format!("|x:vec[i32], y:vec[i32]| tovec(result(@(grain_size: 100)for(zip(x,y),
                dictmerger[i32,i32,+]({}L), |b,i,e| merge(b, e))))",
                if use_local { 100000000 } else { 0 });
    let conf = many_threads_conf();

    const DICT_SIZE: usize = 8192;
    const UNIQUE_KEYS: usize = 256;
    let mut keys = [0; DICT_SIZE];
    let mut vals = [0; DICT_SIZE];

    // Repeated keys will have their values summed.
    for i in 0..DICT_SIZE {
        keys[i] = (i % UNIQUE_KEYS) as i32;
        vals[i] = i as i32;
    }
    let ref input_data = Args {
        x: WeldVec {
            data: &keys as *const i32,
            len: DICT_SIZE as i64,
        },
        y: WeldVec {
            data: &vals as *const i32,
            len: DICT_SIZE as i64,
        },
    };

    let ret_value = compile_and_run(&code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<Pair> };
    let result = unsafe { (*data).clone() };

    assert_eq!(UNIQUE_KEYS as i64, result.len);

    // Build the expected output by summing the values.
    let mut expected = fnv::FnvHashMap::default();
    for i in 0..DICT_SIZE {
        let key = (i % UNIQUE_KEYS) as i32;
        match expected.entry(key) {
            Entry::Occupied(mut ent) => {
                *ent.get_mut() += i as i32;
            }
            Entry::Vacant(mut ent) => {
                ent.insert(i as i32);
            }
        }
    }

    for i in 0..(result.len as isize) {
        let key = unsafe { (*result.data.offset(i)).ele1 };
        let value = unsafe { (*result.data.offset(i)).ele2 };
        let expected_value = expected.get(&key).unwrap();
        assert_eq!(*expected_value, value);
    }
    assert_eq!(result.len, expected.len() as i64);
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn simple_parallel_for_dictmerger_loop_local() {
    simple_parallel_for_dictmerger_loop_helper(true);
}

#[test]
fn simple_parallel_for_dictmerger_loop_global() {
    simple_parallel_for_dictmerger_loop_helper(false);
}

#[test]
fn simple_dict_lookup() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32], y:vec[i32]| let a = result(for(zip(x,y), dictmerger[i32,i32,+],
                |b,i,e| merge(b, e))); lookup(a, 1)";
    let conf = default_conf();

    let keys = [1, 2, 2, 1, 3];
    let vals = [2, 3, 4, 2, 1];
    let ref input_data = Args {
        x: WeldVec {
            data: &keys as *const i32,
            len: keys.len() as i64,
        },
        y: WeldVec {
            data: &vals as *const i32,
            len: vals.len() as i64,
        },
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };

    let output = 4;
    assert_eq!(output, result);
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn string_dict_lookup() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = "|x:vec[i32]| let v = [\"abcdefghi\", \"abcdefghi\", \"abcdefghi\"];
                let d = result(for(zip(v,x), dictmerger[vec[i8],i32,+], |b,i,e| merge(b, e)));
                lookup(d, \"abcdefghi\")";
    let conf = default_conf();

    let input_vec = [1, 1, 1];
    let ref input_data = WeldVec {
        data: &input_vec as *const i32,
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { (*data).clone() };

    let output = 3;
    assert_eq!(output, result);
    unsafe { free_value_and_module(ret_value) };
}

#[test]
fn simple_dict_exists() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let keys = [1, 2, 2, 1, 3];
    let vals = [2, 3, 4, 2, 1];

    let code_true = "|x:vec[i32], y:vec[i32]| let a = result(for(zip(x,y), dictmerger[i32,i32,+],
                |b,i,e| merge(b, e))); keyexists(a, 1)";
    let code_false = "|x:vec[i32], y:vec[i32]| let a = result(for(zip(x,y),
                      dictmerger[i32,i32,+], |b,i,e| merge(b, e))); keyexists(a, 4)";
    let conf = default_conf();

    let ref input_data = Args {
        x: WeldVec {
            data: &keys as *const i32,
            len: keys.len() as i64,
        },
        y: WeldVec {
            data: &vals as *const i32,
            len: vals.len() as i64,
        },
    };

    let ret_value = compile_and_run(code_true, conf, input_data.clone());
    let data = unsafe { weld_value_data(ret_value) as *const bool };
    let result = unsafe { (*data).clone() };

    let output = true;
    assert_eq!(output, result);
    unsafe { free_value_and_module(ret_value) };

    let conf = default_conf();
    let ret_value = compile_and_run(code_false, conf, input_data.clone());
    let data = unsafe { weld_value_data(ret_value) as *const bool };
    let result = unsafe { (*data).clone() };

    let output = false;
    assert_eq!(output, result);
    unsafe { free_value_and_module(ret_value) };
}

