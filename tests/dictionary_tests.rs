//! Tests that use dictionaries. This includes the DictMerger and GroupBuilder
//! builder types, as well as operations on read-only dictionaries (e.g., lookups).

use fnv;

use std::collections::hash_map::Entry;

mod common;
use crate::common::*;

#[repr(C)]
struct I32KeyValArgs {
    x: WeldVec<i32>,
    y: WeldVec<i32>,
}

#[test]
fn simple_for_dictmerger_loop() {
    let code = "|x:vec[i32], y:vec[i32]| tovec(result(for(zip(x,y), dictmerger[i32,i32,+],
                |b,i,e| merge(b, e))))";
    let ref conf = default_conf();
    let keys = vec![1, 2, 2, 1, 3];
    let vals = vec![2, 3, 4, 2, 1];
    let ref input_data = I32KeyValArgs {
        x: WeldVec::from(&keys),
        y: WeldVec::from(&vals),
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<Pair<_, _>>;
    let result = unsafe { (*data).clone() };

    let output_keys = vec![1, 2, 3];
    let output_vals = vec![4, 7, 1];

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
        v2: f32,
    }

    let code = "|x:vec[i32], y:vec[i32]|
                    tovec(result(for(
                        zip(x,y),
                        dictmerger[{i32,i32},{i32,f32},+],
                        |b,i,e| merge(b, {{e.$0, e.$0}, {e.$1, f32(e.$1)}}))))";
    let ref conf = default_conf();
    let keys = vec![1, 2, 2, 1, 3];
    let vals = vec![2, 3, 4, 2, 1];
    let ref input_data = I32KeyValArgs {
        x: WeldVec::from(&keys),
        y: WeldVec::from(&vals),
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<Entry>;
    let result = unsafe { (*data).clone() };

    let output_keys = vec![1, 2, 3];
    let output_vals = vec![4, 7, 1];

    assert_eq!(result.len, output_keys.len() as i64);
    for i in 0..(output_keys.len() as isize) {
        let entry = unsafe { (*result.data.offset(i)).clone() };
        // Check whether we find the entry anywhere in the expected outputs
        let mut success = false;
        for j in 0..(output_keys.len()) {
            if entry.k1 == output_keys[j]
                && entry.k2 == output_keys[j]
                && entry.v1 == output_vals[j]
                && entry.v2 == output_vals[j] as f32
            {
                success = true;
            }
        }
        assert_eq!(success, true);
    }
}

#[test]
fn simple_groupmerger() {
    let code = "|x:vec[i32], y:vec[i32]| tovec(result(for(zip(x,y), groupmerger[i32,i32],
                |b,i,e| merge(b, e))))";

    let ref mut conf = default_conf();

    const DICT_SIZE: usize = 8192;
    const UNIQUE_KEYS: usize = 256;
    let mut keys = vec![0; DICT_SIZE];
    let mut vals = vec![0; DICT_SIZE];

    let mut output: Vec<(i32, Vec<i32>)> = vec![];

    // Repeated keys will have their values summed.
    for i in 0..DICT_SIZE {
        keys[i] = (i % UNIQUE_KEYS) as i32;
        vals[i] = i as i32;

        if output.len() < UNIQUE_KEYS {
            output.push((keys[i], vec![vals[i]]));
        } else {
            output.get_mut(keys[i] as usize).unwrap().1.push(vals[i]);
        }
    }

    let ref input_data = I32KeyValArgs {
        x: WeldVec::from(&keys),
        y: WeldVec::from(&vals),
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<Pair<i32, WeldVec<i32>>>;
    let result = unsafe { (*data).clone() };

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

    let ref conf = default_conf();
    let keys1 = vec![1, 1, 2, 2, 3, 3, 3, 3];
    let keys2 = vec![1, 1, 2, 2, 3, 3, 4, 4];
    let vals = vec![2, 3, 4, 2, 1, 0, 3, 2];

    let ref input_data = Args {
        x: WeldVec::from(&keys1),
        y: WeldVec::from(&keys2),
        z: WeldVec::from(&vals),
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<Pair<Pair<i32, i32>, WeldVec<i32>>>;
    let result = unsafe { (*data).clone() };
    let output = vec![
        ((1, 1), vec![2, 3]),
        ((2, 2), vec![4, 2]),
        ((3, 3), vec![1, 0]),
        ((3, 4), vec![3, 2]),
    ];

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
}

/// Larger dictmerger test with repeated keys
#[test]
fn dictmerger_repeated_keys() {
    let code = "|x:vec[i32], y:vec[i32]| tovec(result(for(zip(x,y),
                dictmerger[i32,i32,+], |b,i,e| merge(b, e))))";
    let ref mut conf = many_threads_conf();

    const DICT_SIZE: usize = 8192;
    const UNIQUE_KEYS: usize = 256;
    let mut keys = vec![0; DICT_SIZE];
    let mut vals = vec![0; DICT_SIZE];

    // Repeated keys will have their values summed.
    for i in 0..DICT_SIZE {
        keys[i] = (i % UNIQUE_KEYS) as i32;
        vals[i] = i as i32;
    }

    let ref input_data = I32KeyValArgs {
        x: WeldVec::from(&keys),
        y: WeldVec::from(&vals),
    };

    let ret_value = compile_and_run(&code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<Pair<_, _>>;
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
            Entry::Vacant(ent) => {
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
}

#[test]
fn simple_dict_lookup() {
    let code = "|x:vec[i32], y:vec[i32]| let a = result(for(zip(x,y), dictmerger[i32,i32,+],
                |b,i,e| merge(b, e))); lookup(a, 1)";
    let ref conf = default_conf();

    let keys = vec![1, 2, 2, 1, 3];
    let vals = vec![2, 3, 4, 2, 1];

    let ref input_data = I32KeyValArgs {
        x: WeldVec::from(&keys),
        y: WeldVec::from(&vals),
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };

    let output = 4;
    assert_eq!(output, result);
}

#[test]
fn simple_dict_optlookup() {
    let code = "|x:vec[i32], y:vec[i32]| let a = result(for(zip(x,y), dictmerger[i32,i32,+],
                |b,i,e| merge(b, e))); {optlookup(a, 1), optlookup(a, 5)}";
    let ref conf = default_conf();

    let keys = vec![1, 2, 2, 1, 3];
    let vals = vec![2, 3, 4, 2, 1];

    #[derive(Clone)]
    struct OptLookupResult {
        flag: u8,
        value: i32,
    };

    #[derive(Clone)]
    struct Output {
        a: OptLookupResult,
        b: OptLookupResult,
    }

    let ref input_data = I32KeyValArgs {
        x: WeldVec::from(&keys),
        y: WeldVec::from(&vals),
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const Output;
    let result = unsafe { (*data).clone() };

    assert!(result.a.flag != 0);
    assert!(result.a.value != 4);

    assert!(result.b.flag == 0);
}

#[test]
fn string_dict_lookup() {
    let code = "|x:vec[i32]| let v = [\"abcdefghi\", \"abcdefghi\", \"abcdefghi\"];
                let d = result(for(zip(v,x), dictmerger[vec[i8],i32,+], |b,i,e| merge(b, e)));
                lookup(d, \"abcdefghi\")";
    let ref conf = default_conf();

    let input_vec = vec![1, 1, 1];
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };

    let output = 3;
    assert_eq!(output, result);
}

#[test]
fn simple_dict_exists() {
    let keys = vec![1, 2, 2, 1, 3];
    let vals = vec![2, 3, 4, 2, 1];

    let code_true = "|x:vec[i32], y:vec[i32]|
    let a = result(for(zip(x,y),
                dictmerger[i32,i32,+],
                |b,i,e| merge(b, e))); 
    keyexists(a, 1) && keyexists(a, 2) && keyexists(a, 3)";
    let code_false = "|x:vec[i32], y:vec[i32]| let a = result(for(zip(x,y),
                      dictmerger[i32,i32,+], |b,i,e| merge(b, e))); keyexists(a, 4)";
    let ref conf = default_conf();

    let ref input_data = I32KeyValArgs {
        x: WeldVec::from(&keys),
        y: WeldVec::from(&vals),
    };

    let ret_value = compile_and_run(code_true, conf, input_data.clone());
    let data = ret_value.data() as *const bool;
    let result = unsafe { (*data).clone() };

    let output = true;
    assert_eq!(output, result);

    let ref conf = default_conf();
    let ret_value = compile_and_run(code_false, conf, input_data.clone());
    let data = ret_value.data() as *const bool;
    let result = unsafe { (*data).clone() };

    let output = false;
    assert_eq!(output, result);
}
