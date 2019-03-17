//! Tests related to creating, running, and freeing Weld modules using the Weld API.
//!
//! This module uses the C API to simplify multi-threaded testing.

use weld;

use std::sync::Arc;
use std::thread;
use weld::*;

mod common;
use crate::common::*;

#[test]
fn multithreaded_module_run() {
    let code = "|v:vec[i32]| result(for(v, appender[i32], |b,i,e| merge(b,e)))";
    let conf = WeldConf::new();

    // Set up input data
    let len: usize = 10 * 1000 * 1000 + 1;
    let mut input_vec = vec![];
    for i in 0..len {
        input_vec.push(i as i32);
    }

    let input_vec = Arc::new(input_vec);
    let module = Arc::new(WeldModule::compile(code, &conf).unwrap());

    // Run several threads, each of which executes the module several times
    let mut threads = vec![];
    let num_threads = 8;
    let num_runs = 4;
    for _ in 0..num_threads {
        let module = Arc::clone(&module);
        let input_vec = Arc::clone(&input_vec);
        threads.push(thread::spawn(move || {
            for _ in 0..num_runs {
                let ref conf = WeldConf::new();
                let input_data = WeldVec::from(input_vec.as_ref());
                let ref input_value = WeldValue::new_from_data(&input_data as *const _ as Data);
                let ref mut context = WeldContext::new(&conf).unwrap();

                unsafe {
                    let ret_value = module.run(context, &input_value.clone()).unwrap();

                    // Check the result
                    let ret_data = ret_value.data() as *const WeldVec<i32>;
                    let result = (*ret_data).clone();
                    assert_eq!(result.len, len as i64);
                    for i in 0..len {
                        assert_eq!(i as i32, *result.data.offset(i as isize));
                    }
                }
            }
        }))
    }

    // Wait for everything to finish, and then clean up
    for t in threads {
        t.join().unwrap();
    }
}
