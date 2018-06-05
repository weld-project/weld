//! Tests related to creating, running, and freeing Weld modules using the Weld API.
//!
//! This module uses the C API to simplify multi-threaded testing.

extern crate libc;
extern crate weld;

use weld::runtime::WeldRuntimeErrno;
use weld::ffi::*;

use std::ffi::CString;
use std::thread;

use libc::{c_char, c_void};

mod common;
use common::*;

/// A wrapper struct to allow passing pointers across threads (they aren't Send/Sync by default).
/// The default #[derive(Copy,Clone)] does not work here unless T has Copy/Clone, so we also
/// implement those traits manually.
struct UnsafePtr<T>(*mut T);
unsafe impl<T> Send for UnsafePtr<T> {}
unsafe impl<T> Sync for UnsafePtr<T> {}
impl<T> Clone for UnsafePtr<T> {
    fn clone(&self) -> UnsafePtr<T> {
        UnsafePtr(self.0)
    }
}
impl<T> Copy for UnsafePtr<T> {}

#[test]
fn multithreaded_module_run() {
    let code = "|v:vec[i32]| result(for(v, appender[i32], |b,i,e| merge(b,e)))";

    let conf_raw = weld_conf_new();
    let conf = UnsafePtr(conf_raw);

    // Set up input data
    let len: usize = 10 * 1000 * 1000 + 1;
    let mut input_vec = vec![];
    for i in 0..len {
        input_vec.push(i as i32);
    }
    let input_data = WeldVec::from(&input_vec);

    unsafe {
        // Compile the module
        let code = CString::new(code).unwrap();
        let input_value = UnsafePtr(weld_value_new(&input_data as *const _ as *const c_void));
        let err = weld_error_new();
        let module = UnsafePtr(weld_module_compile(
            code.into_raw() as *const c_char,
            conf.0,
            err,
        ));
        assert_eq!(weld_error_code(err), WeldRuntimeErrno::Success);

        // Run several threads, each of which executes the module several times
        let mut threads = vec![];
        let num_threads = 8;
        let num_runs = 4;
        for _ in 0..num_threads {
            threads.push(thread::spawn(move || {
                for _ in 0..num_runs {
                    // Run the module
                    let err = weld_error_new();
                    let ret_value = weld_module_run(module.0, conf.0, input_value.0, err);
                    assert_eq!(weld_error_code(err), WeldRuntimeErrno::Success);

                    // Check the result
                    let ret_data = weld_value_data(ret_value) as *const WeldVec<i32>;
                    let result = (*ret_data).clone();
                    assert_eq!(result.len, len as i64);
                    for i in 0..len {
                        assert_eq!(i as i32, *result.data.offset(i as isize));
                    }
                    weld_value_free(ret_value);
                }
            }))
        }

        // Wait for everything to finish, and then clean up
        for t in threads {
            t.join().unwrap();
        }

        weld_module_free(module.0);
        weld_conf_free(conf_raw);
    }
}
