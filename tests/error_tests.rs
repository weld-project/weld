//! Tests for runtime errors that Weld can throw.

extern crate libc;
extern crate weld;

use weld::common::WeldRuntimeErrno;
use weld::*;

use std::ffi::CString;

use libc::c_char;

mod common;
use common::*;

#[test]
fn iters_outofbounds_error_test() {
    let code = "|x:vec[i32]| result(for(iter(x,0L,20000L,1L), appender, |b,i,e| merge(b,e+1)))";
    let conf = many_threads_conf();

    let input_vec = vec![4; 1000 as usize];
    let ref input_data = WeldVec::from(&input_vec);

    let err_value = compile_and_run_error(code, conf, input_data);
    assert_eq!(
        unsafe { weld_error_code(err_value) },
        WeldRuntimeErrno::BadIteratorLength
    );
    unsafe { weld_error_free(err_value) };
}

#[test]
fn outofmemory_error_test() {
    let code = "|x:vec[i32]| result(for(x, vecmerger[i32,+](x), |b,i,e| merge(b,{i,e+1})))";
    let conf = default_conf();

    // Set the memory to something small.
    let key = CString::new("weld.memory.limit").unwrap().into_raw() as *const c_char;
    let value = CString::new("50000").unwrap().into_raw() as *const c_char;
    unsafe { weld_conf_set(conf, key, value) };

    let x = vec![4; 50000 / 4 as usize];
    let ref input_data = WeldVec::from(&x);

    let err_value = compile_and_run_error(code, conf, input_data);
    assert_eq!(
        unsafe { weld_error_code(err_value) },
        WeldRuntimeErrno::OutOfMemory
    );
    unsafe { weld_error_free(err_value) };
}
