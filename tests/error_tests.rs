//! Tests for runtime errors that Weld can throw.

extern crate libc;
extern crate weld;

use weld::runtime::WeldRuntimeErrno;

mod common;
use common::*;

#[test]
#[should_panic]
fn iters_outofbounds_error_test() {
    let code = "|x:vec[i32]| result(for(iter(x,0L,20000L,1L), merger[i32,+], |b,i,e| merge(b,e)))";
    let ref conf = many_threads_conf();

    let input_vec = vec![4; 1000 as usize];
    let ref input_data = WeldVec::from(&input_vec);

    let err_value = compile_and_run_error(code, conf, input_data);
}

#[test]
#[should_panic]
fn outofmemory_error_test() {
    let code = "|x:vec[i32]| result(for(x, vecmerger[i32,+](x), |b,i,e| merge(b,{i,e+1})))";
    let ref mut conf = default_conf();

    // Set the memory to something small.
    conf.set("weld.memory.limit", "50000");

    let x = vec![4; 50000 / 4 as usize];
    let ref input_data = WeldVec::from(&x);

    let err_value = compile_and_run_error(code, conf, input_data);
    assert_eq!(err_value.code(), WeldRuntimeErrno::OutOfMemory);
}
