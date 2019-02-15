//! Tests for runtime errors that Weld can throw.

extern crate libc;
extern crate weld;

use weld::runtime::WeldRuntimeErrno;

mod common;
use common::*;

#[test]
#[should_panic] // XXX The new runtime throws panics for these currently.
fn iters_outofbounds_error_test() {
    let code = "|x:vec[i32]| result(for(iter(x,0L,20000L,1L), merger[i32,+], |b,i,e| merge(b,e)))";
    let ref mut conf = many_threads_conf();

    // Need this to get errors! Awkwardly, this test will segfault without it.
    conf.set("weld.compile.enableBoundsChecks", "true");

    let input_vec = vec![4; 1000 as usize];
    let ref input_data = WeldVec::from(&input_vec);

    let err_value = compile_and_run_error(code, conf, input_data);
    assert_eq!(err_value.code(), WeldRuntimeErrno::ArrayOutOfBounds);
}

#[test]
#[should_panic] // XXX The new runtime throws panics for these currently.
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

#[test]
fn assert_pass_test() {
    let code = "|x: i32| assert(x == 0)";
    let ref mut conf = default_conf();

    let ref input_data: i32 = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i8;
    let result = unsafe { *data };
    assert_eq!(result, 1);
}

#[test]
#[should_panic]
// XXX The new runtime throws panics for these currently.
fn assert_fail_test() {
    let code = "|x: i32| assert(x == 1)";
    let ref mut conf = default_conf();

    let ref input_data: i32 = 0;

    let err_value = compile_and_run_error(code, conf, input_data);
    assert_eq!(err_value.code(), WeldRuntimeErrno::AssertionError);
}
