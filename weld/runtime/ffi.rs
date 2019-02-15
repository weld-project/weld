//! Foreign function interface for the runtime.
//!
//! These are functions with a C-based ABI and calling convention. Generated Weld code will call
//! into these functions.

use super::*;

pub type WeldRuntimeContextRef = *mut WeldRuntimeContext;

#[no_mangle]
/// Initialize the runtime.
///
/// This call currently circumvents an awkward problem with binaries using Rust, where the FFI
/// below used by the Weld runtime is compiled out.
pub unsafe extern "C" fn weld_init() {
    initialize();
}

#[no_mangle]
/// Returns a new runtime handle.
pub unsafe extern "C" fn weld_runst_init(nworkers: int32_t, memlimit: int64_t) -> WeldRuntimeContextRef {
    Box::into_raw(Box::new(WeldRuntimeContext::new(nworkers, memlimit)))
}

#[no_mangle]
/// Delete a run handle and its allocated memory.
pub unsafe extern "C" fn weld_runst_release(run: WeldRuntimeContextRef) {
    Box::from_raw(run);
}

#[no_mangle]
/// Allocate memory within the provided context.
pub unsafe extern "C" fn weld_runst_malloc(run: WeldRuntimeContextRef, size: int64_t) -> Ptr {
    let run = &mut *run;
    run.malloc(size)
}

#[no_mangle]
/// Reallocate memory within the provided context.
///
/// This function has semantics equal to the `realloc` function.
pub unsafe extern "C" fn weld_runst_realloc(run: WeldRuntimeContextRef,
                                     ptr: Ptr,
                                     newsize: int64_t) -> Ptr {
    let run = &mut *run;
    run.realloc(ptr, newsize)
}

#[no_mangle]
/// Free memory allocated in this context.
pub unsafe extern "C" fn weld_runst_free(run: WeldRuntimeContextRef, ptr: Ptr) {
    let run = &mut *run;
    run.free(ptr)
}

#[no_mangle]
/// Set the result pointer.
pub unsafe extern "C" fn weld_runst_set_result(run: WeldRuntimeContextRef, ptr: Ptr) {
    let run = &mut *run;
    run.set_result(ptr)
}

#[no_mangle]
/// Set the errno value.
pub unsafe extern "C" fn weld_runst_set_errno(run: WeldRuntimeContextRef,
                                              errno: WeldRuntimeErrno) {
    let run = &mut *run;
    run.set_errno(errno)
}

#[no_mangle]
/// Get the errno value.
pub unsafe extern "C" fn weld_runst_get_errno(run: WeldRuntimeContextRef) -> WeldRuntimeErrno {
    let run = &mut *run;
    run.errno()
}

#[no_mangle]
/// Get the result pointer.
pub unsafe extern "C" fn weld_runst_get_result(run: WeldRuntimeContextRef) -> Ptr {
    let run = &mut *run;
    run.result()
}

#[no_mangle]
/// Check whether cond is 0 (assertion fails).
pub unsafe extern "C" fn weld_runst_assert(run: WeldRuntimeContextRef,
                                           cond: u8) -> u8 {
    let run = &mut *run;
    if cond == 0 {
        // Doesn't return.
        run.set_errno(WeldRuntimeErrno::AssertionError);
        unreachable!()
    } else {
        1
    }
}

#[no_mangle]
/// Print a value from generated code.
pub unsafe extern "C" fn weld_runst_print(_run: WeldRuntimeContextRef, string: *const c_char) {
    let string = CStr::from_ptr(string).to_str().unwrap();
    println!("{} ", string);
}
