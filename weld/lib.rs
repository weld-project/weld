// Disable dead code macros in "build" mode but keep them on in "test" builds so that we don't
// get spurious warnings for functions that are currently only used in tests. This is useful for
// development but can be removed later.
#![cfg_attr(not(test), allow(dead_code))]

#[macro_use]
extern crate lazy_static;
extern crate regex;
extern crate easy_ll;
extern crate libc;

use std::collections::HashMap;
use std::error::Error;
use libc::{c_char, c_void};
use std::ffi::{CString, CStr};
use std::mem;

/// Utility macro to create an Err result with a WeldError from a format string.
macro_rules! weld_err {
    ( $($arg:tt)* ) => ({
        ::std::result::Result::Err($crate::error::WeldError::new(format!($($arg)*)))
    })
}

// TODO: Not all of these should be public
pub mod ast;
pub mod code_builder;
pub mod error;
pub mod llvm;
pub mod macro_processor;
pub mod parser;
pub mod partial_types;
pub mod pretty_print;
pub mod program;
pub mod sir;
pub mod tokenizer;
pub mod transforms;
pub mod type_inference;
pub mod util;

use std::sync::Mutex;

/// Hold memory information for a run.
struct RunMemoryInfo {
    // Maps addresses to allocation sizes.
    allocations: HashMap<u64, u64>,
    // The memory limit (max amount of live heap allocated memory) for this run.
    mem_limit: i64,
    // The total bytes allocated for this run.
    mem_allocated: i64,
}

impl RunMemoryInfo {
    fn new(limit: i64) -> RunMemoryInfo {
        RunMemoryInfo {
            allocations: HashMap::new(),
            mem_limit: limit,
            mem_allocated: 0,
        }
    }
}

lazy_static! {
    // Maps a run to a list of pointers that must be freed for a given run of a module.
    // The key is a globally unique run ID.
    static ref ALLOCATIONS: Mutex<HashMap<i64, RunMemoryInfo>> = Mutex::new(HashMap::new());
    static ref RUN_ID_NEXT: Mutex<i64> = Mutex::new(0);
}

// Default memory size of 1GB.
const DEFAULT_MEM: i64 = 1000000000;

// C Functions used by the Weld runtime.
extern "C" {
    pub fn malloc(size: libc::size_t) -> *mut c_void;
    pub fn realloc(ptr: *mut c_void, size: libc::size_t) -> *mut c_void;
    pub fn free(ptr: *mut c_void);
}

pub struct WeldError {
    code: i32,
    message: CString,
}

pub struct WeldValue {
    data: *const c_void,
    run_id: Option<i64>,
}

impl WeldError {
    fn new(code: i32, message: &str) -> WeldError {
        WeldError {
            code: code,
            message: CString::new(message).unwrap(),
        }
    }
}

#[no_mangle]
/// Returns a new Weld value.
///
/// The value returned by this method is *not* owned by the runtime, so any data this value refers
/// to must be managed by the caller.
pub extern "C" fn weld_value_new(data: *const c_void) -> *mut WeldValue {
    Box::into_raw(Box::new(WeldValue {
        data: data,
        run_id: None,
    }))
}

#[no_mangle]
/// Returns the Run ID of the value if the value is owned by the Weld runtime, or -1 if the caller
/// owns the value.
pub extern "C" fn weld_value_run(obj: *const WeldValue) -> i64 {
    let obj = unsafe {
        assert!(!obj.is_null());
        &*obj
    };
    if let Some(rid) = obj.run_id {
        rid as i64
    } else {
        -1
    }
}

#[no_mangle]
/// Returns a pointer to the data wrapped by the given Weld value.
pub extern "C" fn weld_value_data(obj: *const WeldValue) -> *const c_void {
    let obj = unsafe {
        assert!(!obj.is_null());
        &*obj
    };
    obj.data
}

#[no_mangle]
/// Frees a Weld value.
///
/// All Weld values must be freed using this call.
/// Weld values which are owned by the runtime also free the data they contain.
/// Weld values which are not owned by the runtime only free the structure used
/// to wrap the data; the actual data itself is owned by the caller.
pub extern "C" fn weld_value_free(obj: *mut WeldValue) {
    let value = unsafe {
        if obj.is_null() {
            return;
        } else {
            &mut *obj
        }
    };

    if let Some(run_id) = value.run_id {
        // Free all the memory associated with the run.
        weld_run_free(run_id);
    }
    mem::drop(obj);
}

#[no_mangle]
/// Given some Weld code and a configuration, returns a runnable Weld module.
pub extern "C" fn weld_module_compile(code: *const c_char,
                                      conf: *const c_char,
                                      err: *mut *mut WeldError)
                                      -> *mut easy_ll::CompiledModule {
    let code = unsafe {
        assert!(!code.is_null());
        CStr::from_ptr(code)
    };
    let code = code.to_str().unwrap().trim();

    let conf = unsafe {
        assert!(!conf.is_null());
        CStr::from_ptr(conf)
    };
    let conf = conf.to_str().unwrap();

    let err = unsafe {
        *err = Box::into_raw(Box::new(WeldError::new(0, "Success")));
        &mut **err
    };

    let module = llvm::compile_program(&parser::parse_program(code).unwrap());
    if let Err(ref e) = module {
        err.code = 1;
        err.message = CString::new(e.description()).unwrap();
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(module.unwrap()))
}

#[no_mangle]
/// Runs a module.
///
/// The module may write a value into the provided error pointer.
pub extern "C" fn weld_module_run(module: *mut easy_ll::CompiledModule,
                                  arg: *const WeldValue,
                                  err: *mut *mut WeldError)
                                  -> *mut WeldValue {
    let module = unsafe {
        assert!(!module.is_null());
        &mut *module
    };

    let arg = unsafe {
        assert!(!arg.is_null());
        &*arg
    };

    // TODO(shoumik): Set the memory limit correctly (config?)
    let mem_limit = DEFAULT_MEM;
    // TODO(shoumik): Set the number of threads correctly
    let threads = 1;
    let my_run_id;

    // Put this in it's own scope so the mutexes are unlocked.
    {
        let mut guarded = ALLOCATIONS.lock().unwrap();
        let mut run_id = RUN_ID_NEXT.lock().unwrap();
        my_run_id = run_id.clone();
        *run_id += 1;
        guarded.insert(my_run_id, RunMemoryInfo::new(mem_limit));
    }

    let input = Box::new(llvm::WeldInputArgs {
        input: arg.data as i64,
        nworkers: threads,
        run_id: my_run_id,
    });
    let ptr = Box::into_raw(input) as i64;

    // TODO(shoumik): Error reporting - at least basic crashes
    let result = module.run(ptr) as *const c_void;
    Box::into_raw(Box::new(WeldValue {
        data: result,
        run_id: Some(my_run_id),
    }))
}

#[no_mangle]
/// Frees a module.
///
/// Freeing a module does not free the memory it may have allocated. Values returned by the module
/// must be freed explicitly using `weld_value_free`.
pub extern "C" fn weld_module_free(ptr: *mut easy_ll::CompiledModule) {
    if ptr.is_null() {
        return;
    }
    unsafe { Box::from_raw(ptr) };
}

#[no_mangle]
/// Returns an error code for a Weld error object.
pub extern "C" fn weld_error_code(err: *mut WeldError) -> i32 {
    let err = unsafe {
        if err.is_null() {
            return 1;
        }
        &mut *err
    };
    err.code as i32
}

#[no_mangle]
/// Returns a C pointer representing a Weld error message.
pub extern "C" fn weld_error_message(err: *mut WeldError) -> *const c_char {
    let err = unsafe {
        if err.is_null() {
            return std::ptr::null();
        }
        &mut *err
    };
    err.message.as_ptr() as *const c_char
}

#[no_mangle]
/// Frees a Weld error object.
pub extern "C" fn weld_error_free(err: *mut WeldError) {
    if err.is_null() {
        return;
    }
    unsafe { Box::from_raw(err) };
}

#[no_mangle]
/// Weld runtime wrapper around malloc.
///
/// This function tracks the pointer returned by malloc in an allocation list. The allocation list
/// is _per run_, where the run is tracked via the `run_id` parameter.
///
/// This should never be called directly; it is meant to be used by the runtime directly.
pub extern "C" fn weld_rt_malloc(run_id: libc::int64_t, size: libc::int64_t) -> *mut c_void {
    let run_id = run_id as i64;
    let mut guarded = ALLOCATIONS.lock().unwrap();
    if !guarded.contains_key(&run_id) {}
    let mem_info = guarded.entry(run_id).or_insert(RunMemoryInfo::new(DEFAULT_MEM));
    if mem_info.mem_allocated + (size as i64) > mem_info.mem_limit {
        println!("weld_rt_malloc: Exceeded memory limit: {}B > {}B",
                 mem_info.mem_allocated,
                 mem_info.mem_limit);
        return std::ptr::null_mut();
    }
    let ptr = unsafe { malloc(size as usize) };
    mem_info.mem_allocated += size;
    mem_info.allocations.insert(ptr as u64, size as u64);
    ptr
}

#[no_mangle]
/// Weld Runtime wrapper around realloc.
///
/// This function will potentially replace an existing pointer in the allocation list with a new
/// one, if realloc returns a new address.
///
/// This should never be called directly; it is meant to be used by the runtime directly.
pub extern "C" fn weld_rt_realloc(run_id: libc::int64_t,

                                  data: *mut c_void,
                                  size: libc::int64_t)
                                  -> *mut c_void {
    let run_id = run_id as i64;
    let mut guarded = ALLOCATIONS.lock().unwrap();
    if !guarded.contains_key(&run_id) {
        panic!("Unseen run {}", run_id);
    }
    let mem_info = guarded.entry(run_id).or_insert(RunMemoryInfo::new(DEFAULT_MEM));
    let old_size = *mem_info.allocations.get(&(data as u64)).unwrap() as i64;
    // Number of additional bytes we'll allocate.
    let plus_bytes = size - old_size;
    if mem_info.mem_allocated + (plus_bytes as i64) > mem_info.mem_limit {
        println!("weld_rt_realloc: Exceeded memory limit: {}B",
                 mem_info.mem_limit);
        // TODO(shoumik): Do something better here.
        return std::ptr::null_mut();
    }
    let ptr = unsafe { realloc(data, size as usize) };
    mem_info.mem_allocated += plus_bytes;
    mem_info.allocations.remove(&(data as u64));
    mem_info.allocations.insert(ptr as u64, size as u64);
    ptr
}

#[no_mangle]
/// Weld Runtime wrapper around free.
///
/// This function removes the pointer from the list of allocated pointers.
///
/// This should never be called directly; it is meant to be used by the runtime directly.
pub extern "C" fn weld_rt_free(run_id: libc::int64_t, data: *mut c_void) {
    let run_id = run_id as i64;
    let mut guarded = ALLOCATIONS.lock().unwrap();
    if !guarded.contains_key(&run_id) {
        panic!("Unseen run {}", run_id);
    }
    let mem_info = guarded.entry(run_id).or_insert(RunMemoryInfo::new(DEFAULT_MEM));
    let bytes_freed = *mem_info.allocations.get(&(data as u64)).unwrap();
    unsafe { free(data) };
    mem_info.mem_allocated -= bytes_freed as i64;
    mem_info.allocations.remove(&(data as u64));

}

/// Frees all memory for a given run. Passing in -1 frees all memory from all runs.
///
/// This is used by internal tests to free memory, as well as by the `weld_value_free` function
/// to free a return value and all associated memory.
pub fn weld_run_free(run_id: i64) {
    let mut guarded = ALLOCATIONS.lock().unwrap();
    if run_id == -1 {
        // Free all memory from all runs.
        let keys: Vec<_> = guarded.keys().map(|i| *i).collect();
        for key in keys {
            {
                let mem_info = guarded.entry(run_id).or_insert(RunMemoryInfo::new(DEFAULT_MEM));
                for entry in mem_info.allocations.iter() {
                    unsafe { free(*entry.0 as *mut c_void) };
                }
            }
            guarded.remove(&key);
        }
    } else {
        if !guarded.contains_key(&run_id) {
            return;
        }
        {
            let mem_info = guarded.entry(run_id).or_insert(RunMemoryInfo::new(DEFAULT_MEM));
            for entry in mem_info.allocations.iter() {
                unsafe { free(*entry.0 as *mut c_void) };
            }
        }
        guarded.remove(&run_id);
    }
}

#[cfg(test)]
mod tests;
