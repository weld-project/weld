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
use std::ffi::CStr;
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
use std::sync::RwLock;
use std::fmt;

const CACHE_BITS: i64 = 6;
const CACHE_LINE: u64 = (1 << (CACHE_BITS as u64));
const MASK: u64 = (!(CACHE_LINE - 1));

/// Hold memory information for a run.
#[derive(Clone, Debug)]
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

/// An errno set by the runtime.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd)]
pub enum WeldRuntimeErrno {
    Success = 0,
    CompileError,
    ArrayOutOfBounds,
    BadIteratorLength,
    MismatchedZipSize,
    OutOfMemory,
    Unknown,
    ErrnoMax,
}

impl fmt::Display for WeldRuntimeErrno {
    /// Just return the errno name.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

lazy_static! {
    // Maps a run to a list of pointers that must be freed for a given run of a module.
    // The key is a globally unique run ID.
    static ref ALLOCATIONS: Mutex<HashMap<i64, RunMemoryInfo>> = Mutex::new(HashMap::new());
    // Error codes for each active run. This is highly inefficient right now; writing a value
    // requires a lock across *all* runs, which is unnecessary.
    static ref WELD_ERRNOS: RwLock<HashMap<i64, WeldRuntimeErrno>> = RwLock::new(HashMap::new());
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

#[repr(C)]
pub struct work_t {
    data: *mut c_void,
    lower: i64,
    upper: i64,
    cur_idx: i64,
    full_task: i32,
    nest_idxs: *mut i64,
    nest_task_ids: *mut i64,
    nest_len: i32,
    task_id: i64,
    fp: extern "C" fn(*mut work_t),
    cont: *mut work_t,
    deps: i32,
    continued: i32,
}

#[repr(C)]
pub struct vec_piece {
    data: *mut c_void,
    size: i64,
    capacity: i64,
    nest_idxs: *mut i64,
    nest_task_ids: *mut i64,
    nest_len: i32,
}

#[repr(C)]
pub struct vec_output {
    data: *mut c_void,
    size: i64,
}

#[link(name = "par", kind = "static")]
extern "C" {
    pub fn my_id_public() -> i32;
    pub fn set_result(res: *mut c_void);
    pub fn get_result() -> *mut c_void;
    pub fn get_nworkers() -> i32;
    pub fn set_nworkers(n: i32);
    pub fn get_runid() -> i64;
    pub fn set_runid(rid: i64);
    pub fn pl_start_loop(w: *mut work_t,
                         body_data: *mut c_void,
                         cont_data: *mut c_void,
                         body: extern "C" fn(*mut work_t),
                         cont: extern "C" fn(*mut work_t),
                         lower: i64,
                         upper: i64,
                         grain_size: i32);
    pub fn execute(run: extern "C" fn(*mut work_t), data: *mut c_void);

    pub fn new_vb(elem_size: i64, starting_cap: i64) -> *mut c_void;
    pub fn new_piece(v: *mut c_void, w: *mut work_t);
    pub fn cur_piece(v: *mut c_void, my_id: i32) -> *mut vec_piece;
    pub fn result_vb(v: *mut c_void) -> vec_output;
    pub fn weld_abort_thread();
}

pub struct WeldError {
    errno: WeldRuntimeErrno,
    message: String,
}

pub struct WeldValue {
    data: *const c_void,
    run_id: Option<i64>,
}

impl WeldError {
    /// Returns a new WeldError where the message is just the errno name.
    fn new(errno: WeldRuntimeErrno) -> WeldError {
        WeldError {
            errno: errno,
            message: errno.to_string(),
        }
    }

    /// Returns a new WeldError with the given message.
    fn new_with_message(errno: WeldRuntimeErrno, message: String) -> WeldError {
        WeldError {
            errno: errno,
            message: message,
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
        *err = Box::into_raw(Box::new(WeldError::new(WeldRuntimeErrno::Success)));
        &mut **err
    };

    if let Err(e) = easy_ll::load_library("target/debug/libweld") {
        println!("load failed: {}", e.description());
    }
    let module = llvm::compile_program(&parser::parse_program(code).unwrap());
    if let Err(ref e) = module {
        err.errno = WeldRuntimeErrno::CompileError;
        err.message = e.description().to_string();
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

    let err = unsafe {
        *err = Box::into_raw(Box::new(WeldError::new(WeldRuntimeErrno::Success)));
        &mut **err
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

    let result = module.run(ptr) as *const c_void;
    let errno = weld_rt_get_errno(my_run_id);
    if errno != WeldRuntimeErrno::Success {
        weld_run_free(my_run_id);
        *err = WeldError::new(errno);
        return std::ptr::null_mut();
    }

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
pub extern "C" fn weld_error_code(err: *mut WeldError) -> libc::int32_t {
    let err = unsafe {
        if err.is_null() {
            return 1;
        }
        &mut *err
    };
    err.errno as libc::int32_t
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
    let mut ptr = std::ptr::null_mut();
    let mut do_allocation = true;
    {
        let mut guarded = ALLOCATIONS.lock().unwrap();
        let mem_info = guarded.entry(run_id).or_insert(RunMemoryInfo::new(DEFAULT_MEM));
        if mem_info.mem_allocated + (size as i64) > mem_info.mem_limit {
            weld_rt_set_errno(run_id, WeldRuntimeErrno::OutOfMemory);
            do_allocation = false;
        }

        if do_allocation {
            ptr = unsafe { malloc(size as usize) };
            mem_info.mem_allocated += size;
            mem_info.allocations.insert(ptr as u64, size as u64);
        }
    }
    // Release the lock before quitting by exiting the above scope.
    if !do_allocation {
        unsafe { weld_abort_thread() };
    }
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
    let mut ptr = std::ptr::null_mut();
    let mut do_allocation = true;
    {
        let mut guarded = ALLOCATIONS.lock().unwrap();
        if !guarded.contains_key(&run_id) {
            panic!("Unseen run {}", run_id);
        }
        let mem_info = guarded.entry(run_id).or_insert(RunMemoryInfo::new(DEFAULT_MEM));
        let old_size = *mem_info.allocations.get(&(data as u64)).unwrap() as i64;
        // Number of additional bytes we'll allocate.
        let plus_bytes = size - old_size;
        if mem_info.mem_allocated + (plus_bytes as i64) > mem_info.mem_limit {
            weld_rt_set_errno(run_id, WeldRuntimeErrno::OutOfMemory);
            do_allocation = false;
        }
        if do_allocation {
            ptr = unsafe { realloc(data, size as usize) };
            mem_info.mem_allocated += plus_bytes;
            mem_info.allocations.remove(&(data as u64));
            mem_info.allocations.insert(ptr as u64, size as u64);
        }
    }
    // Release the lock before quitting by exiting the above scope.
    if !do_allocation {
        unsafe { weld_abort_thread() };
    }
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

#[no_mangle]
/// Returns the errno for a run.
///
/// Worker threads exit if the global `weld_errno` variable is set to a non-zero value,
/// designating a runtime error.
pub extern "C" fn weld_rt_get_errno(run_id: libc::int64_t) -> WeldRuntimeErrno {
    let run_id = run_id as i64;
    let guarded = WELD_ERRNOS.read().unwrap();
    if !guarded.contains_key(&run_id) {
        // TODO(shoumik): Better behavior here.
        return WeldRuntimeErrno::Success;
    }
    *guarded.get(&run_id).unwrap()
}

#[no_mangle]
/// Sets an errno for the given run.
///
/// This signals all threads in the run to stop execution and return.
// TODO(shoumik): Taking a Rust enum as an argument here seems sketchy.
pub extern "C" fn weld_rt_set_errno(run_id: libc::int64_t, errno: WeldRuntimeErrno) {
    if errno > WeldRuntimeErrno::ErrnoMax {
        panic!("Unknown errno {} set", errno);
    }
    let run_id = run_id as i64;
    let mut guarded = WELD_ERRNOS.write().unwrap();
    let entry = guarded.entry(run_id).or_insert(errno);
    *entry = errno;
}

/// Frees all memory for a given run. Passing in -1 frees all memory from all runs.
///
/// This is used by internal tests to free memory, as well as by the `weld_value_free` function
/// to free a return value and all associated memory.
pub fn weld_run_free(run_id: i64) {
    // Create a scope for the lock.
    {
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
}

fn num_cache_blocks(size: i64) -> i64 {
    ((size + ((CACHE_LINE - 1) as i64)) >> CACHE_BITS) + 1
}

#[no_mangle]
pub extern "C" fn new_merger(runid: i64, size: i64, nworkers: i32) -> *mut c_void {
    let num_blocks = num_cache_blocks(size) as i64;
    let total_blocks = num_blocks * (nworkers as i64) * (CACHE_LINE as i64);
    let ptr = weld_rt_malloc(runid, total_blocks as libc::int64_t);
    for i in 0..nworkers {
        let merger_ptr = get_merger_at_index(ptr, size, i);
        unsafe { libc::memset(merger_ptr, 0 as libc::c_int, size as usize) };
    }
    ptr
}

#[no_mangle]
pub extern "C" fn get_merger_at_index(ptr: *mut c_void, size: i64, i: i32) -> *mut libc::c_void {
    let ptr = ptr as *mut libc::uint8_t;
    let ptr =
        unsafe { (((ptr.offset((CACHE_LINE - 1) as isize)) as u64) & MASK) } as *mut libc::uint8_t;
    let num_blocks = num_cache_blocks(size) as i64;
    let offset = num_blocks * (i as i64) * (CACHE_LINE as i64);
    let merger_ptr = unsafe { ptr.offset(offset as isize) } as *mut libc::c_void;
    merger_ptr
}

#[no_mangle]
pub extern "C" fn free_merger(runid: i64, ptr: *mut c_void) {
    weld_rt_free(runid, ptr);
}

#[test]
fn merger_fns() {
    let runid = 0;
    let nworkers = 4;
    let size = 4;
    let ptr = new_merger(runid, size, nworkers);
    let ptr_u64 = ptr as u64;
    let mut aligned_ptr = ptr_u64 >> CACHE_BITS;
    // In case allocated ptr is not already cache-aligned, need next cache-aligned
    // address.
    if ptr_u64 % CACHE_LINE != 0 {
        aligned_ptr += 1;
    }
    aligned_ptr = aligned_ptr << CACHE_BITS;
    for i in 0..nworkers {
        let offset = (num_cache_blocks(size) * (i as i64) * (CACHE_LINE as i64)) as isize;
        let merger_ptr = unsafe { (aligned_ptr as *mut libc::uint8_t).offset(offset) };
        unsafe { *(merger_ptr as *mut u32) = i as u32 };
    }
    for i in 0..nworkers {
        let merger_ptr = get_merger_at_index(ptr, size, i);
        let output = unsafe { *(merger_ptr as *mut u32) };
        // Read value previously stored, make sure they match.
        assert_eq!(output, i as u32);
    }
    free_merger(runid, ptr);
}

#[test]
fn test_num_cache_blocks() {
    assert_eq!(num_cache_blocks(4), 2);
    assert_eq!(num_cache_blocks(8), 2);
    assert_eq!(num_cache_blocks(64), 2);
    assert_eq!(num_cache_blocks(65), 3);
}

#[cfg(test)]
mod tests;
