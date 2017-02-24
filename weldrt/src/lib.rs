// Disable dead code macros in "build" mode but keep them on in "test" builds so that we don't
// get spurious warnings for functions that are currently only used in tests. This is useful for
// development but can be removed later.
#![cfg_attr(not(test), allow(dead_code))]

#[macro_use]
extern crate lazy_static;
extern crate libc;

extern crate weld_common;

use weld_common::WeldRuntimeErrno;

use std::collections::HashMap;
use libc::c_void;

use std::sync::Mutex;
use std::sync::RwLock;

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

lazy_static! {
    // Maps a run to a list of pointers that must be freed for a given run of a module.
    // The key is a globally unique run ID.
    static ref ALLOCATIONS: Mutex<HashMap<i64, RunMemoryInfo>> = Mutex::new(HashMap::new());
    // Error codes for each active run. This is highly inefficient right now; writing a value
    // requires a lock across *all* runs, which is unnecessary.
    static ref WELD_ERRNOS: RwLock<HashMap<i64, WeldRuntimeErrno>> = RwLock::new(HashMap::new());
    // The next run ID to assign.
    static ref RUN_ID_NEXT: Mutex<i64> = Mutex::new(0);
}

// Cache sizes used to align mergers to a cache line.
const CACHE_BITS: i64 = 6;
const CACHE_LINE: u64 = (1 << (CACHE_BITS as u64));
const MASK: u64 = (!(CACHE_LINE - 1));

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

// C Functions used by the Weld runtime.
extern "C" {
    pub fn malloc(size: libc::size_t) -> *mut c_void;
    pub fn realloc(ptr: *mut c_void, size: libc::size_t) -> *mut c_void;
    pub fn free(ptr: *mut c_void);
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

#[no_mangle]
/// Intialize a new run.
///
/// This function *must* be called before any other `weld_rt` functions.
pub extern "C" fn weld_rt_init(mem_limit: i64) {
    let my_run_id: i64;
    // Put this in it's own scope so the mutexes are unlocked.
    {
        let mut guarded = ALLOCATIONS.lock().unwrap();
        let mut run_id = RUN_ID_NEXT.lock().unwrap();
        my_run_id = run_id.clone();
        *run_id += 1;
        guarded.insert(my_run_id, RunMemoryInfo::new(mem_limit));
    }
    unsafe { set_runid(my_run_id) };
    weld_rt_set_errno(my_run_id, WeldRuntimeErrno::Success);
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
        if let Some(mem_info) = guarded.get_mut(&run_id) {
            if mem_info.mem_allocated + (size as i64) > mem_info.mem_limit {
                weld_rt_set_errno(run_id, WeldRuntimeErrno::OutOfMemory);
                do_allocation = false;
            }

            if do_allocation {
                ptr = unsafe { malloc(size as usize) };
                mem_info.mem_allocated += size;
                mem_info.allocations.insert(ptr as u64, size as u64);
            }
        } else {
            weld_rt_set_errno(run_id, WeldRuntimeErrno::RunNotFound);
            do_allocation = false;
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
        if let Some(mem_info) = guarded.get_mut(&run_id) {
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
        } else {
            weld_rt_set_errno(run_id, WeldRuntimeErrno::RunNotFound);
            do_allocation = false;
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
pub unsafe extern "C" fn weld_rt_free(run_id: libc::int64_t, data: *mut c_void) {
    let run_id = run_id as i64;
    let mut abort = false;
    {
        let mut guarded = ALLOCATIONS.lock().unwrap();
        if let Some(mem_info) = guarded.get_mut(&run_id) {
            let bytes_freed = *mem_info.allocations.get(&(data as u64)).unwrap();
            free(data);
            mem_info.mem_allocated -= bytes_freed as i64;
            mem_info.allocations.remove(&(data as u64));
        } else {
            weld_rt_set_errno(run_id, WeldRuntimeErrno::RunNotFound);
            abort = true;
        }
    }
    if abort {
        weld_abort_thread();
    }
}

#[no_mangle]
/// Returns the errno for a run.
///
/// Worker threads exit if the global `weld_errno` variable is set to a non-zero value,
/// designating a runtime error.
pub extern "C" fn weld_rt_get_errno(run_id: libc::int64_t) -> WeldRuntimeErrno {
    let run_id = run_id as i64;
    let guarded = WELD_ERRNOS.read().unwrap();
    return *guarded.get(&run_id).unwrap_or(&WeldRuntimeErrno::RunNotFound);
}

#[no_mangle]
/// Frees all the data from a run, given a run ID.
pub extern "C" fn weld_rt_run_free(run_id: libc::int64_t) {
    let mut guarded = ALLOCATIONS.lock().unwrap();
    if let Some(mem_info) = guarded.get_mut(&run_id) {
        for entry in mem_info.allocations.iter() {
            unsafe { free(*entry.0 as *mut c_void) };
        }
    }
    guarded.remove(&run_id);
}

#[no_mangle]
/// Sets an errno for the given run.
///
/// This signals all threads in the run to stop execution and return.
pub extern "C" fn weld_rt_set_errno(run_id: libc::int64_t, errno: WeldRuntimeErrno) {
    let run_id = run_id as i64;
    let mut guarded = WELD_ERRNOS.write().unwrap();
    let entry = guarded.entry(run_id).or_insert(errno);
    *entry = errno;
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
        let merger_ptr = unsafe { get_merger_at_index(ptr, size, i) };
        unsafe { libc::memset(merger_ptr, 0 as libc::c_int, size as usize) };
    }
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn get_merger_at_index(ptr: *mut c_void,
                                             size: i64,
                                             i: i32)
                                             -> *mut libc::c_void {
    let ptr = ptr as *mut libc::uint8_t;
    let ptr = (((ptr.offset((CACHE_LINE - 1) as isize)) as u64) & MASK) as *mut libc::uint8_t;
    let num_blocks = num_cache_blocks(size) as i64;
    let offset = num_blocks * (i as i64) * (CACHE_LINE as i64);
    let merger_ptr = ptr.offset(offset as isize) as *mut libc::c_void;
    merger_ptr
}

#[no_mangle]
pub unsafe extern "C" fn free_merger(runid: i64, ptr: *mut c_void) {
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
        let merger_ptr = unsafe { get_merger_at_index(ptr, size, i) };
        let output = unsafe { *(merger_ptr as *mut u32) };
        // Read value previously stored, make sure they match.
        assert_eq!(output, i as u32);
    }
    unsafe { free_merger(runid, ptr) };
}

#[test]
fn test_num_cache_blocks() {
    assert_eq!(num_cache_blocks(4), 2);
    assert_eq!(num_cache_blocks(8), 2);
    assert_eq!(num_cache_blocks(64), 2);
    assert_eq!(num_cache_blocks(65), 3);
}
