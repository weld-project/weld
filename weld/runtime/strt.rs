//! A single threaded (st) runtime (rt) for Weld.

extern crate libc;
extern crate fnv;

use libc::{c_char, int32_t, int64_t, size_t};
use libc::{malloc, realloc, free};

use fnv::FnvHashMap;

use std::ffi::CStr;
use std::ptr;
use std::sync::{Once, ONCE_INIT};

use super::WeldRuntimeErrno;

pub type WeldRunRef = *mut WeldRun;

/// An opaque pointer.
type Pointer = *mut libc::c_void;

/// Initialize the Weld runtime only once.
static ONCE: Once = ONCE_INIT;
static mut INITIALIZE_FAILED: bool = false;

/// Maintains information about a single Weld run.
#[derive(Debug)]
pub struct WeldRun {
    allocations: FnvHashMap<Pointer, i64>,
    errno: WeldRuntimeErrno,
    result: Pointer,
    nworkers: i32,
    memlimit: i64,
    allocated: i64,
}

/// Private API used by the FFI.
impl WeldRun {
    fn new(nworkers: i32, memlimit: i64) -> WeldRun {
        WeldRun {
            allocations: FnvHashMap::default(),
            errno: WeldRuntimeErrno::Success,
            result: ptr::null_mut(),
            nworkers: nworkers,
            memlimit: memlimit,
            allocated: 0,
        }
    }

    unsafe fn malloc(&mut self, size: i64) -> Pointer {
        if self.allocated + size > self.memlimit {
            self.errno = WeldRuntimeErrno::OutOfMemory;
            panic!("Weld run ran out of memory (limit={}, attempted to allocate {}",
                   self.memlimit,
                   self.allocated + size);
        }
        let mem = malloc(size as size_t);
        assert!(mem as i64 != 0);
        self.allocated += size;
        self.allocations.insert(mem.clone(), size);
        mem
    }

    unsafe fn realloc(&mut self, pointer: Pointer, size: i64) -> Pointer {
        let allocated = *self.allocations.get(&pointer).unwrap();
        if self.allocated - allocated + size > self.memlimit {
            self.errno = WeldRuntimeErrno::OutOfMemory;
            panic!("Weld run ran out of memory (limit={}, attempted to allocate {}",
                   self.memlimit,
                   self.allocated - allocated + size);
        }

        let mem = realloc(pointer, size as size_t);
        assert!(mem as i64 != 0);
        self.allocated -= allocated;
        self.allocated += size;
        let _ = self.allocations.remove(&pointer);
        self.allocations.insert(mem.clone(), size);
        mem
    }

    unsafe fn free(&mut self, pointer: Pointer) {
        let allocated = *self.allocations.get(&pointer).unwrap();
        free(pointer);
        self.allocated -= allocated;
        self.allocations.remove(&pointer);
    }

    fn set_errno(&mut self, errno: WeldRuntimeErrno) {
        self.errno = errno;
        panic!("Weld runtime threw error: {}", self.errno)
    }

    fn set_result(&mut self, result: Pointer) {
        self.result = result;
    }

    fn result(&self) -> Pointer {
        self.result
    }

    fn errno(&self) -> WeldRuntimeErrno {
        self.errno
    }
}

/// Public read-only API used to query information about the run.
impl WeldRun {
    /// Returns the number of bytes allocated by this Weld run.
    pub fn memory_usage(&self) -> i64 {
        self.allocations.values().sum()
    }

    /// Returns a 64-bit ID identifying this run.
    pub fn run_id(&self) -> i64 {
        0
    }
}

impl Drop for WeldRun {
    fn drop(&mut self) {
        // Free memory allocated by the run.
        unsafe {
            for (pointer, _) in self.allocations.iter() {
                free(*pointer);
            }
        }
    }
}

unsafe fn initialize() {
    // Hack to prevent symbols from being compiled out in a Rust binary.
    let mut x =  weld_runst_init as i64;
    x += weld_runst_set_result as i64;
    x += weld_runst_get_result as i64;
    x += weld_runst_malloc as i64;
    x += weld_runst_realloc as i64;
    x += weld_runst_free as i64;
    x += weld_runst_set_errno as i64;

    use super::link::*;
    x += weld_st_dict_new as i64;
    x += weld_st_dict_get as i64;
    x += weld_st_dict_upsert as i64;
    x += weld_st_dict_keyexists as i64;
    x += weld_st_dict_size as i64;
    x += weld_st_dict_tovec as i64;
    x += weld_st_dict_serialize as i64;

    x += weld_st_gb_new as i64;
    x += weld_st_gb_merge as i64;
    x += weld_st_gb_result as i64;

    trace!("Runtime initialized with hashed values {}", x);
}

#[no_mangle]
/// Initialize the runtime.
///
/// This call currently circumvents an awkward problem with binaries using Rust, where the FFI
/// below used by the Weld runtime is compiled out.
pub unsafe extern "C" fn weld_init() {
    ONCE.call_once(|| initialize());
    if INITIALIZE_FAILED {
        panic!("Weld runtime initialization failed")
    }
}

#[no_mangle]
/// Initialize the runtime if necessary, and then initialize run-specific information.
pub unsafe extern "C" fn weld_runst_init(nworkers: int32_t, memlimit: int64_t) -> WeldRunRef {
    Box::into_raw(Box::new(WeldRun::new(nworkers, memlimit)))
}

#[no_mangle]
/// Allocate memory using `libc` `malloc`, tracking memory usage information.
pub unsafe extern "C" fn weld_runst_malloc(run: WeldRunRef, size: int64_t) -> Pointer {
    let run = &mut *run;
    run.malloc(size)
}

#[no_mangle]
/// Allocate memory using `libc` `remalloc`, tracking memory usage information.
pub unsafe extern "C" fn weld_runst_realloc(run: WeldRunRef,
                                     ptr: Pointer,
                                     newsize: int64_t) -> Pointer {
    let run = &mut *run;
    run.realloc(ptr, newsize)
}

#[no_mangle]
/// Free memory using `libc` `free`, tracking memory usage information.
pub unsafe extern "C" fn weld_runst_free(run: WeldRunRef, ptr: Pointer) {
    let run = &mut *run;
    run.free(ptr)
}

#[no_mangle]
/// Set the result pointer.
pub unsafe extern "C" fn weld_runst_set_result(run: WeldRunRef, ptr: Pointer) {
    let run = &mut *run;
    run.set_result(ptr)
}

#[no_mangle]
/// Set the errno value.
pub unsafe extern "C" fn weld_runst_set_errno(run: WeldRunRef, errno: WeldRuntimeErrno) {
    let run = &mut *run;
    run.set_errno(errno)
}

#[no_mangle]
/// Get the errno value.
pub unsafe extern "C" fn weld_runst_get_errno(run: WeldRunRef) -> WeldRuntimeErrno {
    let run = &mut *run;
    run.errno()
}

#[no_mangle]
/// Get the result pointer.
pub unsafe extern "C" fn weld_runst_get_result(run: WeldRunRef) -> Pointer {
    let run = &mut *run;
    run.result()
}

#[no_mangle]
/// Print a value from generated code.
pub unsafe extern "C" fn weld_runst_print(_run: WeldRunRef, string: *const c_char) {
    let string = CStr::from_ptr(string).to_str().unwrap();
    // XXX We could switch to a custom printer here too.
    println!("{}", string);
}
