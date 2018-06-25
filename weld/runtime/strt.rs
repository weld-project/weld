//! A single threaded (st) runtime (rt) for Weld.

extern crate libc;
extern crate fnv;

use libc::{int32_t, int64_t, size_t};
use libc::{malloc, realloc, free};

use fnv::FnvHashMap;

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
        self.allocated += size;
        let mem = malloc(size as size_t);
        assert!(mem as i64 != 0);
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

        self.allocated -= allocated;
        self.allocated += size;
        let mem = realloc(pointer, size as size_t);
        assert!(mem as i64 != 0);
        self.allocations.insert(mem.clone(), size);
        mem
    }

    unsafe fn free(&mut self, pointer: Pointer) {
        let allocated = *self.allocations.get(&pointer).unwrap();
        self.allocated -= allocated;
        free(pointer);
        self.allocations.remove(&pointer);
    }

    fn set_errno(&mut self, errno: WeldRuntimeErrno) {
        self.errno = errno;
        panic!("Weld runtime threw error: {}", self.errno)
    }

    fn set_result(&mut self, result: Pointer) {
        self.result = result;
    }

    fn result(&mut self) -> Pointer {
        self.result
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
/// Get the result pointer.
pub unsafe extern "C" fn weld_runst_get_result(run: WeldRunRef) -> Pointer {
    let run = &mut *run;
    run.result()
}
