//! Weld's runtime functions.
//!
//! These are functions that are accessed from generated code.

extern crate libc;
extern crate fnv;

use libc::{c_char, int32_t, int64_t, uint64_t, size_t};
use libc::{malloc, realloc, free};

use fnv::FnvHashMap;

use std::ffi::CStr;
use std::fmt;
use std::ptr;
use std::sync::{Once, ONCE_INIT};

use ::DataMut;

pub type WeldRuntimeContextRef = *mut WeldRuntimeContext;

/// Initialize the Weld runtime only once.
static ONCE: Once = ONCE_INIT;
static mut INITIALIZE_FAILED: bool = false;

/// An errno set by the runtime but also used by the Weld API.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd)]
#[repr(u64)]
pub enum WeldRuntimeErrno {
    /// Indicates success.
    ///
    /// This will always be 0.  
    Success = 0,
    /// Invalid configuration.
    ConfigurationError,
    /// Dynamic library load error.
    LoadLibraryError,
    /// Weld compilation error.
    CompileError,
    /// Array out-of-bounds error.
    ArrayOutOfBounds,
    /// A Weld iterator was invalid.
    BadIteratorLength,
    /// Mismatched Zip error.
    ///
    /// This error is thrown if the vectors in a Zip have different lengths.
    MismatchedZipSize,
    /// Out of memory error.
    ///
    /// This error is thrown if the amount of memory allocated by the runtime exceeds the limit set
    /// by the configuration.
    OutOfMemory,
    RunNotFound,
    /// An unknown error.
    Unknown,
    /// A deserialization error.
    ///
    /// This error occurs if a buffer being deserialized has an invalid length.
    DeserializationError,
    /// A key was not found in a dictionary.
    KeyNotFoundError,
    /// Maximum errno value.
    ///
    /// All errors will have a value less than this value and greater than 0.
    ErrnoMax,
}

impl fmt::Display for WeldRuntimeErrno {
    /// Just return the errno name.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}


/// Maintains information about a single Weld run.
#[derive(Debug,PartialEq)]
pub struct WeldRuntimeContext {
    /// Maps pointers to allocation size in bytes.
    allocations: FnvHashMap<DataMut, i64>,
    /// An error code set for the context.
    errno: WeldRuntimeErrno,
    /// A result pointer set by the runtime.
    result: DataMut,
    /// The number of worker threads.
    nworkers: i32,
    /// A memory limit.
    memlimit: i64,
    /// Number of allocated bytes so far.
    ///
    /// This will always be equal to `allocations.values().sum()`.
    allocated: i64,
}

/// Private API used by the FFI.
impl WeldRuntimeContext {
    unsafe fn malloc(&mut self, size: i64) -> DataMut {
        if self.allocated + size > self.memlimit {
            self.errno = WeldRuntimeErrno::OutOfMemory;
            panic!("Weld run ran out of memory (limit={}, attempted to allocate {}",
                   self.memlimit,
                   self.allocated + size);
        }
        let mem = malloc(size as size_t);
        assert!(mem as i64 != 0);
        self.allocated += size;
        trace!("Alloc'd pointer {:?} ({} bytes)", mem, size);
        self.allocations.insert(mem.clone(), size);
        mem
    }

    unsafe fn realloc(&mut self, pointer: DataMut, size: i64) -> DataMut {
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

    fn set_errno(&mut self, errno: WeldRuntimeErrno) {
        self.errno = errno;
        panic!("Weld runtime threw error: {}", self.errno)
    }

    fn set_result(&mut self, result: DataMut) {
        self.result = result;
    }

    fn result(&self) -> DataMut {
        self.result
    }

}

// Public API.
impl WeldRuntimeContext {
    /// Construct a new `WeldRuntimeContext`.
    pub fn new(nworkers: i32, memlimit: i64) -> WeldRuntimeContext {
        WeldRuntimeContext {
            allocations: FnvHashMap::default(),
            errno: WeldRuntimeErrno::Success,
            result: ptr::null_mut(),
            nworkers: nworkers,
            memlimit: memlimit,
            allocated: 0,
        }
    }

    /// Free an allocated data value.
    ///
    /// Panics if the passed value was not allocated by the Weld runtime.
    pub unsafe fn free(&mut self, pointer: DataMut) {
        let allocated = *self.allocations.get(&pointer).unwrap();
        trace!("Freeing pointer {:?} ({} bytes) in runst_free()", pointer, allocated);
        free(pointer);
        self.allocated -= allocated;
        self.allocations.remove(&pointer);
    }

    /// Returns the number of bytes allocated by this Weld run.
    pub fn memory_usage(&self) -> i64 {
        self.allocations.values().sum()
    }

    /// Returns a 64-bit ID identifying this run.
    pub fn run_id(&self) -> i64 {
        0
    }

    /// Returns the error code of this run.
    fn errno(&self) -> WeldRuntimeErrno {
        self.errno
    }

    /// Returns the number of worker threads set for this run.
    pub fn threads(&self) -> i32 {
        self.nworkers
    }

    /// Returns the memory limit of this run.
    pub fn memory_limit(&self) -> i64 {
        self.memlimit
    }
}

impl Drop for WeldRuntimeContext {
    fn drop(&mut self) {
        // Free memory allocated by the run.
        trace!("Allocations: {}", self.allocations.len());
        unsafe {
            for (pointer, size) in self.allocations.iter() {
                trace!("Freeing pointer {:?} ({} bytes) in drop()", *pointer, size);
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
    x += weld_runst_get_errno as i64;
    x += weld_runst_set_errno as i64;

    x += weld_runst_print_int as i64;
    x += weld_runst_print as i64;

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
pub unsafe extern "C" fn weld_runst_init(nworkers: int32_t, memlimit: int64_t) -> WeldRuntimeContextRef {
    Box::into_raw(Box::new(WeldRuntimeContext::new(nworkers, memlimit)))
}

#[no_mangle]
/// Allocate memory using `libc` `malloc`, tracking memory usage information.
pub unsafe extern "C" fn weld_runst_malloc(run: WeldRuntimeContextRef, size: int64_t) -> DataMut {
    let run = &mut *run;
    run.malloc(size)
}

#[no_mangle]
/// Allocate memory using `libc` `remalloc`, tracking memory usage information.
pub unsafe extern "C" fn weld_runst_realloc(run: WeldRuntimeContextRef,
                                     ptr: DataMut,
                                     newsize: int64_t) -> DataMut {
    let run = &mut *run;
    run.realloc(ptr, newsize)
}

#[no_mangle]
/// Free memory using `libc` `free`, tracking memory usage information.
pub unsafe extern "C" fn weld_runst_free(run: WeldRuntimeContextRef, ptr: DataMut) {
    let run = &mut *run;
    run.free(ptr)
}

#[no_mangle]
/// Set the result pointer.
pub unsafe extern "C" fn weld_runst_set_result(run: WeldRuntimeContextRef, ptr: DataMut) {
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
pub unsafe extern "C" fn weld_runst_get_result(run: WeldRuntimeContextRef) -> DataMut {
    let run = &mut *run;
    run.result()
}

#[no_mangle]
/// Print a value from generated code.
pub unsafe extern "C" fn weld_runst_print(_run: WeldRuntimeContextRef, string: *const c_char) {
    let string = CStr::from_ptr(string).to_str().unwrap();
    println!("{} ", string);
}

#[no_mangle]
/// Print a value from generated code.
pub unsafe extern "C" fn weld_runst_print_int(_run: WeldRuntimeContextRef, i: uint64_t) {
    println!("{}", i);
}
