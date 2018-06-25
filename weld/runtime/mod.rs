//! Functions and structs accessed from the Weld runtime.

use std::fmt;

pub use self::link::{weld_runtime_init, weld_run_memory_usage, weld_run_dispose};

pub mod link;
pub mod strt;

pub use self::strt::WeldRun;

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

