//! Common functionality or data types shared by the runtime and the weld compiler.
use std::fmt;

/// An errno set by the runtime but also used by the Weld API.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd)]
#[repr(u64)]
pub enum WeldRuntimeErrno {
    Success = 0, // explicit values for Success and OutOfMemory required due to use by weld_rt C++ code
    ConfigurationError,
    LoadLibraryError,
    CompileError,
    ArrayOutOfBounds,
    BadIteratorLength,
    MismatchedZipSize,
    OutOfMemory = 7,
    RunNotFound,
    Unknown,
    DeserializationError,
    ErrnoMax,
}

impl fmt::Display for WeldRuntimeErrno {
    /// Just return the errno name.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// A logging level in the Weld API.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd)]
#[repr(u64)]
pub enum WeldLogLevel {
    Off = 0,    // Log level values must match weld.h.
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
    Trace = 5
}

impl fmt::Display for WeldLogLevel {
    /// Just return the enum item's name.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
