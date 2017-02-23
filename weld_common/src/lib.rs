
use std::fmt;

/// An errno set by the runtime.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd)]
#[repr(u64)]
pub enum WeldRuntimeErrno {
    Success = 0,
    CompileError,
    ArrayOutOfBounds,
    BadIteratorLength,
    MismatchedZipSize,
    OutOfMemory,
    RunNotFound,
    Unknown,
    ErrnoMax,
}

impl fmt::Display for WeldRuntimeErrno {
    /// Just return the errno name.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
