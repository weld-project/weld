use std::convert::From;
use std::error;
use std::fmt;

use easy_ll::LlvmError;
use common::WeldRuntimeErrno;

/// Error type returned by Weld.
#[derive(Debug,Clone)]
pub struct WeldError {
    description: String,
    code: WeldRuntimeErrno,
}

impl WeldError {
    pub fn new(description: String) -> WeldError {
        WeldError {
            description: description,
            code: WeldRuntimeErrno::CompileError,
        }
    }

    pub fn new_with_errno(description: String, code: WeldRuntimeErrno) -> WeldError {
        WeldError {
            description: description,
            code: code,
        }
    }

    pub fn success() -> WeldError {
        WeldError::new_with_errno("Success".to_string(), WeldRuntimeErrno::Success)
    }

    pub fn code(&self) -> WeldRuntimeErrno {
        self.code
    }
}

impl fmt::Display for WeldError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}

impl error::Error for WeldError {
    fn description(&self) -> &str {
        &self.description
    }

    fn cause(&self) -> Option<&error::Error> {
        None
    }
}

impl From<LlvmError> for WeldError {
    fn from(err: LlvmError) -> WeldError {
        WeldError::new(err.to_string())
    }
}

/// Result type returned by Weld.
pub type WeldResult<T> = Result<T, WeldError>;
