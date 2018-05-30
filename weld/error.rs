use std::convert::From;
use std::error;
use std::fmt;

use easy_ll::LlvmError;

/// Internal macro for creating a compile error.
macro_rules! compile_err {
    ( $($arg:tt)* ) => ({
        ::std::result::Result::Err($crate::error::WeldCompileError::new(format!($($arg)*)))
    })
}

/// A compilation error produced by Weld.
#[derive(Debug, Clone)]
pub struct WeldCompileError(String);

impl WeldCompileError {
    pub fn new<T: Into<String>>(description: T) -> WeldCompileError {
        WeldCompileError(description.into())
    }
}

impl fmt::Display for WeldCompileError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl error::Error for WeldCompileError {
    fn description(&self) -> &str {
        &self.0
    }

    fn cause(&self) -> Option<&error::Error> {
        None
    }
}

impl From<LlvmError> for WeldCompileError {
    fn from(err: LlvmError) -> WeldCompileError {
        WeldCompileError::new(err.to_string())
    }
}

/// Result type returned by Weld.
pub type WeldResult<T> = Result<T, WeldCompileError>;
