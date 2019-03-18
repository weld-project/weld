use std::error;
use std::fmt;

/// Internal macro for creating a compile error.
macro_rules! compile_err {
    ( $($arg:tt)* ) => ({
        ::std::result::Result::Err($crate::error::WeldCompileError::new(format!($($arg)*)))
    })
}

/// Converts a non-fatal error into a log message.
macro_rules! nonfatal {
    ( $arg:expr ) => {{
        if let ::std::result::Result::Err(weld_err) = $arg {
            error!("{}", weld_err.to_string());
        }
    }};
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl error::Error for WeldCompileError {
    fn description(&self) -> &str {
        &self.0
    }

    fn cause(&self) -> Option<&dyn error::Error> {
        None
    }
}

impl From<String> for WeldCompileError {
    fn from(string: String) -> WeldCompileError {
        WeldCompileError(string)
    }
}

/// Result type returned by Weld.
pub type WeldResult<T> = Result<T, WeldCompileError>;
