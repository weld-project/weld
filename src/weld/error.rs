use std::error;
use std::fmt;

/// Error type returned by Weld.
#[derive(Debug)]
pub struct WeldError(pub String);

impl fmt::Display for WeldError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl error::Error for WeldError {
    fn description(&self) -> &str { &self.0 }

    fn cause(&self) -> Option<&error::Error> { None }
}

/// Result type returned by Weld.
pub type WeldResult<T> = Result<T, WeldError>;

/// Utility function to create an Err result with a WeldError.
pub fn weld_err<T>(description: &str) -> WeldResult<T> {
    Err(WeldError(description.to_string()))
}