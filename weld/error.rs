use std::error;
use std::fmt;

/// Error type returned by Weld.
#[derive(Debug)]
pub struct WeldError(String);

impl WeldError {
    pub fn new(description: String) -> WeldError {
        WeldError(description)
    }
}

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
