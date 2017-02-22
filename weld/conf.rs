//! Configurations and defaults for the Weld runtime.

use std::ffi::CString;

// Keys
pub const MEMORY_LIMIT_KEY: &'static str = "weld.memory.limit";
pub const THREADS_KEY: &'static str = "weld.threads";

// Defaults
pub const DEFAULT_MEMORY_LIMIT: i64 = 1000000000;
pub const DEFAULT_THREADS: i64 = 1;

/// Parses the number of threads. Returns the default if the string cannot be parsed.
pub fn parse_threads(s: CString) -> i64 {
    let s = s.into_string().unwrap();
    match s.parse::<i64>() {
        Ok(v) => v,
        Err(_) => DEFAULT_THREADS,
    }
}

/// Parses the memory limit. Returns the default if the string cannot be parsed.
pub fn parse_memory_limit(s: CString) -> i64 {
    let s = s.into_string().unwrap();
    match s.parse::<i64>() {
        Ok(v) => v,
        Err(_) => DEFAULT_MEMORY_LIMIT,
    }
}
