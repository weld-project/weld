//! Configurations and defaults for the Weld runtime.

use std::ffi::CString;

// Keys
pub const MEMORY_LIMIT_KEY: &'static str = "weld.memory.limit";
pub const THREADS_KEY: &'static str = "weld.threads";
pub const OPTIMIZATION_PASSES_KEY: &'static str = "weld.optimization.passes";

// Defaults
pub const DEFAULT_MEMORY_LIMIT: i64 = 1000000000;
pub const DEFAULT_THREADS: i64 = 1;
lazy_static! {
    pub static ref DEFAULT_OPTIMIZATION_PASSES: Vec<String> = {
        let m = ["inline-apply", "inline-let", "inline-zip", "loop-fusion"];
        m.to_vec().iter().map(|e| e.to_string()).collect()
    };
}

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

/// Parses the optimization passes. Returns the default if the string cannot be parsed.
pub fn parse_optimization_passes(s: CString) -> Vec<String> {
    let s = s.into_string().unwrap();
    match s.as_ref() {
        "" => DEFAULT_OPTIMIZATION_PASSES.clone(),
        _ => s.split(",").map(|e| e.to_string()).collect(),
    }
}
