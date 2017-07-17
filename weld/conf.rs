//! Configurations and defaults for the Weld runtime.

use std::ffi::CString;

// Keys
pub const MEMORY_LIMIT_KEY: &'static str = "weld.memory.limit";
pub const THREADS_KEY: &'static str = "weld.threads";
pub const LOG_LEVEL_KEY: &'static str = "weld.log.level";
pub const OPTIMIZATION_PASSES_KEY: &'static str = "weld.optimization.passes";

/// Available logging levels; these should be listed in order of verbosity
/// because code will compare them.
#[derive(PartialEq,Eq,Clone,PartialOrd,Debug)]
pub enum LogLevel {
    None, Debug
}

// Defaults
pub const DEFAULT_MEMORY_LIMIT: i64 = 1000000000;
pub const DEFAULT_THREADS: i64 = 1;
pub const DEFAULT_LOG_LEVEL: LogLevel = LogLevel::None;
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
        Ok(v) if v > 0 => v,
        _ => DEFAULT_THREADS,
    }
}

/// Parses the memory limit. Returns the default if the string cannot be parsed.
pub fn parse_memory_limit(s: CString) -> i64 {
    let s = s.into_string().unwrap();
    match s.parse::<i64>() {
        Ok(v) if v > 0 => v,
        _ => DEFAULT_MEMORY_LIMIT,
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

/// Parses a log level. Returns NONE if it is not recognized.
pub fn parse_log_level(s: CString) -> LogLevel {
    let s = s.into_string().unwrap().to_lowercase();
    match s.as_ref() {
        "debug" => LogLevel::Debug,
        _ => LogLevel::None,
    }
}

#[test]
fn conf_parsing() {
    assert_eq!(parse_threads(CString::new("").unwrap()), DEFAULT_THREADS);
    assert_eq!(parse_threads(CString::new("-1").unwrap()), DEFAULT_THREADS);
    assert_eq!(parse_threads(CString::new("1").unwrap()), 1);
    assert_eq!(parse_threads(CString::new("2").unwrap()), 2);

    assert_eq!(parse_memory_limit(CString::new("").unwrap()), DEFAULT_MEMORY_LIMIT);
    assert_eq!(parse_memory_limit(CString::new("1000000").unwrap()), 1000000);

    assert_eq!(parse_optimization_passes(CString::new("").unwrap()),
        DEFAULT_OPTIMIZATION_PASSES.clone());
    assert_eq!(parse_optimization_passes(CString::new("loop-fusion").unwrap()),
        vec![String::from("loop-fusion")]);

    assert_eq!(parse_log_level(CString::new("debug").unwrap()), LogLevel::Debug);
    assert_eq!(parse_log_level(CString::new("DEBUG").unwrap()), LogLevel::Debug);
    assert_eq!(parse_log_level(CString::new("none").unwrap()), LogLevel::None);
    assert_eq!(parse_log_level(CString::new("").unwrap()), LogLevel::None);
}
