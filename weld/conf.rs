//! Configurations and defaults for the Weld runtime.

use super::WeldConf;
use super::error::WeldResult;
use super::passes::OPTIMIZATION_PASSES;
use super::passes::Pass;

use std::path::{Path, PathBuf};

// Keys used in textual representation of conf
pub const MEMORY_LIMIT_KEY: &'static str = "weld.memory.limit";
pub const THREADS_KEY: &'static str = "weld.threads";
pub const SUPPORT_MULTITHREAD_KEY: &'static str = "weld.compile.multithreadSupport";
pub const TRACE_RUN_KEY: &'static str = "weld.compile.traceExecution";
pub const OPTIMIZATION_PASSES_KEY: &'static str = "weld.optimization.passes";
pub const EXPERIMENTAL_PASSES_KEY: &'static str = "weld.optimization.applyExperimentalTransforms";
pub const SIR_OPT_KEY: &'static str = "weld.optimization.sirOptimization";
pub const LLVM_OPTIMIZATION_LEVEL_KEY: &'static str = "weld.llvm.optimization.level";
pub const DUMP_CODE_KEY: &'static str = "weld.compile.dumpCode";
pub const DUMP_CODE_DIR_KEY: &'static str = "weld.compile.dumpCodeDir";

// Default values of each key
pub const DEFAULT_MEMORY_LIMIT: i64 = 1000000000;
pub const DEFAULT_THREADS: i32 = 1;
pub const DEFAULT_SUPPORT_MULTITHREAD: bool = true;
pub const DEFAULT_SIR_OPT: bool = true;
pub const DEFAULT_LLVM_OPTIMIZATION_LEVEL: u32 = 2;
pub const DEFAULT_DUMP_CODE: bool = false;
pub const DEFAULT_TRACE_RUN: bool = false;
pub const DEFAULT_EXPERIMENTAL_PASSES: bool = false;

lazy_static! {
    pub static ref DEFAULT_OPTIMIZATION_PASSES: Vec<Pass> = {
        let m = ["loop-fusion", "unroll-static-loop", "infer-size", "short-circuit-booleans", "predicate", "vectorize", "fix-iterate"];
        m.iter().map(|e| (*OPTIMIZATION_PASSES.get(e).unwrap()).clone()).collect()
    };
    pub static ref DEFAULT_DUMP_CODE_DIR: PathBuf = Path::new(".").to_path_buf();
}

/// Options for dumping code.
pub struct DumpCodeConf {
    pub enabled: bool,
    pub dir: PathBuf,
}

/// A parsed configuration with correctly typed fields.
pub struct ParsedConf {
    pub memory_limit: i64,
    pub threads: i32,
    pub support_multithread: bool,
    pub trace_run: bool,
    pub enable_sir_opt: bool,
    pub enable_experimental_passes: bool,
    pub optimization_passes: Vec<Pass>,
    pub llvm_optimization_level: u32,
    pub dump_code: DumpCodeConf,
}

/// Parse a configuration from a WeldConf key-value dictionary.
pub fn parse(conf: &WeldConf) -> WeldResult<ParsedConf> {
    let value = get_value(conf, MEMORY_LIMIT_KEY);
    let memory_limit = value.map(|s| parse_memory_limit(&s))
                            .unwrap_or(Ok(DEFAULT_MEMORY_LIMIT))?;

    let value = get_value(conf, THREADS_KEY);
    let threads = value.map(|s| parse_threads(&s))
                       .unwrap_or(Ok(DEFAULT_THREADS))?;

    let value = get_value(conf, OPTIMIZATION_PASSES_KEY);
    let mut passes = value.map(|s| parse_passes(&s))
                      .unwrap_or(Ok(DEFAULT_OPTIMIZATION_PASSES.clone()))?;

    // Insert mandatory passes to the beginning.
    passes.insert(0, OPTIMIZATION_PASSES.get("inline-zip").unwrap().clone());
    passes.insert(0, OPTIMIZATION_PASSES.get("inline-let").unwrap().clone());
    passes.insert(0, OPTIMIZATION_PASSES.get("inline-apply").unwrap().clone());

    let value = get_value(conf, LLVM_OPTIMIZATION_LEVEL_KEY);
    let level = value.map(|s| parse_llvm_optimization_level(&s))
                      .unwrap_or(Ok(DEFAULT_LLVM_OPTIMIZATION_LEVEL))?;

    let value = get_value(conf, SUPPORT_MULTITHREAD_KEY);
    let support_multithread = value.map(|s| parse_bool_flag(&s, "Invalid flag for multithreadSupport"))
                      .unwrap_or(Ok(DEFAULT_SUPPORT_MULTITHREAD))?;

    let value = get_value(conf, DUMP_CODE_DIR_KEY);
    let dump_code_dir = value.map(|s| parse_dir(&s))
                      .unwrap_or(Ok(DEFAULT_DUMP_CODE_DIR.clone()))?;

    let value = get_value(conf, DUMP_CODE_KEY);
    let dump_code_enabled = value.map(|s| parse_bool_flag(&s, "Invalid flag for dumpCode"))
                      .unwrap_or(Ok(DEFAULT_DUMP_CODE))?;

    let value = get_value(conf, SIR_OPT_KEY);
    let sir_opt_enabled = value.map(|s| parse_bool_flag(&s, "Invalid flag for sirOptimization"))
                      .unwrap_or(Ok(DEFAULT_SIR_OPT))?;

    let value = get_value(conf, EXPERIMENTAL_PASSES_KEY);
    let enable_experimental_passes = value.map(|s| parse_bool_flag(&s, "Invalid flag for applyExperimentalPasses"))
                      .unwrap_or(Ok(DEFAULT_EXPERIMENTAL_PASSES))?;

    let value = get_value(conf, TRACE_RUN_KEY);
    let trace_run = value.map(|s| parse_bool_flag(&s, "Invalid flag for trace.run"))
                      .unwrap_or(Ok(DEFAULT_TRACE_RUN))?;

    Ok(ParsedConf {
        memory_limit: memory_limit,
        threads: threads,
        support_multithread: support_multithread,
        trace_run: trace_run,
        enable_sir_opt: sir_opt_enabled,
        enable_experimental_passes: enable_experimental_passes,
        optimization_passes: passes,
        llvm_optimization_level: level,
        dump_code: DumpCodeConf {
            enabled: dump_code_enabled,
            dir: dump_code_dir,
        }
    })
}

fn get_value(conf: &WeldConf, key: &str) -> Option<String> {
    conf.get(key)
        .cloned()
        .map(|v| v.into_string().unwrap())
}

/// Parses a string into a path and checks if the path is a directory in the filesystem.
fn parse_dir(s: &str) -> WeldResult<PathBuf> {
    let path = Path::new(s);
    match path.metadata() {
        Ok(_) if path.is_dir() => Ok(path.to_path_buf()),
        _ => compile_err!("WeldConf: {} is not a directory", s)
    }
}

/// Parse a number of threads.
fn parse_threads(s: &str) -> WeldResult<i32> {
    match s.parse::<i32>() {
        Ok(v) if v > 0 => Ok(v),
        _ => compile_err!("Invalid number of threads: {}", s),
    }
}

/// Parse a boolean flag with a custom error message.
fn parse_bool_flag(s: &str, err: &str) -> WeldResult<bool> {
    match s.parse::<bool>() {
        Ok(v) => Ok(v),
        _ => compile_err!("{}: {}", err, s),
    }
}

/// Parse a memory limit.
fn parse_memory_limit(s: &str) -> WeldResult<i64> {
    match s.parse::<i64>() {
        Ok(v) if v > 0 => Ok(v),
        _ => compile_err!("Invalid memory limit: {}", s),
    }
}

/// Parse a list of optimization passes.
fn parse_passes(s: &str) -> WeldResult<Vec<Pass>> {
    if s.len() == 0 {
        return Ok(vec![]); // Special case because split() creates an empty piece here
    }
    let mut result = vec![];

    for piece in s.split(",") {
        match OPTIMIZATION_PASSES.get(piece) {
            Some(pass) => result.push(pass.clone()),
            None => return compile_err!("Unknown optimization pass: {}", piece)
        }
    }
    Ok(result)
}

/// Parse an LLVM optimization level.
fn parse_llvm_optimization_level(s: &str) -> WeldResult<u32> {
    match s.parse::<u32>() {
        Ok(v) if v <= 3 => Ok(v),
        _ => compile_err!("Invalid LLVM optimization level: {}", s),
    }
}

#[test]
fn conf_parsing() {
    assert_eq!(parse_threads("1").unwrap(), 1);
    assert_eq!(parse_threads("2").unwrap(), 2);
    assert!(parse_threads("-1").is_err());
    assert!(parse_threads("").is_err());

    assert_eq!(parse_memory_limit("1000000").unwrap(), 1000000);
    assert!(parse_memory_limit("-1").is_err());
    assert!(parse_memory_limit("").is_err());

    assert_eq!(parse_passes("loop-fusion,inline-let").unwrap().len(), 2);
    assert_eq!(parse_passes("loop-fusion").unwrap().len(), 1);
    assert_eq!(parse_passes("").unwrap().len(), 0);
    assert!(parse_passes("non-existent-pass").is_err());

    assert_eq!(parse_llvm_optimization_level("0").unwrap(), 0);
    assert_eq!(parse_llvm_optimization_level("2").unwrap(), 2);
    assert!(parse_llvm_optimization_level("5").is_err());
}
