//! Configurations and defaults for the Weld runtime.

use super::WeldConf;
use super::error::WeldResult;
use optimizer::OPTIMIZATION_PASSES;
use optimizer::Pass;

pub mod constants;

use self::constants::*;

lazy_static! {
    pub static ref CONF_OPTIMIZATION_PASSES: Vec<Pass> = {
        CONF_OPTIMIZATION_PASSES_DEFAULT.iter()
            .map(|e| (*OPTIMIZATION_PASSES.get(e).unwrap())
            .clone())
            .collect()
    };
}

/// Options for dumping code.
#[derive(Clone,Debug)]
pub struct DumpCodeConf {
    pub enabled: bool,
    pub dir: String,
}

/// A parsed configuration with correctly typed fields.
#[derive(Clone,Debug)]
pub struct ParsedConf {
    pub memory_limit: i64,
    pub threads: i32,
    pub trace_run: bool,
    pub enable_sir_opt: bool,
    pub enable_experimental_passes: bool,
    pub optimization_passes: Vec<Pass>,
    pub llvm_optimization_level: u32,
    pub dump_code: DumpCodeConf,
    pub enable_bounds_checks: bool,
}

impl ParsedConf {
    /// Returns a new empty ParsedConf.
    ///
    /// All fields of the configuration are set to 0.
    pub fn new() -> ParsedConf {
        ParsedConf {
            memory_limit: 0,
            threads: 0,
            trace_run: false,
            enable_sir_opt: false,
            enable_experimental_passes: false,
            optimization_passes: vec![],
            llvm_optimization_level: 0,
            dump_code: DumpCodeConf {
                enabled: false,
                dir: "".to_string()
            },
            enable_bounds_checks: false,
        }
    }
}

/// Parse a configuration from a WeldConf key-value dictionary.
pub fn parse(conf: &WeldConf) -> WeldResult<ParsedConf> {
    let value = get_value(conf, CONF_MEMORY_LIMIT_KEY);
    let memory_limit = value.map(|s| parse_memory_limit(&s))
                            .unwrap_or(Ok(CONF_MEMORY_LIMIT_DEFAULT))?;

    let value = get_value(conf, CONF_THREADS_KEY);
    let threads = value.map(|s| parse_threads(&s))
                       .unwrap_or(Ok(CONF_THREADS_DEFAULT))?;

    let value = get_value(conf, CONF_OPTIMIZATION_PASSES_KEY);
    let mut passes = value.map(|s| parse_passes(&s))
                      .unwrap_or(Ok(CONF_OPTIMIZATION_PASSES.clone()))?;

    // Insert mandatory passes to the beginning.
    passes.insert(0, OPTIMIZATION_PASSES.get("inline-zip").unwrap().clone());
    passes.insert(0, OPTIMIZATION_PASSES.get("inline-let").unwrap().clone());
    passes.insert(0, OPTIMIZATION_PASSES.get("inline-apply").unwrap().clone());

    let value = get_value(conf, CONF_LLVM_OPTIMIZATION_LEVEL_KEY);
    let level = value.map(|s| parse_llvm_optimization_level(&s))
                      .unwrap_or(Ok(CONF_LLVM_OPTIMIZATION_LEVEL_DEFAULT))?;

    let value = get_value(conf, CONF_DUMP_CODE_DIR_KEY);
    let dump_code_dir = value.unwrap_or(CONF_DUMP_CODE_DIR_DEFAULT.to_string());

    let value = get_value(conf, CONF_DUMP_CODE_KEY);
    let dump_code_enabled = value.map(|s| parse_bool_flag(&s, "Invalid flag for dumpCode"))
                      .unwrap_or(Ok(CONF_DUMP_CODE_DEFAULT))?;

    let value = get_value(conf, CONF_SIR_OPT_KEY);
    let sir_opt_enabled = value.map(|s| parse_bool_flag(&s, "Invalid flag for sirOptimization"))
                      .unwrap_or(Ok(CONF_SIR_OPT_DEFAULT))?;

    let value = get_value(conf, CONF_EXPERIMENTAL_PASSES_KEY);
    let enable_experimental_passes = value.map(|s| parse_bool_flag(&s, "Invalid flag for applyExperimentalPasses"))
                      .unwrap_or(Ok(CONF_EXPERIMENTAL_PASSES_DEFAULT))?;

    let value = get_value(conf, CONF_TRACE_RUN_KEY);
    let trace_run = value.map(|s| parse_bool_flag(&s, "Invalid flag for trace.run"))
                      .unwrap_or(Ok(CONF_TRACE_RUN_DEFAULT))?;

    let value = get_value(conf, CONF_ENABLE_BOUNDS_CHECKS_KEY);
    let enable_bounds_check = value.map(|s| parse_bool_flag(&s, "Invalid flag for enableBoundsChecks"))
                      .unwrap_or(Ok(CONF_ENABLE_BOUNDS_CHECKS_DEFAULT))?;

    Ok(ParsedConf {
        memory_limit: memory_limit,
        threads: threads,
        trace_run: trace_run,
        enable_sir_opt: sir_opt_enabled,
        enable_experimental_passes: enable_experimental_passes,
        optimization_passes: passes,
        llvm_optimization_level: level,
        enable_bounds_checks: enable_bounds_check,
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
