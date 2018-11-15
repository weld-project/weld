//! Configurations and defaults for the Weld runtime.

use super::WeldConf;
use super::error::WeldResult;
use optimizer::OPTIMIZATION_PASSES;
use optimizer::Pass;

use util::dump::{unique_filename, DumpCodeFormat};

use std::collections::HashSet;
use std::str::FromStr;

pub mod constants;

use self::constants::*;

lazy_static! {
    // Parsed list of optimization passes.
    pub static ref CONF_OPTIMIZATION_PASSES: Vec<Pass> = {
        CONF_OPTIMIZATION_PASSES_DEFAULT.iter()
            .map(|e| (*OPTIMIZATION_PASSES.get(e).unwrap())
            .clone())
            .collect()
    };
}

/// Configuration for dumping code.
#[derive(Clone,Debug)]
pub struct DumpCodeConfig {
    /// Toggles code dump.
    pub enabled: bool,
    /// Filename of dumped code.
    pub filename: String,
    /// Directory of dumped code.
    pub directory: String,
    /// Formats to dump.
    pub formats: HashSet<DumpCodeFormat>,
}

impl Default for DumpCodeConfig {
    fn default() -> Self {
        DumpCodeConfig {
            enabled: CONF_DUMP_CODE_DEFAULT,
            filename: unique_filename(),
            directory: CONF_DUMP_CODE_DIR_DEFAULT.to_string(),
            formats: DumpCodeFormat::all().into_iter().collect::<HashSet<_>>(),
        }
    }
}

/// LLVM configuration.
#[derive(Clone,Debug)]
pub struct LLVMConfig {
    /// LLVM optimization level.
    pub opt_level: u32,
    /// Enables the LLVM unroller.
    pub llvm_unroller: bool,
    /// Enables the LLVM vectorizer.
    ///
    /// Requires that target analysis passes are enabled.
    pub llvm_vectorizer: bool,
    /// Enables target analysis passes
    pub target_analysis_passes: bool,
    /// Enables full module optimizations.
    ///
    /// This uses the Clang module optimization pipeline. The specific passes are determined by the
    /// optimization level.
    pub module_optimizations: bool,
    /// Enables per-function optimizations.
    ///
    /// This uses the Clang function optimization pipeline. The specific passes are determined by the
    /// optimization level.
    pub func_optimizations: bool,
}

impl Default for LLVMConfig {
    fn default() -> Self {
        LLVMConfig {
            opt_level: CONF_LLVM_OPTIMIZATION_LEVEL_DEFAULT,
            llvm_unroller: CONF_LLVM_UNROLLER_DEFAULT,
            llvm_vectorizer: CONF_LLVM_VECTORIZER_DEFAULT,
            target_analysis_passes: CONF_LLVM_TARGET_PASSES_DEFAULT,
            module_optimizations: CONF_LLVM_MODULE_OPTS_DEFAULT,
            func_optimizations: CONF_LLVM_FUNC_OPTS_DEFAULT,
        }
    }
}

/// A parsed Weld configuration.
#[derive(Clone,Debug)]
pub struct ParsedConf {
    /// Memory limit for a single context.
    pub memory_limit: i64,
    /// Worker threads to use on backends that support threading.
    pub threads: i32,
    /// Toggles tracing in generated code.
    pub trace_run: bool,
    /// Enables SIR optimizations.
    pub enable_sir_opt: bool,
    /// Enables experimental optimization passes over the Weld IR.
    pub enable_experimental_passes: bool,
    /// Optimization pipeline to use.
    pub optimization_passes: Vec<Pass>,
    /// Enables bounds checking in generated code.
    pub enable_bounds_checks: bool,
    /// LLVM options.
    pub llvm: LLVMConfig,
    /// Options for writing code to a file.
    pub dump_code: DumpCodeConfig,
}

impl Default for ParsedConf {
    fn default() -> Self {
        ParsedConf {
            memory_limit: CONF_MEMORY_LIMIT_DEFAULT,
            threads: CONF_THREADS_DEFAULT,
            trace_run: CONF_TRACE_RUN_DEFAULT,
            enable_sir_opt: CONF_SIR_OPT_DEFAULT,
            enable_experimental_passes: CONF_EXPERIMENTAL_PASSES_DEFAULT,
            optimization_passes: CONF_OPTIMIZATION_PASSES.clone(),
            enable_bounds_checks: CONF_ENABLE_BOUNDS_CHECKS_DEFAULT,
            llvm: LLVMConfig::default(),
            dump_code: DumpCodeConfig::default(),
        }
    }
}

/// Methods for parsing a `String` to `String` map into typed values.
trait ConfigParser {
    /// Retrieves the value mapped to key and parses it to type `F`. The `map` function is applied to
    /// the parsed value. If the value does not exist, the default value is returned.
    fn parse_map<F, T, G>(&self, key: &str, default: T, map: G) -> WeldResult<T>
        where F: FromStr, G: FnMut(F) -> WeldResult<T>;

    /// Retrieves the value mapped to `key` and parse it to type `F`. Returns `default` if key does
    /// not exist.
    fn parse_str<F>(&self, key: &str, default: F) -> WeldResult<F>
        where F: FromStr {
            self.parse_map(key, default, &mut |v| Ok(v))
        }
}

impl ConfigParser for WeldConf {
    fn parse_map<F, T, G>(&self, key: &str, default: T, mut func: G) -> WeldResult<T>
        where F: FromStr, G: FnMut(F) -> WeldResult<T> {
            match self.get(key).cloned().map(|s| s.into_string().unwrap()) {
                Some(field) => {
                    match field.parse::<F>() {
                        Ok(result) => func(result),
                        Err(_) => compile_err!("Invalid configuration value '{}' for '{}'",
                                               field, key)
                    }
                },
                None => Ok(default),
            }
        }
}

impl ParsedConf {
    pub fn parse(conf: &WeldConf) -> WeldResult<ParsedConf> {
        let conf = ParsedConf {
            memory_limit: conf.parse_str(CONF_MEMORY_LIMIT_KEY,
                                         CONF_MEMORY_LIMIT_DEFAULT)?,
            threads: conf.parse_str(CONF_THREADS_KEY,
                                    CONF_THREADS_DEFAULT)?,
            trace_run: conf.parse_str(CONF_TRACE_RUN_KEY,
                                      CONF_TRACE_RUN_DEFAULT)?,
            enable_sir_opt: conf.parse_str(CONF_SIR_OPT_KEY,
                                           CONF_SIR_OPT_DEFAULT)?,
            enable_experimental_passes: conf.parse_str(CONF_EXPERIMENTAL_PASSES_KEY,
                                                       CONF_EXPERIMENTAL_PASSES_DEFAULT)?,
            optimization_passes: conf.parse_map(CONF_OPTIMIZATION_PASSES_KEY,
                                                CONF_OPTIMIZATION_PASSES.clone(),
                                                parse_passes)?,
            enable_bounds_checks: conf.parse_str(CONF_ENABLE_BOUNDS_CHECKS_KEY,
                                                 CONF_ENABLE_BOUNDS_CHECKS_DEFAULT)?,
            llvm: LLVMConfig {
                opt_level: conf.parse_str(CONF_LLVM_OPTIMIZATION_LEVEL_KEY,
                                          CONF_LLVM_OPTIMIZATION_LEVEL_DEFAULT)?,
                llvm_unroller: conf.parse_str(CONF_LLVM_UNROLLER_KEY,
                                              CONF_LLVM_UNROLLER_DEFAULT)?,
                llvm_vectorizer: conf.parse_str(CONF_LLVM_VECTORIZER_KEY,
                                                CONF_LLVM_VECTORIZER_DEFAULT)?,
                target_analysis_passes: conf.parse_str(CONF_LLVM_TARGET_PASSES_KEY,
                                                       CONF_LLVM_TARGET_PASSES_DEFAULT)?,
                module_optimizations: conf.parse_str(CONF_LLVM_MODULE_OPTS_KEY,
                                                     CONF_LLVM_MODULE_OPTS_DEFAULT)?,
                func_optimizations: conf.parse_str(CONF_LLVM_FUNC_OPTS_KEY,
                                                   CONF_LLVM_FUNC_OPTS_DEFAULT)?,
            },
            dump_code: DumpCodeConfig {
                enabled: conf.parse_str(CONF_DUMP_CODE_KEY,
                                        CONF_DUMP_CODE_DEFAULT)?,
                filename: conf.parse_str(CONF_DUMP_CODE_FILENAME_KEY,
                                         unique_filename())?,
                directory: conf.parse_str(CONF_DUMP_CODE_DIR_KEY,
                                          CONF_DUMP_CODE_DIR_DEFAULT.to_string())?,
                formats: conf.parse_map::<String, HashSet<_>, _>(CONF_DUMP_CODE_FORMATS_KEY,
                                        DumpCodeFormat::all().into_iter().collect::<HashSet<_>>(),
                                        parse_dump_code_formats)?,
            }
        };
        Ok(conf)
    }
}

/// Parses a comma separated list of formats.
fn parse_dump_code_formats(s: String) -> WeldResult<HashSet<DumpCodeFormat>> {
    use util::dump::DumpCodeFormat::*;
    s.split(",").map(|s| {
        match s.to_lowercase().as_ref() {
            "weld" => Ok(Weld),
            "weldopt" => Ok(WeldOpt), 
            "llvm" => Ok(LLVM),   
            "llvmopt" => Ok(LLVMOpt),   
            "assembly" => Ok(Assembly),   
            "sir" => Ok(SIR),   
            other => compile_err!("Unknown dumpCode format '{}'", other),
        }
    }).collect::<WeldResult<HashSet<DumpCodeFormat>>>()
}

/// Parse a list of optimization passes.
fn parse_passes(s: String) -> WeldResult<Vec<Pass>> {
    if s.len() == 0 {
        return Ok(vec![]); // Special case because split() creates an empty piece here
    }
    let mut result = vec![];


    // Insert mandatory passes to the beginning.
    //
    // TODO: These shouldn't be passes, since things break if we don't run them...
    result.push(OPTIMIZATION_PASSES.get("inline-zip").unwrap().clone());
    result.push(OPTIMIZATION_PASSES.get("inline-let").unwrap().clone());
    result.push(OPTIMIZATION_PASSES.get("inline-apply").unwrap().clone());

    for piece in s.split(",") {
        match OPTIMIZATION_PASSES.get(piece) {
            Some(pass) => result.push(pass.clone()),
            None => return compile_err!("Unknown optimization pass: {}", piece)
        }
    }
    Ok(result)
}
