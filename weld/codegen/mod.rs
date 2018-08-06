//! Target-specific code generation from Weld.
//!
//! This module supports code generation from Weld to a traditional compiler IR. Usually, code
//! generation for converts the Weld AST to the Weld SIR to facilitate code generation, and then
//! maps each SIR instruction to a compiler IR instruction (e.g., LLVM).
//!
//! For now, we only support LLVM, so this module is just a wrapper for the LLVM module.
//! Eventually we will refactor it so common tasks such as Weld AST optimizations and AST-to-SIR
//! conversion occur here and submodules implement backends for different hardware platforms. For
//! example, an NVPTX GPU-based backend would be a new submodule that emits NVPTX code, and this
//! module will provide the shared Weld optimization and SIR conversion logic that currently lives in the
//! `llvm` module.

extern crate time;

use ast::*;
use conf::ParsedConf;
use conf::Backend;
use error::*;
use optimizer::*;
use runtime::WeldRuntimeErrno;
use sir::*;
use syntax::macro_processor;
use syntax::program::Program;
use util::stats::CompilationStats;

use time::PreciseTime;

// For dumping files.
use std::io::Write;
use std::path::PathBuf;
use std::fs::OpenOptions;

mod llvm;
mod llvm2;

pub use self::llvm::load_library;

/// A wrapper for a struct passed as input to Weld.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct WeldInputArgs {
    /// A pointer to the input data.
    pub input: i64,
    /// Number of worker threads to use.
    pub nworkers: i32,
    /// Memory limit in bytes.
    pub mem_limit: i64,
    /// A run handle if this input is to continue an existing run.
    ///
    /// This value should be 0 if a new run should be initialized.
    pub run: i64,
}

/// A wrapper for outputs passed out of Weld.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct WeldOutputArgs {
    /// A pointer to the output data.
    pub output: i64,
    /// A reference to the run.
    ///
    /// This can either be a numerical identifier or a pointer to some handle.
    pub run: i64,
    /// An error, if one was set.
    pub errno: WeldRuntimeErrno,
}

/// A trait implemented by trait objects for running a Weld program.
pub trait Runnable {
    fn run(&self, arg: i64) -> i64;
}

/// A compiled, runnable module.
pub struct CompiledModule {
    runnable: Box<dyn Runnable>,
    param_types: Vec<Type>,
    return_type: Type,
}

impl CompiledModule {
    /// Run the compiled module.
    ///
    /// This calls the `run` function on the internal `Runnable`.
    pub fn run(&self, arg: i64) -> i64 {
        self.runnable.run(arg)
    }

    /// Get the Weld parameter types of the compiled function.
    pub fn param_types(&self) -> &Vec<Type> {
        &self.param_types
    }

    /// Get the Weld return type of the compiled function.
    pub fn return_type(&self) -> &Type {
        &self.return_type
    }
}

/// Compile a Weld program with a given configuration.
///
/// The given configuration may be modified to reflect the actual configuration used (e.g., if a
/// backend specified in the configuration is not supported). Weld will log cases where the
/// configuration changes.
///
/// This function is the main entry point into compiling a Weld program. It performs the following
/// tasks in order:
///
/// 1. Evaluate macros
/// 2. Infer types
/// 3. Apply optimization passes (either the default set or the ones in `conf`)
/// 4. Convert the AST to SIR
/// 5. Apply SIR optimizations, if enabled.
/// 6. Dump Weld and SIR code, if specified in `conf`.
/// 7. Compile to native code depending on the backend specified in `conf`.
/// 8. Dump native code, if specified in `conf`.
/// 9. Return compiled, runnable module.
///
/// This function is a driver: most of the above tasks are implemented elsewhere.
pub fn compile_program(program: &Program,
                       conf: &mut ParsedConf,
                       stats: &mut CompilationStats) -> WeldResult<CompiledModule> {
    let mut expr = macro_processor::process_program(program)?;
    debug!("After macro substitution:\n{}\n", expr.pretty_print());

    let start = PreciseTime::now();
    expr.uniquify()?;
    let end = PreciseTime::now();
    let mut uniquify_dur = start.to(end);

    let start = PreciseTime::now();
    expr.infer_types()?;
    let end = PreciseTime::now();
    stats.weld_times.push(("Type Inference".to_string(), start.to(end)));
    debug!("After type inference:\n{}\n", expr.pretty_print());

    apply_passes(&mut expr, &conf.optimization_passes, stats, conf.enable_experimental_passes)?;

    let start = PreciseTime::now();
    expr.uniquify()?;
    let end = PreciseTime::now();
    uniquify_dur = uniquify_dur + start.to(end);
    stats.weld_times.push(("Uniquify outside Passes".to_string(), uniquify_dur));
    debug!("Optimized Weld program:\n{}\n", expr.pretty_print());

    let start = PreciseTime::now();
    let mut sir_prog = ast_to_sir(&expr, conf.support_multithread)?;
    let end = PreciseTime::now();
    stats.weld_times.push(("AST to SIR".to_string(), start.to(end)));
    debug!("SIR program:\n{}\n", &sir_prog);

    let start = PreciseTime::now();
    if conf.enable_sir_opt {
        use sir::optimizations;
        info!("Applying SIR optimizations");
        optimizations::fold_constants::fold_constants(&mut sir_prog)?;
    }
    let end = PreciseTime::now();
    debug!("Optimized SIR program:\n{}\n", &sir_prog);
    stats.weld_times.push(("SIR Optimization".to_string(), start.to(end)));

    // Dump files if needed.
    let ref timestamp = format!("{}", time::now().to_timespec().sec);
    if conf.dump_code.enabled {
        info!("Writing code to directory '{}' with timestamp {}", &conf.dump_code.dir.display(), timestamp);
        write_code(expr.pretty_print(), "weld", timestamp, &conf.dump_code.dir);
        write_code(sir_prog.to_string(), "sir", timestamp, &conf.dump_code.dir);
    }

    let runnable = match conf.backend {
        // Check if the single threaded runtime supports the SIR program.
        Backend::LLVMSingleThreadBackend => {
            if let Some(reason) = llvm2::unsupported(&sir_prog) {
                // If fallback is enabled, we can fall back to the old runtime. Otherwise, raise an
                // error.
                if conf.enable_fallback {
                    warn!("ST Found unsupported SIR statement '{}': falling back to MT runtime",
                          reason);
                    conf.backend = Backend::LLVMWorkStealingBackend;
                    conf.support_multithread = false;
                    conf.threads = 1;
                    llvm::compile(&sir_prog, conf, stats, timestamp)
                } else {
                    return compile_err!("Unsupported SIR statement '{}' for ST runtime:
                    set weld.compile.enableFallback=true to automatically fall back to MT runtime or use different backend", reason);
                }
            } else {
                if conf.support_multithread {
                    warn!("Compiling with ST backend, but multithread support is set to true");
                }
                llvm2::compile(&sir_prog, conf, stats, timestamp)
            }
        }
        Backend::LLVMWorkStealingBackend => llvm::compile(&sir_prog, conf, stats, timestamp),
        _ => compile_err!("Unsupported backend variant {:?}", conf.backend),
    }?;

    if let Type::Function(ref param_tys, ref return_ty) = expr.ty {
        Ok(CompiledModule {
            runnable: runnable,
            param_types: param_tys.clone(),
            return_type: *return_ty.clone(),
        })
    } else {
        unreachable!()
    }
}

/// Writes code to a file specified by `PathBuf`. Writes a log message if it failed.
fn write_code<T: AsRef<str>, U: AsRef<str>, V: AsRef<str>>(code: T, ext: U, prefix: V, dir_path: &PathBuf) {
    let mut options = OpenOptions::new();
    options.write(true)
        .create_new(true)
        .create(true);

    let ref mut path = dir_path.clone();
    path.push(format!("code-{}", prefix.as_ref()));
    path.set_extension(ext.as_ref());

    let ref path_str = format!("{}", path.display());
    match options.open(path) {
        Ok(ref mut file) => {
            if let Err(_) = file.write_all(code.as_ref().as_bytes()) {
                error!("Write failed: could not write code to file {}", path_str);
            }
        }
        Err(_) => {
            error!("Open failed: could not write code to file {}", path_str);
        }
    }
}
