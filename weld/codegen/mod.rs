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
use conf::{Backend, ParsedConf};
use error::*;
use runtime::WeldRuntimeErrno;
use sir::*;
use util::stats::CompilationStats;
use util::write_code;

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
}

impl CompiledModule {
    /// Run the compiled module.
    ///
    /// This calls the `run` function on the internal `Runnable`.
    pub fn run(&self, arg: i64) -> i64 {
        self.runnable.run(arg)
    }

}

/// Compile a Weld SIR program with a given configuration.
///
/// This function dispatches to the backend specified in `conf` to generate a compiled, runnable
/// module. Statistics about compilation (e.g., time to generate code) are written into `stats`. If
/// the `dumpCode` option is enabled, code is dumped to a filename `code-<timestamp>.[ll|S].
pub fn compile_program(program: &SirProgram,
                       conf: &mut ParsedConf,
                       stats: &mut CompilationStats,
                       timestamp: &str) -> WeldResult<CompiledModule> {
    let runnable = match conf.backend {
        // Check if the single threaded runtime supports the SIR program.
        Backend::LLVMSingleThreadBackend => {
            if let Some(reason) = llvm2::unsupported(&program) {
                // If fallback is enabled, we can fall back to the old runtime. Otherwise, raise an
                // error.
                if conf.enable_fallback {
                    warn!("ST Found unsupported SIR statement '{}': falling back to MT runtime",
                          reason);
                    conf.backend = Backend::LLVMWorkStealingBackend;
                    conf.support_multithread = false;
                    conf.threads = 1;
                    llvm::compile(&program, conf, stats, timestamp)
                } else {
                    return compile_err!("Unsupported SIR statement '{}' for ST runtime", reason);
                }
            } else {
                if conf.support_multithread {
                    warn!("Compiling with ST backend, but multithread support is set to true");
                }
                llvm2::compile(&program, conf, stats, timestamp)
            }
        }
        Backend::LLVMWorkStealingBackend => llvm::compile(&program, conf, stats, timestamp),
        _ => compile_err!("Unsupported backend variant {:?}", conf.backend),
    }?;

    let result = CompiledModule {
        runnable: runnable,
    };
    Ok(result)
}
