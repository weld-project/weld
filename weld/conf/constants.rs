//! Defines the constant key names for various Weld properties.
//!
//! This module is re-exported at the top-level.

/// Bytes a single `WeldContext` is allowed to allocate.
///
/// If a run allocates more memory than specified by this parameter across *all runs using the
/// context*, the run will fail with an `OutOfMemoryError`.
///
/// This parameter should be set in a configuration passed to a `WeldContext`.
pub const CONF_MEMORY_LIMIT_KEY: &'static str = "weld.memory.limit";

/// Specifies the number of threads to use during execution.
///
/// This parameter should be set in a configuration passed to a `WeldContext`.
pub const CONF_THREADS_KEY: &'static str = "weld.threads";

/// Specifies whether tracing should be enabled when compiling the program.
///
/// Tracing will print, during runtime, the internal Weld instruction before executing it. This has
/// a very large performance overhead and will print lots of output on large inputs: it should only
/// be enabled for debugging (e.g., to identify a crash).
///
/// This parameter should be set for compilation.
pub const CONF_TRACE_RUN_KEY: &'static str = "weld.compile.traceExecution";

/// Specifies an ordered list of the optimizations to apply to a Weld program.
///
/// This supercedes the default optimization set.
///
/// This parameter should be set for compilation.
pub const CONF_OPTIMIZATION_PASSES_KEY: &'static str = "weld.optimization.passes";

/// Enables experimental (unstable) optimizations.
///
/// Experimental optimizations may produce adverse effects in certain programs, as they are
/// unstable/still being tested. They may also improve the performance of a program, however.
///
/// This parameter should be set for compilation.
pub const CONF_EXPERIMENTAL_PASSES_KEY: &'static str =
    "weld.optimization.applyExperimentalTransforms";

/// Enables internal Sequential IR (SIR) optimizations.
///
/// This enables optimizations over the internal "sequential" Weld IR. Certain optimizations are
/// easier to apply over this IR, and are thus implemented over the SIR rather than the expression
/// tree.
///
/// This parameter should be set for compilation.
pub const CONF_SIR_OPT_KEY: &'static str = "weld.optimization.sirOptimization";

/// Set the LLVM optimization level.
///
/// This parameter should be set for compilation.
pub const CONF_LLVM_OPTIMIZATION_LEVEL_KEY: &'static str = "weld.llvm.optimization.level";

/// Toggles LLVM loop unrolling.
///
/// This parameter should be set for compilation.
pub const CONF_LLVM_UNROLLER_KEY: &'static str = "weld.llvm.optimization.unroller";

/// Toggles LLVM vectorization.
///
/// Requires targetPasses to be true to have an effect.
///
/// This parameter should be set for compilation.
pub const CONF_LLVM_VECTORIZER_KEY: &'static str = "weld.llvm.optimization.vectorizer";

/// Toggles LLVM target passes.
///
/// This parameter should be set for compilation.
pub const CONF_LLVM_TARGET_PASSES_KEY: &'static str = "weld.llvm.optimization.targetPasses";

/// Toggles full module optimizations.
///
/// This parameter should be set for compilation.
pub const CONF_LLVM_MODULE_OPTS_KEY: &'static str = "weld.llvm.optimization.moduleOpts";

/// Toggles per-function optimizations.
///
/// This parameter should be set for compilation.
pub const CONF_LLVM_FUNC_OPTS_KEY: &'static str = "weld.llvm.optimization.funcOpts";

/// Sets the symbol name of the entry-point function.
pub const CONF_LLVM_RUN_FUNC_NAME_KEY: &'static str = "weld.llvm.runFunctionName";

/// Enables dumping code during compilation.
///
/// This will produce several files in the directory specified by `weld.compile.dumpCodeDir`:
///
/// * The un-optimized Weld program passed to the compiler
/// * THe optimized Weld program.
/// * The internal SIR representation of the optimized program.
/// * Unoptimized backend code (e.g., LLVM) generated by the Weld backend.
/// * Optimized backend code (e.g.,) after LLVM passes.
/// * Assembly code for target architecture (e.g,. x64 assembly)
///
/// This parameter should be set for compilation.
pub const CONF_DUMP_CODE_KEY: &'static str = "weld.compile.dumpCode";

/// Specifies the directory to dump code into.
///
/// This parameter should be set for compilation.
pub const CONF_DUMP_CODE_DIR_KEY: &'static str = "weld.compile.dumpCodeDir";

/// Specifies the filename prefix for dumped code.
///
/// This parameter should be set for compilation.
pub const CONF_DUMP_CODE_FILENAME_KEY: &'static str = "weld.compile.dumpCodeFilename";

/// Specifies the formats to dump for dumped code.
///
/// This key does not have an explicit default value: if it is not specified, all formats
/// are dumped. The possible formats are:
///
/// weld,weldopt,llvm,llvmopt,sir,assembly.
///
/// This parameter should be set for compilation.
pub const CONF_DUMP_CODE_FORMATS_KEY: &'static str = "weld.compile.dumpCodeFormats";

/// Enables runtime bounds checking for loops before executing them.
///
/// This parameter should be set for compilation.
pub const CONF_ENABLE_BOUNDS_CHECKS_KEY: &'static str = "weld.compile.enableBoundsChecks";

/// Default memory limit.
pub const CONF_MEMORY_LIMIT_DEFAULT: i64 = 1000000000;

/// Default number of threads.
pub const CONF_THREADS_DEFAULT: i32 = 1;

/// Default setting for SIR optimization.
pub const CONF_SIR_OPT_DEFAULT: bool = true;

/// Default LLVM optimization level.
pub const CONF_LLVM_OPTIMIZATION_LEVEL_DEFAULT: u32 = 2;

/// Default LLVM loop unroller setting.
pub const CONF_LLVM_UNROLLER_DEFAULT: bool = true;

/// Default LLVM loop vectorizer setting.
pub const CONF_LLVM_VECTORIZER_DEFAULT: bool = true;

/// Default LLVM target analysis passes.
pub const CONF_LLVM_TARGET_PASSES_DEFAULT: bool = true;

/// Default LLVM module passes setting.
pub const CONF_LLVM_MODULE_OPTS_DEFAULT: bool = true;

/// Default LLVM function passes setting.
pub const CONF_LLVM_FUNC_OPTS_DEFAULT: bool = true;

/// Default symbol name for LLVM entry-point function.
pub const CONF_LLVM_RUN_FUNC_NAME_DEFAULT: &'static str = "run";

/// Default setting for whether to dump code.
pub const CONF_DUMP_CODE_DEFAULT: bool = false;

/// Default setting for whether to trace SIR instructions.
pub const CONF_TRACE_RUN_DEFAULT: bool = false;

/// Default setting for whether to enable experimental (unstable) optimizations.
pub const CONF_EXPERIMENTAL_PASSES_DEFAULT: bool = false;

/// Default setting for whether to enable bounds checking.
pub const CONF_ENABLE_BOUNDS_CHECKS_DEFAULT: bool = false;

/// Default directory for dumping code.
pub const CONF_DUMP_CODE_DIR_DEFAULT: &'static str = ".";

/// Default set of optimization passes.
pub const CONF_OPTIMIZATION_PASSES_DEFAULT: &[&'static str] = &[
    "inline-zip",
    "inline-let",
    "inline-apply",
    "loop-fusion",
    "unroll-static-loop",
    "infer-size",
    "algebraic",
    "inline-literals",
    "cse",
    "short-circuit-booleans",
    "predicate",
    "vectorize",
];
