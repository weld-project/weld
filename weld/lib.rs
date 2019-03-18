//!
//! Weld is a runtime for improving the performance of data-intensive applications. It optimizes
//! across libraries and functions by expressing the core computations in libraries using a small
//! common intermediate representation, similar to CUDA and OpenCL.
//!
//! # Using Weld
//!
//! Fundamentally, Weld is a small programming language that supports _parallel loops_ and
//! _builders_, which are declarative objects that specify how to buid results. The parallel loops
//! can be used in conjunction with the builders to build a result in parallel.
//!
//! This crate contains the Weld compiler and runtime, though users only interact with the
//! compiler. Users use Weld by constructing a Weld program (currently as a string), compiling
//! the string into a runnable _module_, and then running the module with in-memory data.
//!
//! Weld JITs code into the current process using LLVM. As a result, Weld users must have a version
//! of LLVM installed on their machine (currently, Weld uses LLVM 3.8).
//!
//! ## Example
//!
//! The following program shows a minimal Weld program that adds two numbers:
//!
//! ```rust,no_run
//! # extern crate weld;
//! #
//! # use weld::*;
//! #
//! #[repr(C)]
//! struct MyArgs {
//!     a: i32,
//!     b: i32,
//! }
//!
//! let code = "|a: i32, b: i32| a + b";
//! let ref conf = WeldConf::new();
//! let mut module = WeldModule::compile(code, conf).unwrap();
//!
//! // Weld accepts a packed C struct as an argument.
//! let ref args = MyArgs { a: 1, b: 50 };
//! let ref input = WeldValue::new_from_data(args as *const _ as Data);
//!
//! // Running a Weld module and reading a value out of it is unsafe!
//! unsafe {
//!     // Run the module, which returns a wrapper `WeldValue`.
//!     let result = module.run(conf, input).unwrap();
//!     // The data is just a pointer: cast it to the expected type
//!     let data = result.data() as *const i32;
//!
//!     let result = (*data).clone();
//!     assert_eq!(args.a + args.b, result);
//! }
//! ```
//! 
//! Users write a Weld program as a string, compile it into a module, and then pass packed
//! arguments into it to run the JITed code. The result is a pointer that represents the output of
//! the Weld program: we can cast that to the appropriate pointer type and read it by
//! dereferencing.
//!
//! ## Modules
//!
//! The `WeldModule` is the main entry point into Weld. Users can compile Weld programs using
//! `WeldModule::compile`, and then run compiled programs using `WeldModule::run`.
//!
//! The module functions can be configured in several ways. This configuration is controlled using
//! the `WeldConf` struct, which is effectively a dictionary of `String` key/value pairs that
//! control how a Weld program is compiled and run.
//!
//! ## Values
//!
//! Since Weld JITs code and implements a custom runtime, data passed in and out of it must be in a
//! specific, C-compatible packed format. The [Weld Github](http://github.com/weld-project/weld)
//! contains a plethora of information on how data should be formatted when passed into Weld, but
//! in short, it is **not** safe to simply pass Rust objects into Weld.
//!
//! `WeldModule` accepts and returns a wrapper struct called `WeldValue`, which wraps an opaque
//! `*const void` that Weld reads depending on the argument and return types of the Weld program.
//! Weld's main `run` function is thus `unsafe`: users need to guarantee that the data passed into
//! Weld is properly formatted!
//!
//! ### Passing Rust Values into Weld
//!
//! Currently, users need to manually munge Rust values into a format that Weld understands, as
//! specified [here](https://github.com/weld-project/weld/blob/master/docs/api.md). Eventually, we
//! may add a module in this crate that contains wrappers for some useful types. The current Rust
//! types can be passed safely into Weld already:
//!
//! * Primitive types such as `i8`, `i16`, and `f32`. These have a 1-1 correspondance with Weld.
//! * Rust structs with `repr(C)`.
//! 
//! Notably, `Vec<T>` _cannot_ be passed without adhering to the custom Weld format. Currently,
//! that format is defined as:
//!
//! ```
//! #[repr(C)]
//! struct WeldVec<T> {
//!     ptr: *const T,
//!     len: i64,
//! }
//! ```
//!
//! There is thus a straightforward conversion from `Vec<T>` to a `WeldVec<T>`.

#![cfg_attr(not(test), allow(dead_code))]

#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate log;

extern crate regex;
extern crate libc;
extern crate env_logger;
extern crate chrono;
extern crate fnv;
extern crate time;
extern crate code_builder;

use libc::{free, c_void};
use self::time::PreciseTime;

use std::error::Error;
use std::default::Default;
use std::ffi::{CString, CStr};
use std::fmt;

/// A macro for creating a `WeldError` with a message and an unknown error code.
#[macro_export]
macro_rules! weld_err {
    ( $($arg:tt)* ) => ({
        ::std::result::Result::Err($crate::WeldError::new_unknown(format!($($arg)*)))
    })
}

#[macro_use]
mod error;
mod annotation;
mod codegen;
mod optimizer;
mod syntax;
mod sir;
mod conf;

// Public interfaces.
pub mod ast;
pub mod util;
pub mod ffi;
pub mod runtime;

// Tests.
#[cfg(test)]
mod tests;

use conf::Backend;
use runtime::{WeldRuntimeErrno, WeldRun};
use util::stats::CompilationStats;

/// A wrapper for a C pointer.
pub type Data = *const libc::c_void;

/// A wrapper for a mutable C pointer.
pub type DataMut = *mut libc::c_void;

/// An identifier that uniquely identifies a call to `WeldModule::run`.
pub type RunId = i64;

/// An error when compiling or running a Weld program.
#[derive(Debug,Clone)]
pub struct WeldError {
    message: CString,
    code: WeldRuntimeErrno,
}

/// A `Result` that uses `WeldError`.
pub type WeldResult<T> = Result<T, WeldError>;

impl WeldError {
    /// Creates a new error with a particular message and error code.
    pub fn new<T: Into<Vec<u8>>>(message: T, code: WeldRuntimeErrno) -> WeldError {
        WeldError {
            message: CString::new(message).unwrap(),
            code: code,
        }
    }

    /// Creates a new error with a particular message and an `Unknown` error code.
    pub fn new_unknown<T: Into<Vec<u8>>>(message: T) -> WeldError {
        WeldError {
            message: CString::new(message).unwrap(),
            code: WeldRuntimeErrno::Unknown,
        }
    }

    /// Creates a new error with a particular message indicating success.
    ///
    /// The error returned by this function has a message "Success" and the error code `Success`.
    pub fn new_success() -> WeldError {
        WeldError::new(CString::new("Success").unwrap(), WeldRuntimeErrno::Success)
    }

    /// Returns the error code of this `WeldError`.
    pub fn code(&self) -> WeldRuntimeErrno {
        self.code
    }

    /// Returns the error message of this `WeldError`.
    pub fn message(&self) -> &CStr {
        self.message.as_ref()
    }
}

impl Default for WeldError {
    fn default() -> WeldError {
        WeldError {
            message: CString::new("").unwrap(),
            code: WeldRuntimeErrno::Success,
        }
    }
}

// Conversion from a compilation error to an external WeldError.
impl From<error::WeldCompileError> for WeldError {
    fn from(err: error::WeldCompileError) -> WeldError {
        WeldError::new(CString::new(err.description()).unwrap(), WeldRuntimeErrno::CompileError)
    }
}

/// A wrapper for data passed into and out of Weld.
#[derive(Debug,Clone)]
pub struct WeldValue {
    data: Data,
    run: Option<RunId>,
    backend: Backend,
}

impl WeldValue {
    /// Creates a new `WeldValue` with a particular data pointer.
    ///
    /// This function is used to wrap data that will be passed into Weld. Data passed into Weld
    /// should be in a standard format that Weld understands: this is usually some kind of packed C
    /// structure with a particular field layout.
    pub fn new_from_data(data: Data) -> WeldValue {
        WeldValue {
            data: data,
            run: None,
            backend: Backend::Unknown,
        }
    }

    /// Returns the data pointer of this `WeldValue`.
    pub fn data(&self) -> Data {
        self.data
    }

    /// Returns the run ID of this value if it has one.
    ///
    /// A `WeldValue` will only have a run ID if it was _returned_ by a Weld program. That is, a
    /// `WeldValue` that is created using `WeldValue::new_from_data` will always have a `run_id` of
    /// `None`.
    pub fn run_id(&self) -> Option<RunId> {
        match self.backend {
            Backend::LLVMWorkStealingBackend => self.run.clone(),
            Backend::LLVMSingleThreadBackend => {
                let run = unsafe { &mut *(self.run.unwrap() as *mut WeldRun) };
                Some(run.run_id())
            }
            Backend::Unknown => None,
        }
    }

    /// Returns the memory usage of this value in bytes.
    ///
    /// This equivalently returns the amount of memory allocated by a Weld run. If the value was
    /// not returned by Weld, returns `None`.
    pub fn memory_usage(&self) -> Option<i64> {
        match self.backend {
            Backend::LLVMWorkStealingBackend => {
                self.run.map(|v| unsafe { runtime::weld_run_memory_usage(v) } )
            } 
            Backend::LLVMSingleThreadBackend => {
                let run = unsafe { &mut *(self.run.unwrap() as *mut WeldRun) };
                Some(run.memory_usage())
            }
            Backend::Unknown => None,
        }
    }
}

// Custom Drop implementation that disposes a run.
impl Drop for WeldValue {
    fn drop(&mut self) {
        match self.backend {
            Backend::LLVMWorkStealingBackend => {
                if let Some(run_id) = self.run {
                    unsafe { runtime::weld_run_dispose(run_id); }
                }
            }
            Backend::LLVMSingleThreadBackend => {
                unsafe { Box::from_raw(self.run.unwrap() as *mut WeldRun) };
            }
            Backend::Unknown => (),
        }
    }
}

/// A struct used to configure compilation and the Weld runtime.
#[derive(Debug,Clone)]
pub struct WeldConf {
    dict: fnv::FnvHashMap<String, CString>,
}

impl WeldConf {
    /// Creates a new empty `WeldConf`.
    ///
    /// Weld configurations are unstructured key/value pairs. The configuration is used to modify
    /// how a Weld program is compiled (e.g., setting multi-thread support, configuring
    /// optimization passes, etc.) and how a Weld program is run (e.g., a memory limit, the number
    /// of threads to allocate the run, etc.).
    pub fn new() -> WeldConf {
        WeldConf { dict: fnv::FnvHashMap::default() }
    }

    /// Adds a configuration to this `WeldConf`.
    ///
    /// This method does not perform any checks to ensure that the key/value pairs are valid. If
    /// a `WeldConf` contains an invalid configuration option, the `WeldModule` methods that
    /// compile and run modules will fail with an error.
    pub fn set<K: Into<String>, V: Into<Vec<u8>>>(&mut self, key: K, value: V) {
        self.dict.insert(key.into(), CString::new(value).unwrap());
    }

    /// Get the value of the given key, or `None` if it is not set.
    pub fn get(&self, key: &str) -> Option<&CString> {
        self.dict.get(key)
    }
}

/// A compiled runnable Weld module.
pub struct WeldModule {
    /// A compiled, runnable module.
    llvm_module: codegen::CompiledModule,
    /// The backend the compiled module uses.
    ///
    /// This is passed to the `WeldValue` produced when the module is run.
    backend: Backend,
    /// The Weld parameter types this modules accepts.
    param_types: Vec<ast::Type>,
    /// The Weld return type of this module.
    return_type: ast::Type,
}

impl WeldModule {
    /// Creates a compiled `WeldModule` with a Weld program and configuration.
    ///
    /// A compiled module encapsulates JIT'd code in the current process. This function takes a
    /// string reprsentation of Weld code, parses it, and compiles it into machine code. The passed
    /// `WeldConf` can be used to configure how the code is compiled (see `conf.rs` for a list of
    /// compilation options). Each configuration option has a default value, so setting
    /// configuration options is optional.
    pub fn compile<S: AsRef<str>>(code: S, conf: &WeldConf) -> WeldResult<WeldModule> {
        use self::ast::*;
        let mut stats = CompilationStats::new();
        let ref mut conf = conf::parse(conf)?;
        let code = code.as_ref();

        // For dumping code, if enabled.
        let ref timestamp = format!("{}", time::now().to_timespec().sec);

        // Configuration.
        debug!("{:?}", conf);

        // Parse the string into a Weld AST.
        let start = PreciseTime::now();
        let program = syntax::parser::parse_program(code)?;
        let end = PreciseTime::now();
        stats.weld_times.push(("Parsing".to_string(), start.to(end)));

        // Substitute macros in the parsed program.
        let mut expr = syntax::macro_processor::process_program(&program)?;
        debug!("After macro substitution:\n{}\n", expr.pretty_print());

        // Dump the generated Weld program before applying any analyses.
        if conf.dump_code.enabled {
            info!("Writing code to directory '{}' with timestamp {}", &conf.dump_code.dir.display(), timestamp);
            util::write_code(expr.pretty_print(), "weld", timestamp, &conf.dump_code.dir);
        }

        // Uniquify symbol names.
        let start = PreciseTime::now();
        expr.uniquify()?;
        let end = PreciseTime::now();
        let mut uniquify_dur = start.to(end);

        // Infer types of expressions.
        let start = PreciseTime::now();
        expr.infer_types()?;
        let end = PreciseTime::now();
        stats.weld_times.push(("Type Inference".to_string(), start.to(end)));
        debug!("After type inference:\n{}\n", expr.pretty_print());

        // Apply optimization passes.
        optimizer::apply_passes(&mut expr,
                                &conf.optimization_passes,
                                &mut stats,
                                conf.enable_experimental_passes)?;

        // Uniquify again.
        let start = PreciseTime::now();
        expr.uniquify()?;
        let end = PreciseTime::now();
        uniquify_dur = uniquify_dur + start.to(end);
        stats.weld_times.push(("Uniquify outside Passes".to_string(), uniquify_dur));
        debug!("Optimized Weld program:\n{}\n", expr.pretty_print());

        // Convert the AST to SIR.
        let start = PreciseTime::now();
        let mut sir_prog = sir::ast_to_sir(&expr, conf.support_multithread)?;
        let end = PreciseTime::now();
        stats.weld_times.push(("AST to SIR".to_string(), start.to(end)));
        debug!("SIR program:\n{}\n", &sir_prog);

        // If enabled, apply SIR optimizations.
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
        if conf.dump_code.enabled {
            util::write_code(expr.pretty_print(), "weld", format!("{}-opt", timestamp), &conf.dump_code.dir);
            util::write_code(sir_prog.to_string(), "sir", timestamp, &conf.dump_code.dir);
        }

        // Generate code.
        let compiled_module = codegen::compile_program(&sir_prog,
                                                       conf,
                                                       &mut stats,
                                                       timestamp)?;
        debug!("\n{}\n", stats.pretty_print());

        let (param_types, return_type) = if let Type::Function(ref param_tys, ref return_ty) = expr.ty {
            (param_tys.clone(), *return_ty.clone())
        } else {
            unreachable!()
        };

        Ok(WeldModule { 
            llvm_module: compiled_module,
            backend: conf.backend.clone(),
            param_types: param_types,
            return_type: return_type,
        })
    }

    /// Run this `WeldModule` with a configuration and argument.
    ///
    /// This is the entry point for running a Weld program. The `WeldConf` specifies how runtime
    /// options for running the program (e.g., number of threads): see `conf.rs` for a list of
    /// runtime options. Each configuration option has a default value, so setting configuration
    /// options is optional.
    ///
    /// # Weld Arguments
    ///
    /// This function takes a `WeldValue` initialized using `WeldValue::new_from_data` or another Weld
    /// program. The value must encapsulate a valid pointer in a "Weld-compatible" format as
    /// specified by the [specification](https://github.com/weld-project/weld/blob/master/docs/api.md).
    /// This method is, as a result, `unsafe` because passing invalid data into a Weld program will
    /// cause undefined behavior.
    ///
    /// Note that most Rust values cannot be passed into Weld directly. For example, it is *not*
    /// safe to simply pass a raw pointer to a `Vec<T>` into Weld directly.
    pub unsafe fn run(&mut self, conf: &WeldConf, arg: &WeldValue) -> WeldResult<WeldValue> {
        let ref parsed_conf = conf::parse(conf)?;

        // This is the required input format of data passed into a compiled module.
        let input = Box::new(codegen::WeldInputArgs {
                             input: arg.data as i64,
                             nworkers: parsed_conf.threads,
                             mem_limit: parsed_conf.memory_limit,
                             run: 0,
                         });
        let ptr = Box::into_raw(input) as i64;

        // Runs the Weld program.
        let raw = self.llvm_module.run(ptr) as *const codegen::WeldOutputArgs;
        let result = (*raw).clone();

        // Free the boxed input.
        let _ = Box::from_raw(ptr as *mut codegen::WeldInputArgs);

        let value = WeldValue {
            data: result.output as *const c_void,
            run: Some(result.run),
            backend: self.backend.clone(),
        };

        // Check whether the run was successful -- if not, free the data in the module, andn return
        // an error indicating what went wrong.
        if result.errno != WeldRuntimeErrno::Success {
            // The WeldValue is automatically dropped and freed here.
            let message = CString::new(format!("Weld program failed with error {:?}", result.errno)).unwrap();
            Err(WeldError::new(message, result.errno))
        } else {
            // Weld allocates the output using libc malloc, but we cloned it, so free the output struct
            // here.
            free(raw as DataMut);
            Ok(value)
        }
    }

    /// Returns the Weld arguments types of this `WeldModule`.
    pub fn param_types(&self) -> Vec<ast::Type> {
        self.param_types.clone()
    }

    /// Returns the Weld return type of this `WeldModule`.
    pub fn return_type(&self) -> ast::Type {
        self.return_type.clone()
    }
}

/// A logging level for the compiler.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd)]
#[repr(u64)]
pub enum WeldLogLevel {
    Off = 0,
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl fmt::Display for WeldLogLevel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl From<WeldLogLevel> for log::LogLevelFilter {
    fn from(level: WeldLogLevel) -> log::LogLevelFilter {
        match level {
            WeldLogLevel::Error => log::LogLevelFilter::Error,
            WeldLogLevel::Warn => log::LogLevelFilter::Warn,
            WeldLogLevel::Info => log::LogLevelFilter::Info,
            WeldLogLevel::Debug => log::LogLevelFilter::Debug,
            WeldLogLevel::Trace => log::LogLevelFilter::Trace,
            _ => log::LogLevelFilter::Off
        }
    }
}

impl From<log::LogLevelFilter> for WeldLogLevel {
    fn from(level: log::LogLevelFilter) -> WeldLogLevel {
        match level {
            log::LogLevelFilter::Error => WeldLogLevel::Error,
            log::LogLevelFilter::Warn => WeldLogLevel::Warn,
            log::LogLevelFilter::Info => WeldLogLevel::Info,
            log::LogLevelFilter::Debug => WeldLogLevel::Debug,
            log::LogLevelFilter::Trace => WeldLogLevel::Trace,
            _ => WeldLogLevel::Off
        }
    }
}

/// Load a dynamic library that a Weld program can access.
///
/// The dynamic library is a C dynamic library identified by its filename.
pub fn load_linked_library<S: AsRef<str>>(filename: S) -> WeldResult<()> {
    codegen::load_library(filename.as_ref()).map_err(|e| WeldError::from(e))
}

/// Enables logging to stderr in Weld with the given log level.
///
/// This function is ignored if it has already been called once, or if some other code in the
/// process has initialized logging using Rust's `log` crate.
pub fn set_log_level(level: WeldLogLevel) {
    use util::colors::*;
    use util::colors::Color::*;

    let filter: log::LogLevelFilter = level.into();
    let format = |rec: &log::LogRecord| {
        let prefix = match rec.level() {
            log::LogLevel::Error => format_color(Red, "error"),
            log::LogLevel::Warn => format_color(Yellow, "warn"),
            log::LogLevel::Info => format_color(Yellow, "info"),
            log::LogLevel::Debug => format_color(Green, "debug"),
            log::LogLevel::Trace => format_color(Green, "trace"),
        };
        let date = chrono::Local::now().format("%T%.3f");
        format!("[{}] {}: {}", prefix, date, rec.args())
    };

    let mut builder = env_logger::LogBuilder::new();
    builder.format(format);
    builder.filter(None, filter);
    builder.init().unwrap_or(());
}
