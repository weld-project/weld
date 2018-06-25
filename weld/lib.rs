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

use libc::c_void;
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
// TODO these probably shouldn't all be public...
pub mod ast;
pub mod util;
pub mod ffi;
pub mod runtime;

// Tests.
#[cfg(test)]
mod tests;

use runtime::WeldRuntimeErrno;
use util::stats::CompilationStats;

// XXX This should ideally be runtime-configurable!
use codegen::llvm as backend;

// This is needed to free the output struct of a Weld run.
extern "C" {
    fn free(ptr: *mut c_void);
}

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
    run_id: Option<RunId>,
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
            run_id: None,
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
        self.run_id
    }

    /// Returns the memory usage of this value.
    ///
    /// This equivalently returns the amount of memory allocated by a Weld run. If the value was
    /// not returned by Weld, returns `None`.
    pub fn memory_usage(&self) -> Option<i64> {
        self.run_id.map(|v| unsafe { runtime::weld_run_memory_usage(v) } )
    }
}

// Custom Drop implementation that disposes a run.
impl Drop for WeldValue {
    fn drop(&mut self) {
        if let Some(run_id) = self.run_id {
            unsafe { runtime::weld_run_dispose(run_id); }
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
    llvm_module: backend::CompiledModule,
}

impl WeldModule {
    /// Creates a compiled `WeldModule` with a Weld program and configuration.
    ///
    /// A compiled module encapsulates JIT'd code in the current process. This function takes a
    /// string reprsentation of Weld code, parses it, and compiles it into machine code. The passed
    /// `WeldConf` can be used to configure how the code is compiled (see `conf.rs` for a list of
    /// compilation options). Each configuration option has a default value, so setting
    /// configuration options is optional.
    #[allow(unreachable_code)]
    pub fn compile<S: AsRef<str>>(code: S, conf: &WeldConf) -> WeldResult<WeldModule> {
        let mut stats = CompilationStats::new();
        let ref parsed_conf = conf::parse(conf)?;
        let code = code.as_ref();

        // TODO create a macro like this to measure steps...
        // TODO move compilation stuff in llvm::codegen::mod.rs to here.
        // let program = profile!("Parsing", stats, syntax::parser::parse_program(code)?);

        let start = PreciseTime::now();
        let program = syntax::parser::parse_program(code)?;
        let end = PreciseTime::now();
        stats.weld_times.push(("Parsing".to_string(), start.to(end)));

        // Print the code generated by the new codegen.
        let module = backend::compile_program(&program, parsed_conf, &mut stats)?;
        debug!("\n{}\n", stats.pretty_print());

        Ok(WeldModule { llvm_module: module })
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
    /// Note that most Rust values cannot be passed into Weld directly. That is, it is *not* safe
    /// to simply pass a pointer to a `Vec<T>` into Weld directly.
    pub unsafe fn run(&mut self, conf: &WeldConf, arg: &WeldValue) -> WeldResult<WeldValue> {
        let ref parsed_conf = conf::parse(conf)?;

        // This is the required input format of data passed into a compiled module.
        let input = Box::new(backend::WeldInputArgs {
                             input: arg.data as i64,
                             nworkers: parsed_conf.threads,
                             mem_limit: parsed_conf.memory_limit,
                         });
        let ptr = Box::into_raw(input) as i64;

        // Runs the Weld program.
        let raw = self.llvm_module.run(ptr) as *const backend::WeldOutputArgs;
        let result = (*raw).clone();

        let value = WeldValue {
            data: result.output as *const c_void,
            // TODO change this to Option<WeldRun>!
            run_id: Some(0),
        };

        // Check whether the run was successful -- if not, free the data in the module, andn return
        // an error indicating what went wrong.
        if result.errno != WeldRuntimeErrno::Success {
            // The WeldValue is automatically dropped and freed here.
            let message = CString::new(format!("Weld program failed with error {:?}", result.errno)).unwrap();
            Err(WeldError::new(message, result.errno))
        } else {
            // Weld allocates the output using malloc, but we cloned it, so free the output struct
            // here.
            free(raw as DataMut);
            Ok(value)
        }
    }

    /// Returns the Weld arguments types of this `WeldModule`.
    pub fn param_types(&self) -> Vec<ast::Type> {
        self.llvm_module.param_types().clone()
    }

    /// Returns the Weld return type of this `WeldModule`.
    pub fn return_type(&self) -> ast::Type {
        self.llvm_module.return_type().clone()
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
    use error::WeldCompileError;
    codegen::llvm::load_library(filename.as_ref()).map_err(|e| WeldError::from(WeldCompileError::from(e)))
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
