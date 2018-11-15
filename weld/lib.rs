//!
//! Weld is a runtime for improving the performance of data-intensive applications. It optimizes
//! across libraries and functions by expressing the core computations in libraries using a small
//! common intermediate representation, similar to CUDA and OpenCL.
//!
//! # Using Weld
//!
//! Weld is a small programming language that supports _parallel loops_ and
//! _builders_, which are declarative objects that specify how to buid results. The parallel loops
//! can be used in conjunction with the builders to build a result in parallel.
//!
//! This crate contains the Weld compiler and runtime, though users only interact with the
//! compiler. Users use Weld by constructing a Weld program (currently as a string), compiling
//! the string into a runnable _module_, and then running the module with in-memory data.
//!
//! Weld JITs code into the current process using LLVM. As a result, Weld users must have a version
//! of LLVM installed on their machine (currently, Weld uses LLVM 6).
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
//! // A context manages memory.
//! let ref mut context = WeldContext::new(conf).unwrap();
//!
//! // Running a Weld module and reading a value out of it is unsafe!
//! unsafe {
//!     // Run the module, which returns a wrapper `WeldValue`.
//!     let result = module.run(context, input).unwrap();
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
//!
//! The `data` module defines layouts of Weld-compatible types, and also contains some methods for
//! converting Rust values into Weld values.
//!
//! ## Contexts
//!
//! A context manages state such as allocation information. A context is passed into
//! `WeldModule::run` and updated by the compiled Weld program.
//!
//! The `WeldContext` struct wraps a context. Contexts are internally reference counted because
//! values produced by Weld hold references to the context in which they are allocated. The memory
//! backing a `WeldContext` is freed when all references to the context are dropped.
//!
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
extern crate uuid;

use self::time::PreciseTime;

use std::error::Error;
use std::default::Default;
use std::ffi::{CString, CStr};
use std::fmt;

use std::rc::Rc;
use std::cell::RefCell;

use uuid::Uuid;

/// A macro for creating a `WeldError` with a message and an unknown error code.
#[macro_export]
macro_rules! weld_err {
    ( $($arg:tt)* ) => ({
        ::std::result::Result::Err($crate::WeldError::new_unknown(format!($($arg)*)))
    })
}


/// A build ID.
///
/// If Weld was compiled in a non-standard manner (i.e., without Cargo), this will be unknown.
pub const BUILD: &'static str = env!("BUILD_ID");

/// Weld version.
///
/// If Weld was compiled in a non-standard manner (i.e., without Cargo), this will be unknown.
pub const VERSION: Option<&'static str> = option_env!("CARGO_PKG_VERSION");

#[macro_use]
mod error;

mod codegen;
mod conf;
mod optimizer;
mod syntax;
mod sir;
mod util;

// Public interfaces.
pub mod ast;
pub mod ffi;
pub mod runtime;
pub mod data;

pub use conf::constants::*;

// Tests.
#[cfg(test)]
mod tests;

use conf::ParsedConf;
use runtime::WeldRuntimeContext;
use util::dump::{DumpCodeFormat, write_code};
use util::stats::CompilationStats;

// Error codes are exposed publicly.
pub use runtime::WeldRuntimeErrno;

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

/// A context for a Weld program.
///
/// Contexts are internally reference counted, so cloning a context will produce a reference to the
/// same internal object. The reference-counted internal object is protected via a `RefCell` to
/// prevent double-mutable-borrows: this is necessary because contexts may not be passed into
/// multiple `WeldModule::run` calls in parallel, even if they are cloned (since cloned contexts
/// point to the same underlying object).
///
/// Contexts are *not* thread-safe, and thus do not implement `Send+Sync`.
#[derive(Clone,Debug,PartialEq)]
pub struct WeldContext {
    context: Rc<RefCell<WeldRuntimeContext>>,
}

// Public API.
impl WeldContext {
    /// Returns a new `WeldContext` with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is malformed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use weld::{WeldConf, WeldContext};
    ///
    /// // Create a new default configuration.
    /// let ref mut conf = WeldConf::new();
    ///
    /// // Set 1KB memory limit, 2 worker threads.
    /// conf.set("weld.memory.limit", "1024");
    /// conf.set("weld.threads", "2");
    ///
    /// // Create a context.
    /// let context = WeldContext::new(conf).unwrap();
    /// ```
    pub fn new(conf: &WeldConf) -> WeldResult<WeldContext> {
        let ref mut conf = ParsedConf::parse(conf)?;
        let threads = conf.threads;
        let mem_limit = conf.memory_limit;

        let run = WeldRuntimeContext::new(threads as i32, mem_limit);
        Ok(WeldContext {
            context: Rc::new(RefCell::new(run))
        })
    }

    /// Returns the memory used by this context.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use weld::{WeldConf, WeldContext};
    ///
    /// let context = WeldContext::new(&WeldConf::new()).unwrap();
    /// assert_eq!(context.memory_usage(), 0);
    /// ```
    pub fn memory_usage(&self) -> i64 {
        self.context.borrow().memory_usage()
    }

    /// Returns the memory limit of this context.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use weld::{WeldConf, WeldContext};
    ///
    /// let ref mut conf = WeldConf::new();
    ///
    /// // Set 1KB memory limit, 2 worker threads.
    /// conf.set("weld.memory.limit", "1024");
    ///
    /// let context = WeldContext::new(conf).unwrap();
    /// assert_eq!(context.memory_limit(), 1024);
    /// ```
    pub fn memory_limit(&self) -> i64 {
        self.context.borrow().memory_limit()
    }
}

impl WeldError {
    /// Creates a new error with a message and error code.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use weld::{WeldError, WeldRuntimeErrno};
    ///
    /// let err = WeldError::new("A new error", WeldRuntimeErrno::Unknown);
    /// ```
    pub fn new<T: Into<Vec<u8>>>(message: T, code: WeldRuntimeErrno) -> WeldError {
        WeldError {
            message: CString::new(message).unwrap(),
            code: code,
        }
    }

    /// Creates a new error with a particular message and an `Unknown` error code.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use weld::{WeldError, WeldRuntimeErrno};
    ///
    /// let err = WeldError::new_unknown("A new unknown error");
    /// assert_eq!(err.code(), WeldRuntimeErrno::Unknown);
    /// ```
    pub fn new_unknown<T: Into<Vec<u8>>>(message: T) -> WeldError {
        WeldError {
            message: CString::new(message).unwrap(),
            code: WeldRuntimeErrno::Unknown,
        }
    }

    /// Creates a new error with a particular message indicating success.
    ///
    /// The error returned by this function has a message "Success" and the error code
    /// `WeldRuntimeErrno::Success`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use weld::{WeldError, WeldRuntimeErrno};
    ///
    /// let err = WeldError::new_success();
    /// assert_eq!(err.code(), WeldRuntimeErrno::Success);
    /// assert_eq!(err.message().to_str().unwrap(), "Success");
    /// ```
    pub fn new_success() -> WeldError {
        WeldError::new(CString::new("Success").unwrap(), WeldRuntimeErrno::Success)
    }

    /// Returns the error code of this `WeldError`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use weld::{WeldError, WeldRuntimeErrno};
    ///
    /// let err = WeldError::new("An out of memory error", WeldRuntimeErrno::OutOfMemory);
    /// assert_eq!(err.code(), WeldRuntimeErrno::OutOfMemory);
    /// ```
    pub fn code(&self) -> WeldRuntimeErrno {
        self.code
    }

    /// Returns the error message of this `WeldError`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use weld::{WeldError, WeldRuntimeErrno};
    ///
    /// let err = WeldError::new("Custom error", WeldRuntimeErrno::Unknown);
    /// assert_eq!(err.message().to_str().unwrap(), "Custom error");
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
///
/// Values produced by Weld (i.e., as a return value from `WeldModule::run`) hold a reference to
/// the context they are allocated in.
#[derive(Debug,Clone)]
pub struct WeldValue {
    data: Data,
    run: Option<RunId>,
    context: Option<WeldContext>,
}

impl WeldValue {
    /// Creates a new `WeldValue` with a particular data pointer.
    ///
    /// This function is used to wrap data that will be passed into Weld. Data passed into Weld
    /// should be in a standard format that Weld understands: this is usually some kind of packed C
    /// structure with a particular field layout.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use weld::{Data, WeldValue};
    ///
    /// let vec = vec![1, 2, 3];
    /// let value = WeldValue::new_from_data(vec.as_ptr() as Data);
    /// ```
    pub fn new_from_data(data: Data) -> WeldValue {
        WeldValue {
            data: data,
            run: None,
            context: None,
        }
    }

    /// Returns the data pointer of this `WeldValue`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use weld::{Data, WeldValue};
    ///
    /// let vec = vec![1, 2, 3];
    /// let value = WeldValue::new_from_data(vec.as_ptr() as Data);
    ///
    /// assert_eq!(vec.as_ptr() as Data, value.data() as Data);
    /// ```
    pub fn data(&self) -> Data {
        self.data
    }

    /// Returns the context of this value.
    ///
    /// This method will return `None` if the value does not have a context (e.g., if it was
    /// initialized using `new_from_data`). If it does have a context, the reference count of the
    /// context is increased -- the context will thus not be dropped until this reference is
    /// dropped.
    ///
    /// # Examples
    ///
    /// Values created using `new_from_data` return `None`:
    ///
    /// ```rust
    /// use weld::{Data, WeldValue};
    ///
    /// let vec = vec![1, 2, 3];
    /// let value = WeldValue::new_from_data(vec.as_ptr() as Data);
    ///
    /// assert!(value.context().is_none());
    /// ```
    ///
    /// Values created by Weld return the context that owns the data:
    ///
    /// ```rust,no_run
    /// use weld::*;
    /// use std::cell::Cell;
    ///
    /// // Wrap in Cell so we can get a raw pointer
    /// let input = Cell::new(1 as i32);
    ///
    /// let ref conf = WeldConf::new();
    /// let mut module = WeldModule::compile("|x: i32| x + 1", conf).unwrap();
    ///
    /// let ref input_value = WeldValue::new_from_data(input.as_ptr() as Data);
    ///
    /// let ref mut context = WeldContext::new(conf).unwrap();
    /// let result = unsafe { module.run(context, input_value).unwrap() };
    ///
    /// assert!(result.context().is_some());
    /// assert_eq!(result.context().as_mut().unwrap(), context);
    /// ```
    pub fn context(&self) -> Option<WeldContext> {
        self.context.clone()
    }

    /// Returns the run ID of this value if it has one.
    ///
    /// A `WeldValue` will only have a run ID if it was _returned_ by a Weld program. That is, a
    /// `WeldValue` that is created using `WeldValue::new_from_data` will always have a `run_id` of
    /// `None`.
    pub fn run_id(&self) -> Option<RunId> {
        Some(0)
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
    ///
    /// # Examples
    ///
    /// ```rust
    /// use weld::WeldConf;
    ///
    /// let conf = WeldConf::new();
    /// ```
    pub fn new() -> WeldConf {
        WeldConf { dict: fnv::FnvHashMap::default() }
    }

    /// Adds a configuration to this `WeldConf`.
    ///
    /// This method does not perform any checks to ensure that the key/value pairs are valid. If
    /// a `WeldConf` contains an invalid configuration option, the `WeldModule` methods that
    /// compile and run modules will fail with an error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use weld::WeldConf;
    ///
    /// let mut conf = WeldConf::new();
    /// conf.set("weld.memory.limit", "1024");
    ///
    /// // Invalid key/value pairs are also allowed but may raise errors.
    /// conf.set("weld.madeUpConfig", "madeUpValue");
    /// ```
    pub fn set<K: Into<String>, V: Into<Vec<u8>>>(&mut self, key: K, value: V) {
        self.dict.insert(key.into(), CString::new(value).unwrap());
    }

    /// Get the value of the given key, or `None` if it is not set.
    ///
    /// # Examples
    ///
    ///```rust
    /// use weld::WeldConf;
    ///
    /// let mut conf = WeldConf::new();
    /// conf.set("weld.memory.limit", "1024");
    ///
    /// let value = conf.get("weld.memory.limit").unwrap().to_str().unwrap();
    /// assert_eq!(value, "1024");
    ///
    /// let value = conf.get("non-existant key");
    /// assert!(value.is_none());
    /// ```
    pub fn get(&self, key: &str) -> Option<&CString> {
        self.dict.get(key)
    }
}

/// A compiled runnable Weld module.
#[derive(Debug)]
pub struct WeldModule {
    /// A compiled, runnable module.
    llvm_module: codegen::CompiledModule,
    /// The Weld parameter types this modules accepts.
    param_types: Vec<ast::Type>,
    /// The Weld return type of this module.
    return_type: ast::Type,
    /// A unique identifier for a module.
    module_id: Uuid,
}

impl WeldModule {
    /// Creates a compiled `WeldModule` with a Weld program and configuration.
    ///
    /// A compiled module encapsulates JIT'd code in the current process. This function takes a
    /// string reprsentation of Weld code, parses it, and compiles it into machine code. The passed
    /// `WeldConf` can be used to configure how the code is compiled (see `conf.rs` for a list of
    /// compilation options). Each configuration option has a default value, so setting
    /// configuration options is optional.
    ///
    /// # Errors
    ///
    /// * If the provided code does not compile (e.g., due to a syntax error), a compile error is
    /// returned.
    /// * If the provided configuration has an invalid configuration option, a compile
    /// error is returned.
    ///
    /// # Examples
    ///
    /// Compiling a valid program:
    ///
    /// ```rust,no_run
    /// use weld::*;
    ///
    /// let ref conf = WeldConf::new();
    /// let code = "|| 1";
    ///
    /// let mut module = WeldModule::compile(code, conf);
    /// assert!(module.is_ok());
    /// ```
    ///
    /// Invalid programs or configurations will return compile errors:
    ///
    /// ```rust,no_run
    /// # use weld::*;
    /// let ref mut conf = WeldConf::new();
    ///
    /// // Type error in program!
    /// let mut module = WeldModule::compile("|| 1 + f32(1)", conf);
    /// assert!(module.is_err());
    ///
    /// let err = module.unwrap_err();
    /// assert_eq!(err.code(), WeldRuntimeErrno::CompileError);
    ///
    /// conf.set("weld.memory.limit", "invalidLimit");
    /// let mut module = WeldModule::compile("|| 1", conf);
    /// assert!(module.is_err());
    ///
    /// let err = module.unwrap_err();
    /// assert_eq!(err.code(), WeldRuntimeErrno::CompileError);
    /// ```
    pub fn compile<S: AsRef<str>>(code: S, conf: &WeldConf) -> WeldResult<WeldModule> {
        use self::ast::*;

        let e2e_start = PreciseTime::now();
        let mut stats = CompilationStats::new();
        let ref mut conf = ParsedConf::parse(conf)?;
        let code = code.as_ref();

        let uuid = Uuid::new_v4();

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

        let unoptimized_code = expr.pretty_print();
        info!("Compiling module with UUID={}, code\n{}",
              uuid.to_hyphenated(),
              unoptimized_code);

        // Dump the generated Weld program before applying any analyses.
        nonfatal!(write_code(&unoptimized_code, DumpCodeFormat::Weld, &conf.dump_code));

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
        let mut sir_prog = sir::ast_to_sir(&expr)?;
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

        nonfatal!(write_code(expr.pretty_print(), DumpCodeFormat::WeldOpt, &conf.dump_code));
        nonfatal!(write_code(sir_prog.to_string(), DumpCodeFormat::SIR, &conf.dump_code));

        // Generate code.
        let compiled_module = codegen::compile_program(&sir_prog, conf, &mut stats)?;
        debug!("\n{}\n", stats.pretty_print());

        let (param_types, return_type) = if let Type::Function(ref param_tys, ref return_ty) = expr.ty {
            (param_tys.clone(), *return_ty.clone())
        } else {
            unreachable!()
        };

        let end = PreciseTime::now();
        let duration = e2e_start.to(end);
        let us = duration.num_microseconds().unwrap_or(std::i64::MAX);
        let e2e_ms: f64 = us as f64 / 1000.0;
        info!("Compiled module with UUID={} in {} ms",
              uuid.to_hyphenated(),
              e2e_ms);

        Ok(WeldModule {
            llvm_module: compiled_module,
            param_types: param_types,
            return_type: return_type,
            module_id: uuid,
        })
    }

    /// Run this `WeldModule` with a context and argument.
    ///
    /// This is the entry point for running a Weld program. The argument is a `WeldValue` that
    /// encapsulates a pointer to the argument. See the section below about how this argument
    /// should be structured.
    ///
    /// The context captures _state_: in particular, it holds the memory allocated by a `run`.
    /// Contexts can be reused across runs and modules. Contexts are primarily useful for passing
    /// mutable state---builders---in and out of Weld and updating them in place. For example, a
    /// program can compute some partial result, return a builder, and then pass the builder as a
    /// `WeldValue` back into `run` _with the same context_ to continue updating that builder.
    ///
    /// Contexts are not thread-safe---this is enforced in Rust by having this function take a
    /// mutable reference to a context. If a context is cloned, this constraint is maintained via
    /// _interior mutability_: contexts internally hold a `RefCell` that is mutably borrowed by
    /// this function, so a panic will be thrown if multiple callers try to `run` a module with the
    /// same context.
    ///
    /// # Structuring Arguments
    ///
    /// This function takes a `WeldValue` initialized using `WeldValue::new_from_data` or another Weld
    /// program. The value must encapsulate a valid pointer in a "Weld-compatible" format as
    /// specified by the [specification](https://github.com/weld-project/weld/blob/master/docs/api.md).
    /// This method is, as a result, `unsafe` because passing invalid data into a Weld program will
    /// cause undefined behavior.
    ///
    /// Note that most Rust values cannot be passed into Weld directly. For example, it is *not*
    /// safe to simply pass a raw pointer to a `Vec<T>` into Weld directly.
    ///
    /// # Errors
    ///
    /// This method may return any of the errors specified in `WeldRuntimeErrno`, if a runtime
    /// error occurs during the execution of the program. Currently, the implementation panics if a
    /// runtime error is thrown.
    ///
    /// # Panics
    ///
    /// The current implementation panics whenever the runtime throws an error. This function will
    /// also panic if the same context is passed to `run` at once (this is possible if, e.g., if a
    /// context is cloned).
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use weld::*;
    /// use std::cell::Cell;
    ///
    /// // Wrap in Cell so we can get a raw pointer
    /// let input = Cell::new(1 as i32);
    /// let ref conf = WeldConf::new();
    ///
    /// // Program that adds one to an i32.
    /// let mut module = WeldModule::compile("|x: i32| x + 1", conf).unwrap();
    /// let ref input_value = WeldValue::new_from_data(input.as_ptr() as Data);
    /// let ref mut context = WeldContext::new(conf).unwrap();
    ///
    /// // Running is unsafe, since we're outside of Rust in JIT'd code, operating over
    /// // raw pointers.
    /// let result = unsafe { module.run(context, input_value).unwrap() };
    ///
    /// assert!(result.context().is_some());
    ///
    /// // Unsafe to read raw pointers!
    /// unsafe {
    ///     // The data is just a raw pointer: cast it to the expected type.
    ///     let data = result.data() as *const i32;
    ///
    ///     let result = (*data).clone();
    ///     assert_eq!(input.get() + 1, result);
    /// }
    /// ```
    pub unsafe fn run(&self,
                      context: &mut WeldContext,
                      arg: &WeldValue) -> WeldResult<WeldValue> {

        let start = PreciseTime::now();
        let nworkers = context.context.borrow().threads();
        let mem_limit = context.context.borrow().memory_limit();

         // Borrow the inner context mutably since we pass a mutable pointer to it to the compiled
        // module. This enforces the single-mutable-borrow rule manually for contexts.
        let mut context_borrowed = context.context.borrow_mut();

        let (raw, result) = {
            // This is the required input format of data passed into a compiled module.
            let input = Box::new(codegen::WeldInputArgs {
                input: arg.data as i64,
                nworkers: nworkers,
                mem_limit: mem_limit,
                run: context.context.as_ptr() as i64,
            });
            let ptr = Box::into_raw(input) as i64;

            // Runs the Weld program.
            let raw = self.llvm_module.run(ptr) as *const codegen::WeldOutputArgs;
            let result = (*raw).clone();

            // Free the boxed input.
            let _ = Box::from_raw(ptr as *mut codegen::WeldInputArgs);

            (raw, result)
        };

        let value = WeldValue {
            data: result.output as Data,
            run: None,
            context: Some(context.clone()),
        };

        let end = PreciseTime::now();
        let duration = start.to(end);
        let us = duration.num_microseconds().unwrap_or(std::i64::MAX);
        let ms: f64 = us as f64 / 1000.0;
        debug!("Ran module UUID={} in {} ms",
              self.module_id.to_hyphenated(), ms);

        // Check whether the run was successful -- if not, free the data in the module, andn return
        // an error indicating what went wrong.
        if result.errno != WeldRuntimeErrno::Success {
            // The WeldValue is automatically dropped and freed here.
            let message = CString::new(format!("Weld program failed with error {:?}", result.errno)).unwrap();
            Err(WeldError::new(message, result.errno))
        } else {
            // Free the WeldOutputArgs struct.
            context_borrowed.free(raw as *mut u8);
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

    info!("Weld Version {} (Build {})", VERSION.unwrap_or("unknown"), BUILD);
}
