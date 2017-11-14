// get spurious warnings for functions that are currently only used in tests. This is useful for
// development but can be removed later.
#![cfg_attr(not(test), allow(dead_code))]

#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate log;

extern crate regex;
extern crate libc;
extern crate env_logger;
extern crate chrono;
extern crate time;

use self::time::{PreciseTime, Duration};

use std::collections::HashMap;
use std::error::Error;
use libc::{c_char, c_void};
use std::ffi::{CStr, CString};

use common::WeldRuntimeErrno;
use common::WeldLogLevel;

/// Utility macro to create an Err result with a WeldError from a format string.
#[macro_export]
macro_rules! weld_err {
    ( $($arg:tt)* ) => ({
        ::std::result::Result::Err($crate::error::WeldError::new(format!($($arg)*)))
    })
}


pub mod annotations;
pub mod ast;
pub mod code_builder;
pub mod common;
pub mod error;
pub mod llvm;
pub mod macro_processor;
pub mod passes;
pub mod parser;
pub mod partial_types;
pub mod pretty_print;
pub mod program;
pub mod runtime;
pub mod sir;
pub mod tokenizer;
pub mod transforms;
pub mod type_inference;
pub mod conf;
pub mod util;
pub mod exprs;
pub mod expr_hash;

pub mod easy_ll;

extern "C" {
    pub fn free(ptr: *mut c_void);
}

/// A clean alias for a compiled LLVM module.
pub type WeldModule = llvm::CompiledModule;

/// An error passed as an opaque pointer using the runtime API.
pub struct WeldError {
    errno: WeldRuntimeErrno,
    message: CString,
}

impl WeldError {
    /// Returns a new WeldError where the message is just the errno name.
    fn new(errno: WeldRuntimeErrno) -> WeldError {
        WeldError {
            errno: errno,
            message: CString::new(errno.to_string()).unwrap(),
        }
    }
}

/// A value passed as an opaque pointer using the runtime API. `WeldValue` encapsulates some raw
/// data along with a `run_id` which designates the run a value is associated with. If the `run_id`
/// is `None` the data is owned by a caller outside the runtime.
pub struct WeldValue {
    data: *const c_void,
    module: Option<*mut WeldModule>,
    run_id: Option<i64>,
}

/// A configuration passed into the runtime to set compilation or execution parameters. This
/// struct has two methods other than the ones for allocation: `get` and `set`. `get` returns
/// the current string value of a configuration parameter, while `set` associates a string
/// parameter with a value.
pub struct WeldConf {
    dict: HashMap<CString, CString>,
}

impl WeldConf {
    fn new() -> WeldConf {
        WeldConf { dict: HashMap::new() }
    }
}

/// Tracks various compile-time statistics throughout the compiler.
pub struct CompilationStats {
    /// Running times for various Weld compiler components.
    pub weld_times: Vec<(String, Duration)>,
    /// Running times for Weld optimization passes.
    pub pass_times: Vec<(String, Duration)>,
    /// Running times for various LLVM components.
    pub llvm_times: Vec<(String, Duration)>,
}

impl CompilationStats {
    pub fn new() -> CompilationStats {
        CompilationStats {
            weld_times: Vec::new(),
            pass_times: Vec::new(),
            llvm_times: Vec::new(),
        }
    }

    /// Formats a duration for printing the statistics, in terms of milliseconds and microseconds.
    fn format_time(duration: &Duration) -> f64 {
        if duration.num_milliseconds() == 0 {
            if let Some(v) = duration.num_microseconds() {
                (v as f64) / 1000.0
            } else {
                0.0
            }
        } else {
            duration.num_milliseconds() as f64
        }
    }

    /// Returns pretty-printed statistics stored in `self`.
    pub fn pretty_print(&self) -> String {
        let mut result = String::new();
        result.push_str("Weld Compiler:\n");
        let mut total = Duration::milliseconds(0);
        for &(ref name, ref dur) in self.weld_times.iter() {
            result.push_str(&format!("\t{}: {:.3} ms\n", name, CompilationStats::format_time(dur)));
            total = total + *dur;
        }
        result.push_str(&format!("\t\x1b[0;32mWeld Compiler Total\x1b[0m {} ms\n", CompilationStats::format_time(&total)));

        let mut total = Duration::milliseconds(0);
        result.push_str("Weld Optimization Passes:\n");
        for &(ref name, ref dur) in self.pass_times.iter() {
            result.push_str(&format!("\t{}: {:.3} ms\n", name, CompilationStats::format_time(dur)));
            total = total + *dur;
        }
        result.push_str(&format!("\t\x1b[0;32mWeld Optimization Passes Total\x1b[0m {} ms\n", CompilationStats::format_time(&total)));

        let mut total = Duration::milliseconds(0);
        result.push_str("LLVM:\n");
        for &(ref name, ref dur) in self.llvm_times.iter() {
            result.push_str(&format!("\t{}: {:.3} ms\n", name, CompilationStats::format_time(dur)));
            total = total + *dur;
        }
        result.push_str(&format!("\t\x1b[0;32mLLVM Total\x1b[0m {} ms\n", CompilationStats::format_time(&total)));

        result
    }
}

#[no_mangle]
/// Return a new Weld configuration.
pub extern "C" fn weld_conf_new() -> *mut WeldConf {
    Box::into_raw(Box::new(WeldConf::new()))
}

#[no_mangle]
/// Free a Weld configuration.
pub unsafe extern "C" fn weld_conf_free(ptr: *mut WeldConf) {
    assert!(!ptr.is_null());
    Box::from_raw(ptr);
}

#[no_mangle]
/// Get a value for a given configuration.
pub unsafe extern "C" fn weld_conf_get(ptr: *const WeldConf, key: *const c_char) -> *const c_char {
    assert!(!ptr.is_null());
    let conf = &*ptr;

    let key = CStr::from_ptr(key);
    let key = key.to_owned();
    match conf.dict.get(&key) {
        Some(k) => k.as_ptr() as *const c_char,
        None => std::ptr::null_mut() as *const c_char,
    }
}

#[no_mangle]
/// Set the value of `key` to `value`.
pub unsafe extern "C" fn weld_conf_set(ptr: *mut WeldConf,
                                       key: *const c_char,
                                       value: *const c_char) {
    assert!(!ptr.is_null());
    let conf = &mut *ptr;

    let key = CStr::from_ptr(key);
    let key = key.to_owned();
    let val = CStr::from_ptr(value);
    let val = val.to_owned();

    conf.dict.insert(key, val);
}

#[no_mangle]
/// Returns a new Weld value.
///
/// The value returned by this method is *not* owned by the runtime, so any data this value refers
/// to must be managed by the caller.
pub extern "C" fn weld_value_new(data: *const c_void) -> *mut WeldValue {
    Box::into_raw(Box::new(WeldValue {
                               data: data,
                               module: None,
                               run_id: None,
                           }))
}

#[no_mangle]
/// Returns the Run ID of the value if the value is owned by the Weld runtime, or -1 if the caller
/// owns the value.
pub unsafe extern "C" fn weld_value_run(obj: *const WeldValue) -> i64 {
    assert!(!obj.is_null());
    let obj = &*obj;
    if let Some(rid) = obj.run_id {
        rid as i64
    } else {
        -1
    }
}

#[no_mangle]
/// Returns a pointer to the data wrapped by the given Weld value.
pub unsafe extern "C" fn weld_value_data(obj: *const WeldValue) -> *const c_void {
    assert!(!obj.is_null());
    let obj = &*obj;
    obj.data
}

#[no_mangle]
/// Returns a pointer to the module owning the given Weld value.
pub unsafe extern "C" fn weld_value_module(obj: *mut WeldValue) -> *mut WeldModule {
    assert!(!obj.is_null());
    let obj = &*obj;
    obj.module.unwrap()
}

#[no_mangle]
/// Frees a Weld value.
///
/// All Weld values must be freed using this call.
/// Weld values which are owned by the runtime also free the data they contain.
/// Weld values which are not owned by the runtime only free the structure used
/// to wrap the data; the actual data itself is owned by the caller.
pub unsafe extern "C" fn weld_value_free(obj: *mut WeldValue) {
    if obj.is_null() {
        return;
    }
    let value = &mut *obj;
    if let Some(run_id) = value.run_id {
        runtime::weld_run_dispose(run_id);
    }
    Box::from_raw(obj);
}

#[no_mangle]
/// Gets the memory allocated to a Weld value, which generally includes all memory allocated during
/// the execution of the module that created it (Weld does not currently free intermediate values
/// while running a module). Returns -1 if the value is not part of a valid Weld run.
pub unsafe extern "C" fn weld_value_memory_usage(obj: *mut WeldValue) -> libc::int64_t {
    if obj.is_null() {
        return -1 as libc::int64_t;
    }
    let value = &mut *obj;
    if let Some(run_id) = value.run_id {
        return runtime::weld_run_memory_usage(run_id);
    } else {
        return -1 as libc::int64_t;
    }
}

#[no_mangle]
/// Given some Weld code and a configuration, returns a runnable Weld module.
pub unsafe extern "C" fn weld_module_compile(code: *const c_char,
                                             conf: *const WeldConf,
                                             err_ptr: *mut WeldError)
                                             -> *mut WeldModule {
    info!("Started weld_module_compile");
    assert!(!code.is_null());
    assert!(!err_ptr.is_null());
    let err = &mut *err_ptr;

    let mut stats = CompilationStats::new();

    let conf = conf::parse(&*conf);
    if let Err(e) = conf {
        err.errno = WeldRuntimeErrno::ConfigurationError;
        err.message = CString::new(e.description().to_string()).unwrap();
        return std::ptr::null_mut();
    }
    let conf = conf.unwrap();

    let code = CStr::from_ptr(code);
    let code = code.to_str().unwrap().trim();

    let start = PreciseTime::now();
    info!("Started parsing program");
    let parsed = parser::parse_program(code);
    info!("Done parsing program");
    let end = PreciseTime::now();
    stats.weld_times.push(("Parsing".to_string(), start.to(end)));

    if let Err(e) = parsed {
        err.errno = WeldRuntimeErrno::CompileError;
        err.message = CString::new(e.description().to_string()).unwrap();
        return std::ptr::null_mut();
    }

    info!("Started compiling program");
    let module = llvm::compile_program(&parsed.unwrap(), &conf, &mut stats);
    info!("Done compiling program");

    if let Err(ref e) = module {
        err.errno = WeldRuntimeErrno::CompileError;
        err.message = CString::new(e.description().to_string()).unwrap();
        return std::ptr::null_mut();
    }

    debug!("\n{}\n", stats.pretty_print());

    info!("Done weld_module_compile");
    Box::into_raw(Box::new(module.unwrap()))
}

#[no_mangle]
/// Runs a module.
///
/// The module may write a value into the provided error pointer.
pub unsafe extern "C" fn weld_module_run(module: *mut WeldModule,
                                         conf: *const WeldConf,
                                         arg: *const WeldValue,
                                         err_ptr: *mut WeldError)
                                         -> *mut WeldValue {
    assert!(!module.is_null());
    assert!(!conf.is_null());
    assert!(!arg.is_null());
    assert!(!err_ptr.is_null());

    let module_ref = &mut *module;
    let module_callable = module_ref.llvm_mut();
    let arg = &*arg;
    let err = &mut *err_ptr;

    let conf = conf::parse(&*conf);
    if let Err(e) = conf {
        err.errno = WeldRuntimeErrno::ConfigurationError;
        err.message = CString::new(e.description().to_string()).unwrap();
        return std::ptr::null_mut();
    }
    let conf = conf.unwrap();

    #[derive(Clone)]
    struct I32Vec {
        data: *const i32,
        one: i64,
    }

    let input = Box::new(llvm::WeldInputArgs {
                             input: arg.data as i64,
                             nworkers: conf.threads,
                             mem_limit: conf.memory_limit,
                         });
    let ptr = Box::into_raw(input) as i64;
    // result_raw is allocated with ordinary malloc, hence the free below
    let result_raw = module_callable.run(ptr) as *const llvm::WeldOutputArgs;
    let result = (*result_raw).clone();

    let ret = Box::into_raw(Box::new(WeldValue {
                                         data: result.output as *const c_void,
                                         module: Some(module),
                                         run_id: Some(result.run_id),
                                     }));

    if result.errno != WeldRuntimeErrno::Success {
        weld_value_free(ret);
        err.errno = result.errno;
        err.message = CString::new(result.errno.to_string()).unwrap();
        free(result_raw as *mut c_void);
        std::ptr::null_mut()
    } else {
        free(result_raw as *mut c_void);
        ret
    }
}

#[no_mangle]
/// Frees a module.
///
/// Freeing a module does not free the memory it may have allocated. Values returned by the module
/// must be freed explicitly using `weld_value_free`.
pub unsafe extern "C" fn weld_module_free(ptr: *mut WeldModule) {
    if ptr.is_null() {
        return;
    }
    Box::from_raw(ptr);
}

#[no_mangle]
/// Return a new Weld error object.
pub extern "C" fn weld_error_new() -> *mut WeldError {
    Box::into_raw(Box::new(WeldError::new(WeldRuntimeErrno::Success)))
}

#[no_mangle]
/// Returns an error code for a Weld error object.
pub unsafe extern "C" fn weld_error_code(err: *mut WeldError) -> WeldRuntimeErrno {
    // TODO(shoumik): Handle this better.
    if err.is_null() {
        return WeldRuntimeErrno::Success;
    }
    let err = &mut *err;
    err.errno
}

#[no_mangle]
/// Returns a Weld error message.
pub unsafe extern "C" fn weld_error_message(err: *mut WeldError) -> *const c_char {
    if err.is_null() {
        return std::ptr::null();
    }
    let err = &mut *err;
    err.message.as_ptr() as *const c_char
}

#[no_mangle]
/// Frees a Weld error object.
pub unsafe extern "C" fn weld_error_free(err: *mut WeldError) {
    if err.is_null() {
        return;
    }
    Box::from_raw(err);
}

#[no_mangle]
/// Loads a dynamically linked library and makes it available to the LLVM compiler and JIT.
/// This function is safe to call multiple times on the same library file.
pub unsafe extern "C" fn weld_load_library(filename: *const c_char, err: *mut WeldError) {
    assert!(!err.is_null());
    let err = &mut *err;
    let filename = CStr::from_ptr(filename);
    let filename = filename.to_str().unwrap();
    if let Err(e) = easy_ll::load_library(filename) {
        err.errno = WeldRuntimeErrno::LoadLibraryError;
        err.message = CString::new(e.description()).unwrap();
    }
}

#[no_mangle]
/// Enables logging to stderr in Weld with the given log level.
/// This function is ignored if it has already been called once, or if some other code in the
/// process has initialized logging using Rust's `log` crate.
pub extern "C" fn weld_set_log_level(level: WeldLogLevel) {
    let filter = match level {
        WeldLogLevel::Error => log::LogLevelFilter::Error,
        WeldLogLevel::Warn => log::LogLevelFilter::Warn,
        WeldLogLevel::Info => log::LogLevelFilter::Info,
        WeldLogLevel::Debug => log::LogLevelFilter::Debug,
        WeldLogLevel::Trace => log::LogLevelFilter::Trace,
        _ => log::LogLevelFilter::Off
    };

    let format = |rec: &log::LogRecord| {
        let prefix = match rec.level() {
            log::LogLevel::Error => "\x1b[0;31merror\x1b[0m",
            log::LogLevel::Warn => "\x1b[0;33mwarn\x1b[0m",
            log::LogLevel::Info => "\x1b[0;33minfo\x1b[0m",
            log::LogLevel::Debug => "\x1b[0;32mdebug\x1b[0m",
            log::LogLevel::Trace => "\x1b[0;32mtrace\x1b[0m",
        };
        let date = chrono::Local::now().format("%T%.3f");
        format!("[{}] {}: {}", prefix, date, rec.args())
    };

    let mut builder = env_logger::LogBuilder::new();
    builder.format(format);
    builder.filter(None, filter);
    builder.init().unwrap_or(());
}

#[cfg(test)]
mod tests;
