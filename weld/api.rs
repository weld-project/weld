
extern crate libc;
extern crate fnv;
extern crate time;

use libc::c_void;
use self::time::PreciseTime;

use super::common::WeldRuntimeErrno;
use super::runtime;

use super::error;
use super::conf;
use super::llvm;
use super::parser;
use super::CompilationStats;

use std::error::Error;
use std::default::Default;
use std::ffi::{CString, CStr};

// This is needed to free the output struct of a Weld run.
extern "C" {
    fn free(ptr: *mut c_void);
}

/// A wrapper for a C pointer.
pub type Data = *const libc::c_void;

/// A wrapper for a mutable C pointer.
pub type DataMut = *mut libc::c_void;

/// A run ID that uniquely identifies a call to `WeldModule::run`.
pub type RunId = i64;

/// An error produced by Weld.
pub struct WeldError {
    message: CString,
    code: WeldRuntimeErrno,
}

/// A wrapper representing the API-exposed `WeldError`.
type WeldResult<T> = Result<T, WeldError>;

impl WeldError {
    /// Return a new error.
    pub fn new(message: CString, code: WeldRuntimeErrno) -> WeldError {
        WeldError {
            message: message,
            code: code,
        }
    }

    /// Returns a new error indicating success.
    pub fn new_success() -> WeldError {
        WeldError::new(CString::new("Success").unwrap(), WeldRuntimeErrno::Success)
    }

    /// Returns the error code.
    pub fn code(&self) -> WeldRuntimeErrno {
        self.code
    }

    /// Returns the error message.
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

impl From<error::WeldError> for WeldError {
    fn from(err: error::WeldError) -> WeldError {
        WeldError::new(CString::new(err.description()).unwrap(), WeldRuntimeErrno::CompileError)
    }
}

/// Wraps a value produced or consumed by a Weld program. Values that are passed into Weld can be
/// initialized using `WeldValue::new_from_data`. Weld also returns an output wrapped a
/// `WeldValue`.
pub struct WeldValue {
    data: Data,
    run_id: Option<RunId>,
}

impl WeldValue {
    /// Wrap a value represented as a C pointer as a `WeldValue`.
    pub fn new_from_data(data: Data) -> WeldValue {
        WeldValue {
            data: data,
            run_id: None,
        }
    }

    /// Returns the data pointer of this value.
    pub fn data(&self) -> Data {
        self.data
    }

    /// Returns the run ID of this value, if it has one. A `WeldValue` will only have a run ID if
    /// it was returned by a Weld program.
    pub fn run_id(&self) -> Option<RunId> {
        self.run_id
    }

    /// Returns the memory usage of this value. This equivalently returns the amount of memory
    /// allocated by a Weld run. If the value was not returned by Weld, returns `None`.
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

/// Describes a Weld configuration, which is a dictionary of key-value pairs.
pub struct WeldConf {
    dict: fnv::FnvHashMap<String, CString>,
}

impl WeldConf {
    /// Returns a new, empty configuration.
    pub fn new() -> WeldConf {
        WeldConf { dict: fnv::FnvHashMap::default() }
    }

    /// Add a configuration option to `self.`
    pub fn set<K: Into<String>, V: Into<Vec<u8>>>(&mut self, key: K, value: V) {
        self.dict.insert(key.into(), CString::new(value).unwrap());
    }

    /// Get the value of the given key, or `None` if it is not set.
    pub fn get(&self, key: &str) -> Option<&CString> {
        self.dict.get(key)
    }
}

/// A compiled Weld module.
pub struct WeldModule {
    llvm_module: llvm::CompiledModule,
}

impl WeldModule {

    /// Compiles Weld code represented as a string into a `WeldModule`, with the given
    /// configuration.
    pub fn compile<S: AsRef<str>>(code: S, conf: &WeldConf) -> WeldResult<WeldModule> {
        let mut stats = CompilationStats::new();
        let ref parsed_conf = conf::parse(conf)?;
        let code = code.as_ref();

        let start = PreciseTime::now();
        let program = parser::parse_program(code)?;
        let end = PreciseTime::now();
        stats.weld_times.push(("Parsing".to_string(), start.to(end)));

        let module = llvm::compile_program(&program, parsed_conf, &mut stats)?;
        debug!("\n{}\n", stats.pretty_print());

        Ok(WeldModule { llvm_module: module })
    }

    /// Run `self` with a given runtime configuraotion and argument, returning a `WeldValue`
    /// wrapping the return value (or an error if something went wrong).
    pub unsafe fn run(&mut self, conf: &WeldConf, arg: &WeldValue) -> WeldResult<WeldValue> {
        let callable = self.llvm_module.llvm_mut();
        let ref parsed_conf = conf::parse(conf)?;

        // This is the required input format of data passed into a compiled module.
        let input = Box::new(llvm::WeldInputArgs {
                             input: arg.data as i64,
                             nworkers: parsed_conf.threads,
                             mem_limit: parsed_conf.memory_limit,
                         });
        let ptr = Box::into_raw(input) as i64;

        // Runs the Weld program.
        let raw = callable.run(ptr) as *const llvm::WeldOutputArgs;
        let result = (*raw).clone();

        let value = WeldValue {
            data: result.output as *const c_void,
            run_id: Some(result.run_id),
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
}
