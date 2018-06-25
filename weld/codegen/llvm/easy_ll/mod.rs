//! A very simple wrapper for LLVM that can JIT functions written as IR strings.

extern crate llvm_sys as llvm;
extern crate libc;
extern crate time;

use error::*;

use std::error::Error;
use std::ffi::{CStr, CString, NulError};
use std::fmt;
use std::result::Result;
use std::mem;
use std::ops::Drop;
use std::os::raw::c_char;
use std::sync::{Once, ONCE_INIT};

use std::ptr;

use self::llvm::support::LLVMLoadLibraryPermanently;
use self::llvm::prelude::{LLVMContextRef, LLVMModuleRef, LLVMMemoryBufferRef};
use self::llvm::execution_engine::{LLVMExecutionEngineRef, LLVMMCJITCompilerOptions, LLVMGetExecutionEngineTargetMachine};

use self::llvm::target_machine::{LLVMCodeGenFileType, LLVMTargetMachineEmitToMemoryBuffer};
use self::llvm::core::{LLVMGetBufferStart, LLVMPrintModuleToString};
use self::llvm::analysis::LLVMVerifierFailureAction;
use self::llvm::transforms::pass_manager_builder as pmb;

use codegen::Runnable;

use time::{Duration, PreciseTime};

#[cfg(test)]
mod tests;

// Helper objects to make sure we only initialize once
static ONCE: Once = ONCE_INIT;
static mut INITIALIZE_FAILED: bool = false;

/// Error type returned by easy_ll.
#[derive(Debug)]
pub struct LlvmError(String);

impl LlvmError {
    pub fn new(description: &str) -> LlvmError {
        LlvmError(description.to_string())
    }

    pub fn to_string(self) -> String {
        self.0
    }
}

impl fmt::Display for LlvmError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for LlvmError {
    fn description(&self) -> &str {
        &self.0
    }

    fn cause(&self) -> Option<&Error> {
        None
    }
}

impl From<NulError> for LlvmError {
    fn from(_: NulError) -> LlvmError {
        LlvmError::new("Null byte in string")
    }
}

#[derive(Debug)]
/// Stores timing information for various LLVM compilation stages.
pub struct LlvmTimingInfo {
    pub times: Vec<(String, Duration)>,
}

impl LlvmTimingInfo {
    pub fn new() -> LlvmTimingInfo {
        LlvmTimingInfo {
            times: Vec::new(),
        }
    }
}

/// The type of function pointer we'll return. We only support functions that take and return i64.
type I64Func = extern "C" fn(i64) -> i64;

/// A compiled module returned by `compile_module`, wrapping a `run` function that takes `i64`
/// and returns `i64`. This structure includes (and manages) an LLVM execution engine, which is
/// freed when this structure is dropped.
#[derive(Debug)]
pub struct CompiledModule {
    context: LLVMContextRef,
    engine: Option<LLVMExecutionEngineRef>,
    run_function: Option<I64Func>,
}

/// Generated code saved a string. This can be dumped to a file later.
#[derive(Debug)]
pub struct CodeDump {
    pub optimized_llvm: String,
    pub assembly: String,
}

#[derive(Debug)]
/// The return type of `compile_module`.
pub struct Compiled {
   pub module: CompiledModule,
   pub code: Option<CodeDump>,
   pub timing: LlvmTimingInfo,
}

impl Runnable for CompiledModule {
    fn run(&self, arg: i64) -> i64 {
        (self.run_function.unwrap())(arg)
    }
}

impl Drop for CompiledModule {
    /// Disposes of the LLVM execution engine and compiled function.
    fn drop(&mut self) {
        unsafe {
            self.engine.map(|e| llvm::execution_engine::LLVMDisposeExecutionEngine(e));
            llvm::core::LLVMContextDispose(self.context);
        }
    }
}

/// Loads a dynamic library from a file using LLVMLoadLibraryPermanently. It is safe to call
/// this function multiple times for the same library.
pub fn load_library(libname: &str) -> WeldResult<()> {
    let c_string = CString::new(libname.clone()).unwrap();
    let c_string_raw = c_string.into_raw() as *const c_char;
    if unsafe { LLVMLoadLibraryPermanently(c_string_raw) } == 0 {
        Ok(())
    } else {
        compile_err!("Couldn't load library {}", libname)
    }
}

/// Compile a string of LLVM IR (in human readable format) into a `CompiledModule` that can then
/// be executed. The LLVM IR should contain an entry point function called `run` that takes `i64`
/// and returns `i64`, which will be called by `CompiledModule::run`.
pub fn compile_module(
        code: &str,
        optimization_level: u32,
        dump_code: bool,
        bc_file: Option<&[u8]>)
        -> Result<Compiled, LlvmError> {

    let mut timing = LlvmTimingInfo::new();

    unsafe {
        // Initialize LLVM
        ONCE.call_once(|| initialize());
        if INITIALIZE_FAILED {
            return Err(LlvmError::new("LLVM initialization failed"));
        }

        // Create an LLVM context
        let context = llvm::core::LLVMContextCreate();
        if context.is_null() {
            return Err(LlvmError::new("LLVMContextCreate returned null"));
        }
        debug!("Done creating LLVM context");

        let start = PreciseTime::now();
        // Create a CompiledModule to wrap the context and our result (will clean it on Drop).
        let mut result = CompiledModule {
            context: context,
            engine: None,
            run_function: None,
        };

        // Parse the IR to get an LLVMModuleRef
        let module = parse_module_str(context, code)?;
        let end = PreciseTime::now();
        timing.times.push(("IR Parsing".to_string(), start.to(end)));
        debug!("Done parsing module");

        // Parse the bytecode file and link it.
        let start = PreciseTime::now();
        if let Some(s) = bc_file {
            let bc_module = parse_module_bytes(context, s)?;
            debug!("Done parsing bytecode file");
            llvm::linker::LLVMLinkModules2(module, bc_module);
            debug!("Done linking bytecode file");
        }
        let end = PreciseTime::now();
        timing.times.push(("Bytecode Linking".to_string(), start.to(end)));

        // Validate the module
        let start = PreciseTime::now();
        verify_module(module)?;
        check_run_function(module)?;
        let end = PreciseTime::now();
        timing.times.push(("Module Verification".to_string(), start.to(end)));
        debug!("Done validating module");

        // Optimize the module.
        let start = PreciseTime::now();
        optimize_module(module, optimization_level)?;
        let end = PreciseTime::now();
        timing.times.push(("Module Optimization".to_string(), start.to(end)));
        debug!("Done optimizing module");

        // Create an execution engine for the module and find its run function
        let start = PreciseTime::now();
        let engine = create_exec_engine(module, optimization_level)?;
        let end = PreciseTime::now();
        timing.times.push(("Create Exec Engine".to_string(), start.to(end)));
        debug!("Done creating execution engine");

        // Find the run function
        let start = PreciseTime::now();
        result.engine = Some(engine);
        result.run_function = Some(find_function(engine, "run")?);
        let end = PreciseTime::now();
        timing.times.push(("Find Run Func Address".to_string(), start.to(end)));
        debug!("Done generating/finding run function");

        let code = if dump_code {
            let ir = output_llvm_ir(module)?;
            let assembly = output_target_machine_assembly(engine, module)?;
            Some(CodeDump { optimized_llvm: ir, assembly: assembly })
        } else {
            None
        };

        let result = Compiled {
            module: result,
            code: code,
            timing: timing
        };
        Ok(result)
    }
}

/// Initialize LLVM or save an error message in `INITIALIZE_FAILED` if this does not work.
/// We call this function only once in cases some steps are expensive.
fn initialize() {
    unsafe {
        if llvm::target::LLVM_InitializeNativeTarget() != 0 {
            INITIALIZE_FAILED = true;
            return;
        }
        if llvm::target::LLVM_InitializeNativeAsmPrinter() != 0 {
            INITIALIZE_FAILED = true;
            return;
        }
        if llvm::target::LLVM_InitializeNativeAsmParser() != 0 {
            INITIALIZE_FAILED = true;
            return;
        }
        llvm::execution_engine::LLVMLinkInMCJIT();
    }
}

unsafe fn parse_module_helper(context: LLVMContextRef,
                              buffer: LLVMMemoryBufferRef)
                              -> Result<LLVMModuleRef, LlvmError> {
    // Parse IR into a module
    let mut module = 0 as LLVMModuleRef;
    let mut error_str = 0 as *mut c_char;

    let result_code =
        llvm::ir_reader::LLVMParseIRInContext(context, buffer, &mut module, &mut error_str);
    if result_code != 0 {
        let msg = format!("Compile error: {}",
                          CStr::from_ptr(error_str).to_str().unwrap());
        return Err(LlvmError(msg));
    }

    Ok(module)
}

/// Parse a buffer of IR bytecode into an `LLVMModuleRef` for the given context.
unsafe fn parse_module_bytes(context: LLVMContextRef,
                             code: &[u8])
                             -> Result<LLVMModuleRef, LlvmError> {
    // Create an LLVM memory buffer around the code
    let code_len = code.len();
    let name = CString::new("module")?;
    let buffer = llvm::core::LLVMCreateMemoryBufferWithMemoryRange(code.as_ptr() as *const i8,
                                                                   code_len,
                                                                   name.as_ptr(),
                                                                   0);

    if buffer.is_null() {
        return Err(LlvmError::new("LLVMCreateMemoryBufferWithMemoryRange failed"));
    }

    parse_module_helper(context, buffer)
}

/// Parse a string of IR code into an `LLVMModuleRef` for the given context.
unsafe fn parse_module_str(context: LLVMContextRef,
                           code: &str)
                           -> Result<LLVMModuleRef, LlvmError> {
    // Create an LLVM memory buffer around the code
    let code_len = code.len();
    let name = CString::new("module")?;
    let code = CString::new(code)?;
    let buffer = llvm::core::LLVMCreateMemoryBufferWithMemoryRange(code.as_ptr(),
                                                                   code_len,
                                                                   name.as_ptr(),
                                                                   0);
    if buffer.is_null() {
        return Err(LlvmError::new("LLVMCreateMemoryBufferWithMemoryRange failed"));
    }

    parse_module_helper(context, buffer)
}

/// Parse a file of IR code into an `LLVMModuleRef` for the given context.
#[allow(dead_code)]
unsafe fn parse_module_file(context: LLVMContextRef,
                            file: &str)
                            -> Result<LLVMModuleRef, LlvmError> {
    let mut buffer = 0 as LLVMMemoryBufferRef;
    let mut error_str = 0 as *mut c_char;
    let file_name = CString::new(file).unwrap();
    let result_code = llvm::core::LLVMCreateMemoryBufferWithContentsOfFile(file_name.as_ptr(),
                                                                           &mut buffer,
                                                                           &mut error_str);
    if result_code != 0 {
        let msg = format!("Error reading module file {}: {}",
                          file,
                          CStr::from_ptr(error_str).to_str().unwrap());
        return Err(LlvmError(msg));
    }
    if buffer.is_null() {
        return Err(LlvmError::new("LLVMCreateMemoryBufferWithContentsOfFile failed"));
    }
    parse_module_helper(context, buffer)
}

/// Verify a module using LLVM's verifier (for basic block structure, etc).
unsafe fn verify_module(module: LLVMModuleRef) -> Result<(), LlvmError> {
    let mut error_str = 0 as *mut c_char;
    let result_code =
        llvm::analysis::LLVMVerifyModule(module,
                                         LLVMVerifierFailureAction::LLVMReturnStatusAction,
                                         &mut error_str);
    if result_code != 0 {
        let msg = format!("Module verification failed: {}",
                          CStr::from_ptr(error_str).to_str().unwrap());
        return Err(LlvmError(msg));
    }
    Ok(())
}

/// Check that a module has a "run" function of type i64 -> i64.
unsafe fn check_run_function(module: LLVMModuleRef) -> Result<(), LlvmError> {
    let run = CString::new("run").unwrap();
    let func = llvm::core::LLVMGetNamedFunction(module, run.as_ptr());
    if func.is_null() {
        return Err(LlvmError::new("No run function in module"));
    }
    let c_str = llvm::core::LLVMPrintTypeToString(llvm::core::LLVMTypeOf(func));
    let func_type = CStr::from_ptr(c_str).to_str().unwrap();
    if func_type != "i64 (i64)*" {
        return Err(LlvmError(format!("Run function has wrong type: {}", func_type)));
    }
    Ok(())
}

/// Optimize an LLVM module using a given LLVM optimization level.
unsafe fn optimize_module(module: LLVMModuleRef, optimization_level: u32)
        -> Result<(), LlvmError> {
    let manager = llvm::core::LLVMCreatePassManager();
    if manager.is_null() {
        return Err(LlvmError::new("LLVMCreatePassManager returned null"));
    }
    let builder = pmb::LLVMPassManagerBuilderCreate();
    if builder.is_null() {
        return Err(LlvmError::new("LLVMPassManagerBuilderCreate returned null"));
    }

    pmb::LLVMPassManagerBuilderSetOptLevel(builder, optimization_level);
    pmb::LLVMPassManagerBuilderPopulateLTOPassManager(builder, manager, 1, 1);
    pmb::LLVMPassManagerBuilderDispose(builder);
    llvm::core::LLVMRunPassManager(manager, module);
    llvm::core::LLVMDisposePassManager(manager);
    Ok(())
}

/// Create an MCJIT execution engine for a given module.
unsafe fn create_exec_engine(module: LLVMModuleRef, optimization_level: u32)
        -> Result<LLVMExecutionEngineRef, LlvmError> {
    let mut engine = 0 as LLVMExecutionEngineRef;
    let mut error_str = 0 as *mut c_char;
    let mut options: LLVMMCJITCompilerOptions = mem::uninitialized();
    let options_size = mem::size_of::<LLVMMCJITCompilerOptions>();
    llvm::execution_engine::LLVMInitializeMCJITCompilerOptions(&mut options, options_size);
    options.OptLevel = optimization_level;
    let result_code = llvm::execution_engine::LLVMCreateMCJITCompilerForModule(
        &mut engine, module, &mut options, options_size, &mut error_str);
    if result_code != 0 {
        let msg = format!("Creating execution engine failed: {}",
                          CStr::from_ptr(error_str).to_str().unwrap());
        return Err(LlvmError(msg));
    }
    Ok(engine)
}

/// Get a pointer to a named function in an execution engine.
unsafe fn find_function(engine: LLVMExecutionEngineRef, name: &str) -> Result<I64Func, LlvmError> {
    let c_name = CString::new(name).unwrap();
    let func_addr = llvm::execution_engine::LLVMGetFunctionAddress(engine, c_name.as_ptr());
    if func_addr == 0 {
        return Err(LlvmError(format!("No function named {} in module", name)));
    }
    let function: I64Func = mem::transmute(func_addr);
    Ok(function)
}


/// Outputs the target machine assembly based on the given engine and module.
unsafe fn output_target_machine_assembly(engine: LLVMExecutionEngineRef, module: LLVMModuleRef)
    -> Result<String, LlvmError> {
    // We create a pointer to a MemoryBuffer, and pass its address to be modified by
    // EmitToMemoryBuffer.
    let mut output_buf : self::llvm::prelude::LLVMMemoryBufferRef = ptr::null_mut();
    let mut err = ptr::null_mut();
    let cur_target = LLVMGetExecutionEngineTargetMachine(engine);
    let file_type :LLVMCodeGenFileType = LLVMCodeGenFileType::LLVMAssemblyFile;
    let res = LLVMTargetMachineEmitToMemoryBuffer(cur_target, module, file_type, &mut err, &mut output_buf);
    if res == 1 {
        let x = CStr::from_ptr(err as *mut c_char).to_string_lossy().into_owned();
        return Err(LlvmError::new(format!("Getting LLVM IR failed with error {}", &x).as_ref()));
    }
    let start = LLVMGetBufferStart(output_buf);
    let c_str: &CStr = CStr::from_ptr(start as *mut c_char);
    Ok(c_str.to_string_lossy().into_owned())
}

/// Outputs the LLVM IR for the given module.
unsafe fn output_llvm_ir(module: LLVMModuleRef) -> Result<String, LlvmError> {
    let start = LLVMPrintModuleToString(module);
    let c_str: &CStr = CStr::from_ptr(start as *mut c_char);
    Ok(c_str.to_str().unwrap().to_owned())
}
