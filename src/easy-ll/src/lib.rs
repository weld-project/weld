extern crate llvm_sys as llvm;

use std::error::Error;
use std::ffi::{CStr, CString, NulError};
use std::fmt;
use std::result::Result;
use std::ops::Drop;
use std::os::raw::c_char;
use std::sync::{Once, ONCE_INIT};

use llvm::execution_engine::LLVMMCJITCompilerOptions;
use llvm::analysis::LLVMVerifierFailureAction;
use llvm::transforms::pass_manager_builder;

#[cfg(test)]
mod tests;

static ONCE: Once = ONCE_INIT;
static mut initialize_failed: bool = false;

#[derive(Debug)]
pub struct LlvmError(String);

impl LlvmError {
    fn new(description: &str) -> LlvmError { LlvmError(description.to_string()) }
}

impl fmt::Display for LlvmError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for LlvmError {
    fn description(&self) -> &str { &self.0 }

    fn cause(&self) -> Option<&Error> { None }
}

impl From<NulError> for LlvmError {
    fn from(_: NulError) -> LlvmError { LlvmError::new("Null byte in string") }
}

#[derive(Debug)]
pub struct CompiledModule {
    context: llvm::prelude::LLVMContextRef,
    engine: Option<llvm::execution_engine::LLVMExecutionEngineRef>,
    function: Option<extern "C" fn(u64) -> u64>
}

impl CompiledModule {
    pub fn run(&self, arg: u64) -> u64 {
        (self.function.unwrap())(arg)
    }
}

impl Drop for CompiledModule {
    fn drop(&mut self) {
        unsafe {
            self.engine.map(|e| {
                llvm::execution_engine::LLVMDisposeExecutionEngine(e)
            });
            llvm::core::LLVMContextDispose(self.context);
        }
    }
}

fn initialize() {
    unsafe {
        if llvm::target::LLVM_InitializeNativeTarget() != 0 {
            initialize_failed = true;
            return;
        }
        if llvm::target::LLVM_InitializeNativeAsmPrinter() != 0 {
            initialize_failed = true;
            return;
        }
        if llvm::target::LLVM_InitializeNativeAsmParser() != 0 {
            initialize_failed = true;
            return;
        }
        llvm::execution_engine::LLVMLinkInMCJIT();
    }
}

pub fn compile_module(name: &str, code: &str) -> Result<CompiledModule, LlvmError> {
    unsafe {
        // Initialize LLVM
        ONCE.call_once(|| initialize());
        if initialize_failed {
            return Err(LlvmError::new("LLVM initialization failed"))
        }

        // Create an LLVM context
        let context = llvm::core::LLVMContextCreate();
        if context.is_null() {
            return Err(LlvmError::new("LLVMContextCreate returned null"))
        }
        
        // Create a module
        let mut result = CompiledModule { context: context, engine: None, function: None };
        
        // Create a memory buffer to hold the code
        let code_len = code.len();
        let name = try!(CString::new(name));
        let code = try!(CString::new(code));
        let buffer = llvm::core::LLVMCreateMemoryBufferWithMemoryRange(
            code.as_ptr(), code_len, name.as_ptr(), 0);
        if buffer.is_null() {
            return Err(LlvmError::new("LLVMCreateMemoryBufferWithMemoryRange failed"))
        }

        // Parse IR into a module
        let mut module = 0 as llvm::prelude::LLVMModuleRef;
        let mut error_str = 0 as *mut c_char;
        let result_code = llvm::ir_reader::LLVMParseIRInContext(
            context, buffer, &mut module, &mut error_str);
        if result_code != 0 {
            let msg = format!("Compile error: {}", CStr::from_ptr(error_str).to_str().unwrap());
            return Err(LlvmError(msg));
        }

        // Verify module
        let mut error_str = 0 as *mut c_char;
        let result_code = llvm::analysis::LLVMVerifyModule(
            module, LLVMVerifierFailureAction::LLVMReturnStatusAction, &mut error_str);
        if result_code != 0 {
            let msg = format!("Module verification failed: {}",
                CStr::from_ptr(error_str).to_str().unwrap());
            return Err(LlvmError(msg));
        }

        // Optimize module
        let manager = llvm::core::LLVMCreatePassManager();
        if manager.is_null() {
            return Err(LlvmError::new("LLVMCreatePassManager returned null"))
        }
        let builder = pass_manager_builder::LLVMPassManagerBuilderCreate();
        if builder.is_null() {
            return Err(LlvmError::new("LLVMPassManagerBuilderCreate returned null"))
        }
        pass_manager_builder::LLVMPassManagerBuilderPopulateFunctionPassManager(builder, manager);
        pass_manager_builder::LLVMPassManagerBuilderPopulateModulePassManager(builder, manager);
        pass_manager_builder::LLVMPassManagerBuilderPopulateLTOPassManager(builder, manager, 1, 1);
        pass_manager_builder::LLVMPassManagerBuilderDispose(builder);
        llvm::core::LLVMRunPassManager(manager, module);
        llvm::core::LLVMDisposePassManager(manager);
        
        // Create an execution engine
        let mut engine = 0 as llvm::execution_engine::LLVMExecutionEngineRef;
        let mut error_str = 0 as *mut c_char;
        let mut options: LLVMMCJITCompilerOptions = std::mem::uninitialized();
        options.OptLevel = 2;
        let options_size = std::mem::size_of::<LLVMMCJITCompilerOptions>();
        llvm::execution_engine::LLVMInitializeMCJITCompilerOptions(&mut options, options_size);
        let result_code = llvm::execution_engine::LLVMCreateMCJITCompilerForModule(
            &mut engine, module, &mut options, options_size, &mut error_str);
        if result_code != 0 {
            let msg = format!("Creating execution engine failed: {}",
                CStr::from_ptr(error_str).to_str().unwrap());
            return Err(LlvmError(msg));
        }
        result.engine = Some(engine);

        // Find the "run" function
        let func_address = llvm::execution_engine::LLVMGetFunctionAddress(
            engine, CString::new("run").unwrap().into_raw());
        if func_address == 0 {
            return Err(LlvmError::new("No run function in module"))
        }
        let func: extern fn(u64) -> u64 = std::mem::transmute(func_address);
        result.function = Some(func);

        Ok(result)
    }
}