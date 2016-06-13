extern crate llvm_sys as llvm;

use std::error::Error;
use std::ffi::{CStr, CString, NulError};
use std::fmt;
use std::result::Result;
use std::ops::Drop;
use std::os::raw::c_char;

#[cfg(test)]
mod tests;

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
    entry: Option<&'static fn(u64) -> u64>
}

impl Drop for CompiledModule {
    fn drop(&mut self) {
        println!("Dropping {:?}", self);
        unsafe {
            self.engine.map(|e| {
                llvm::execution_engine::LLVMDisposeExecutionEngine(e)
            });
            llvm::core::LLVMContextDispose(self.context);
        }
    }
}

pub fn initialize() -> Result<(), LlvmError> {
    unsafe {
        if llvm::target::LLVM_InitializeNativeTarget() != 0 {
            return Err(LlvmError::new("LLVM_InitializeNativeTarget failed"));
        }
        if llvm::target::LLVM_InitializeNativeAsmPrinter() != 0 {
            return Err(LlvmError::new("LLVM_InitializeNativeAsmPrinter failed"));
        }
        if llvm::target::LLVM_InitializeNativeAsmParser() != 0 {
            return Err(LlvmError::new("LLVM_InitializeNativeAsmParser failed"));
        }
        llvm::execution_engine::LLVMLinkInMCJIT();
    }
    Ok(())
}

pub fn compile_module(name: &str, code: &str) -> Result<CompiledModule, LlvmError> {
    unsafe {
        // Create an LLVM context
        let context = llvm::core::LLVMContextCreate();
        if context.is_null() {
            return Err(LlvmError::new("LLVMContextCreate returned null"))
        }
        
        // Create a module
        let mut module = CompiledModule { context: context, engine: None, entry: None };
        
        // Create a memory buffer to hold the code
        let code_len = code.len();
        let name = try!(CString::new(name));
        let code = try!(CString::new(code));
        let buffer = llvm::core::LLVMCreateMemoryBufferWithMemoryRange(
            code.as_ptr(), code_len, name.as_ptr(), 0);
        if buffer.is_null() {
            return Err(LlvmError::new("LLVMCreateMemoryBufferWithMemoryRange failed"))
        }

        println!("HERE 0");
        // Parse IR into a module
        let mut module_ref = 0 as llvm::prelude::LLVMModuleRef;
        let mut error_str = 0 as *mut c_char;
        let result_code = llvm::ir_reader::LLVMParseIRInContext(
            context, buffer, &mut module_ref, &mut error_str);
        if result_code != 0 {
            let message = format!("Parse error: {}", CStr::from_ptr(error_str).to_str().unwrap());
            return Err(LlvmError(message));
        }

        // Create an execution engine
        println!("HERE 1");
        let mut engine_ref = 0 as llvm::execution_engine::LLVMExecutionEngineRef;
        let mut error_str = 0 as *mut c_char;
        let result_code = llvm::execution_engine::LLVMCreateMCJITCompilerForModule(
            &mut engine_ref,
            module_ref,
            0 as *mut llvm::execution_engine::LLVMMCJITCompilerOptions,
            0,
            &mut error_str);
        println!("HERE 2 {} {:?}", result_code, error_str);
        if result_code != 0 {
            //llvm::core::LLVMDisposeModule(module_ref);
            let message = format!("Creating execution engine failed: {}",
                CStr::from_ptr(error_str).to_str().unwrap());
            return Err(LlvmError(message));
        }
        module.engine = Some(engine_ref);

        Err(LlvmError::new("EEK"))
    }
}