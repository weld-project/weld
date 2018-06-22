//! Extensions to compile generated code into a runnable module.

extern crate llvm_sys;

use conf::ParsedConf;

use self::llvm::execution_engine::*;
use self::llvm::transforms::pass_manager_builder::*;
use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;

use self::llvm::support::LLVMLoadLibraryPermanently;
use self::llvm::target_machine::{LLVMCodeGenFileType, LLVMTargetMachineEmitToMemoryBuffer};

static ONCE: Once = ONCE_INIT;
static mut INITIALIZE_FAILED: bool = false;

/// The callable function type.
type I64Func = extern "C" fn(i64) -> i64;

pub unsafe fn compile(context: LLVMContextRef,
               module: LLVMModuleRef,
               conf: &ParsedConf) -> WeldResult<()> {

    ONCE.call_once(|| initialize());
    if INITIALIZE_FAILED {
        unreachable!()
    }

    verify_module(module)?;
    optimize_module(module, conf.llvm_optimization_level)?;
    
    // Takes ownership of the module.
    let engine = create_exec_engine(module, conf.llvm_optimization_level)?;
    let run_func = find_function(engine, "run")?;
}

/// Verify a module using LLVM's verifier.
unsafe fn verify_module(module: LLVMModuleRef) -> WeldResult<()> {
    use llvm_sys::analysis::LLVMVerifyModule;
    use llvm_sys::analysis::LLVMVerifierFailureAction::*;
    let mut error_str = 0 as *mut c_char;
    let result_code = LLVMVerifyModule(module,
                                       LLVMReturnStatusAction,
                                       &mut error_str);
    if result_code != 0 {
        let err = CStr::from_ptr(error_str).to_str().unwrap());
        compile_err!("{}", format!("Module verification failed: {}", err);
    } else {
        Ok(())
    }
}

/// Optimize an LLVM module using a given LLVM optimization level.
unsafe fn optimize_module(module: LLVMModuleRef, level: u32) -> WeldResult<()> {
    let manager = LLVMCreatePassManager();
    assert!(!manager.is_null());
    let builder = pmb::LLVMPassManagerBuilderCreate();
    assert!(!builder.is_null());

    LLVMPassManagerBuilderSetOptLevel(builder, level);
    LLVMPassManagerBuilderPopulateLTOPassManager(builder, manager, 1, 1);
    LLVMPassManagerBuilderDispose(builder);
    LLVMRunPassManager(manager, module);
    LLVMDisposePassManager(manager);
    Ok(())
}

/// Create an MCJIT execution engine for a given module.
unsafe fn create_exec_engine(module: LLVMModuleRef, level: u32) -> WeldResult<()> {
    let mut engine = mem::uninitialized();
    let mut error_str = mem::uninitialized();
    let mut options: LLVMMCJITCompilerOptions = mem::uninitialized();
    let options_size = mem::size_of::<LLVMMCJITCompilerOptions>();
    LLVMInitializeMCJITCompilerOptions(&mut options, options_size);
    options.OptLevel = level;

    let result_code = LLVMCreateMCJITCompilerForModule(&mut engine,
                                                       module,
                                                       &mut options,
                                                       options_size,
                                                       &mut error_str);
    if result_code != 0 {
        compile_err!("Creating execution engine failed: {}",
                          CStr::from_ptr(error_str).to_str().unwrap())
    } else {
        Ok(engine)
    }
}

/// Get a pointer to a named function in an execution engine.
unsafe fn find_function(engine: LLVMExecutionEngineRef, name: &str) -> WeldResult<I64Func> {
    let c_name = CString::new(name).unwrap();
    let func_addr = LLVMGetFunctionAddress(engine, c_name.as_ptr());
    if func_addr == 0 {
        return compile_err!("No function named {} in module", name);
    }
    let function: I64Func = mem::transmute(func_addr);
    Ok(function)
}

/// Outputs the target machine assembly based on the given engine and module.
unsafe fn output_target_machine_assembly(engine: LLVMExecutionEngineRef,
                                         module: LLVMModuleRef) -> WeldResult<String> {
    let mut output_buf: LLVMMemoryBufferRef = ptr::null_mut();
    let mut err = ptr::null_mut();
    let cur_target = LLVMGetExecutionEngineTargetMachine(engine);
    let file_type :LLVMCodeGenFileType = LLVMCodeGenFileType::LLVMAssemblyFile;
    let res = LLVMTargetMachineEmitToMemoryBuffer(cur_target,
                                                  module,
                                                  file_type,
                                                  &mut err,
                                                  &mut output_buf);
    if res == 1 {
        let err = CStr::from_ptr(err as *mut c_char).to_str().unwrap();
        return compile_err!("Machine code generation failed with error {}", err);
    }
    let start = LLVMGetBufferStart(output_buf);
    let c_str = CStr::from_ptr(start as *mut c_char);
    Ok(c_str.to_string_lossy().into_owned())
}

/// Outputs the LLVM IR for the given module.
unsafe fn output_llvm_ir(module: LLVMModuleRef) -> WeldResult<String> {
    let start = LLVMPrintModuleToString(module);
    let c_str: &CStr = CStr::from_ptr(start as *mut c_char);
    Ok(c_str.to_str().unwrap().to_owned())
}
