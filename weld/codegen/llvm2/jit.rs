
//!
//! This module manages verifying the generated LLVM module, optimizing it using the LLVM
//! optimization passes, and compiling it to machine code.

extern crate llvm_sys;
extern crate time;
extern crate libc;

use std::ffi::{CStr, CString};
use std::mem;
use std::ptr;
use std::sync::{Once, ONCE_INIT};

use libc::c_char;

use self::time::PreciseTime;

use conf::ParsedConf;
use error::*;
use util::stats::CompilationStats;

use self::llvm_sys::core::*;
use self::llvm_sys::prelude::*;
use self::llvm_sys::execution_engine::*;
use self::llvm_sys::target::*;
use self::llvm_sys::target_machine::*;

use codegen::Runnable;

use codegen::llvm2::llvm_exts::*;

static ONCE: Once = ONCE_INIT;
static mut INITIALIZE_FAILED: bool = false;

/// The callable function type.
type I64Func = extern "C" fn(i64) -> i64;

/// A compiled, runnable LLVM module.
pub struct CompiledModule {
    context: LLVMContextRef,
    module: LLVMModuleRef,
    engine: LLVMExecutionEngineRef,
    run_function: I64Func,
}

// The codegen interface requires that modules implement this trait. This allows supporting
// multiple backends via dynamic dispatch.
impl Runnable for CompiledModule {
    fn run(&self, arg: i64) -> i64 {
        (self.run_function)(arg)
    }
}

// LLVM modules are thread-safe.
unsafe impl Send for CompiledModule {}
unsafe impl Sync for CompiledModule {}

impl CompiledModule {
    /// Dumps assembly for this module.
    pub fn asm(&self) -> WeldResult<String> {
        unsafe {
            let mut output_buf = ptr::null_mut();
            let mut err = ptr::null_mut();
            let target = LLVMGetExecutionEngineTargetMachine(self.engine);
            let file_type = LLVMCodeGenFileType::LLVMAssemblyFile;
            let res = LLVMTargetMachineEmitToMemoryBuffer(target,
                                                          self.module,
                                                          file_type,
                                                          &mut err,
                                                          &mut output_buf);
            if res == 1 {
                let err = CStr::from_ptr(err as *mut c_char).to_str().unwrap();
                return compile_err!("Machine code generation failed with error {}", err);
            }
            LLVMDisposeMessage(err);

            let start = LLVMGetBufferStart(output_buf);
            let c_str = CStr::from_ptr(start as *mut c_char);
            Ok(c_str.to_string_lossy().into_owned())
        }
    }

    /// Dumps the optimized LLVM IR for this module.
    pub fn llvm(&self) -> WeldResult<String> {
        unsafe { 
            let start = LLVMPrintModuleToString(self.module);
            let c_str: &CStr = CStr::from_ptr(start as *mut c_char);
            Ok(c_str.to_str().unwrap().to_owned())
        }
    }
}

impl Drop for CompiledModule {
    fn drop(&mut self) {
        unsafe {
            // Engine owns the module, so do not drop it explicitly.
            LLVMDisposeExecutionEngine(self.engine);
            LLVMContextDispose(self.context);
        }
    }
}

pub unsafe fn init() {
    ONCE.call_once(|| initialize());
    if INITIALIZE_FAILED {
        unreachable!()
    }
}

/// Compile a constructed module in the given LLVM context.
pub unsafe fn compile(context: LLVMContextRef,
               module: LLVMModuleRef,
               conf: &ParsedConf,
               stats: &mut CompilationStats) -> WeldResult<CompiledModule> {
    init();

    let start = PreciseTime::now();
    verify_module(module)?;
    let end = PreciseTime::now();
    stats.llvm_times.push(("Module Verification".to_string(), start.to(end)));

    let start = PreciseTime::now();
    optimize_module(module, conf)?;
    let end = PreciseTime::now();
    stats.llvm_times.push(("Module Optimization".to_string(), start.to(end)));
    
    let start = PreciseTime::now();
    // Takes ownership of the module.
    let engine = create_exec_engine(module, conf)?;
    let end = PreciseTime::now();
    stats.llvm_times.push(("Create Exec Engine".to_string(), start.to(end)));

    let start = PreciseTime::now();
    let run_func = find_function(engine, &conf.llvm.run_func_name)?;
    let end = PreciseTime::now();
    stats.llvm_times.push(("Find Run Func Address".to_string(), start.to(end)));

    let result = CompiledModule {
        context: context,
        module: module,
        engine: engine,
        run_function: run_func,
    };
    Ok(result)
}

/// Initialize LLVM.
///
/// This function should only be called once.
unsafe fn initialize() {
    use self::llvm_sys::target::*;
    if LLVM_InitializeNativeTarget() != 0 {
        INITIALIZE_FAILED = true;
        return;
    }
    if LLVM_InitializeNativeAsmPrinter() != 0 {
        INITIALIZE_FAILED = true;
        return;
    }
    if LLVM_InitializeNativeAsmParser() != 0 {
        INITIALIZE_FAILED = true;
        return;
    }

    // No version that just initializes the current one?
    LLVM_InitializeAllTargetInfos();
    LLVMLinkInMCJIT();

    use self::llvm_sys::initialization::*;

    let registry = LLVMGetGlobalPassRegistry();
    LLVMInitializeCore(registry);
    LLVMInitializeAnalysis(registry);
    LLVMInitializeCodeGen(registry);
    LLVMInitializeIPA(registry);
    LLVMInitializeIPO(registry);
    LLVMInitializeInstrumentation(registry);
    LLVMInitializeObjCARCOpts(registry);
    LLVMInitializeScalarOpts(registry);
    LLVMInitializeTarget(registry);
    LLVMInitializeTransformUtils(registry);
    LLVMInitializeVectorization(registry);
}

unsafe fn target_machine() -> WeldResult<LLVMTargetMachineRef> {
    let mut target = mem::uninitialized();
    let mut err = ptr::null_mut();
    let result = LLVMGetTargetFromTriple(PROCESS_TRIPLE.as_ptr(), &mut target, &mut err);
    if result == 1 {
        let err = CStr::from_ptr(err as *mut c_char).to_str().unwrap();
        return compile_err!("Target initialization failed with error {}", err);
    }
    LLVMDisposeMessage(err);
    Ok(LLVMCreateTargetMachine(target,
                            PROCESS_TRIPLE.as_ptr(),
                            HOST_CPU_NAME.as_ptr(),
                            HOST_CPU_FEATURES.as_ptr(),
                            LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
                            LLVMRelocMode::LLVMRelocDefault,
                            LLVMCodeModel::LLVMCodeModelDefault))
}

pub unsafe fn set_triple_and_layout(module: LLVMModuleRef) -> WeldResult<()> {
    LLVMSetTarget(module, PROCESS_TRIPLE.as_ptr() as *const _);
    debug!("Set module target {:?}", PROCESS_TRIPLE.to_str().unwrap());
    let target_machine = target_machine()?;
    let layout = LLVMCreateTargetDataLayout(target_machine);
    LLVMSetModuleDataLayout(module, layout);
    LLVMDisposeTargetMachine(target_machine);
    Ok(())
}

/// Verify a module using LLVM's verifier.
unsafe fn verify_module(module: LLVMModuleRef) -> WeldResult<()> {
    use self::llvm_sys::analysis::LLVMVerifyModule;
    use self::llvm_sys::analysis::LLVMVerifierFailureAction::*;
    let mut error_str = 0 as *mut c_char;
    let result_code = LLVMVerifyModule(module,
                                       LLVMReturnStatusAction,
                                       &mut error_str);
    if result_code != 0 {
        let err = CStr::from_ptr(error_str).to_str().unwrap();
        compile_err!("{}", format!("Module verification failed: {}", err))
    } else {
        Ok(())
    }
}

/// Optimize an LLVM module using a given LLVM optimization level.
///
/// This function is currently modeled after the `AddOptimizationPasses` in the LLVM `opt` tool:
/// https://github.com/llvm-mirror/llvm/blob/master/tools/opt/opt.cpp
unsafe fn optimize_module(module: LLVMModuleRef, conf: &ParsedConf) -> WeldResult<()> {
    info!("Optimizing LLVM module");
    use self::llvm_sys::transforms::pass_manager_builder::*;
    let mpm = LLVMCreatePassManager();
    let fpm = LLVMCreateFunctionPassManagerForModule(module);

    // Target specific analyses so LLVM can query the backend.
    let target_machine = target_machine()?;

    let target = LLVMGetTargetMachineTarget(target_machine);

    // Log some information about the machine...
    // TODO this leaks stuff
    let cpu = CStr::from_ptr(LLVMGetTargetMachineCPU(target_machine)).to_str().unwrap();
    let description = CStr::from_ptr(LLVMGetTargetDescription(target)).to_str().unwrap();
    let features = CStr::from_ptr(LLVMGetTargetMachineFeatureString(target_machine)).to_str().unwrap();

    debug!("CPU: {}, Description: {} Features: {}", cpu, description, features);
    let start = PreciseTime::now();

    if conf.llvm.target_analysis_passes {
        LLVMAddTargetLibraryInfo(LLVMExtTargetLibraryInfo(), mpm);
        LLVMAddAnalysisPasses(target_machine, mpm);
        LLVMExtAddTargetPassConfig(target_machine, mpm);
        LLVMAddAnalysisPasses(target_machine, fpm);
    }

    // TODO set the size and inliner threshold depending on the optimization level. Right now, we
    // set the inliner to be as aggressive as the -O3 inliner in Clang.
    let builder = LLVMPassManagerBuilderCreate();
    LLVMPassManagerBuilderSetOptLevel(builder, conf.llvm.opt_level);
    LLVMPassManagerBuilderSetSizeLevel(builder, 0);
    LLVMPassManagerBuilderSetDisableUnrollLoops(builder, if conf.llvm.llvm_unroller { 0 } else { 1 });
    LLVMExtPassManagerBuilderSetDisableVectorize(builder, if conf.llvm.llvm_vectorizer { 0 } else { 1 });
    // 250 should correspond to OptLevel = 3
    LLVMPassManagerBuilderUseInlinerWithThreshold(builder, 250);

    if conf.llvm.func_optimizations {
        LLVMPassManagerBuilderPopulateFunctionPassManager(builder, fpm);
    }

    if conf.llvm.module_optimizations {
        LLVMPassManagerBuilderPopulateModulePassManager(builder, mpm);
    }

    LLVMPassManagerBuilderDispose(builder);
    let end = PreciseTime::now();
    debug!("LLVM Constructed PassManager in {} ms", start.to(end).num_milliseconds());

    let start = PreciseTime::now();
    let mut func = LLVMGetFirstFunction(module);
    while func != ptr::null_mut() {
        LLVMRunFunctionPassManager(fpm, func);
        func = LLVMGetNextFunction(func);
    }
    LLVMFinalizeFunctionPassManager(fpm);
    let end = PreciseTime::now();
    debug!("LLVM Function Passes Ran in {} ms", start.to(end).num_milliseconds());

    let start = PreciseTime::now();
    LLVMRunPassManager(mpm, module);
    let end = PreciseTime::now();
    debug!("LLVM Module Passes Ran in {} ms", start.to(end).num_milliseconds());

    LLVMDisposePassManager(fpm);
    LLVMDisposePassManager(mpm);
    LLVMDisposeTargetMachine(target_machine);

    Ok(())
}

/// Create an MCJIT execution engine for a given module.
unsafe fn create_exec_engine(module: LLVMModuleRef,
                             conf: &ParsedConf) -> WeldResult<LLVMExecutionEngineRef> {
    let mut engine = mem::uninitialized();
    let mut error_str = mem::uninitialized();
    let mut options: LLVMMCJITCompilerOptions = mem::uninitialized();
    let options_size = mem::size_of::<LLVMMCJITCompilerOptions>();
    LLVMInitializeMCJITCompilerOptions(&mut options, options_size);
    options.OptLevel = conf.llvm.opt_level;
    options.CodeModel = LLVMCodeModel::LLVMCodeModelDefault;

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
