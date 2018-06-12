//! A port of the LLVM backend with optimized single-threaded performance.
//!
//! Unlike the other backend, this one directly uses the LLVM APIs.

extern crate fnv;
extern crate time;
extern crate libc;
extern crate llvm_sys;

use std::fmt;
use std::ptr;
use std::ffi::{CStr, CString};

use fnv::FnvHashMap;
use libc::{c_char, c_double, c_ulonglong};
use time::PreciseTime;

use ast::*;
use conf::ParsedConf;
use error::*;
use optimizer::*;
use sir::*;
use syntax::program::Program;
use util::stats::CompilationStats;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;

/// Convert a string literal into a C string.
macro_rules! c_str {
    ($s:expr) => (
        concat!($s, "\0").as_ptr() as *const i8
    );
}

type SymbolMap = FnvHashMap<Symbol, LLVMValueRef>;

pub struct CompiledModule;

// XXX Why is this here...
pub fn apply_opt_passes(expr: &mut Expr,
                        opt_passes: &Vec<Pass>,
                        stats: &mut CompilationStats,
                        use_experimental: bool) -> WeldResult<()> {

    for pass in opt_passes {
        let start = PreciseTime::now();
        pass.transform(expr, use_experimental)?;
        let end = PreciseTime::now();
        stats.pass_times.push((pass.pass_name(), start.to(end)));
        debug!("After {} pass:\n{}", pass.pass_name(), expr.pretty_print());
    }
    Ok(())
}

pub fn compile_program(program: &Program,
                       conf: &ParsedConf,
                       stats: &mut CompilationStats) -> WeldResult<CompiledModule> {
    use syntax::macro_processor;
    use sir;

    let mut expr = macro_processor::process_program(program)?;
    debug!("After macro substitution:\n{}\n", expr.pretty_print());

    let start = PreciseTime::now();
    expr.uniquify()?;
    let end = PreciseTime::now();

    let mut uniquify_dur = start.to(end);

    let start = PreciseTime::now();
    expr.infer_types()?;
    let end = PreciseTime::now();
    stats.weld_times.push(("Type Inference".to_string(), start.to(end)));
    debug!("After type inference:\n{}\n", expr.pretty_print());

    apply_opt_passes(&mut expr, &conf.optimization_passes, stats, conf.enable_experimental_passes)?;

    let start = PreciseTime::now();
    expr.uniquify()?;
    let end = PreciseTime::now();
    uniquify_dur = uniquify_dur + start.to(end);
    stats.weld_times.push(("Uniquify outside Passes".to_string(), uniquify_dur));
    debug!("Optimized Weld program:\n{}\n", expr.pretty_print());

    let start = PreciseTime::now();
    let mut sir_prog = sir::ast_to_sir(&expr, conf.support_multithread)?;
    let end = PreciseTime::now();
    stats.weld_times.push(("AST to SIR".to_string(), start.to(end)));
    debug!("SIR program:\n{}\n", &sir_prog);

    let start = PreciseTime::now();
    if conf.enable_sir_opt {
        use sir::optimizations;
        info!("Applying SIR optimizations");
        optimizations::fold_constants::fold_constants(&mut sir_prog)?;
    }
    let end = PreciseTime::now();
    debug!("Optimized SIR program:\n{}\n", &sir_prog);
    stats.weld_times.push(("SIR Optimization".to_string(), start.to(end)));

    // XXX Generate code here.
    let mut codegen = LlvmGenerator::new(conf.clone());
    unsafe { codegen.generate(&sir_prog)?; }
    println!("{}", codegen);

    Ok(CompiledModule)
}

struct LlvmGenerator {
    conf: ParsedConf,
    context: LLVMContextRef,
    module: LLVMModuleRef,
    builder: LLVMBuilderRef,
}

impl LlvmGenerator {
    fn new(conf: ParsedConf) -> LlvmGenerator {
        let (context, module, builder) = unsafe {
            let context = LLVMContextCreate();
            let module = LLVMModuleCreateWithName(c_str!("main"));
            let builder =  LLVMCreateBuilderInContext(context);
            (context, module, builder)
        };

        LlvmGenerator {
            conf: conf,
            context: context,
            module: module,
            builder: builder,
        }
    }

    unsafe fn generate(&mut self, program: &SirProgram) -> WeldResult<()> {
        for func in program.funcs.iter() {
            self.generate_function(program, func)?;
        }
        Ok(())
    }

    unsafe fn llvm_type(&mut self, ty: &Type) -> WeldResult<LLVMTypeRef> {
        use ast::Type::*;
        use ast::ScalarKind::*;
        let result = match *ty {
            Scalar(kind) => match kind {
                Bool => LLVMInt1Type(),
                I8 | U8 => LLVMInt8Type(),
                I16 | U16 => LLVMInt16Type(),
                I32 | U32 => LLVMInt32Type(),
                I64 | U64 => LLVMInt64Type(),
                F32 => LLVMFloatType(),
                F64 => LLVMDoubleType(),
            }
            _ => unimplemented!(),
        };
        Ok(result)
    }

    unsafe fn generate_function(&mut self, program: &SirProgram, func: &SirFunction) -> WeldResult<()> {
        // Maps a symbol name in a function to the function's LLVM values.
        let ref mut symbol_map = SymbolMap::default();
        let mut argument_types = vec![];
        for (_, ty) in func.params.iter() {
            argument_types.push(self.llvm_type(ty)?);
        }
        let return_type = LLVMVoidTypeInContext(self.context);
        let function_type = LLVMFunctionType(return_type, argument_types.as_mut_ptr(), argument_types.len() as u32, 0);

        // TODO - Need a name generator for functions!
        let function_name = CString::new(format!("f{}", func.id)).unwrap();
        let function = LLVMAddFunction(self.module, function_name.as_ptr(), function_type);

        // Add the function parameters to the symbol map.
        for (i, (symbol, _)) in func.params.iter().enumerate() {
            symbol_map.insert(symbol.clone(), LLVMGetParam(function, i as u32));
        }

        // Create the entry basic block, where we define alloca'd variables.
        let entry_bb = LLVMAppendBasicBlockInContext(self.context, function, c_str!("entry"));
        LLVMPositionBuilderAtEnd(self.builder, entry_bb);

        // Generate local variables.
        for (symbol, ty) in func.locals.iter() {
            let name = CString::new(symbol.to_string()).unwrap();
            let value = LLVMBuildAlloca(self.builder, self.llvm_type(ty)?, name.as_ptr()); 
            symbol_map.insert(symbol.clone(), value);
        }
        
        // Generate the first basic block.
        let first_bb = self.generate_basic_block(program, func, &func.blocks[0], function, symbol_map)?;

        // Jump from locals to the first basic block.
        LLVMPositionBuilderAtEnd(self.builder, entry_bb);
        LLVMBuildBr(self.builder, first_bb);
        LLVMPositionBuilderAtEnd(self.builder, first_bb);

        // Finally, generate the remaining basic blocks.
        for bb in func.blocks.iter().skip(1) {
            let _ = self.generate_basic_block(program, func, bb, function, symbol_map)?;
        }
        Ok(())
    }

    unsafe fn generate_basic_block(&mut self, program: &SirProgram,
                                   func: &SirFunction,
                                   bb: &BasicBlock,
                                   function: LLVMValueRef,
                                   symbol_map: &mut SymbolMap) -> WeldResult<LLVMBasicBlockRef> {
        debug!("Generate basic block called");
        let name = CString::new(format!("b{}", bb.id)).unwrap();
        let block = LLVMAppendBasicBlockInContext(self.context, function, name.as_ptr());
        LLVMPositionBuilderAtEnd(self.builder, block);

        for statement in bb.statements.iter() {
            self.generate_statement(program, func, statement, symbol_map)?;
        }
        self.generate_terminator(program, func, &bb.terminator, symbol_map)?;
        Ok(block)
    }

    unsafe fn generate_statement(&mut self, program: &SirProgram,
                                   func: &SirFunction,
                                   statement: &Statement,
                                   symbol_map: &mut SymbolMap) -> WeldResult<LLVMValueRef> {
        use sir::StatementKind::*;

        match statement.kind {
            AssignLiteral(ref value) => {
                use ast::Type::{Scalar, Simd};
                use ast::LiteralKind::*;
                let output = statement.output.as_ref().unwrap();
                let pointer = *symbol_map.get(output).unwrap();
                let output_type = func.symbol_type(output)?;
                let literal_type = self.llvm_type(output_type)?;
                let constant = if let Simd(_) = output_type {
                    unimplemented!()
                } else {
                    match *value {
                       BoolLiteral(val) => LLVMConstInt(literal_type, val as c_ulonglong, 0),
                       I8Literal(val) => LLVMConstInt(literal_type, val as c_ulonglong, 1),
                       I16Literal(val) => LLVMConstInt(literal_type, val as c_ulonglong, 1),
                       I32Literal(val) => LLVMConstInt(literal_type, val as c_ulonglong, 1),
                       I64Literal(val) => LLVMConstInt(literal_type, val as c_ulonglong, 1),
                       U8Literal(val) => LLVMConstInt(literal_type, val as c_ulonglong, 0),
                       U16Literal(val) => LLVMConstInt(literal_type, val as c_ulonglong, 0),
                       U32Literal(val) => LLVMConstInt(literal_type, val as c_ulonglong, 0),
                       U64Literal(val) => LLVMConstInt(literal_type, val as c_ulonglong, 0),
                       F32Literal(val) => LLVMConstReal(literal_type, f32::from_bits(val) as c_double),
                       F64Literal(val) => LLVMConstReal(literal_type, f64::from_bits(val) as c_double),
                       StringLiteral(_) => unimplemented!()
                    }
                };
                Ok(LLVMBuildStore(self.builder, constant, pointer)) 
            }
            _ => unimplemented!(),
        }
    }

    unsafe fn generate_terminator(&mut self, program: &SirProgram,
                                   func: &SirFunction,
                                   terminator: &Terminator,
                                   symbol_map: &mut SymbolMap) -> WeldResult<()> {
        use sir::Terminator::*;
        match *terminator {
            ProgramReturn(ref sym) => {
                LLVMBuildRetVoid(self.builder);
            }
            _ => unimplemented!(),
        }
        Ok(())
    }
}

impl Drop for LlvmGenerator {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeBuilder(self.builder);
            LLVMDisposeModule(self.module);
            LLVMContextDispose(self.context);
        }
    }
}

impl fmt::Display for LlvmGenerator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = unsafe {
            CStr::from_ptr(LLVMPrintModuleToString(self.module) as *mut c_char)
        };
        write!(f, "{}", s.to_str().unwrap())
    }
}
