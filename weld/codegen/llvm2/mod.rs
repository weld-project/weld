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

struct BasicBlockContext {
    block: LLVMBasicBlockRef,
    generated: bool,
}

impl BasicBlockContext {
    fn new(block: LLVMBasicBlockRef, generated: bool) -> BasicBlockContext {
        BasicBlockContext {
            block: block,
            generated: generated,
        }
    }
}

struct FunctionContext<'a> {
    sir_program: &'a SirProgram,
    sir_function: &'a SirFunction,
    llvm_function: LLVMValueRef,
    builder: LLVMBuilderRef,
    symbols: FnvHashMap<Symbol, LLVMValueRef>,
    blocks: FnvHashMap<BasicBlockId, BasicBlockContext>,
}

impl<'a> FunctionContext<'a> {
    pub fn new(sir_program: &'a SirProgram,
               sir_function: &'a SirFunction,
               llvm_context: LLVMContextRef,
               llvm_function: LLVMValueRef) -> FunctionContext<'a> {
        FunctionContext {
            sir_program: sir_program,
            sir_function: sir_function,
            llvm_function: llvm_function,
            builder: unsafe { LLVMCreateBuilderInContext(llvm_context) },
            symbols: FnvHashMap::default(),
            blocks: FnvHashMap::default(),
        }
    }
}

impl<'a> Drop for FunctionContext<'a> {
    fn drop(&mut self) {
        unsafe { LLVMDisposeBuilder(self.builder); }
    }
}

struct LlvmGenerator {
    conf: ParsedConf,
    context: LLVMContextRef,
    module: LLVMModuleRef,
}

impl LlvmGenerator {
    fn new(conf: ParsedConf) -> LlvmGenerator {
        unsafe {
            LlvmGenerator {
                conf: conf,
                context: LLVMContextCreate(),
                module: LLVMModuleCreateWithName(c_str!("main")),
            }
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
        let mut argument_types = vec![];
        for (_, ty) in func.params.iter() {
            argument_types.push(self.llvm_type(ty)?);
        }

        let return_type = LLVMVoidTypeInContext(self.context);
        let function_type = LLVMFunctionType(return_type, argument_types.as_mut_ptr(), argument_types.len() as u32, 0);
        let function_name = CString::new(format!("f{}", func.id)).unwrap();
        let function = LLVMAddFunction(self.module, function_name.as_ptr(), function_type);

        let ref mut context = FunctionContext::new(program, func, self.context, function);

        // Add the function parameters to the symbol map.
        for (i, (symbol, _)) in func.params.iter().enumerate() {
            context.symbols.insert(symbol.clone(), LLVMGetParam(function, i as u32));
        }

        // Create the entry basic block, where we define alloca'd variables.
        let entry_bb = LLVMAppendBasicBlockInContext(self.context, context.llvm_function, c_str!("entry"));
        LLVMPositionBuilderAtEnd(context.builder, entry_bb);

        // Generate local variables.
        for (symbol, ty) in func.locals.iter() {
            let name = CString::new(symbol.to_string()).unwrap();
            let value = LLVMBuildAlloca(context.builder, self.llvm_type(ty)?, name.as_ptr()); 
            context.symbols.insert(symbol.clone(), value);
        }

        // Generate the basic block references. We do this first so we can forward reference blocks
        // if necessary.
        for bb in func.blocks.iter() {
            let name = CString::new(format!("b{}", bb.id)).unwrap();
            let block = LLVMAppendBasicBlockInContext(self.context, context.llvm_function, name.as_ptr());
            context.blocks.insert(bb.id, BasicBlockContext::new(block, false));
        }
        
        // Generate the first basic block.
        let first_bb = self.generate_basic_block(program, context, &func.blocks[0])?;

        // Jump from locals to the first basic block.
        LLVMPositionBuilderAtEnd(context.builder, entry_bb);
        LLVMBuildBr(context.builder, first_bb);
        LLVMPositionBuilderAtEnd(context.builder, first_bb);

        // Finally, generate the remaining basic blocks. The basic blocks will handle returning
        // from the function when necessary.
        for bb in func.blocks.iter().skip(1) {
            let _ = self.generate_basic_block(program, context, bb)?;
        }
        Ok(())
    }

    unsafe fn generate_basic_block(&mut self,
                                   program: &SirProgram,
                                   context: &mut FunctionContext,
                                   bb: &BasicBlock) -> WeldResult<LLVMBasicBlockRef> {

        let block = { 
            let block_context = context.blocks.get_mut(&bb.id).unwrap();
            if block_context.generated {
                return Ok(block_context.block);
            }
            block_context.generated = true;
            block_context.block
        };

        LLVMPositionBuilderAtEnd(context.builder, block);
        for statement in bb.statements.iter() {
            self.generate_statement(program, context, statement)?;
        }
        self.generate_terminator(program, context, &bb)?;
        Ok(block)
    }

    unsafe fn generate_statement(&mut self, program: &SirProgram,
                                 context: &mut FunctionContext,
                                 statement: &Statement) -> WeldResult<LLVMValueRef> {
        use sir::StatementKind::*;
        match statement.kind {
            AssignLiteral(ref value) => {
                // use ast::Type::{Scalar, Simd};
                use ast::LiteralKind::*;
                let output = statement.output.as_ref().unwrap();
                let pointer = *context.symbols.get(output).unwrap();
                let output_type = context.sir_function.symbol_type(output)?;
                let literal_type = self.llvm_type(output_type)?;
                let constant = if let Type::Simd(_) = output_type {
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
                Ok(LLVMBuildStore(context.builder, constant, pointer)) 
            }
            _ => unimplemented!(),
        }
    }

    unsafe fn generate_terminator(&mut self, program: &SirProgram,
                                   context: &mut FunctionContext,
                                   bb: &BasicBlock) -> WeldResult<()> {
        use sir::Terminator::*;
        match bb.terminator {
            ProgramReturn(ref sym) => {
                LLVMBuildRetVoid(context.builder);
            }
            Branch { ref cond, ref on_true, ref on_false } => {
                let cond_value = *context.symbols.get(cond).unwrap();

                let on_true_block = context.blocks.get(&on_true).unwrap().block;
                let on_false_block = context.blocks.get(&on_false).unwrap().block;
                LLVMPositionBuilderAtEnd(context.builder, context.blocks.get(&bb.id).unwrap().block);
                let _ = LLVMBuildCondBr(context.builder, cond_value, on_true_block, on_false_block);
            }
            JumpBlock(ref id) => {
                let block = context.blocks.get(id).unwrap().block;
                LLVMPositionBuilderAtEnd(context.builder, context.blocks.get(&bb.id).unwrap().block);
                LLVMBuildBr(context.builder, block);
            }
            _ => unimplemented!(),
        };
        Ok(())
    }
}

impl Drop for LlvmGenerator {
    fn drop(&mut self) {
        unsafe {
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
