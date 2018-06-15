//! An LLVM backend currently optimized for single-threaded execution.
extern crate fnv;
extern crate time;
extern crate libc;
extern crate llvm_sys;

use std::fmt;
use std::ffi::{CStr, CString};

use fnv::FnvHashMap;
use libc::c_char;
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

static NULL_NAME:[c_char; 1] = [0];

/// Convert a string literal into a C string.
macro_rules! c_str {
    ($s:expr) => (
        concat!($s, "\0").as_ptr() as *const i8
    );
}

// Traits implementing code generation for various expressions.
mod gen_numeric;

pub struct CompiledModule;

/// TODO
type VectorMethods = i32;
/// TODO
type DictMethods = i32;

/// A struct holding the global codegen state for an SIR program.
pub struct LlvmGenerator {
    /// A configuration for generating code.
    conf: ParsedConf,
    /// An LLVM Context for isolating code generation.
    context: LLVMContextRef,
    /// The main LLVM module to which code is added.
    module: LLVMModuleRef,
    /// A map that tracks references to an SIR function's LLVM function.
    functions: FnvHashMap<FunctionId, LLVMValueRef>,
    /// A map tracking generated vectors.
    ///
    /// The key maps the vector's element type to the vector's type reference and methods on it.
    vectors: FnvHashMap<Type, VectorMethods>,
    /// A map tracking generated dictionaries.
    ///
    /// The key maps the vector's element type to the vector's type reference and methods on it.
    dictionaries: FnvHashMap<Type, DictMethods>,
}

impl LlvmGenerator {
    /// Generate code for an SIR program.
    unsafe fn generate(conf: ParsedConf, program: &SirProgram) -> WeldResult<LlvmGenerator> {
        let mut gen = LlvmGenerator {
            conf: conf,
            context: LLVMContextCreate(),
            module: LLVMModuleCreateWithName(c_str!("main")),
            functions: FnvHashMap::default(),
            vectors: FnvHashMap::default(),
            dictionaries: FnvHashMap::default(),
        };

        // Declare each function first to create a reference to it.
        for func in program.funcs.iter() {
            gen.declare_function(func)?;
        }

        // Generate each function in turn.
        for func in program.funcs.iter() {
            gen.generate_function(program, func)?;
        }

        // Generate the entry point.
        gen.generate_entry(&program)?;
        Ok(gen)
    }

    /// Generate the entry point to the module.
    unsafe fn generate_entry(&mut self, program: &SirProgram) -> WeldResult<()> {
        Ok(())
    }

    /// Declare a function in the SIR module and track its reference.
    ///
    /// This method does not generate code for the function.
    unsafe fn declare_function(&mut self, func: &SirFunction) -> WeldResult<()> {
        // Convert each argument to an SIR function.
        let mut arg_tys = vec![];
        for (_, ty) in func.params.iter() {
            arg_tys.push(self.llvm_type(ty)?);
        }

        // All SIR functions have a void return type.
        let ret_ty = LLVMVoidTypeInContext(self.context);
        let func_ty = LLVMFunctionType(ret_ty, arg_tys.as_mut_ptr(), arg_tys.len() as u32, 0);
        let name = CString::new(format!("f{}", func.id)).unwrap();
        let function = LLVMAddFunction(self.module, name.as_ptr(), func_ty);

        self.functions.insert(func.id, function);
        Ok(())
    }

    /// Generate code for a defined SIR `function` from `program`.
    unsafe fn generate_function(&mut self, program: &SirProgram, func: &SirFunction) -> WeldResult<()> {
        let function = *self.functions.get(&func.id).unwrap();
        if LLVMCountParams(function) != func.params.len() as u32 {
            return compile_err!("Internal error");
        }

        // Create a context for the function.
        let ref mut context = FunctionContext::new(self.context, program, func, function);

        // Create the entry basic block, where we define alloca'd variables.
        let entry_bb = LLVMAppendBasicBlockInContext(self.context, context.llvm_function, c_str!("entry"));
        LLVMPositionBuilderAtEnd(context.builder, entry_bb);

        // Add the function parameters. Function parameters are stored in alloca'd variables. The
        // function parameters are always enumerated alphabetically sorted by symbol name.
        for (symbol, ty) in func.params.iter() {
            debug!("Adding param symbol {} to context", symbol);
            let name = CString::new(symbol.to_string()).unwrap();
            let value = LLVMBuildAlloca(context.builder, self.llvm_type(ty)?, name.as_ptr()); 
            context.symbols.insert(symbol.clone(), value);
        }

        // Generate local variables.
        for (symbol, ty) in func.locals.iter() {
            debug!("Adding local symbol {} to context", symbol);
            let name = CString::new(symbol.to_string()).unwrap();
            let value = LLVMBuildAlloca(context.builder, self.llvm_type(ty)?, name.as_ptr()); 
            context.symbols.insert(symbol.clone(), value);
        }

        // Store the parameter values in the alloca'd symbols.
        for (i, (symbol, _)) in func.params.iter().enumerate() {
            let pointer = context.get_value(symbol)?;
            let value = LLVMGetParam(function, i as u32);
            LLVMBuildStore(context.builder, value, pointer);
        }

        // Generate the basic block references by appending each basic block to the function. We do
        // this first so we can forward reference blocks if necessary.
        for bb in func.blocks.iter() {
            let name = CString::new(format!("b{}", bb.id)).unwrap();
            let block = LLVMAppendBasicBlockInContext(self.context, context.llvm_function, name.as_ptr());
            context.blocks.insert(bb.id, block);
        }

        // Jump from locals to the first basic block.
        LLVMPositionBuilderAtEnd(context.builder, entry_bb);
        LLVMBuildBr(context.builder, context.get_block(&func.blocks[0].id)?);

        // Generate code for the basic blocks in order.
        for bb in func.blocks.iter() {
            self.generate_basic_block(context, bb)?;
        }

        Ok(())
    }

    /// Generate code for a basic block.
    ///
    /// This function should only be called once per basic block.
    unsafe fn generate_basic_block(&mut self, context: &mut FunctionContext, bb: &BasicBlock) -> WeldResult<()> {
        LLVMPositionBuilderAtEnd(context.builder, context.get_block(&bb.id)?);
        for statement in bb.statements.iter() {
            self.generate_statement(context, statement)?;
        }
        self.generate_terminator(context, &bb)?;
        Ok(())
    }

    /// Generate code for an SIR statement.
    unsafe fn generate_statement(&mut self, context: &mut FunctionContext, statement: &Statement) -> WeldResult<()> {
        use sir::StatementKind::*;
        let ref output = statement.output.clone().unwrap_or(Symbol::new("unused", 0));
        match statement.kind {
            Assign(ref value) => {
                let loaded = self.load(context.builder, context.get_value(value)?)?;
                LLVMBuildStore(context.builder, loaded, context.get_value(output)?);
                Ok(())
            }
            AssignLiteral(_) => {
                use self::gen_numeric::NumericExpressionGen;
                self.gen_literal(context, statement)
            }
            BinOp { .. } => {
                use self::gen_numeric::NumericExpressionGen;
                self.gen_binop(context, statement)
            }
            GetField { ref value, index } => {
                let pointer = context.get_value(output)?;
                let value_pointer = context.get_value(value)?;
                let elem_pointer = LLVMBuildStructGEP(context.builder, value_pointer, index, NULL_NAME.as_ptr());
                let elem = self.load(context.builder, elem_pointer)?;
                LLVMBuildStore(context.builder, elem, pointer);
                Ok(())
            }
            MakeStruct(ref elems) => {
                let pointer = context.get_value(output)?;
                for (i, elem) in elems.iter().enumerate() {
                    let elem_pointer = LLVMBuildStructGEP(context.builder, pointer, i as u32, NULL_NAME.as_ptr());
                    let value = self.load(context.builder, context.get_value(elem)?)?;
                    LLVMBuildStore(context.builder, value, elem_pointer);
                }
                Ok(())
            }
            _ => unimplemented!(),
        }
    }

    /// Generate code for a terminator within an SIR basic block.
    unsafe fn generate_terminator(&mut self, context: &mut FunctionContext, bb: &BasicBlock) -> WeldResult<()> {
        use sir::Terminator::*;
        match bb.terminator {
            ProgramReturn(ref _sym) => {
                LLVMBuildRetVoid(context.builder);
            }
            Branch { ref cond, ref on_true, ref on_false } => {
                LLVMPositionBuilderAtEnd(context.builder, context.get_block(&bb.id)?);
                let _ = LLVMBuildCondBr(context.builder,
                                        context.get_value(cond)?,
                                        context.get_block(&on_true)?,
                                        context.get_block(&on_false)?);
            }
            JumpBlock(ref id) => {
                LLVMPositionBuilderAtEnd(context.builder, context.get_block(&bb.id)?);
                LLVMBuildBr(context.builder, context.get_block(id)?);
            }
            JumpFunction(ref _func) => {
                unimplemented!()
            }
            ParallelFor(ref _parfor) => {
                unimplemented!()
            }
            EndFunction => {
                LLVMBuildRetVoid(context.builder);
            }
            Crash => {
                // Set errno?
                LLVMBuildRetVoid(context.builder);
            }
        };
        Ok(())
    }

    // Common instructions or wrappers.

    /// Loads a value.
    ///
    /// This method includes a check to ensure that `pointer` is actually a pointer: otherwise, the
    /// LLVM API seg-faults.
    unsafe fn load(&mut self, builder: LLVMBuilderRef, pointer: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        use self::llvm_sys::LLVMTypeKind;
        if LLVMGetTypeKind(LLVMTypeOf(pointer)) != LLVMTypeKind::LLVMPointerTypeKind {
            compile_err!("Non-pointer type passed to load")
        } else {
            Ok(LLVMBuildLoad(builder, pointer, NULL_NAME.as_ptr()))
        }
    }

    // TODO move to own file...?
    unsafe fn llvm_type(&mut self, ty: &Type) -> WeldResult<LLVMTypeRef> {
        use ast::Type::*;
        use ast::ScalarKind::*;
        let result = match *ty {
            Builder(_, _) => {
                unimplemented!()
            }
            Dict(_, _) => {
                unimplemented!()
            }
            Scalar(kind) => match kind {
                Bool => LLVMInt1TypeInContext(self.context),
                I8 | U8 => LLVMInt8TypeInContext(self.context),
                I16 | U16 => LLVMInt16TypeInContext(self.context),
                I32 | U32 => LLVMInt32TypeInContext(self.context),
                I64 | U64 => LLVMInt64TypeInContext(self.context),
                F32 => LLVMFloatTypeInContext(self.context),
                F64 => LLVMDoubleTypeInContext(self.context),
            }
            Simd(kind) => {
                let base = self.llvm_type(&Scalar(kind))?;
                // TODO set the vector width...
                LLVMVectorType(base, 4)
            }
            Struct(ref elems) => {
                let mut llvm_types: Vec<_> = elems.iter().map(&mut |t| self.llvm_type(t)).collect::<WeldResult<_>>()?;
                // XXX Do we want to name structs? We also need to track struct names here if we do...
                // let named = LLVMStructCreateNamed(self.context, c_str!("s"));
                // LLVMStructSetBody(named, llvm_types.as_mut_ptr(), llvm_types.len() as u32, 0);
                // named
                LLVMStructTypeInContext(self.context, llvm_types.as_mut_ptr(), llvm_types.len() as u32, 0)
            }
            Vector(ref elem_type) => {
                // TODO build vector methods and track names...
                let mut layout = [LLVMPointerType(self.llvm_type(elem_type)?, 0), self.llvm_type(&Scalar(I64))?];
                let vector = LLVMStructCreateNamed(self.context, c_str!("vec"));
                LLVMStructSetBody(vector, layout.as_mut_ptr(), layout.len() as u32, 0);
                vector
            }
            Function(_, _) | Unknown => {
                return compile_err!("Invalid type {} for code generation", ty)
            }
        };
        Ok(result)
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

pub struct FunctionContext<'a> {
    sir_program: &'a SirProgram,
    sir_function: &'a SirFunction,
    llvm_function: LLVMValueRef,
    symbols: FnvHashMap<Symbol, LLVMValueRef>,
    blocks: FnvHashMap<BasicBlockId, LLVMBasicBlockRef>,
    builder: LLVMBuilderRef,
}

impl<'a> FunctionContext<'a> {
    pub fn new(llvm_context: LLVMContextRef,
               sir_program: &'a SirProgram,
               sir_function: &'a SirFunction,
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

    pub fn get_value(&self, sym: &Symbol) -> WeldResult<LLVMValueRef> {
        self.symbols.get(sym).cloned().ok_or(WeldCompileError::new("Undefined symbol in function codegen"))
    }

    pub fn get_block(&self, id: &BasicBlockId) -> WeldResult<LLVMBasicBlockRef> {
        self.blocks.get(id).cloned().ok_or(WeldCompileError::new("Undefined basic block in function codegen"))
    }
}

impl<'a> Drop for FunctionContext<'a> {
    fn drop(&mut self) {
        unsafe { LLVMDisposeBuilder(self.builder); }
    }
}

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

    let codegen = unsafe { LlvmGenerator::generate(conf.clone(), &sir_prog)? };
    println!("{}", codegen);
    Ok(CompiledModule)
}
