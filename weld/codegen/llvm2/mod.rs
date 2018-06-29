//! An LLVM backend currently optimized for single-threaded execution.
//!
//! The `LlvmGenerator` struct is responsible for converting an SIR program into an LLVM module.
//! This module is then JIT'd and returned into a runnable executable.
extern crate fnv;
extern crate time;
extern crate libc;
extern crate llvm_sys;

use std::fmt;
use std::ffi::{CStr, CString};

use fnv::FnvHashMap;
use libc::{c_char, c_double, c_ulonglong};

use conf::ParsedConf;
use error::*;
use sir::*;
use util::stats::CompilationStats;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;
use self::llvm_sys::LLVMLinkage;

use super::*;

static NULL_NAME:[c_char; 1] = [0];

// TODO This should be based on the type!
pub const LLVM_VECTOR_WIDTH: u32 = 4;

/// Convert a string literal into a C string.
macro_rules! c_str {
    ($s:expr) => (
        concat!($s, "\0").as_ptr() as *const i8
    );
}

// Traits implementing code generation for various expressions.
mod builder;
mod intrinsic;
mod jit;
mod numeric;
mod vector;

use self::builder::merger;
use self::builder::appender;

pub fn compile(program: &SirProgram,
               conf: &ParsedConf,
               stats: &mut CompilationStats,
               dump_prefix: &str) -> WeldResult<Box<dyn Runnable>> {

    info!("Compiling using single thread runtime");

    use runtime::strt;

    let codegen = unsafe { LlvmGenerator::generate(conf.clone(), &program)? };
    if conf.dump_code.enabled {
        write_code(codegen.to_string(), "ll", dump_prefix, &conf.dump_code.dir);
    }
    trace!("{}", codegen);

    unsafe {
        strt::weld_init();
    }

    let module = unsafe { jit::compile(codegen.context, codegen.module, conf, stats)? };

    if conf.dump_code.enabled {
        write_code(module.asm()?, "S", format!("{}-opt", dump_prefix), &conf.dump_code.dir);
        write_code(module.llvm()?, "ll", format!("{}-opt", dump_prefix), &conf.dump_code.dir);
    }
    Ok(Box::new(module))
}

trait LlvmInputArg {
    unsafe fn llvm_type(context: LLVMContextRef) -> LLVMTypeRef; 
    fn input_index() -> u32;
    fn nworkers_index() -> u32;
    fn memlimit_index() -> u32;
}

impl LlvmInputArg for WeldInputArgs {
    unsafe fn llvm_type(context: LLVMContextRef) -> LLVMTypeRef {
        let mut types = [
            LLVMInt64TypeInContext(context),
            LLVMInt32TypeInContext(context),
            LLVMInt64TypeInContext(context)
        ];
        let args = LLVMStructCreateNamed(context, c_str!("input_args_t"));
        LLVMStructSetBody(args, types.as_mut_ptr(), types.len() as u32, 0);
        args
    }

    fn input_index() -> u32 {
        0
    }

    fn nworkers_index() -> u32 {
        1
    }

    fn memlimit_index() -> u32 {
        2
    }
}

trait LlvmOutputArg {
    unsafe fn llvm_type(context: LLVMContextRef) -> LLVMTypeRef; 
    fn output_index() -> u32;
    fn run_index() -> u32;
    fn errno_index() -> u32;
}

impl LlvmOutputArg for WeldOutputArgs {
    unsafe fn llvm_type(context: LLVMContextRef) -> LLVMTypeRef {
        let mut types = [
            LLVMInt64TypeInContext(context),
            LLVMInt64TypeInContext(context),
            LLVMInt64TypeInContext(context)
        ];
        let args = LLVMStructCreateNamed(context, c_str!("output_args_t"));
        LLVMStructSetBody(args, types.as_mut_ptr(), types.len() as u32, 0);
        args
    }

    fn output_index() -> u32 {
        0
    }

    fn run_index() -> u32 {
        1
    }

    fn errno_index() -> u32 {
        2
    }
}

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
    /// The key maps the element type to the vector's type reference and methods on it.
    vectors: FnvHashMap<Type, vector::Vector>,
    /// A map tracking generated mergers.
    ///
    /// The key maps the merger type to the merger's type reference and methods on it.
    mergers: FnvHashMap<BuilderKind, merger::Merger>,
    /// A map tracking generated appenders.
    ///
    /// The key maps the appender type to the appender's type reference and methods on it.
    appenders: FnvHashMap<BuilderKind, appender::Appender>,
    /// A map tracking generated dictionaries.
    ///
    /// The key maps the vector's element type to the vector's type reference and methods on it.
    dictionaries: FnvHashMap<Type, u32>,
    /// Intrinsics defined in the module.
    intrinsics: intrinsic::Intrinsics,
}

/// Defines helper methods for LLVM code generation.
///
/// The main methods here have default implementations: implemenators only need to implement the
/// `module` and `context` methods.
pub trait CodeGenExt {
    /// Returns the module used by this code generator.
    fn module(&self) -> LLVMModuleRef;

    /// Returns the context used by this code generator.
    fn context(&self) -> LLVMContextRef;

    /// Loads a value.
    ///
    /// This method includes a check to ensure that `pointer` is actually a pointer: otherwise, the
    /// LLVM API throws a segmentation fault.
    unsafe fn load(&mut self,
                   builder: LLVMBuilderRef,
                   pointer: LLVMValueRef) -> WeldResult<LLVMValueRef> {
        use self::llvm_sys::LLVMTypeKind;
        if LLVMGetTypeKind(LLVMTypeOf(pointer)) != LLVMTypeKind::LLVMPointerTypeKind {
            unreachable!()
        } else {
            let loaded = LLVMBuildLoad(builder, pointer, NULL_NAME.as_ptr());
            if LLVMGetTypeKind(LLVMTypeOf(loaded)) == LLVMTypeKind::LLVMVectorTypeKind {
                LLVMSetAlignment(loaded, 1);
            }
            Ok(loaded)
        }
    }



    /// Generates code to define a function with the given return type and argument type.
    ///
    /// Returns a reference to the function, a builder used to build the function body, and the
    unsafe fn define_function<T: Into<Vec<u8>>>(&mut self,
                                      ret_ty: LLVMTypeRef,
                                      arg_tys: &mut [LLVMTypeRef],
                                      name: T) -> (LLVMValueRef, LLVMBuilderRef, LLVMBasicBlockRef) {
        let func_ty = LLVMFunctionType(ret_ty, arg_tys.as_mut_ptr(), arg_tys.len() as u32, 0);
        let name = CString::new(name).unwrap();
        let function = LLVMAddFunction(self.module(), name.as_ptr(), func_ty); 
        let builder = LLVMCreateBuilderInContext(self.context());
        let block = LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));
        LLVMPositionBuilderAtEnd(builder, block);
        LLVMSetLinkage(function, LLVMLinkage::LLVMPrivateLinkage);
        (function, builder, block)
    }

    /// Converts a `LiteralKind` into a constant LLVM scalar literal value.
    ///
    /// This method does not generate any code.
    unsafe fn scalar_literal(&self, kind: &LiteralKind) -> LLVMValueRef {
        use ast::LiteralKind::*;
        match *kind {
            BoolLiteral(val) => self.bool(val),
            I8Literal(val) => self.i8(val),
            I16Literal(val) => self.i16(val),
            I32Literal(val) => self.i32(val),
            I64Literal(val) => self.i64(val),
            U8Literal(val) => self.u8(val),
            U16Literal(val) => self.u16(val),
            U32Literal(val) => self.u32(val),
            U64Literal(val) => self.u64(val),
            F32Literal(val) => self.f32(f32::from_bits(val)),
            F64Literal(val) => self.f64(f64::from_bits(val)),
            StringLiteral(_) => unimplemented!()
        }
    }

    /// Returns the identity for a given scalar kind and binary operator.
    unsafe fn binop_identity(&self, op: BinOpKind, kind: ScalarKind) -> WeldResult<LLVMValueRef> {
        use ast::BinOpKind::*;
        use ast::ScalarKind::*;
        match kind {
            _ if kind.is_integer() => {
                let ty = LLVMIntTypeInContext(self.context(), kind.bits());
                let signed = kind.is_signed() as i32;
                match op {
                    Add => Ok(LLVMConstInt(ty, 0, signed)),
                    Multiply => Ok(LLVMConstInt(ty, 1, signed)),
                    Max => Ok(LLVMConstInt(ty, ::std::u64::MIN, signed)),
                    Min => Ok(LLVMConstInt(ty, ::std::u64::MAX, signed)),
                    _ => unreachable!(),
                }
            }
            F32  => {
                let ty = self.f32_type();
                match op {
                    Add => Ok(LLVMConstReal(ty, 0.0)),
                    Multiply => Ok(LLVMConstReal(ty, 1.0)),
                    Max => Ok(LLVMConstReal(ty, ::std::f32::MIN as c_double)),
                    Min => Ok(LLVMConstReal(ty, ::std::f32::MAX as c_double)),
                    _ => unreachable!(),
                }
            }
            F64 => {
                let ty = self.f64_type();
                match op {
                    Add => Ok(LLVMConstReal(ty, 0.0)),
                    Multiply => Ok(LLVMConstReal(ty, 1.0)),
                    Max => Ok(LLVMConstReal(ty, ::std::f64::MIN)),
                    Min => Ok(LLVMConstReal(ty, ::std::f64::MAX)),
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        }
    }

    /// Returns the constant size of a type.
    unsafe fn size_of(&self, ty: LLVMTypeRef) -> LLVMValueRef {
        let size_pointer = LLVMConstGEP(self.null_ptr(ty), [self.i32(1)].as_mut_ptr(), 1);
        LLVMConstPtrToInt(size_pointer, self.i64_type())
    }

    unsafe fn bool_type(&self) -> LLVMTypeRef {
        LLVMInt1TypeInContext(self.context())
    }

    unsafe fn i8_type(&self) -> LLVMTypeRef {
        LLVMInt8TypeInContext(self.context())
    }

    unsafe fn u8_type(&self) -> LLVMTypeRef {
        LLVMInt8TypeInContext(self.context())
    }

    unsafe fn i16_type(&self) -> LLVMTypeRef {
        LLVMInt16TypeInContext(self.context())
    }

    unsafe fn u16_type(&self) -> LLVMTypeRef {
        LLVMInt16TypeInContext(self.context())
    }

    unsafe fn i32_type(&self) -> LLVMTypeRef {
        LLVMInt32TypeInContext(self.context())
    }

    unsafe fn u32_type(&self) -> LLVMTypeRef {
        LLVMInt32TypeInContext(self.context())
    }

    unsafe fn i64_type(&self) -> LLVMTypeRef {
        LLVMInt64TypeInContext(self.context())
    }

    unsafe fn u64_type(&self) -> LLVMTypeRef {
        LLVMInt64TypeInContext(self.context())
    }

    unsafe fn f32_type(&self) -> LLVMTypeRef {
        LLVMFloatTypeInContext(self.context())
    }

    unsafe fn f64_type(&self) -> LLVMTypeRef {
        LLVMDoubleTypeInContext(self.context())
    }

    unsafe fn void_type(&self) -> LLVMTypeRef {
        LLVMVoidTypeInContext(self.context())
    }

    unsafe fn run_handle_type(&self) -> LLVMTypeRef {
        LLVMPointerType(self.i8_type(), 0)
    }

    unsafe fn bool(&self, v: bool) -> LLVMValueRef {
        LLVMConstInt(self.bool_type(), v as c_ulonglong, 0)
    }

    unsafe fn i8(&self, v: i8) -> LLVMValueRef {
        LLVMConstInt(self.i8_type(), v as c_ulonglong, 1)
    }

    unsafe fn u8(&self, v: u8) -> LLVMValueRef {
        LLVMConstInt(self.u8_type(), v as c_ulonglong, 0)
    }

    unsafe fn i16(&self, v: i16) -> LLVMValueRef {
        LLVMConstInt(self.i16_type(), v as c_ulonglong, 1)
    }

    unsafe fn u16(&self, v: u16) -> LLVMValueRef {
        LLVMConstInt(self.u16_type(), v as c_ulonglong, 0)
    }

    unsafe fn i32(&self, v: i32) -> LLVMValueRef {
        LLVMConstInt(self.i32_type(), v as c_ulonglong, 1)
    }

    unsafe fn u32(&self, v: u32) -> LLVMValueRef {
        LLVMConstInt(self.u32_type(), v as c_ulonglong, 0)
    }

    unsafe fn i64(&self, v: i64) -> LLVMValueRef {
        LLVMConstInt(self.i64_type(), v as c_ulonglong, 1)
    }

    unsafe fn u64(&self, v: u64) -> LLVMValueRef {
        LLVMConstInt(self.u64_type(), v as c_ulonglong, 0)
    }

    unsafe fn f32(&self, v: f32) -> LLVMValueRef {
        LLVMConstReal(self.f32_type(), v as c_double)
    }

    unsafe fn f64(&self, v: f64) -> LLVMValueRef {
        LLVMConstReal(self.f64_type(), v as c_double)
    }

    unsafe fn null_ptr(&self, ty: LLVMTypeRef) -> LLVMValueRef {
        LLVMConstPointerNull(LLVMPointerType(ty, 0))
    }
}

impl CodeGenExt for LlvmGenerator {
    fn module(&self) -> LLVMModuleRef {
        self.module
    }

    fn context(&self) -> LLVMContextRef {
        self.context
    }
}

impl LlvmGenerator {
    /// Generate code for an SIR program.
    unsafe fn generate(conf: ParsedConf, program: &SirProgram) -> WeldResult<LlvmGenerator> {
        let context = LLVMContextCreate();
        let module = LLVMModuleCreateWithNameInContext(c_str!("main"), context);
        // Adds the default intrinsic definitions.
        let intrinsics = intrinsic::Intrinsics::defaults(context, module);

        let mut gen = LlvmGenerator {
            conf: conf,
            context: context,
            module: module,
            functions: FnvHashMap::default(),
            vectors: FnvHashMap::default(),
            mergers: FnvHashMap::default(),
            appenders: FnvHashMap::default(),
            dictionaries: FnvHashMap::default(),
            intrinsics: intrinsics,
        };

        // Declare each function first to create a reference to it. Loop body functions are only
        // called by their ParallelForData terminators, so those are generated on-the-fly during
        // loop code generation.
        for func in program.funcs.iter().filter(|f| !f.loop_body) {
            gen.declare_sir_function(func)?;
        }

        // Generate each non-loop body function in turn. Loop body functions are constructed when
        // the For loop is generated, with the loop control flow injected into the function.
        for func in program.funcs.iter().filter(|f| !f.loop_body) {
            gen.gen_sir_function(program, func)?;
        }

        // Generates a callable entry function in the module.
        gen.gen_entry(program)?;

        Ok(gen)
    }

    /// Generates a print call with the given string.
    unsafe fn gen_print<T: AsRef<CStr>>(&mut self,
                                        builder: LLVMBuilderRef,
                                        run: LLVMValueRef,
                                        string: T) -> WeldResult<()> {
        let string = LLVMBuildGlobalStringPtr(builder,
                                              string.as_ref().as_ptr(), c_str!(""));
        let pointer = LLVMConstBitCast(string, LLVMPointerType(self.i8_type(), 0));
        let _ = self.intrinsics.call_weld_run_print(builder, run, pointer);
        Ok(())
    }

    /// Generates the entry point to the Weld program.
    ///
    /// The entry function takes an `i64` and returns an `i64`. Both represent pointers that
    /// point to a `WeldInputAgs` and `WeldOutputArgs` respectively.
    unsafe fn gen_entry(&mut self, program: &SirProgram) -> WeldResult<()> {
        use ast::Type::Struct;

        let input_type = WeldInputArgs::llvm_type(self.context);
        let output_type = WeldOutputArgs::llvm_type(self.context);

        let name = CString::new("run").unwrap();
        let func_ty = LLVMFunctionType(self.i64_type(), [self.i64_type()].as_mut_ptr(), 1, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), func_ty);
        LLVMSetLinkage(function, LLVMLinkage::LLVMExternalLinkage);

        let builder = LLVMCreateBuilderInContext(self.context);
        let block = LLVMAppendBasicBlockInContext(self.context, function, c_str!(""));
        LLVMPositionBuilderAtEnd(builder, block);

        let argument = LLVMGetParam(function, 0);
        let pointer = LLVMBuildIntToPtr(builder, argument, LLVMPointerType(input_type, 0), c_str!(""));

        let nworkers_pointer = LLVMBuildStructGEP(builder, pointer, WeldInputArgs::nworkers_index(), c_str!("nworkers"));
        let nworkers = self.load(builder, nworkers_pointer)?;
        let memlimit_pointer = LLVMBuildStructGEP(builder, pointer, WeldInputArgs::memlimit_index(), c_str!("memlimit"));
        let memlimit = self.load(builder, memlimit_pointer)?;


        let run = self.intrinsics.call_weld_run_init(builder, nworkers, memlimit, None);

        let arg_pointer = LLVMBuildStructGEP(builder, pointer, WeldInputArgs::input_index(), c_str!("argptr"));
        // Still a pointer, but now as an integer.
        let arg_pointer = self.load(builder, arg_pointer)?;
        // The first SIR function is the entry point.
        let ref arg_ty = Struct(program.top_params.iter().map(|p| p.ty.clone()).collect());
        let llvm_arg_ty = self.llvm_type(arg_ty)?;
        let arg_struct_pointer = LLVMBuildIntToPtr(builder, arg_pointer, LLVMPointerType(llvm_arg_ty, 0), c_str!("arg"));

        // Function arguments are sorted by symbol name - arrange the inputs in the proper order.
        let mut params: Vec<(&Symbol, u32)> = program.top_params.iter()
            .enumerate()
            .map(|(i, p)| (&p.name, i as u32))
            .collect();

        params.sort();

        let mut func_args = vec![];
        for (_, i) in params.iter() {
            let pointer = LLVMBuildStructGEP(builder, arg_struct_pointer, *i, c_str!("param"));
            let value = self.load(builder, pointer)?;
            func_args.push(value);
        }
        // Push the run handle.
        func_args.push(run);

        // Run the Weld program.
        let entry_function = *self.functions.get(&program.funcs[0].id).unwrap();
        let _ = LLVMBuildCall(builder, entry_function, func_args.as_mut_ptr(), func_args.len() as u32, c_str!(""));

        let result = self.intrinsics.call_weld_run_get_result(builder, run, None);
        let result = LLVMBuildPtrToInt(builder, result, self.i64_type(), c_str!("result"));
        let errno = self.intrinsics.call_weld_run_get_errno(builder, run, Some(c_str!("errno")));
        let run_int = LLVMBuildPtrToInt(builder, run, self.i64_type(), c_str!("run"));

        let mut output = LLVMGetUndef(output_type);
        output = LLVMBuildInsertValue(builder, output, result, WeldOutputArgs::output_index(), c_str!(""));
        output = LLVMBuildInsertValue(builder, output, run_int, WeldOutputArgs::run_index(), c_str!(""));
        output = LLVMBuildInsertValue(builder, output, errno, WeldOutputArgs::errno_index(), c_str!(""));

        let return_pointer = LLVMBuildMalloc(builder, output_type, c_str!(""));
        LLVMBuildStore(builder, output, return_pointer);
        let return_value  = LLVMBuildPtrToInt(builder, return_pointer, self.i64_type(), c_str!(""));
        LLVMBuildRet(builder, return_value);

        LLVMDisposeBuilder(builder);
        Ok(())
    }

    /// Build the list of argument and return type for an SIR function.
    unsafe fn argument_types(&mut self, func: &SirFunction) -> WeldResult<Vec<LLVMTypeRef>> {
        let mut types = vec![];
        for (_, ty) in func.params.iter() {
            types.push(self.llvm_type(ty)?);
        }
        Ok(types)
    }

    /// Declare a function in the SIR module and track its reference.
    ///
    /// In addition to the SIR-defined parameters, the runtime adds an `i8*` pointer as the last
    /// argument to each function, representing the handle to the run data. This handle is always
    /// guaranteed to be the last argument.
    ///
    /// This method only defines functions and does not generate code for the function.
    unsafe fn declare_sir_function(&mut self, func: &SirFunction) -> WeldResult<()> {
        let mut arg_tys = self.argument_types(func)?;
        arg_tys.push(self.run_handle_type());
        let ret_ty = self.llvm_type(&func.return_type)?;
        let func_ty = LLVMFunctionType(ret_ty, arg_tys.as_mut_ptr(), arg_tys.len() as u32, 0);
        let name = CString::new(format!("f{}", func.id)).unwrap();
        let function = LLVMAddFunction(self.module, name.as_ptr(), func_ty);
        LLVMSetLinkage(function, LLVMLinkage::LLVMPrivateLinkage);

        self.functions.insert(func.id, function);
        Ok(())
    }

    /// Generates the Allocas for a function.
    ///
    /// The allocas should generally be generated in the entry block of the function. The caller
    /// should ensure that the context builder is appropriately positioned.
    unsafe fn gen_allocas(&mut self, context: &mut FunctionContext) -> WeldResult<()> {
        // Add the function parameters, which are stored in alloca'd variables. The
        // function parameters are always enumerated alphabetically sorted by symbol name.
        for (symbol, ty) in context.sir_function.params.iter() {
            let name = CString::new(symbol.to_string()).unwrap();
            let value = LLVMBuildAlloca(context.builder, self.llvm_type(ty)?, name.as_ptr()); 
            context.symbols.insert(symbol.clone(), value);
        }

        // alloca the local variables.
        for (symbol, ty) in context.sir_function.locals.iter() {
            let name = CString::new(symbol.to_string()).unwrap();
            let value = LLVMBuildAlloca(context.builder, self.llvm_type(ty)?, name.as_ptr()); 
            context.symbols.insert(symbol.clone(), value);
        }
        Ok(())
    }

    /// Generates code to store function parameters in alloca'd variables.
    unsafe fn gen_store_parameters(&mut self, context: &mut FunctionContext) -> WeldResult<()> {
        // Store the parameter values in the alloca'd symbols.
        for (i, (symbol, _)) in context.sir_function.params.iter().enumerate() {
            let pointer = context.get_value(symbol)?;
            let value = LLVMGetParam(context.llvm_function, i as u32);
            LLVMBuildStore(context.builder, value, pointer);
        }
        Ok(())
    }

    /// Generates code to define each basic block in the function.
    ///
    /// This function does not actually generate the basic block code: it only adds the basic
    /// blocks to the context so they can be forward referenced if necessary.
    unsafe fn gen_basic_block_defs(&mut self, context: &mut FunctionContext) -> WeldResult<()> {
        for bb in context.sir_function.blocks.iter() {
            let name = CString::new(format!("b{}", bb.id)).unwrap();
            let block = LLVMAppendBasicBlockInContext(self.context,
                                                      context.llvm_function,
                                                      name.as_ptr());
            context.blocks.insert(bb.id, block);
        }
        Ok(())
    }

    /// Generate code for a defined SIR `function` from `program`.
    ///
    /// This function specifically generates code for non-loop body functions.
    unsafe fn gen_sir_function(&mut self, program: &SirProgram, func: &SirFunction) -> WeldResult<()> {
        let function = *self.functions.get(&func.id).unwrap();
        // + 1 to account for the run handle.
        if LLVMCountParams(function) != (1 + func.params.len()) as u32 {
            unreachable!()
        }

        // Create a context for the function.
        let ref mut context = FunctionContext::new(self.context, program, func, function);

        // Create the entry basic block, where we define alloca'd variables.
        let entry_bb = LLVMAppendBasicBlockInContext(self.context,
                                                     context.llvm_function,
                                                     c_str!(""));
        LLVMPositionBuilderAtEnd(context.builder, entry_bb);

        self.gen_allocas(context)?;
        self.gen_store_parameters(context)?;
        self.gen_basic_block_defs(context)?;

        // Jump from locals to the first basic block.
        LLVMPositionBuilderAtEnd(context.builder, entry_bb);
        LLVMBuildBr(context.builder, context.get_block(&func.blocks[0].id)?);

        // Generate code for the basic blocks in order.
        for bb in func.blocks.iter() {
            LLVMPositionBuilderAtEnd(context.builder, context.get_block(&bb.id)?);
            for statement in bb.statements.iter() {
                self.gen_statement(context, statement)?;
            }
            self.gen_terminator(context, &bb, None)?;
        }
        Ok(())
    }

    /// Generate code for an SIR statement.
    unsafe fn gen_statement(&mut self, context: &mut FunctionContext, statement: &Statement) -> WeldResult<()> {
        use ast::Type::*;
        use sir::StatementKind::*;
        let ref output = statement.output.clone().unwrap_or(Symbol::new("unused", 0));

        if self.conf.trace_run {
            self.gen_print(context.builder, context.get_run(), CString::new(format!("{}", statement)).unwrap())?;
        }

        match statement.kind {
            Assign(ref value) => {
                let loaded = self.load(context.builder, context.get_value(value)?)?;
                LLVMBuildStore(context.builder, loaded, context.get_value(output)?);
                Ok(())
            }
            AssignLiteral(_) => {
                use self::numeric::NumericExpressionGen;
                self.gen_assign_literal(context, statement)
            }
            BinOp { .. } => {
                use self::numeric::NumericExpressionGen;
                self.gen_binop(context, statement)
            }
            Broadcast(ref child) => {
                let output_pointer = context.get_value(output)?;
                let child_value = self.load(context.builder, context.get_value(child)?)?;
                let ty = self.llvm_type(context.sir_function.symbol_type(output)?)?;
                let mut result = LLVMGetUndef(ty);
                for i in 0..LLVM_VECTOR_WIDTH {
                    result = LLVMBuildInsertElement(context.builder, result, child_value, self.i32(i as i32), c_str!(""));
                }
                LLVMBuildStore(context.builder, result, output_pointer);
                Ok(())
            }
            Cast(_, _) => {
                use self::numeric::NumericExpressionGen;
                self.gen_cast(context, statement)
            }
            CUDF { ref symbol_name, ref args } => {
                let output_pointer = context.get_value(output)?;
                let return_ty = self.llvm_type(context.sir_function.symbol_type(output)?)?;
                let mut arg_tys = vec![];
                for arg in args.iter() {
                    arg_tys.push(self.llvm_type(context.sir_function.symbol_type(arg)?)?);
                }
                self.intrinsics.add(symbol_name, return_ty, &mut arg_tys);

                let mut arg_values = vec![];
                for arg in args.iter() {
                    arg_values.push(self.load(context.builder, context.get_value(arg)?)?);
                }
                let result = self.intrinsics.call(context.builder, symbol_name, &mut arg_values)?;
                LLVMBuildStore(context.builder, result, output_pointer);
                Ok(())
            }
            Deserialize(_) => {
                unimplemented!() 
            }
            GetField { ref value, index } => {
                let output_pointer = context.get_value(output)?;
                let value_pointer = context.get_value(value)?;
                let elem_pointer = LLVMBuildStructGEP(context.builder, value_pointer, index, NULL_NAME.as_ptr());
                let elem = self.load(context.builder, elem_pointer)?;
                LLVMBuildStore(context.builder, elem, output_pointer);
                Ok(())
            }
            KeyExists { .. } => {
                unimplemented!()
            }
            Length(ref child) => {
                let output_pointer = context.get_value(output)?;
                let child_value = self.load(context.builder, context.get_value(child)?)?;
                let child_type = context.sir_function.symbol_type(child)?;
                if let Vector(ref elem_type) = *child_type {
                    let mut methods = self.vectors.get_mut(elem_type).unwrap();
                    let result = methods.gen_size(context.builder, child_value)?;
                    LLVMBuildStore(context.builder, result, output_pointer);
                    Ok(())
                } else {
                    unreachable!()
                }
            }
            Lookup { ref child, ref index } => {
                let output_pointer = context.get_value(output)?;
                let child_value = self.load(context.builder, context.get_value(child)?)?;
                let index_value = self.load(context.builder, context.get_value(index)?)?;
                let child_type = context.sir_function.symbol_type(child)?;
                if let Vector(ref elem_type) = *child_type {
                    let pointer = {
                        let mut methods = self.vectors.get_mut(elem_type).unwrap();
                        methods.gen_at(context.builder, child_value, index_value)?
                    };
                    let result = self.load(context.builder, pointer)?;
                    LLVMBuildStore(context.builder, result, output_pointer);
                    Ok(())
                } else if let Dict(_, _) = *child_type {
                    unimplemented!() 
                } else {
                    unreachable!()
                }
            }
            MakeStruct(ref elems) => {
                let output_pointer = context.get_value(output)?;
                for (i, elem) in elems.iter().enumerate() {
                    let elem_pointer = LLVMBuildStructGEP(context.builder, output_pointer, i as u32, NULL_NAME.as_ptr());
                    let value = self.load(context.builder, context.get_value(elem)?)?;
                    LLVMBuildStore(context.builder, value, elem_pointer);
                }
                Ok(())
            }
            MakeVector(ref elems) => {
                let output_pointer = context.get_value(output)?;
                let output_type = context.sir_function.symbol_type(output)?;
                let size = self.i64(elems.len() as i64);
                if let Vector(ref elem_type) = *output_type {
                    let vector = {
                        let mut methods = self.vectors.get_mut(elem_type).unwrap();
                        methods.gen_new(context.builder, &mut self.intrinsics, context.get_run(), size)?
                    };
                    for (i, elem) in elems.iter().enumerate() {
                        let index = self.i64(i as i64);
                        // Scope to prevent borrow error with self.load...
                        let vec_pointer = {
                            let mut methods = self.vectors.get_mut(elem_type).unwrap();
                            methods.gen_at(context.builder, vector, index)?
                        };
                        let loaded = self.load(context.builder, context.get_value(elem)?)?;
                        LLVMBuildStore(context.builder, loaded, vec_pointer);
                    }
                    LLVMBuildStore(context.builder, vector, output_pointer);
                    Ok(())
                } else {
                    unreachable!()
                }
            }
            Merge { .. } => {
                use self::builder::BuilderExpressionGen;
                self.gen_merge(context, statement)
            }
            Negate(_) => {
                use self::numeric::NumericExpressionGen;
                self.gen_negate(context, statement)
            }
            NewBuilder { .. } => {
                use self::builder::BuilderExpressionGen;
                self.gen_new_builder(context, statement)
            }
            Res(_) => {
                use self::builder::BuilderExpressionGen;
                self.gen_result(context, statement)
            }
            Select { ref cond, ref on_true, ref on_false } => {
                let output_pointer = context.get_value(output)?;
                let cond = self.load(context.builder, context.get_value(cond)?)?;
                let on_true = self.load(context.builder, context.get_value(on_true)?)?;
                let on_false = self.load(context.builder, context.get_value(on_false)?)?;
                let result = LLVMBuildSelect(context.builder, cond, on_true, on_false, c_str!(""));
                LLVMBuildStore(context.builder, result, output_pointer);
                Ok(())
            }
            Serialize(_) => {
                unimplemented!() 
            }
            Slice { ref child, ref index, ref size } => {
                let output_pointer = context.get_value(output)?;
                let child_value = self.load(context.builder, context.get_value(child)?)?;
                let index_value = self.load(context.builder, context.get_value(index)?)?;
                let size_value = self.load(context.builder, context.get_value(size)?)?;
                let child_type = context.sir_function.symbol_type(child)?;
                if let Vector(ref elem_type) = *child_type {
                    let result = {
                        let mut methods = self.vectors.get_mut(elem_type).unwrap();
                        methods.gen_slice(context.builder, child_value, index_value, size_value)?
                    };
                    LLVMBuildStore(context.builder, result, output_pointer);
                    Ok(())
                } else {
                    unreachable!()
                }
            }
            Sort { .. } => {
                unimplemented!() 
            }
            ToVec(_) => {
                unimplemented!() 
            }
            UnaryOp { .. }  => {
                use self::numeric::NumericExpressionGen;
                self.gen_unaryop(context, statement)
            }
        }
    }

    /// Generate code for a terminator within an SIR basic block.
    ///
    /// `loop_terminator` is an optional tuple that is present only when generating loop body
    /// functions. The first argument is a basic block to jump to to continue looping instead of
    /// returning from a function.  The second argument is a value representing the loop's builder
    /// argument pointer. In cases where the loop body function returns, this function stores the
    /// resulting builder into this pointer. The function is guaranteed to return a builder (since
    /// For loop functions must return builders derived from their input builder).
    unsafe fn gen_terminator(&mut self,
                                  context: &mut FunctionContext,
                                  bb: &BasicBlock,
                                  loop_terminator: Option<(LLVMBasicBlockRef, LLVMValueRef)>) -> WeldResult<()> {

        if self.conf.trace_run {
            self.gen_print(context.builder, context.get_run(), CString::new(format!("{}", bb.terminator)).unwrap())?;
        }

        use sir::Terminator::*;
        match bb.terminator {
            ProgramReturn(ref sym) => {
                let value = self.load(context.builder, context.get_value(sym)?)?;
                let run = context.get_run();
                let ty = LLVMTypeOf(value);
                let size = self.size_of(ty);
                let bytes = self.intrinsics.call_weld_run_malloc(context.builder, run, size, None);
                let pointer = LLVMBuildBitCast(context.builder, bytes, LLVMPointerType(ty, 0), c_str!(""));
                LLVMBuildStore(context.builder, value, pointer);
                let _ = self.intrinsics.call_weld_run_set_result(context.builder, run, bytes, None);
                LLVMBuildRet(context.builder, value);
            }
            Branch { ref cond, ref on_true, ref on_false } => {
                let cond = self.load(context.builder, context.get_value(cond)?)?;
                let _ = LLVMBuildCondBr(context.builder,
                                        cond,
                                        context.get_block(&on_true)?,
                                        context.get_block(&on_false)?);
            }
            JumpBlock(ref id) => {
                LLVMBuildBr(context.builder, context.get_block(id)?);
            }
            JumpFunction(ref func) => {
                let ref sir_function = context.sir_program.funcs[*func];
                let mut arguments = vec![];
                for (symbol, _) in sir_function.params.iter() {
                    let value = self.load(context.builder, context.get_value(symbol)?)?;
                    arguments.push(value);
                }
                arguments.push(context.get_run());
                let jump_function = *self.functions.get(func).unwrap();
                let result = LLVMBuildCall(context.builder,
                              jump_function,
                              arguments.as_mut_ptr(),
                              arguments.len() as u32,
                              c_str!(""));
                // XXX Should we return the builder here?
                LLVMBuildRet(context.builder, result);
            }
            ParallelFor(ref parfor) => {
                use self::builder::BuilderExpressionGen;
                let updated_builder = self.gen_for(context, parfor)?;
                if let Some((jumpto, loop_builder)) = loop_terminator {
                    LLVMBuildStore(context.builder, updated_builder, loop_builder);
                    LLVMBuildBr(context.builder, jumpto);
                } else {
                    // Continuation will continue the program - this ends the current function.
                    LLVMBuildRet(context.builder, updated_builder);
                }
            }
            EndFunction(ref sym) => {
                if let Some((jumpto, loop_builder)) = loop_terminator {
                    let pointer = context.get_value(sym)?;
                    let updated_builder = self.load(context.builder, pointer)?;
                    LLVMBuildStore(context.builder, updated_builder, loop_builder);
                    LLVMBuildBr(context.builder, jumpto);
                } else {
                    let pointer = context.get_value(sym)?;
                    let return_value = self.load(context.builder, pointer)?;
                    LLVMBuildRet(context.builder, return_value);
                }
            }
            Crash => {
                use runtime::WeldRuntimeErrno;
                let errno = self.i64(WeldRuntimeErrno::Unknown as i64);
                self.intrinsics.call_weld_run_set_errno(context.builder,
                                                        context.get_run(),
                                                        errno,
                                                        None);
                LLVMBuildUnreachable(context.builder);
            }
        };
        Ok(())
    }

    unsafe fn llvm_type(&mut self, ty: &Type) -> WeldResult<LLVMTypeRef> {
        use ast::Type::*;
        use ast::ScalarKind::*;
        let result = match *ty {
            Builder(_, _) => {
                use self::builder::BuilderExpressionGen;
                self.builder_type(ty)?
            }
            Dict(_, _) => {
                unimplemented!()
            }
            Scalar(kind) => match kind {
                Bool => self.bool_type(),
                I8 | U8 => self.i8_type(),
                I16 | U16 => self.i16_type(),
                I32 | U32 => self.i32_type(),
                I64 | U64 => self.i64_type(),
                F32 => self.f32_type(),
                F64 => self.f64_type(),
            }
            Simd(kind) => {
                let base = self.llvm_type(&Scalar(kind))?;
                LLVMVectorType(base, LLVM_VECTOR_WIDTH)
            }
            Struct(ref elems) => {
                let mut llvm_types: Vec<_> = elems.iter()
                    .map(&mut |t| self.llvm_type(t)).collect::<WeldResult<_>>()?;
                LLVMStructTypeInContext(self.context,
                                        llvm_types.as_mut_ptr(),
                                        llvm_types.len() as u32, 0)
            }
            Vector(ref elem_type) => {
                // Vectors are a named type, so only generate the name once.
                if !self.vectors.contains_key(elem_type) {
                    let llvm_elem_type = self.llvm_type(elem_type)?;
                    let vector = vector::Vector::define("vec",
                                                        llvm_elem_type,
                                                        self.context,
                                                        self.module);
                    self.vectors.insert(elem_type.as_ref().clone(), vector);
                }
                self.vectors.get(elem_type).unwrap().vector_ty
            }
            Function(_, _) | Unknown => unreachable!(),
        };
        Ok(result)
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
        self.symbols
            .get(sym)
            .cloned()
            .ok_or(WeldCompileError::new("Undefined symbol in function codegen"))
    }

    pub fn get_block(&self, id: &BasicBlockId) -> WeldResult<LLVMBasicBlockRef> {
        self.blocks.get(id)
            .cloned()
            .ok_or(WeldCompileError::new("Undefined basic block in function codegen"))
    }

    /// Get the handle to the run.
    pub fn get_run(&self) -> LLVMValueRef {
        unsafe { LLVMGetLastParam(self.llvm_function) }
    }
}

impl<'a> Drop for FunctionContext<'a> {
    fn drop(&mut self) {
        unsafe { LLVMDisposeBuilder(self.builder); }
    }
}
