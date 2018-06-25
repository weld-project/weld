//! An LLVM backend currently optimized for single-threaded execution.
//!
//! The `LlvmGenerator` struct is responsible for converting an SIR program into an LLVM module.
//! This module is then JIT'd and returned into a runnable executable.
extern crate fnv;
extern crate time;
extern crate libc;
extern crate llvm_sys;

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

use std::fmt;
use std::ffi::{CStr, CString};

use fnv::FnvHashMap;
use libc::{c_char, c_double, c_ulonglong};
use time::PreciseTime;

use ast::*;
use conf::ParsedConf;
use error::*;
use optimizer::*;
use runtime::WeldRuntimeErrno;
use sir::*;
use syntax::program::Program;
use util::stats::CompilationStats;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;
use self::llvm_sys::LLVMLinkage;

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

pub use self::jit::CompiledModule;

use self::builder::merger;


/// A wrapper for a struct passed as input to the Weld runtime.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct WeldInputArgs {
    pub input: i64,
    pub nworkers: i32,
    pub mem_limit: i64,
}

impl WeldInputArgs {
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

/// A wrapper for outputs passed out of the Weld runtime.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct WeldOutputArgs {
    pub output: i64,
    pub run_id: i64,
    pub errno: WeldRuntimeErrno,
}

impl WeldOutputArgs {
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

    fn runid_index() -> u32 {
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
            Ok(LLVMBuildLoad(builder, pointer, NULL_NAME.as_ptr()))
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
        let block = LLVMAppendBasicBlockInContext(self.context(), function, c_str!("entry"));
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

        let run_id = self.intrinsics.call_weld_run_init(builder, nworkers, memlimit, None);

        let arg_pointer = LLVMBuildStructGEP(builder, pointer, WeldInputArgs::input_index(), c_str!("argptr"));
        // Still a pointer, but now as an integer.
        let arg_pointer = self.load(builder, arg_pointer)?;
        // The first SIR function is the entry point.
        let ref arg_ty = Struct(program.top_params.iter().map(|p| p.ty.clone()).collect());
        let llvm_arg_ty = self.llvm_type(arg_ty)?;
        let arg_struct_pointer = LLVMBuildIntToPtr(builder, arg_pointer, LLVMPointerType(llvm_arg_ty, 0), c_str!("arg"));

        let mut func_args = vec![];
        for (i, _) in program.top_params.iter().enumerate() {
            let pointer = LLVMBuildStructGEP(builder, arg_struct_pointer, i as u32, c_str!("param"));
            let value = self.load(builder, pointer)?;
            func_args.push(value);
        }

        // Run the Weld program.
        let entry_function = *self.functions.get(&program.funcs[0].id).unwrap();
        let _ = LLVMBuildCall(builder, entry_function, func_args.as_mut_ptr(), func_args.len() as u32, c_str!(""));

        let result = self.intrinsics.call_weld_run_get_result(builder, run_id, None);
        let result = LLVMBuildPtrToInt(builder, result, self.i64_type(), c_str!(""));

        let mut output = LLVMGetUndef(output_type);
        output = LLVMBuildInsertValue(builder, output, result, WeldOutputArgs::output_index(), c_str!("result"));
        output = LLVMBuildInsertValue(builder, output, run_id, WeldOutputArgs::runid_index(), c_str!("runid"));
        // TODO Get and set the actual errno.
        output = LLVMBuildInsertValue(builder, output, self.i64(0), WeldOutputArgs::errno_index(), c_str!("errno"));

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
    /// Since the SIR does not expose runtime-related parameters (e.g., thread IDs and number of
    /// threads), this function may additionally inject additional parameters into the function
    /// parameter list if the function is a loop body. Invocations to those functions must be
    /// managed appropriately to ensure that the parameters added here are passed during call
    /// generation.
    ///
    /// This method only defines functions and does not generate code for the function.
    unsafe fn declare_sir_function(&mut self, func: &SirFunction) -> WeldResult<()> {
        let mut arg_tys = self.argument_types(func)?;
        let ret_ty = self.void_type();
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
        if LLVMCountParams(function) != func.params.len() as u32 {
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
                        methods.gen_new(context.builder, &mut self.intrinsics, size)?
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
                unimplemented!() 
            }
            NewBuilder { .. } => {
                use self::builder::BuilderExpressionGen;
                self.gen_new_builder(context, statement)
            }
            Res(_) => {
                use self::builder::BuilderExpressionGen;
                self.gen_result(context, statement)
            }
            Select { .. } => {
                unimplemented!() 
            }
            Serialize(_) => {
                unimplemented!() 
            }
            Slice { .. } => {
                unimplemented!() 
            }
            Sort { .. } => {
                unimplemented!() 
            }
            ToVec(_) => {
                unimplemented!() 
            }
            UnaryOp { .. }  => {
                unimplemented!() 
            }
        }
    }

    /// Generate code for a terminator within an SIR basic block.
    unsafe fn gen_terminator(&mut self,
                                  context: &mut FunctionContext,
                                  bb: &BasicBlock,
                                  loop_terminator: Option<LLVMBasicBlockRef>) -> WeldResult<()> {
        use sir::Terminator::*;
        match bb.terminator {
            ProgramReturn(ref sym) => {
                let value = self.load(context.builder, context.get_value(sym)?)?;
                let run_id = self.intrinsics.call_weld_run_get_run_id(context.builder, None);
                let ty = LLVMTypeOf(value);
                let size = self.size_of(ty);
                let bytes = self.intrinsics.call_weld_run_malloc(context.builder, run_id, size, None);
                let pointer = LLVMBuildBitCast(context.builder, bytes, LLVMPointerType(ty, 0), c_str!(""));
                LLVMBuildStore(context.builder, value, pointer);
                let _ = self.intrinsics.call_weld_run_set_result(context.builder, bytes, None);
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
            ParallelFor(ref parfor) => {
                use self::builder::BuilderExpressionGen;
                self.gen_for(context, parfor)?;
            }
            EndFunction => {
                if let Some(bb) = loop_terminator {
                    LLVMBuildBr(context.builder, bb);
                } else {
                    LLVMBuildRetVoid(context.builder);
                }
            }
            Crash => {
                // Set errno?
                LLVMBuildRetVoid(context.builder);
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
}

impl<'a> Drop for FunctionContext<'a> {
    fn drop(&mut self) {
        unsafe { LLVMDisposeBuilder(self.builder); }
    }
}

/// Writes code to a file specified by `PathBuf`. Writes a log message if it failed.
fn write_code(code: &str, ext: &str, timestamp: &str, dir_path: &PathBuf) {
    let mut options = OpenOptions::new();
    options.write(true)
        .create_new(true)
        .create(true);
    let ref mut path = dir_path.clone();
    path.push(format!("code-{}", timestamp));
    path.set_extension(ext);

    let ref path_str = format!("{}", path.display());
    match options.open(path) {
        Ok(ref mut file) => {
            if let Err(_) = file.write_all(code.as_bytes()) {
                error!("Write failed: could not write code to file {}", path_str);
            }
        }
        Err(_) => {
            error!("Open failed: could not write code to file {}", path_str);
        }
    }
}

// XXX Why is this here...
pub fn apply_opt_passes(expr: &mut Expr,
                        opt_passes: &Vec<Pass>,
                        stats: &mut CompilationStats,
                        use_experimental: bool) -> WeldResult<()> {

    for pass in opt_passes {
        if pass.pass_name() == "vectorize" {
            continue;
        }
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

    let ref timestamp = format!("{}", time::now().to_timespec().sec);
    // Dump files if needed. Do this here in case the actual LLVM code gen fails.
    if conf.dump_code.enabled {
        info!("Writing code to directory '{}' with timestamp {}", &conf.dump_code.dir.display(), timestamp);
        write_code(expr.pretty_print().as_ref(), "weld", timestamp, &conf.dump_code.dir);
        write_code(&format!("{}", &sir_prog), "sir", timestamp, &conf.dump_code.dir);
        write_code(&format!("{}", &codegen), "ll", timestamp, &conf.dump_code.dir);
    }

    println!("{}", codegen);

    let module = unsafe { jit::compile(codegen.context, codegen.module, conf)? };
    if conf.dump_code.enabled {
        write_code(&module.asm()?, "S", &format!("{}-opt", timestamp), &conf.dump_code.dir);
        write_code(&module.llvm()?, "ll", &format!("{}-opt", timestamp), &conf.dump_code.dir);
    }
    Ok(module)
}
