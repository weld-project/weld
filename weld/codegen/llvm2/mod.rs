//! An LLVM backend currently optimized for single-threaded execution.
//!
//! The `LlvmGenerator` struct is responsible for converting an SIR program into an LLVM module.
//! The LLVM module is then JIT'd and returned as a runnable executable.
//!
//! # Overview
//!
//! This code generator is divided into a number of submodules, most of which implement extension
//! traits on top of `LlvmGenerator`. For example, the `hash` module implements the `GenHash`
//! trait, whose sole implemenator is `LlvmGenerator`. The `gen_hash` function in the `GenHash`
//! trait thus adds using state maintained in the `LlvmGenerator` to hash Weld types.
//! `LlvmGenerator` tracks code that has been generated already: for most extension traits, this
//! usually involves some state to ensure that the same code is not generated twice.
//!
//! # The `CodeGenExt` trait
//!
//! The `CodeGenExt` trait contains a number of helper functions for generating LLVM code,
//! retrieving types, etc. Implementors should implement the `module` and `context` functions: all
//! other methods in the trait have standard default implementations that should not be overridden.
//!
//! ## Submodules
//!
//! * The `builder` module provides code generation for the builder types.  `builder` also contains
//! extension traits for generating builder-related expressions (Result, Merge, and For).
//!
//! * The `dict` and `vector` modules define the layout of dictionaries and vectors, and also
//! provide methods over them.
//!
//! * The `eq` module defines equality-check code generation.
//!
//! * The `hash` module implements hashing.
//!
//! * The `intrinsics` module manages intrinsics, or functions that are declared but not generated.
//! This module adds a number of "default" intrinsics, such as the Weld runtime functions (prefixed
//! with `weld_strt_`), `memcpy`, and so forth.
//!
//! * The `jit` module manages compiling a constructed LLVM module into a runnable executable.
//! Among other things, it manages LLVM optimization passes and module verification.
//!
//! The `llvm_exts` modules uses `libllvmext` to provide LLVM functionality that `llvm_sys` (and by
//! extension, the `llvm-c` API) does not provide. It is effectively a wrapper around a few
//! required C++ library calls.
//!
//! * The `numeric` module generates code for numeric expressions such as binary and unary
//! operators, comparisons, etc.
//!
//! * The `serde` module generates code for serializing and deserializing types.
//!
//! * The `target` module provides parsed target specific feature information.

extern crate fnv;
extern crate time;
extern crate libc;
extern crate llvm_sys;
extern crate lazy_static;

use std::fmt;
use std::mem;
use std::ffi::{CStr, CString};

use fnv::FnvHashMap;
use libc::{c_char, c_double, c_uint, c_ulonglong};

use conf::ParsedConf;
use error::*;
use sir::*;
use util::stats::CompilationStats;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;
use self::llvm_sys::LLVMLinkage;

use super::*;

lazy_static! {
    /// Name of the run handle struct in generated code.
    static ref RUN_HANDLE_NAME: CString = CString::new("RunHandle").unwrap();
}

/// Width of a SIMD vector.
// TODO This should be based on the type!
pub const LLVM_VECTOR_WIDTH: u32 = 4;

/// Calling convention for SIR function.
pub const SIR_FUNC_CALL_CONV: u32 = llvm_sys::LLVMCallConv::LLVMFastCallConv as u32;

/// Convert a string literal into a C string.
macro_rules! c_str {
    ($s:expr) => (
        concat!($s, "\0").as_ptr() as *const i8
    );
}

mod builder;
mod cmp;
mod dict;
mod eq;
mod hash;
mod intrinsic;
mod jit;
mod llvm_exts;
mod numeric;
mod serde;
mod target;
mod vector;

use self::builder::appender;
use self::builder::merger;

/// Loads a dynamic library from a file using LLVMLoadLibraryPermanently.
///
/// It is safe to call this function multiple times for the same library.
pub fn load_library(libname: &str) -> WeldResult<()> {
    let c_string = CString::new(libname.clone()).unwrap();
    let c_string_raw = c_string.into_raw() as *const c_char;
    if unsafe { llvm_sys::support::LLVMLoadLibraryPermanently(c_string_raw) } == 0 {
        Ok(())
    } else {
        compile_err!("Couldn't load library {}", libname)
    }
}

/// Returns the size of a type in bytes.
pub fn size_of(ty: &Type) -> usize {
    unsafe {
        let mut gen = LlvmGenerator::new(ParsedConf::default()).unwrap();
        gen.size_of_ty(ty)
    }
}

/// Compile Weld SIR into a runnable module.
///
/// The runnable module is wrapped as a trait object which the `CompiledModule` struct in `codegen`
/// calls.
pub fn compile(program: &SirProgram,
               conf: &ParsedConf,
               stats: &mut CompilationStats) -> WeldResult<Box<dyn Runnable + Send + Sync>> {

    use runtime;
    use util::dump::{write_code, DumpCodeFormat};

    info!("Compiling using single thread runtime");

    let codegen = unsafe { LlvmGenerator::generate(conf.clone(), &program)? };

    nonfatal!(write_code(codegen.to_string(), DumpCodeFormat::LLVM, &conf.dump_code));

    unsafe {
        runtime::ffi::weld_init();
    }

    let module = unsafe { jit::compile(codegen.context, codegen.module, conf, stats)? };

    nonfatal!(write_code(module.asm()?, DumpCodeFormat::Assembly, &conf.dump_code));
    nonfatal!(write_code(module.llvm()?, DumpCodeFormat::LLVMOpt, &conf.dump_code));

    Ok(Box::new(module))
}

/// A helper trait that defines the LLVM type and structure of an input to the runtime.
trait LlvmInputArg {
    /// LLVM type of the input struct.
    unsafe fn llvm_type(context: LLVMContextRef) -> LLVMTypeRef;
    /// Index of the data pointer in the struct.
    fn input_index() -> u32;
    /// Index of the number of workers value in the struct.
    fn nworkers_index() -> u32;
    /// Index of the memory limit value in the struct.
    fn memlimit_index() -> u32;
    /// Index of run handle pointer in the struct.
    fn run_index() -> u32;
}

impl LlvmInputArg for WeldInputArgs {
    unsafe fn llvm_type(context: LLVMContextRef) -> LLVMTypeRef {
        let mut types = [
            LLVMInt64TypeInContext(context),
            LLVMInt32TypeInContext(context),
            LLVMInt64TypeInContext(context),
            LLVMInt64TypeInContext(context),
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

    fn run_index() -> u32 {
        3
    }
}

/// A helper trait that defines the LLVM type and structure of an output from the runtime.
trait LlvmOutputArg {
    /// LLVM type of the output struct.
    unsafe fn llvm_type(context: LLVMContextRef) -> LLVMTypeRef;
    /// Index of the output data pointer in the struct.
    fn output_index() -> u32;
    /// Index of the run ID/data pointer in the struct.
    fn run_index() -> u32;
    /// Index of the errno pointer in the struct.
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

/// Specifies whether a type contains a pointer in generated code.
pub trait HasPointer {
    fn has_pointer(&self) -> bool;
}

impl HasPointer for Type {
    fn has_pointer(&self) -> bool {
        use ast::Type::*;
        match *self {
            Scalar(_) => false,
            Simd(_) => false,
            Vector(_) => true,
            Dict(_, _) => true,
            Builder(_, _) => true,
            Struct(ref tys) => tys.iter().any(|ref t| t.has_pointer()),
            Function(_, _) | Unknown => unreachable!(),
        }
    }
}

/// A struct holding the global codegen state for an SIR program.
pub struct LlvmGenerator {
    /// A configuration for generating code.
    conf: ParsedConf,
    /// Target-specific information used during code generation.
    target: target::Target,
    /// An LLVM Context for isolating code generation.
    context: LLVMContextRef,
    /// The main LLVM module to which code is added.
    module: LLVMModuleRef,
    /// A map that tracks references to an SIR function's LLVM function.
    functions: FnvHashMap<FunctionId, LLVMValueRef>,
    /// A map tracking generated vectors.
    ///
    /// The key maps the *element type* to the vector's type reference and methods on it.
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
    /// The key maps the dictionary's `Dict` type to the type reference and methods on it.
    dictionaries: FnvHashMap<Type, dict::Dict>,
    /// Common intrinsics defined in the module.
    ///
    /// An intrinsic is any function defined outside of module (i.e., is not code generated).
    intrinsics: intrinsic::Intrinsics,
    /// Generated string literal values.
    strings: FnvHashMap<CString, LLVMValueRef>,
    /// Equality functions on various types.
    eq_fns: FnvHashMap<Type, LLVMValueRef>,
    /// Opaque, externally visible wrappers for equality functions.
    ///
    /// These are used by the dicitonary.
    opaque_eq_fns: FnvHashMap<Type, LLVMValueRef>,
    /// Comparison functions on various types.
    cmp_fns: FnvHashMap<Type, LLVMValueRef>,
    /// Opaque comparison functions that contain a key function, indexed by the ID of the key function.
    hash_fns: FnvHashMap<Type, LLVMValueRef>,
    /// Serialization functions on various types.
    serialize_fns: FnvHashMap<Type, LLVMValueRef>,
    /// Deserialization functions on various types.
    deserialize_fns: FnvHashMap<Type, LLVMValueRef>,
    /// Names of structs for readability.
    struct_names: FnvHashMap<Type, CString>,
    /// Counter for unique struct names.
    struct_index: u32,
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
            let loaded = LLVMBuildLoad(builder, pointer, c_str!(""));
            if LLVMGetTypeKind(LLVMTypeOf(loaded)) == LLVMTypeKind::LLVMVectorTypeKind {
                LLVMSetAlignment(loaded, 1);
            }
            Ok(loaded)
        }
    }

    /// Get a constant zero-value of the given type.
    unsafe fn zero(&self, ty: LLVMTypeRef) -> LLVMValueRef {
        use self::llvm_sys::LLVMTypeKind::*;
        use std::ptr;
        match LLVMGetTypeKind(ty) {
            LLVMFloatTypeKind => self.f32(0.0),
            LLVMDoubleTypeKind => self.f64(0.0),
            LLVMIntegerTypeKind => LLVMConstInt(ty, 0, 0),
            LLVMStructTypeKind => {
                let num_fields = LLVMCountStructElementTypes(ty) as usize;
                let mut fields = vec![ ptr::null_mut() ; num_fields ];
                LLVMGetStructElementTypes(ty, fields.as_mut_ptr());

                let mut value = LLVMGetUndef(ty);
                for (i, field) in fields.into_iter().enumerate() {
                    value = LLVMConstInsertValue(value, self.zero(field), [i as u32].as_mut_ptr(), 1);
                }
                value
            }
            LLVMPointerTypeKind => self.null_ptr(ty),
            LLVMVectorTypeKind => {
                let size = LLVMGetVectorSize(ty);
                let zero = self.zero(LLVMGetElementType(ty));
                let mut constants = vec![ zero; size as usize ];
                LLVMConstVector(constants.as_mut_ptr(), size)
            }
            // Other types are not used in the backend.
            other => panic!("Unsupported type kind {:?} in CodeGenExt::zero()", other)
        }
    }

    /// Returns the type of a hash code.
    unsafe fn hash_type(&self) -> LLVMTypeRef {
        self.i32_type()
    }

    /// Returns the type of the key comparator over opaque pointers.
    unsafe fn opaque_cmp_type(&self) -> LLVMTypeRef {
        let mut arg_tys = [self.void_pointer_type(), self.void_pointer_type()];
        let fn_type = LLVMFunctionType(self.i32_type(), arg_tys.as_mut_ptr(), arg_tys.len() as u32, 0);
        LLVMPointerType(fn_type, 0)
    }

    /// Generates code to define a function with the given return type and argument type.
    ///
    /// Returns a reference to the function, a builder used to build the function body, and the
    /// entry basic block. This method uses the default private linkage type, meaning functions
    /// generated using this method cannot be passed or called outside of the module.
    unsafe fn define_function<T: Into<Vec<u8>>>(&mut self,
                                      ret_ty: LLVMTypeRef,
                                      arg_tys: &mut [LLVMTypeRef],
                                      name: T) -> (LLVMValueRef, LLVMBuilderRef, LLVMBasicBlockRef) {
        self.define_function_with_visibility(ret_ty, arg_tys, LLVMLinkage::LLVMPrivateLinkage, name)
    }

    /// Generates code to define a function with the given return type and argument type.
    ///
    /// Returns a reference to the function, a builder used to build the function body, and the
    /// entry basic block.
    unsafe fn define_function_with_visibility<T: Into<Vec<u8>>>(&mut self,
                                      ret_ty: LLVMTypeRef,
                                      arg_tys: &mut [LLVMTypeRef],
                                      visibility: LLVMLinkage,
                                      name: T) -> (LLVMValueRef, LLVMBuilderRef, LLVMBasicBlockRef) {
        let func_ty = LLVMFunctionType(ret_ty, arg_tys.as_mut_ptr(), arg_tys.len() as u32, 0);
        let name = CString::new(name).unwrap();
        let function = LLVMAddFunction(self.module(), name.as_ptr(), func_ty);
        // Add the default attributes to all functions.
        llvm_exts::LLVMExtAddDefaultAttrs(self.context(), function);

        let builder = LLVMCreateBuilderInContext(self.context());
        let block = LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));
        LLVMPositionBuilderAtEnd(builder, block);
        LLVMSetLinkage(function, visibility);
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
            // Handled by the `gen_numeric`.
            StringLiteral(_) => unreachable!(),
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
        LLVMSizeOf(ty)
    }

    /// Returns the constant size of a type in bits.
    ///
    /// Unlike `size_of`, this returns the size of the value at compile time.
    unsafe fn size_of_bits(&self, ty: LLVMTypeRef) -> u64 {
        let layout = llvm_sys::target::LLVMGetModuleDataLayout(self.module());
        llvm_sys::target::LLVMSizeOfTypeInBits(layout, ty) as u64
    }

    /// Returns the LLVM type corresponding to size_t on this architecture.
    unsafe fn size_t_type(&self) -> WeldResult<LLVMTypeRef> {
        Ok(LLVMIntTypeInContext(self.context(), mem::size_of::<libc::size_t>() as c_uint))
    }

    /// Computes the next power of two for the given value.
    ///
    /// `value` must be either an `i32` or `i64` type.
    /// Uses the algorithm from https://graphics.stanford.edu/~seander/bithacks.html.
    unsafe fn next_pow2(&self, builder: LLVMBuilderRef, value: LLVMValueRef) -> LLVMValueRef {
        use self::llvm_sys::LLVMTypeKind;
        let ty = LLVMTypeOf(value);
        assert!(LLVMGetTypeKind(ty) == LLVMTypeKind::LLVMIntegerTypeKind);
        let bits = LLVMGetIntTypeWidth(ty);
        let one = LLVMConstInt(ty, 1 as c_ulonglong, 0);
        let mut result = LLVMBuildSub(builder, value, one, c_str!(""));
        let mut shift_amount = 1;
        while shift_amount < bits {
            let amount = LLVMConstInt(ty, shift_amount as c_ulonglong, 0);
            let shift = LLVMBuildAShr(builder, result, amount, c_str!(""));
            result = LLVMBuildOr(builder, result, shift, c_str!(""));
            shift_amount *= 2;
        }
        LLVMBuildAdd(builder, result, one, c_str!(""))
    }

    /// Convert a boolean to an `i1`.
    ///
    /// If the boolean is a vector, a vector of `i1` is produced.
    unsafe fn bool_to_i1(&self, builder: LLVMBuilderRef, v: LLVMValueRef) -> LLVMValueRef {
        let type_kind = LLVMGetTypeKind(LLVMTypeOf(v));
        let mut zero = self.bool(false);
        if type_kind == llvm_sys::LLVMTypeKind::LLVMVectorTypeKind {
            let mut zeroes = [zero; LLVM_VECTOR_WIDTH as usize];
            zero = LLVMConstVector(zeroes.as_mut_ptr(), zeroes.len() as u32);
        }
        LLVMBuildICmp(builder, llvm_sys::LLVMIntPredicate::LLVMIntNE, v, zero, c_str!(""))
    }

    /// Convert an `i1` to a boolean.
    ///
    /// If the input is a vector, a vector of `boolean` is produced.
    unsafe fn i1_to_bool(&self, builder: LLVMBuilderRef, v: LLVMValueRef) -> LLVMValueRef {
        let type_kind = LLVMGetTypeKind(LLVMTypeOf(v));
        if type_kind == llvm_sys::LLVMTypeKind::LLVMVectorTypeKind {
            LLVMBuildZExt(builder, v, LLVMVectorType(self.bool_type(), LLVM_VECTOR_WIDTH), c_str!(""))
        } else {
            LLVMBuildZExt(builder, v, self.bool_type(), c_str!(""))
        }
    }

    /// Booleans are represented as `i8`.
    ///
    /// For instructions that require `i1` (e.g, conditional branching or select), the caller
    /// should truncate this type to `i1_type` manually. The distinction between booleans and `i1`
    /// is that boolean types are "externally visible", whereas `i1`s only appear in internal code.
    unsafe fn bool_type(&self) -> LLVMTypeRef {
        LLVMInt8TypeInContext(self.context())
    }

    unsafe fn i1_type(&self) -> LLVMTypeRef {
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

    unsafe fn void_pointer_type(&self) -> LLVMTypeRef {
        LLVMPointerType(self.i8_type(), 0)
    }

    unsafe fn run_handle_type(&self) -> LLVMTypeRef {
        let mut ty = LLVMGetTypeByName(self.module(), RUN_HANDLE_NAME.as_ptr());
        if ty.is_null() {
            let mut layout = [self.i8_type()];
            ty = LLVMStructCreateNamed(self.context(), RUN_HANDLE_NAME.as_ptr());
            LLVMStructSetBody(ty, layout.as_mut_ptr(), layout.len() as u32, 0);
        }
        LLVMPointerType(ty, 0)
    }

    unsafe fn bool(&self, v: bool) -> LLVMValueRef {
        LLVMConstInt(self.bool_type(), if v { 1 } else { 0 }, 0)
    }

    unsafe fn i1(&self, v: bool) -> LLVMValueRef {
        LLVMConstInt(self.i1_type(), if v { 1 } else { 0 }, 0)
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
    /// Initialize a new LlvmGenerator.
    unsafe fn new(conf: ParsedConf) -> WeldResult<LlvmGenerator> {
        let context = LLVMContextCreate();
        let module = LLVMModuleCreateWithNameInContext(c_str!("main"), context);

        // These methods *must* be called before using any of the `CodeGenExt` extension methods.
        jit::init();
        jit::set_triple_and_layout(module)?;

        // Adds the default intrinsic definitions.
        let intrinsics = intrinsic::Intrinsics::defaults(context, module);

        let target = target::Target::from_llvm_strings(
            llvm_exts::PROCESS_TRIPLE.to_str().unwrap(),
            llvm_exts::HOST_CPU_NAME.to_str().unwrap(),
            llvm_exts::HOST_CPU_FEATURES.to_str().unwrap()
            )?;

        debug!("LlvmGenerator features: {}", target.features);

        Ok(LlvmGenerator {
            conf: conf,
            context: context,
            module: module,
            target: target,
            functions: FnvHashMap::default(),
            vectors: FnvHashMap::default(),
            mergers: FnvHashMap::default(),
            appenders: FnvHashMap::default(),
            dictionaries: FnvHashMap::default(),
            strings: FnvHashMap::default(),
            eq_fns: FnvHashMap::default(),
            opaque_eq_fns: FnvHashMap::default(),
            cmp_fns: FnvHashMap::default(),
            hash_fns: FnvHashMap::default(),
            serialize_fns: FnvHashMap::default(),
            deserialize_fns: FnvHashMap::default(),
            struct_names: FnvHashMap::default(),
            struct_index: 0,
            intrinsics: intrinsics,
        })
    }

    /// Generate code for an SIR program.
    unsafe fn generate(conf: ParsedConf, program: &SirProgram) -> WeldResult<LlvmGenerator> {
        let mut gen = LlvmGenerator::new(conf)?;

        // Declare each function first to create a reference to it. Loop body functions are only
        // called by their ParallelForData terminators, so those are generated on-the-fly during
        // loop code generation.
        for func in program.funcs.iter().filter(|f| !f.loop_body) {
            gen.declare_sir_function(func)?;
        }

        // Generate each non-loop body function in turn. Loop body functions are constructed when
        // the For loop terminator is generated, with the loop control flow injected into the function.
        for func in program.funcs.iter().filter(|f| !f.loop_body) {
            gen.gen_sir_function(program, func)?;
        }

        // Generates a callable entry function in the module.
        gen.gen_entry(program)?;
        Ok(gen)
    }

    /// Generates a global string literal and returns a `i8*` to it.
    unsafe fn gen_global_string(&mut self, builder: LLVMBuilderRef, string: CString) -> LLVMValueRef {
        let ptr = string.as_ptr();
        self.strings.entry(string).or_insert_with(|| {
            LLVMBuildGlobalStringPtr(builder, ptr, c_str!(""))
        }).clone()
    }

    /// Generates a print call with the given string.
    unsafe fn gen_print(&mut self,
                        builder: LLVMBuilderRef,
                        run: LLVMValueRef,
                        string: CString) -> WeldResult<()> {
        let string = self.gen_global_string(builder, string);
        let pointer = LLVMConstBitCast(string, LLVMPointerType(self.i8_type(), 0));
        let _ = self.intrinsics.call_weld_run_print(builder, run, pointer);
        Ok(())
    }

    /// Generates the entry point to the Weld program.
    ///
    /// The entry function takes an `i64` and returns an `i64`. Both represent pointers that
    /// point to a `WeldInputArgs` and `WeldOutputArgs` respectively.
    unsafe fn gen_entry(&mut self, program: &SirProgram) -> WeldResult<()> {
        use ast::Type::Struct;

        let input_type = WeldInputArgs::llvm_type(self.context);
        let output_type = WeldOutputArgs::llvm_type(self.context);

        let name = CString::new("run").unwrap();
        let func_ty = LLVMFunctionType(self.i64_type(), [self.i64_type()].as_mut_ptr(), 1, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), func_ty);

        // Add the default attributes to all functions.
        llvm_exts::LLVMExtAddDefaultAttrs(self.context(), function);

        // This function is the global entry point into the program, so we must give it externally
        // visible linkage.
        LLVMSetLinkage(function, LLVMLinkage::LLVMExternalLinkage);

        let builder = LLVMCreateBuilderInContext(self.context);
        let entry_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!(""));
        let init_run_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!(""));
        let get_arg_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!(""));

        LLVMPositionBuilderAtEnd(builder, entry_block);
        let argument = LLVMGetParam(function, 0);
        let pointer = LLVMBuildIntToPtr(builder, argument, LLVMPointerType(input_type, 0), c_str!(""));

        // Check whether we already have an existing run.
        let run_pointer = LLVMBuildStructGEP(builder, pointer, WeldInputArgs::run_index(), c_str!(""));
        let run_pointer = self.load(builder, run_pointer)?;
        let run_arg = LLVMBuildIntToPtr(builder, run_pointer, self.run_handle_type(), c_str!(""));
        let null = LLVMConstNull(self.run_handle_type());
        let null_check = LLVMBuildICmp(builder, llvm_sys::LLVMIntPredicate::LLVMIntEQ, run_arg, null, c_str!(""));
        LLVMBuildCondBr(builder, null_check, init_run_block, get_arg_block);

        LLVMPositionBuilderAtEnd(builder, init_run_block);
        let nworkers_pointer = LLVMBuildStructGEP(builder, pointer, WeldInputArgs::nworkers_index(), c_str!("nworkers"));
        let nworkers = self.load(builder, nworkers_pointer)?;
        let memlimit_pointer = LLVMBuildStructGEP(builder, pointer, WeldInputArgs::memlimit_index(), c_str!("memlimit"));
        let memlimit = self.load(builder, memlimit_pointer)?;
        let run_new = self.intrinsics.call_weld_run_init(builder, nworkers, memlimit, None);
        LLVMBuildBr(builder, get_arg_block);

        LLVMPositionBuilderAtEnd(builder, get_arg_block);
        let run = LLVMBuildPhi(builder, self.run_handle_type(), c_str!(""));
        let mut blocks = [entry_block, init_run_block];
        let mut values = [run_arg, run_new];
        LLVMAddIncoming(run, values.as_mut_ptr(), blocks.as_mut_ptr(), blocks.len() as u32);

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
        let inst = LLVMBuildCall(builder, entry_function,
                                 func_args.as_mut_ptr(), func_args.len() as u32, c_str!(""));
        LLVMSetInstructionCallConv(inst, SIR_FUNC_CALL_CONV);

        let result = self.intrinsics.call_weld_run_get_result(builder, run, None);
        let result = LLVMBuildPtrToInt(builder, result, self.i64_type(), c_str!("result"));
        let errno = self.intrinsics.call_weld_run_get_errno(builder, run, Some(c_str!("errno")));
        let run_int = LLVMBuildPtrToInt(builder, run, self.i64_type(), c_str!("run"));

        let mut output = LLVMGetUndef(output_type);
        output = LLVMBuildInsertValue(builder, output, result, WeldOutputArgs::output_index(), c_str!(""));
        output = LLVMBuildInsertValue(builder, output, run_int, WeldOutputArgs::run_index(), c_str!(""));
        output = LLVMBuildInsertValue(builder, output, errno, WeldOutputArgs::errno_index(), c_str!(""));

        let return_size = self.size_of(output_type);
        let return_pointer = self.intrinsics.call_weld_run_malloc(builder, run, return_size, None);
        let return_pointer = LLVMBuildBitCast(builder, return_pointer, LLVMPointerType(output_type, 0), c_str!(""));

        LLVMBuildStore(builder, output, return_pointer);
        let return_value  = LLVMBuildPtrToInt(builder, return_pointer, self.i64_type(), c_str!(""));
        LLVMBuildRet(builder, return_value);

        LLVMDisposeBuilder(builder);
        Ok(())
    }

    /// Build the list of argument types for an SIR function.
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

        // Add attributes, set linkage, etc.
        llvm_exts::LLVMExtAddDefaultAttrs(self.context(), function);
        LLVMSetLinkage(function, LLVMLinkage::LLVMPrivateLinkage);
        LLVMSetFunctionCallConv(function, SIR_FUNC_CALL_CONV);

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

    /// Generate code for a single SIR statement.
    ///
    /// The code is generated at the position specified by the function context.
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

                // A CUDF with declaration Name[R](T1, T2, T3) has a signature `void Name(T1, T2, T3, R)`.
                for arg in args.iter() {
                    arg_tys.push(LLVMPointerType(self.llvm_type(context.sir_function.symbol_type(arg)?)?, 0));
                }
                arg_tys.push(LLVMPointerType(return_ty, 0));

                let fn_ret_ty = self.void_type();
                self.intrinsics.add(symbol_name, fn_ret_ty, &mut arg_tys);

                let mut arg_values = vec![];
                for arg in args.iter() {
                    arg_values.push(context.get_value(arg)?);
                }

                arg_values.push(output_pointer);
                let _ = self.intrinsics.call(context.builder, symbol_name, &mut arg_values)?;

                Ok(())
            }
            Deserialize(_) => {
                use self::serde::SerDeGen;
                self.gen_deserialize(context, statement)
            }
            GetField { ref value, index } => {
                let output_pointer = context.get_value(output)?;
                let value_pointer = context.get_value(value)?;
                let elem_pointer = LLVMBuildStructGEP(context.builder, value_pointer, index, c_str!(""));
                let elem = self.load(context.builder, elem_pointer)?;
                LLVMBuildStore(context.builder, elem, output_pointer);
                Ok(())
            }
            KeyExists { ref child, ref key } => {
                use self::hash::GenHash;
                let output_pointer = context.get_value(output)?;
                let child_value = self.load(context.builder, context.get_value(child)?)?;
                let key_pointer = context.get_value(key)?;
                let child_type = context.sir_function.symbol_type(child)?;
                let hash = if let Dict(ref key, _) = *child_type {
                    self.gen_hash(key, context.builder, key_pointer, None)?
                } else {
                    unreachable!()
                };

                let result = {
                    let mut methods = self.dictionaries.get_mut(child_type).unwrap();
                    methods.gen_key_exists(context.builder,
                                           child_value,
                                           context.get_value(key)?,
                                           hash)?
                };
                LLVMBuildStore(context.builder, result, output_pointer);
                Ok(())
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
                } else if let Dict(_, _) = *child_type {
                    let pointer = {
                        let mut methods = self.dictionaries.get_mut(child_type).unwrap();
                        methods.gen_size(context.builder, child_value)?
                    };
                    let result = self.load(context.builder, pointer)?;
                    LLVMBuildStore(context.builder, result, output_pointer);
                    Ok(())
                } else {
                    unreachable!()
                }
            }
            Lookup { ref child, ref index } => {
                let output_pointer = context.get_value(output)?;
                let child_value = self.load(context.builder, context.get_value(child)?)?;
                let child_type = context.sir_function.symbol_type(child)?;
                if let Vector(_) = *child_type {
                    use self::vector::VectorExt;
                    let index_value = self.load(context.builder, context.get_value(index)?)?;
                    let pointer = self.gen_at(context.builder, child_type, child_value, index_value)?;
                    let result = self.load(context.builder, pointer)?;
                    LLVMBuildStore(context.builder, result, output_pointer);
                    Ok(())
                } else if let Dict(ref key, _) = *child_type {
                    use self::hash::GenHash;
                    let hash = self.gen_hash(key, context.builder, context.get_value(index)?, None)?;
                    let result = {
                        let mut methods = self.dictionaries.get_mut(child_type).unwrap();
                        let slot = methods.gen_lookup(context.builder,
                                                      &mut self.intrinsics,
                                                      child_value,
                                                      context.get_value(index)?,
                                                      hash,
                                                      context.get_run())?;
                        let value_pointer = methods.slot_ty.value(context.builder, slot);
                        LLVMBuildLoad(context.builder, value_pointer, c_str!(""))
                    };
                    LLVMBuildStore(context.builder, result, output_pointer);
                    Ok(())
                } else {
                    unreachable!()
                }
            }
            OptLookup { ref child, ref index } => {
                let output_pointer = context.get_value(output)?;
                let child_value = self.load(context.builder, context.get_value(child)?)?;
                let child_type = context.sir_function.symbol_type(child)?;
                if let Dict(ref key, _) = *child_type {
                    use self::hash::GenHash;
                    let hash = self.gen_hash(key, context.builder, context.get_value(index)?, None)?;
                    let (filled, value) = {
                        let mut methods = self.dictionaries.get_mut(child_type).unwrap();
                        let slot = methods.gen_opt_lookup(context.builder,
                                                      child_value,
                                                      context.get_value(index)?,
                                                      hash)?;
                        let filled = methods.slot_ty.filled(context.builder, slot);
                        let value_pointer = methods.slot_ty.value(context.builder, slot);
                        // NOTE: This could be an invalid (zeroed value) -- code should check the
                        // boolean.
                        let loaded_value = LLVMBuildLoad(context.builder, value_pointer, c_str!(""));

                        (filled, loaded_value)
                    };

                    let filled = self.i1_to_bool(context.builder, filled);

                    let filled_output_pointer = LLVMBuildStructGEP(context.builder, output_pointer, 0, c_str!(""));
                    LLVMBuildStore(context.builder, filled, filled_output_pointer);
                    let value_output_pointer = LLVMBuildStructGEP(context.builder, output_pointer, 1, c_str!(""));
                    LLVMBuildStore(context.builder, value, value_output_pointer);
                    Ok(())
                } else {
                    unreachable!()
                }
            }
            MakeStruct(ref elems) => {
                let output_pointer = context.get_value(output)?;
                for (i, elem) in elems.iter().enumerate() {
                    let elem_pointer = LLVMBuildStructGEP(context.builder,
                                                          output_pointer,
                                                          i as u32,
                                                          c_str!(""));
                    let value = self.load(context.builder, context.get_value(elem)?)?;
                    LLVMBuildStore(context.builder, value, elem_pointer);
                }
                Ok(())
            }
            MakeVector(ref elems) => {
                use self::vector::VectorExt;
                let output_pointer = context.get_value(output)?;
                let output_type = context.sir_function.symbol_type(output)?;
                let size = self.i64(elems.len() as i64);
                let vector = self.gen_new(context.builder, output_type, size, context.get_run())?;
                for (i, elem) in elems.iter().enumerate() {
                    let index = self.i64(i as i64);
                    let vec_pointer = self.gen_at(context.builder, output_type, vector, index)?;
                    let loaded = self.load(context.builder, context.get_value(elem)?)?;
                    LLVMBuildStore(context.builder, loaded, vec_pointer);
                }
                LLVMBuildStore(context.builder, vector, output_pointer);
                Ok(())
            }
            Merge { .. } => {
                use self::builder::BuilderExpressionGen;
                self.gen_merge(context, statement)
            }
            Negate(_) => {
                use self::numeric::NumericExpressionGen;
                self.gen_negate(context, statement)
            }
            Not(_) => {
                use self::numeric::NumericExpressionGen;
                self.gen_not(context, statement)
            }
            NewBuilder { .. } => {
                use self::builder::BuilderExpressionGen;
                self.gen_new_builder(context, statement)
            }
            ParallelFor(_) => {
                use self::builder::BuilderExpressionGen;
                self.gen_for(context, statement)
            }
            Res(_) => {
                use self::builder::BuilderExpressionGen;
                self.gen_result(context, statement)
            }
            Select { ref cond, ref on_true, ref on_false } => {
                let output_pointer = context.get_value(output)?;
                let cond = self.load(context.builder, context.get_value(cond)?)?;
                let cond = self.bool_to_i1(context.builder, cond);
                let on_true = self.load(context.builder, context.get_value(on_true)?)?;
                let on_false = self.load(context.builder, context.get_value(on_false)?)?;
                let result = LLVMBuildSelect(context.builder, cond, on_true, on_false, c_str!(""));
                LLVMBuildStore(context.builder, result, output_pointer);
                Ok(())
            }
            Serialize(_) => {
                use self::serde::SerDeGen;
                self.gen_serialize(context, statement)
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
            Sort { ref child, ref cmpfunc } => {

                let output_pointer = context.get_value(output)?;
                let output_type = context.sir_function.symbol_type(
                    statement.output.as_ref().unwrap())?;

                if let Vector(ref elem_ty) = *output_type {
                    use self::vector::VectorExt;

                    let child_value = self.load(context.builder, context.get_value(child)?)?;

                    // Sort clones the vector at the moment.
                    let output_value = self.gen_clone(context.builder, output_type, child_value, context.get_run())?;

                    let zero = self.zero(self.i64_type());
                    let elems = self.gen_at(context.builder, output_type, output_value, zero)?;
                    let elems_ptr = LLVMBuildBitCast(context.builder, elems,
                                                     self.void_pointer_type(),
                                                     c_str!(""));
                    let size = self.gen_size(context.builder, output_type, output_value)?;
                    let elem_ll_ty = self.llvm_type(elem_ty)?;
                    let ty_size = self.size_of(elem_ll_ty);

                    use self::cmp::GenCmp;
                    let cmpfunc_ll_fn = self.functions[cmpfunc];

                    let run = context.get_run();

                    // Generate the comparator from the provided custom code.
                    let comparator = self.gen_custom_cmp(elem_ll_ty,
                                                         *cmpfunc,
                                                         cmpfunc_ll_fn)?;

                    // args to qsort_r are: base array pointer, num elements,
                    // element size, comparator function, run handle.
                    //
                    // MacOS and Linux pass arguments to qsort_r in different order.
                    let (mut args, mut arg_tys) = if cfg!(target_os = "macos") {
                        let mut args = vec![elems_ptr, size, ty_size, run, comparator];
                        let mut arg_tys = vec![
                            LLVMTypeOf(elems_ptr),
                            LLVMTypeOf(size),
                            LLVMTypeOf(ty_size),
                            LLVMTypeOf(run),
                            LLVMTypeOf(comparator)
                        ];
                        (args, arg_tys)
                    } else if cfg!(target_os = "linux") {
                        let mut args = vec![elems_ptr, size, ty_size, comparator, run];
                        let mut arg_tys = vec![
                            LLVMTypeOf(elems_ptr),
                            LLVMTypeOf(size),
                            LLVMTypeOf(ty_size),
                            LLVMTypeOf(comparator),
                            LLVMTypeOf(run)
                        ];
                        (args, arg_tys)
                    } else {
                        unimplemented!("Sort not available on this platform.");
                    };

                    // Generate the call to qsort.
                    let void_type = self.void_type();
                    self.intrinsics.add("qsort_r", void_type, &mut arg_tys);
                    self.intrinsics.call(context.builder, "qsort_r", &mut args)?;

                    LLVMBuildStore(context.builder, output_value, output_pointer);

                    Ok(())
                } else {
                    unreachable!()
                }
            }
            ToVec(ref child) => {
                let output_pointer = context.get_value(output)?;
                let child_value = self.load(context.builder, context.get_value(child)?)?;
                let child_type = context.sir_function.symbol_type(child)?;
                // This is the type of the resulting key/value vector (vec[{K,V}])
                let output_type = context.sir_function.symbol_type(
                    statement.output.as_ref().unwrap())?;
                let elem = if let Vector(ref elem) = *output_type {
                    elem
                } else {
                    unreachable!()
                };

                let _ = self.llvm_type(output_type)?;
                let result = {
                    let mut vector_methods = self.vectors.get_mut(elem).unwrap();
                    let mut methods = self.dictionaries.get_mut(child_type).unwrap();
                    methods.gen_to_vec(context.builder,
                                       &mut self.intrinsics,
                                       vector_methods,
                                       child_value,
                                       context.get_run())?
                };
                LLVMBuildStore(context.builder, result, output_pointer);
                Ok(())
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
    /// returning from a function. The second argument is a value representing the loop's builder
    /// argument pointer. In cases where the loop body function returns, this function stores the
    /// resulting builder into this pointer. The function is guaranteed to return a builder (since
    /// For loop functions must return builders derived from their input builder).
    ///
    /// This function does not make any assumptions about which *LLVM basic block* the
    /// builder is positioned in, as long as the builder is logically within the passed SIR basic
    /// block.
    unsafe fn gen_terminator(&mut self,
                                  context: &mut FunctionContext,
                                  bb: &BasicBlock,
                                  loop_terminator: Option<(LLVMBasicBlockRef, LLVMValueRef)>) -> WeldResult<()> {

        if self.conf.trace_run {
            self.gen_print(context.builder,
                           context.get_run(),
                           CString::new(format!("{}", bb.terminator)).unwrap())?;
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
                let cond = self.bool_to_i1(context.builder, cond);
                let _ = LLVMBuildCondBr(context.builder,
                                        cond,
                                        context.get_block(&on_true)?,
                                        context.get_block(&on_false)?);
            }
            JumpBlock(ref id) => {
                LLVMBuildBr(context.builder, context.get_block(id)?);
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

    /// Returns the LLVM type for a Weld Type.
    ///
    /// This method may generate auxillary code before returning the type. For example, for complex
    /// data structures, this function may generate a definition for the data structure first.
    unsafe fn llvm_type(&mut self, ty: &Type) -> WeldResult<LLVMTypeRef> {
        use ast::Type::*;
        use ast::ScalarKind::*;
        let result = match *ty {
            Builder(_, _) => {
                use self::builder::BuilderExpressionGen;
                self.builder_type(ty)?
            }
            Dict(ref key, ref value) => {
                use self::eq::GenEq;
                if !self.dictionaries.contains_key(ty) {
                    let key_ty = self.llvm_type(key)?;
                    let value_ty = self.llvm_type(value)?;
                    let key_comparator = self.gen_eq_fn(key)?;
                    let dict = dict::Dict::define("dict",
                                                  key_ty,
                                                  key_comparator,
                                                  value_ty,
                                                  self.context,
                                                  self.module);
                    self.dictionaries.insert(ty.clone(), dict);
                }
                self.dictionaries.get(ty).unwrap().dict_ty
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
                if !self.struct_names.contains_key(ty) {
                    let name = CString::new(format!("s{}", self.struct_index)).unwrap();
                    self.struct_index += 1;
                    let mut llvm_types: Vec<_> = elems.iter()
                        .map(&mut |t| self.llvm_type(t)).collect::<WeldResult<_>>()?;
                    let struct_ty = LLVMStructCreateNamed(self.context, name.as_ptr());
                    LLVMStructSetBody(struct_ty, llvm_types.as_mut_ptr(), llvm_types.len() as u32, 0);
                    self.struct_names.insert(ty.clone(), name);
                }
                LLVMGetTypeByName(self.module, self.struct_names.get(ty).cloned().unwrap().as_ptr())
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

    unsafe fn size_of_ty(&mut self, ty: &Type) -> usize {
        let ll_ty = self.llvm_type(ty).unwrap();
        (self.size_of_bits(ll_ty) / 8) as usize
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

/// A context for generating code in an SIR function.
///
/// The function context holds the position at which new code should be generated.
pub struct FunctionContext<'a> {
    /// A reference to the SIR program.
    sir_program: &'a SirProgram,
    /// The SIR function to which this context belongs.
    ///
    /// Equivalently, this context represents code generation state for this function.
    sir_function: &'a SirFunction,
    /// An LLVM reference to this function.
    llvm_function: LLVMValueRef,
    /// The LLVM values for symbols defined in this function.
    ///
    /// These symbols are the ones defined in the SIR (i.e., locals and parameters). The symbols
    /// values are thus all alloca'd pointers.
    symbols: FnvHashMap<Symbol, LLVMValueRef>,
    /// A mapping from SIR basic blocks to LLVM basic blocks.
    blocks: FnvHashMap<BasicBlockId, LLVMBasicBlockRef>,
    /// The LLVM builder, which marks where to insert new code.
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

    /// Returns the LLVM value for a symbol in this function.
    pub fn get_value(&self, sym: &Symbol) -> WeldResult<LLVMValueRef> {
        self.symbols
            .get(sym)
            .cloned()
            .ok_or(WeldCompileError::new(format!("Undefined symbol {} in function codegen", sym)))
    }

    /// Returns the LLVM basic block for a basic block ID in this function.
    pub fn get_block(&self, id: &BasicBlockId) -> WeldResult<LLVMBasicBlockRef> {
        self.blocks.get(id)
            .cloned()
            .ok_or(WeldCompileError::new("Undefined basic block in function codegen"))
    }

    /// Get the handle to the run.
    ///
    /// The run handle is always the last argument of an SIR function.
    pub fn get_run(&self) -> LLVMValueRef {
        unsafe { LLVMGetLastParam(self.llvm_function) }
    }
}

impl<'a> Drop for FunctionContext<'a> {
    fn drop(&mut self) {
        unsafe { LLVMDisposeBuilder(self.builder); }
    }
}
