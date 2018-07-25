//! Defines the builder structures and methods over them.
//!
//! Specifically, this module provides an extension trait that generates code for:
//!
//! * `NewBuilder`
//! * `Merge`
//! * `Res`
//!
//! The module additionally encapsulates the builder data structures.

extern crate llvm_sys;

use annotation::Annotations;

use ast::*;
use ast::BuilderKind::*;
use ast::Type::*;

use error::*;

use sir::*;
use sir::StatementKind::*;

use self::llvm_sys::prelude::*;
use self::llvm_sys::core::*;

use super::{CodeGenExt, FunctionContext, LlvmGenerator};

use super::hash;
use super::numeric;

mod for_loop;

pub mod appender;
pub mod dictmerger;
pub mod merger;
pub mod vecmerger;

/// A trait for generating builder code.
///
/// This trait primarily unwraps the builder kind in the statement and muxes functionality to the
/// various builder implementations.
pub trait BuilderExpressionGen {
    /// Generates code for the `NewBuilder` statement.
    unsafe fn gen_new_builder(&mut self, ctx: &mut FunctionContext, statement: &Statement) -> WeldResult<()>;
    /// Generates code for the `Merge` statement.
    unsafe fn gen_merge(&mut self, ctx: &mut FunctionContext, statement: &Statement) -> WeldResult<()>;
    /// Generates code for the `Result` statement.
    unsafe fn gen_result(&mut self, ctx: &mut FunctionContext, statement: &Statement) -> WeldResult<()>;
    /// Generates code for the `ParallelFor` terminator.
    unsafe fn gen_for(&mut self, ctx: &mut FunctionContext, parfor: &ParallelForData) -> WeldResult<LLVMValueRef>;
    /// Generates code to define builder types.
    unsafe fn builder_type(&mut self, builder: &Type) -> WeldResult<LLVMTypeRef>;
}

/// Encapsulates the fields of a `NewBuilder` statement.
struct NewBuilderStatement<'a> {
    output: &'a Symbol,
    arg: Option<&'a Symbol>,
    ty: &'a Type,
    kind: &'a BuilderKind,
    annotations: &'a Annotations,
}

impl<'a> NewBuilderStatement<'a> {
    fn extract(statement: &'a Statement, func: &'a SirFunction) -> WeldResult<NewBuilderStatement<'a>> {
        if statement.output.is_none() {
            unreachable!()
        }
        if let NewBuilder { ref arg, .. } = statement.kind {
            let output = statement.output.as_ref().unwrap();
            let builder_type = func.symbol_type(output)?;
            if let Builder(ref kind, ref annotations) = *builder_type {
                let result = NewBuilderStatement {
                    output: output,
                    arg: arg.as_ref(),
                    ty: builder_type,
                    kind: kind,
                    annotations: annotations,
                };
                return Ok(result);
            }
        }
        unreachable!()
    }
}

/// Encapsulates the fields of a `Merge` statement.
struct MergeStatement<'a> {
    builder: &'a Symbol,
    value: &'a Symbol,
    kind: &'a BuilderKind,
    annotations: &'a Annotations,
}

impl<'a> MergeStatement<'a> {
    fn extract(statement: &'a Statement, func: &'a SirFunction) -> WeldResult<MergeStatement<'a>> {
        if let Merge { ref builder, ref value } = statement.kind {
            let builder_type = func.symbol_type(builder)?;
            if let Builder(ref kind, ref annotations) = *builder_type {
                let result = MergeStatement {
                    builder: builder,
                    value: value,
                    kind: kind,
                    annotations: annotations,
                };
                return Ok(result);
            }
        }
        unreachable!()
    }
}

/// Encapsulates the fields of a `Res` statement.
struct ResStatement<'a> {
    output: &'a Symbol,
    builder: &'a Symbol,
    kind: &'a BuilderKind,
    annotations: &'a Annotations,
}

impl<'a> ResStatement<'a> {
    fn extract(statement: &'a Statement, func: &'a SirFunction) -> WeldResult<ResStatement<'a>> {
        if statement.output.is_none() {
            unreachable!()
        }
        if let Res(ref builder) = statement.kind {
            let builder_type = func.symbol_type(builder)?;
            if let Builder(ref kind, ref annotations) = *builder_type {
                let result = ResStatement {
                    output: statement.output.as_ref().unwrap(),
                    builder: builder,
                    kind: kind,
                    annotations: annotations,
                };
                return Ok(result);
            }
        }
        unreachable!()
    }
}

impl BuilderExpressionGen for LlvmGenerator {
    unsafe fn gen_new_builder(&mut self, ctx: &mut FunctionContext, statement: &Statement) -> WeldResult<()> {
        let nb = NewBuilderStatement::extract(statement, ctx.sir_function)?;
        let output_pointer = ctx.get_value(nb.output)?;
        match *nb.kind {
            Appender(_) => {
                // The argument is either the provided one or the default capacity.
                let argument = if let Some(arg) = nb.arg {
                    self.load(ctx.builder, ctx.get_value(arg)?)?
                } else {
                    self.i64(appender::DEFAULT_CAPACITY)
                };
                let appender = {
                    let mut methods = self.appenders.get_mut(nb.kind).unwrap();
                    let run = ctx.get_run();
                    methods.gen_new(ctx.builder, &mut self.intrinsics, run, argument)?
                };
                LLVMBuildStore(ctx.builder, appender, output_pointer);
                Ok(())
            }
            Merger(_, _) => {
                // The argument is either the provided one or the identity.
                let argument = if let Some(arg) = nb.arg {
                    self.load(ctx.builder, ctx.get_value(arg)?)?
                } else {
                    let mut methods = self.mergers.get_mut(nb.kind).unwrap();
                    methods.binop_identity(methods.op, methods.scalar_kind)?
                };
                let merger = {
                    let mut methods = self.mergers.get_mut(nb.kind).unwrap();
                    methods.gen_new(ctx.builder, argument)?
                };
                LLVMBuildStore(ctx.builder, merger, output_pointer);
                Ok(())
            }
            DictMerger(ref key, ref val, _) => {
                let ref dict_type = Dict(key.clone(), val.clone());
                let default_capacity = self.i64(dictmerger::DEFAULT_CAPACITY);
                let dictmerger = {
                    let mut methods = self.dictionaries.get_mut(dict_type).unwrap();
                    methods.gen_new(ctx.builder, &self.dict_intrinsics, ctx.get_run(), default_capacity)?
                };
                LLVMBuildStore(ctx.builder, dictmerger, output_pointer);
                Ok(())
            }
            _ => {
                unimplemented!()
            }
        }
    }

    unsafe fn gen_merge(&mut self, ctx: &mut FunctionContext, statement: &Statement) -> WeldResult<()> {
        let m = MergeStatement::extract(statement, ctx.sir_function)?;
        let builder_pointer = ctx.get_value(m.builder)?;
        match *m.kind {
            Appender(_) => {
                let merge_value = self.load(ctx.builder, ctx.get_value(m.value)?)?;
                let mut methods = self.appenders.get_mut(m.kind).unwrap();
                let _ = methods.gen_merge(ctx.builder, &mut self.intrinsics, ctx.get_run(), builder_pointer, merge_value)?;
                Ok(())
            }
            Merger(_, _) => {
                let merge_value = self.load(ctx.builder, ctx.get_value(m.value)?)?;
                let mut methods = self.mergers.get_mut(m.kind).unwrap();
                let _ = methods.gen_merge(ctx.builder, builder_pointer, merge_value)?;
                Ok(())
            }
            DictMerger(ref key, ref val, ref binop) => {
                use ast::Type::Scalar;
                use self::hash::GenHash;

                // Build the default value that we upsert if the key is not present in the
                // dictionary yet.
                let default = match *val.as_ref() {
                    Scalar(ref kind) => self.binop_identity(*binop, *kind)?,
                    Struct(ref elems) => {
                        let mut default = LLVMGetUndef(self.llvm_type(val)?);
                        for (i, elem) in elems.iter().enumerate() {
                            if let Scalar(ref kind) = *elem {
                                let mut indices = [i as u32];
                                default = LLVMConstInsertValue(default,
                                                               self.binop_identity(*binop, *kind)?,
                                                               indices.as_mut_ptr(),
                                                               indices.len() as u32);
                            } else {
                                unreachable!()
                            }
                        }
                        default
                    }
                    _ => unreachable!()
                };

                // The type of the merge value is {key, value} so use GEP to extract
                // the key and the key pointer.
                let merge_type = ctx.sir_function.symbol_type(m.value)?;
                let (key_pointer, value_pointer) = match *merge_type {
                    // We need this to make sure LLVM doesn't freak out when we use GEP on a
                    // non-struct type or access something out-of-bounds.
                    Struct(ref elems) if elems.len() == 2 => {
                        let merge_value = ctx.get_value(m.value)?;
                        let key_pointer = LLVMBuildStructGEP(ctx.builder, merge_value, 0, c_str!(""));
                        let val_pointer = LLVMBuildStructGEP(ctx.builder, merge_value, 1, c_str!(""));
                        (key_pointer, val_pointer)
                    }
                    _ => unreachable!()
                };

                let hash_fn = self.gen_hash_fn(key)?;
                let mut args = [self.load(ctx.builder, key_pointer)?];
                let hash = LLVMBuildCall(ctx.builder, hash_fn, args.as_mut_ptr(), args.len() as u32, c_str!(""));

                let builder_loaded = self.load(ctx.builder, builder_pointer)?;

                let ref dict_type = Dict(key.clone(), val.clone());
                let slot_value_pointer = {
                    let mut methods = self.dictionaries.get_mut(dict_type).unwrap();
                    methods.gen_upsert(ctx.builder,
                                       &self.dict_intrinsics,
                                       ctx.get_run(),
                                       builder_loaded,
                                       key_pointer,
                                       hash,
                                       default)?
                };

                // Generate the merge code. We either load the values and add them, or, if the
                // values are structs, we load each element at a time and apply the binop.
                match *val.as_ref() {
                    Scalar(_) => {
                        let merge_value = self.load(ctx.builder, value_pointer)?;
                        let slot_value = self.load(ctx.builder, slot_value_pointer)?;
                        let merged = numeric::gen_binop(ctx.builder, *binop, merge_value, slot_value, val)?;
                        LLVMBuildStore(ctx.builder, merged, slot_value_pointer);
                    }
                    Struct(ref elems) => {
                        for (i, elem) in elems.iter().enumerate() {
                            let merge_elem_pointer = LLVMBuildStructGEP(ctx.builder,
                                                                        value_pointer,
                                                                        i as u32,
                                                                        c_str!(""));
                            let slot_elem_pointer = LLVMBuildStructGEP(ctx.builder,
                                                                       slot_value_pointer,
                                                                       i as u32,
                                                                       c_str!(""));
                            let merge_value = self.load(ctx.builder, merge_elem_pointer)?;
                            let slot_value = self.load(ctx.builder, slot_elem_pointer)?;
                            let merged = numeric::gen_binop(ctx.builder, *binop, merge_value, slot_value, elem)?;
                            LLVMBuildStore(ctx.builder, merged, slot_elem_pointer);
                        }
                    }
                    _ => unreachable!()
                };
                Ok(())
            }
            _ => unimplemented!()
        }
    }

    unsafe fn gen_result(&mut self, ctx: &mut FunctionContext, statement: &Statement) -> WeldResult<()> {
        let m = ResStatement::extract(statement, ctx.sir_function)?;
        let output_pointer = ctx.get_value(m.output)?;
        let builder_pointer = ctx.get_value(m.builder)?;
        match *m.kind {
            Appender(ref elem_type) => {
                let ref vector = Vector(elem_type.clone());
                let vector_type = self.llvm_type(vector)?;
                let result = {
                    let mut methods = self.appenders.get_mut(m.kind).unwrap();
                    methods.gen_result(ctx.builder, vector_type, builder_pointer)?
                };
                LLVMBuildStore(ctx.builder, result, output_pointer);
                Ok(())
            }
            Merger(_, _) => {
                let result = {
                    let mut methods = self.mergers.get_mut(m.kind).unwrap();
                    methods.gen_result(ctx.builder, builder_pointer)?
                };
                LLVMBuildStore(ctx.builder, result, output_pointer);
                Ok(())
            }
            DictMerger(_, _, _) => {
                // A dictmerger just updates a dictionary in-place, so return the produced
                // dictionary.
                let builder_loaded = self.load(ctx.builder, builder_pointer)?;
                LLVMBuildStore(ctx.builder, builder_loaded, output_pointer);
                Ok(())
            }
            _ => {
                unimplemented!()
            }
        }
    }

    unsafe fn gen_for(&mut self,
                      ctx: &mut FunctionContext,
                      parfor: &ParallelForData) -> WeldResult<LLVMValueRef> {
        use self::for_loop::ForLoopGenInternal;
        self.gen_for_internal(ctx, parfor)
    }

    unsafe fn builder_type(&mut self, builder: &Type) -> WeldResult<LLVMTypeRef> {
        if let Builder(ref kind, _) = *builder {
            match *kind {
                Merger(ref elem_type, ref binop) => {
                    if !self.mergers.contains_key(kind) {
                        let scalar_kind = if let Scalar(ref kind) = *elem_type.as_ref() {
                            *kind
                        } else {
                            unreachable!()
                        };
                        let llvm_elem_type = self.llvm_type(elem_type)?;
                        let merger = merger::Merger::define("merger",
                                                            *binop,
                                                            llvm_elem_type,
                                                            scalar_kind,
                                                            self.context,
                                                            self.module);
                        self.mergers.insert(kind.clone(), merger);
                    }
                    Ok(self.mergers.get(kind).unwrap().merger_ty)
                }
                Appender(ref elem_type) => {
                    if !self.appenders.contains_key(kind) {
                        let llvm_elem_type = self.llvm_type(elem_type)?;
                        let appender = appender::Appender::define("appender",
                                                            llvm_elem_type,
                                                            self.context,
                                                            self.module);
                        self.appenders.insert(kind.clone(), appender);
                    }
                    Ok(self.appenders.get(kind).unwrap().appender_ty)
                }
                DictMerger(ref key, ref val, _) => {
                    let ref dict_type = Dict(key.clone(), val.clone());
                    self.llvm_type(dict_type)
                }
                _ => unimplemented!()
            }
        } else {
            unreachable!()
        }
    }
}
