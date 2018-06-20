//! Code generation for the parallel for loop.


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

/*
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ParallelForData {
    pub data: Vec<ParallelForIter>,
    pub builder: Symbol,
    pub data_arg: Symbol,
    pub builder_arg: Symbol,
    pub idx_arg: Symbol,
    pub body: FunctionId,
    pub cont: FunctionId,
    pub innermost: bool,
    /// If `true`, always invoke parallel runtime for the loop.
    pub always_use_runtime: bool,
    pub grain_size: Option<i32>
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ParallelForIter {
    pub data: Symbol,
    pub start: Option<Symbol>,
    pub end: Option<Symbol>,
    pub stride: Option<Symbol>,
    pub kind: IterKind,
    // NdIter specific fields
    pub strides: Option<Symbol>,
    pub shape: Option<Symbol>,
}
*/

/// An internal trait for generating parallel code.
pub trait ForLoopGenInternal {
    /// Entry point to generating a for loop.
    ///
    /// This is the only function in the trait that should be called -- all other methods are
    /// helpers. 
    unsafe fn gen_for_internal(&mut self, ctx: &mut FunctionContext, parfor: &ParallelForData) -> WeldResult<()>; 
    /// Generates a bounds check for the given iterator.
    unsafe fn gen_bounds_check(&mut self, ctx: &mut FunctionContext, iterator: &ParallelForIter) -> WeldResult<()>;
}

impl ForLoopGenInternal for LlvmGenerator {
    unsafe fn gen_for_internal(&mut self, ctx: &mut FunctionContext, parfor: &ParallelForData) -> WeldResult<()> {
        Ok(())
    }

    unsafe fn gen_bounds_check(&mut self, ctx: &mut FunctionContext, iterator: &ParallelForIter) -> WeldResult<()> {
        Ok(())
    }

    unsafe fn gen_loop_body_function(&mut self, ctx: &mut FunctionContext, iterator: &ParallelForIter) -> WeldResult<()> {
        Ok(())
    }
}
