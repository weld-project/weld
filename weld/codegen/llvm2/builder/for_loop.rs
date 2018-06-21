//! Code generation for the parallel for loop.


extern crate llvm_sys;

use std::ffi::CString;

use ast::*;
use ast::IterKind::*;
use ast::Type::*;
use error::*;

use sir::*;

use self::llvm_sys::{LLVMIntPredicate, LLVMLinkage};
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
    unsafe fn gen_for_internal(&mut self,
                               ctx: &mut FunctionContext,
                               parfor: &ParallelForData) -> WeldResult<()>; 
    /// Generates the loop body.
    ///
    /// This generates both the loop control flow and the executing body of the loop.
    unsafe fn gen_loop_body_function(&mut self,
                                     program: &SirProgram,
                                     func: &SirFunction,
                                     parfor: &ParallelForData) -> WeldResult<()>;
    /// Generates a bounds check for the given iterator.
    ///
    /// Returns a value representing the number of iterations the iterator will produce.
    unsafe fn gen_bounds_check(&mut self,
                               ctx: &mut FunctionContext,
                               iterator: &ParallelForIter) -> WeldResult<LLVMValueRef>;
    /// Generates code to check whether the number of iterations in each value is the same.
    unsafe fn gen_check_equal(&mut self,
                              ctx: &mut FunctionContext,
                              iterations: &Vec<LLVMValueRef>) -> WeldResult<()>;
    /// Generates code to load potentially zipped elements at index `i` into `e`.
    /// 
    /// `e` must be a pointer, and `i` must be a loaded index argument of type `i64`.
    unsafe fn gen_loop_element(&mut self,
                                    _ctx: &mut FunctionContext, 
                                    i: LLVMValueRef,
                                    e: LLVMValueRef,
                                    parfor: &ParallelForData) -> WeldResult<()>;
}

impl ForLoopGenInternal for LlvmGenerator {
    unsafe fn gen_for_internal(&mut self, ctx: &mut FunctionContext, parfor: &ParallelForData) -> WeldResult<()> {
        let ref mut num_iterations = vec![];
        for iter in parfor.data.iter() {
            let iterations = self.gen_bounds_check(ctx, iter)?;
            num_iterations.push(iterations);
        }

        assert!(num_iterations.len() > 0);

        // Make sure each iterator produces the same number of iterations.
        if num_iterations.len() > 1 {
            self.gen_check_equal(ctx, num_iterations)?;
        }

        let ref sir_function = ctx.sir_program.funcs[parfor.body];
        assert!(sir_function.loop_body);

        self.gen_loop_body_function(ctx.sir_program, sir_function, parfor)?;

        let body_function = *self.functions.get(&parfor.body).unwrap();

        // The parameters of the body function have symbol names that must exist in the current
        // context.
        let mut arguments = vec![];
        for (symbol, _) in sir_function.params.iter() {
            let value = self.load(ctx.builder, ctx.get_value(symbol)?)?;
            arguments.push(value);
        }
        let iterations = num_iterations[0];
        arguments.push(iterations);

        let builder = LLVMBuildCall(ctx.builder, body_function, arguments.as_mut_ptr(), arguments.len() as u32, c_str!(""));
        LLVMBuildStore(ctx.builder, builder, ctx.get_value(&parfor.builder)?);

        // Create a new exit block, jump to it, and then load the arguments to the continuation and
        // call it. The call is a tail call.
        let block = LLVMAppendBasicBlockInContext(self.context, ctx.llvm_function, c_str!("exit"));
        LLVMBuildBr(ctx.builder, block);
        LLVMPositionBuilderAtEnd(ctx.builder, block);

        // Call the continuation, which can be built as a tail call.
        let ref sir_function = ctx.sir_program.funcs[parfor.cont];
        let mut arguments = vec![];
        for (symbol, _) in sir_function.params.iter() {
            let value = self.load(ctx.builder, ctx.get_value(symbol)?)?;
            arguments.push(value);
        }
        let cont_function = *self.functions.get(&parfor.cont).unwrap();

        let _ = LLVMBuildCall(ctx.builder, cont_function, arguments.as_mut_ptr(), arguments.len() as u32, c_str!(""));
        LLVMSetTailCall(LLVMGetLastInstruction(block), 1);
        LLVMBuildRetVoid(ctx.builder);
        Ok(())
    }

    /// Generate a loop body function.
    ///
    /// A loop body function has the following layout:
    ///
    /// { builders } FuncName(arg1, arg2, ...):
    /// entry:
    ///     alloca all variables
    ///     store builder in parfor.builder_arg
    ///     br loop.begin
    /// loop.begin:
    ///     i = <initializer code>
    ///     br loop.entry
    /// loop.entry:
    ///     if i >= end:
    ///         br loop.exit
    ///     else
    ///         br loop.body
    /// loop.body:
    ///     e = < load elements > based on i
    ///     < generate function body>, replace EndFunction with Br loop.check
    /// loop.check:
    ///     update i
    ///     br loop.entry
    /// loop.exit:
    ///     return { builders }
    unsafe fn gen_loop_body_function(&mut self, program: &SirProgram, func: &SirFunction, parfor: &ParallelForData) -> WeldResult<()> {
        // Construct the return type, which is the builders passed into the function sorted by the
        // argument parameter names.
        let builders: Vec<Type> = func.params.values()
            .filter(|v| v.is_builder())
            .map(|v| v.clone())
            .collect();

        // Each loop provides a single builder expression (which could be a struct of builders).
        // The loop's output is by definition derived from this builder.
        assert_eq!(builders.len(), 1);
        let ref weld_ty = builders[0];

        let mut arg_tys = vec![];
        for (_, ty) in func.params.iter() {
            arg_tys.push(self.llvm_type(ty)?);
        }
        // The last argument is the *total* number of iterations across all threads (in a
        // multi-threaded setting) that this loop will execute for.
        arg_tys.push(self.i64_type());

        let num_iterations_index = (arg_tys.len() - 1) as u32;

        let ret_ty = self.llvm_type(weld_ty)?;
        let func_ty = LLVMFunctionType(ret_ty, arg_tys.as_mut_ptr(), arg_tys.len() as u32, 0);
        let name = CString::new(format!("f{}_loop", func.id)).unwrap();
        let function = LLVMAddFunction(self.module, name.as_ptr(), func_ty);
        LLVMSetLinkage(function, LLVMLinkage::LLVMPrivateLinkage);
        self.functions.insert(func.id, function);

        // Create a context for the function.
        let ref mut context = FunctionContext::new(self.context, program, func, function);

        // Create the entry basic block, where we define alloca'd variables.
        let entry_bb = LLVMAppendBasicBlockInContext(self.context, context.llvm_function, c_str!("entry"));
        LLVMPositionBuilderAtEnd(context.builder, entry_bb);

        // Add the function parameters. Function parameters are stored in alloca'd variables. The
        // function parameters are always enumerated alphabetically sorted by symbol name.
        //
        // Note that the call to llvm_type here is also important, since it ensures that each type
        // is defined when generating statements.
        for (symbol, ty) in func.params.iter() {
            let name = CString::new(symbol.to_string()).unwrap();
            let value = LLVMBuildAlloca(context.builder, self.llvm_type(ty)?, name.as_ptr()); 
            context.symbols.insert(symbol.clone(), value);
        }

        // Generate local variables.
        for (symbol, ty) in func.locals.iter() {
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

        // Append the loop basic blocks.
        let loop_begin_bb = LLVMAppendBasicBlockInContext(self.context, context.llvm_function, c_str!("loop.begin"));
        let loop_entry_bb = LLVMAppendBasicBlockInContext(self.context, context.llvm_function, c_str!("loop.entry"));

        // Generate the basic block references by appending each basic block to the function. We do
        // this first so we can forward reference blocks if necessary.
        for bb in func.blocks.iter() {
            let name = CString::new(format!("b{}", bb.id)).unwrap();
            let block = LLVMAppendBasicBlockInContext(self.context, context.llvm_function, name.as_ptr());
            context.blocks.insert(bb.id, block);
        }

        // Finally, add the loop end basic blocks.
        let loop_end_bb = LLVMAppendBasicBlockInContext(self.context, context.llvm_function, c_str!("loop.end"));
        let loop_exit_bb = LLVMAppendBasicBlockInContext(self.context, context.llvm_function, c_str!("loop.exit"));

        // Build the loop.

        LLVMPositionBuilderAtEnd(context.builder, entry_bb);
        LLVMBuildBr(context.builder, loop_begin_bb);

        LLVMPositionBuilderAtEnd(context.builder, loop_begin_bb);

        LLVMBuildStore(context.builder,
                        self.load(context.builder, context.get_value(&parfor.builder)?)?,
                        context.get_value(&parfor.builder_arg)?);

        // XXX set the start to a function of the thread ID in a multi-threaded setting.
        LLVMBuildStore(context.builder,
                       self.i64(0),
                       context.get_value(&parfor.idx_arg)?);

        LLVMBuildBr(context.builder, loop_entry_bb);
        LLVMPositionBuilderAtEnd(context.builder, loop_entry_bb);

        let max = LLVMGetParam(context.llvm_function, num_iterations_index);
        let i = self.load(context.builder, context.get_value(&parfor.idx_arg)?)?;
        let continue_cond = LLVMBuildICmp(context.builder, LLVMIntPredicate::LLVMIntSGE, max, i, c_str!(""));

        // First body block of the SIR function.
        let first_body_block = *context.blocks.get(&0).unwrap();

        let _ = LLVMBuildCondBr(context.builder, continue_cond, first_body_block, loop_exit_bb);

        LLVMPositionBuilderAtEnd(context.builder, first_body_block);

        // Load the elements.
        let e = context.get_value(&parfor.data_arg)?;
        self.gen_loop_element(context, i, e, parfor)?;

        // Generate the body.
        for bb in func.blocks.iter() {
            LLVMPositionBuilderAtEnd(context.builder, context.get_block(&bb.id)?);
            for statement in bb.statements.iter() {
                self.generate_statement(context, statement)?;
            }
            self.generate_terminator(context, &bb, Some(loop_end_bb))?;
        }

        // The EndFunction terminators in the loop body jump to this block, so no explicit jumps to
        // it are necessary.
        LLVMPositionBuilderAtEnd(context.builder, loop_end_bb);

        let i = self.load(context.builder, context.get_value(&parfor.idx_arg)?)?;
        LLVMBuildNSWAdd(context.builder, i, self.i64(1), c_str!(""));
        LLVMBuildBr(context.builder, loop_entry_bb);

        LLVMPositionBuilderAtEnd(context.builder, loop_exit_bb);
        let updated_builder = self.load(context.builder, context.get_value(&parfor.builder)?)?;
        LLVMBuildRet(context.builder, updated_builder);
        Ok(())
    }

    unsafe fn gen_loop_element(&mut self,
                                    ctx: &mut FunctionContext,
                                    i: LLVMValueRef,
                                    e: LLVMValueRef,
                                    parfor: &ParallelForData) -> WeldResult<()> {

        let mut values = vec![];
        for iter in parfor.data.iter() {
            match iter.kind {
                ScalarIter if iter.start.is_some() => {
                    let start = self.load(ctx.builder, ctx.get_value(iter.start.as_ref().unwrap())?)?;
                    let stride = self.load(ctx.builder, ctx.get_value(iter.stride.as_ref().unwrap())?)?;

                    // Index = (start + stride * i)
                    let tmp = LLVMBuildNSWMul(ctx.builder, stride, i, c_str!(""));
                    let i = LLVMBuildNSWAdd(ctx.builder, start, tmp, c_str!(""));

                    let vector = self.load(ctx.builder, ctx.get_value(&iter.data)?)?;
                    let vector_type = ctx.sir_function.symbol_type(&iter.data)?;
                    let element_pointer = if let Vector(ref elem_type) = *vector_type {
                        let mut methods = self.vectors.get_mut(elem_type).unwrap();
                        methods.generate_at(ctx.builder, vector, i)?
                    } else {
                        unreachable!()
                    };
                    let element = self.load(ctx.builder, element_pointer)?;
                    values.push(element);
                }
                ScalarIter => {
                    // Iterates over the full vector: i is the array index.
                    //
                    // TODO this is a common pattern, would be nice to push this into a function.
                    let vector = self.load(ctx.builder, ctx.get_value(&iter.data)?)?;
                    let vector_type = ctx.sir_function.symbol_type(&iter.data)?;
                    let element_pointer = if let Vector(ref elem_type) = *vector_type {
                        let mut methods = self.vectors.get_mut(elem_type).unwrap();
                        methods.generate_at(ctx.builder, vector, i)?
                    } else {
                        unreachable!()
                    };
                    let element = self.load(ctx.builder, element_pointer)?;
                    values.push(element);
                }
                SimdIter if iter.start.is_some() => unimplemented!(),
                SimdIter => {
                    unimplemented!()
                }
                FringeIter if iter.start.is_some() => unimplemented!(),
                FringeIter => {
                    unimplemented!()
                }
                RangeIter => {
                    unimplemented!()
                }
                NdIter => unimplemented!(),
            }
        }

        assert!(values.len() > 0);

        if values.len() > 1 {
            for (i, value) in values.into_iter().enumerate() {
                let pointer = LLVMBuildStructGEP(ctx.builder, e, i as u32, c_str!(""));
                LLVMBuildStore(ctx.builder, value, pointer);
            }
        } else {
            LLVMBuildStore(ctx.builder, values[0], e);
        }
        Ok(())
    }

    unsafe fn gen_bounds_check(&mut self, _ctx: &mut FunctionContext, _iterator: &ParallelForIter) -> WeldResult<LLVMValueRef> {
        Ok(self.i64(10))
    }

    unsafe fn gen_check_equal(&mut self, ctx: &mut FunctionContext, iterations: &Vec<LLVMValueRef>) -> WeldResult<()> {
        Ok(())
    }
}
