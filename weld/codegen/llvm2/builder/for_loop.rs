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

use codegen::llvm2::LLVM_VECTOR_WIDTH;
use super::{CodeGenExt, FunctionContext, LlvmGenerator};

/// An internal trait for generating parallel For loops.
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
    ///
    /// If the number of iterations is not the same, the module raises an error and exits.
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
        // Generate a bounds check and determine the number of iterations each iterator will
        // produce.
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
        // The body function has an additional arguement representing the number of iterations.
        arguments.push(iterations);

        // Call the body function, which runs the loop and updates the builder. The updated builder
        // is returned to the current function.
        let builder = LLVMBuildCall(ctx.builder, body_function, arguments.as_mut_ptr(), arguments.len() as u32, c_str!(""));
        LLVMBuildStore(ctx.builder, builder, ctx.get_value(&parfor.builder)?);

        // Create a new exit block, jump to it, and then load the arguments to the continuation and
        // call it.
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
    unsafe fn gen_loop_body_function(&mut self,
                                     program: &SirProgram,
                                     func: &SirFunction,
                                     parfor: &ParallelForData) -> WeldResult<()> {
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

        let mut arg_tys = self.argument_types(func)?;
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
        let entry_bb = LLVMAppendBasicBlockInContext(self.context,
                                                     context.llvm_function,
                                                     c_str!("entry"));
        LLVMPositionBuilderAtEnd(context.builder, entry_bb);

        self.gen_allocas(context)?;
        self.gen_store_parameters(context)?;

        // Append the loop start basic blocks.
        let loop_begin_bb = LLVMAppendBasicBlockInContext(self.context,
                                                          context.llvm_function,
                                                          c_str!("loop.begin"));
        let loop_entry_bb = LLVMAppendBasicBlockInContext(self.context,
                                                          context.llvm_function,
                                                          c_str!("loop.entry"));

        // Add the SIR function basic blocks.
        self.gen_basic_block_defs(context)?;

        // Finally, add the loop end basic blocks.
        let loop_end_bb = LLVMAppendBasicBlockInContext(self.context,
                                                        context.llvm_function,
                                                        c_str!("loop.end"));
        let loop_exit_bb = LLVMAppendBasicBlockInContext(self.context,
                                                         context.llvm_function,
                                                         c_str!("loop.exit"));

        // Build the loop.
        LLVMPositionBuilderAtEnd(context.builder, entry_bb);
        LLVMBuildBr(context.builder, loop_begin_bb);

        LLVMPositionBuilderAtEnd(context.builder, loop_begin_bb);
        LLVMBuildStore(context.builder,
                        self.load(context.builder, context.get_value(&parfor.builder)?)?,
                        context.get_value(&parfor.builder_arg)?);
        LLVMBuildStore(context.builder,
                       self.i64(0),
                       context.get_value(&parfor.idx_arg)?);

        LLVMBuildBr(context.builder, loop_entry_bb);
        LLVMPositionBuilderAtEnd(context.builder, loop_entry_bb);

        let max = LLVMGetParam(context.llvm_function, num_iterations_index);
        let i = self.load(context.builder, context.get_value(&parfor.idx_arg)?)?;
        let continue_cond = LLVMBuildICmp(context.builder,
                                          LLVMIntPredicate::LLVMIntSGT, max, i, c_str!(""));

        // First body block of the SIR function.
        let first_body_block = *context.blocks.get(&0).unwrap();

        let _ = LLVMBuildCondBr(context.builder, continue_cond, first_body_block, loop_exit_bb);
        LLVMPositionBuilderAtEnd(context.builder, first_body_block);

        // Load the loop element.
        let e = context.get_value(&parfor.data_arg)?;
        self.gen_loop_element(context, i, e, parfor)?;

        // Generate the body - this resembles the usual SIR function generation, but we pass a
        // basic block ID to gen_terminator to change the `EndFunction` terminators to a basic
        // block jump to the end of the loop.
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

        // Increment the iteration variable i.
        let i = self.load(context.builder, context.get_value(&parfor.idx_arg)?)?;
        let updated = LLVMBuildNSWAdd(context.builder, i, self.i64(1), c_str!(""));
        LLVMBuildStore(context.builder, updated, context.get_value(&parfor.idx_arg)?);
        LLVMBuildBr(context.builder, loop_entry_bb);

        // The last basic block loads the updated builder and returns it.
        LLVMPositionBuilderAtEnd(context.builder, loop_exit_bb);
        let updated_builder = self.load(context.builder, context.get_value(&parfor.builder_arg)?)?;
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
                    let i = LLVMBuildNSWMul(ctx.builder, i, self.i64(LLVM_VECTOR_WIDTH as i64), c_str!(""));
                    let vector = self.load(ctx.builder, ctx.get_value(&iter.data)?)?;
                    let vector_type = ctx.sir_function.symbol_type(&iter.data)?;
                    let element_pointer = if let Vector(ref elem_type) = *vector_type {
                        let mut methods = self.vectors.get_mut(elem_type).unwrap();
                        methods.generate_vat(ctx.builder, vector, i)?
                    } else {
                        unreachable!()
                    };
                    let element = self.load(ctx.builder, element_pointer)?;
                    values.push(element);
                }
                FringeIter if iter.start.is_some() => unimplemented!(),
                FringeIter => {
                    let vector = self.load(ctx.builder, ctx.get_value(&iter.data)?)?;
                    let vector_type = ctx.sir_function.symbol_type(&iter.data)?;
                    let size = if let Vector(ref elem_type) = *vector_type {
                        let mut methods = self.vectors.get_mut(elem_type).unwrap();
                        methods.generate_size(ctx.builder, vector)?
                    } else {
                        unreachable!()
                    };

                    // Start = Len(vector) - Len(vector) % VECTOR_WIDTH
                    // Index = start + i
                    let tmp = LLVMBuildSRem(ctx.builder, size, self.i64(LLVM_VECTOR_WIDTH as i64), c_str!(""));
                    let start = LLVMBuildNSWSub(ctx.builder, size, tmp, c_str!(""));
                    let i = LLVMBuildNSWAdd(ctx.builder, start, i, c_str!(""));

                    let element_pointer = if let Vector(ref elem_type) = *vector_type {
                        let mut methods = self.vectors.get_mut(elem_type).unwrap();
                        methods.generate_at(ctx.builder, vector, i)?
                    } else {
                        unreachable!()
                    };

                    let element = self.load(ctx.builder, element_pointer)?;
                    values.push(element);
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

    unsafe fn gen_bounds_check(&mut self, ctx: &mut FunctionContext, iter: &ParallelForIter) -> WeldResult<LLVMValueRef> {
        match iter.kind {
            ScalarIter if iter.start.is_some() => {
                unimplemented!()
            }
            ScalarIter => {
                // The number of iterations is the size of the vector. No explicit bounds check is
                // necessary here.
                let vector = self.load(ctx.builder, ctx.get_value(&iter.data)?)?;
                let vector_type = ctx.sir_function.symbol_type(&iter.data)?;
                if let Vector(ref elem_type) = *vector_type {
                    let mut methods = self.vectors.get_mut(elem_type).unwrap();
                    methods.generate_size(ctx.builder, vector)
                } else {
                    unreachable!()
                }
            }
            SimdIter if iter.start.is_some() => unimplemented!(),
            SimdIter => {
                // The number of iterations is the size of the vector. No explicit bounds check is
                // necessary here. TODO definitely want a function for this...
                let vector = self.load(ctx.builder, ctx.get_value(&iter.data)?)?;
                let vector_type = ctx.sir_function.symbol_type(&iter.data)?;
                let size = if let Vector(ref elem_type) = *vector_type {
                    let mut methods = self.vectors.get_mut(elem_type).unwrap();
                    methods.generate_size(ctx.builder, vector)?
                } else {
                    unreachable!()
                };
                Ok(LLVMBuildSDiv(ctx.builder, size, self.i64(LLVM_VECTOR_WIDTH as i64), c_str!("")))
            }
            FringeIter if iter.start.is_some() => unimplemented!(),
            FringeIter => {
                let vector = self.load(ctx.builder, ctx.get_value(&iter.data)?)?;
                let vector_type = ctx.sir_function.symbol_type(&iter.data)?;
                let size = if let Vector(ref elem_type) = *vector_type {
                    let mut methods = self.vectors.get_mut(elem_type).unwrap();
                    methods.generate_size(ctx.builder, vector)?
                } else {
                    unreachable!()
                };
                Ok(LLVMBuildSRem(ctx.builder, size, self.i64(LLVM_VECTOR_WIDTH as i64), c_str!("")))
            }
            NdIter | RangeIter => unimplemented!()
        }
    }

    unsafe fn gen_check_equal(&mut self, ctx: &mut FunctionContext, iterations: &Vec<LLVMValueRef>) -> WeldResult<()> {
        // TODO
        Ok(())
    }
}
