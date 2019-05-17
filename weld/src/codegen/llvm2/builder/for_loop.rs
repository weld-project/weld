//! Code generation for the parallel for loop.
//!
//! This backend currently only generates single threaded code, but its structure is amenable to
//! parallelization (e.g., loops are still divided out into their own function, albeit marked with
//! `alwaysinline` at the moment).
//!
//! The `GenForLoopInternal` is the main workhorse of this module, and provides methods for
//! building a loop, creating bounds checks, loading elements, and so forth.

use llvm_sys;

use std::ffi::CString;

use crate::ast::IterKind::*;
use crate::ast::*;
use crate::error::*;
use crate::runtime::WeldRuntimeErrno;
use crate::sir::*;

use self::llvm_sys::core::*;
use self::llvm_sys::prelude::*;
use self::llvm_sys::{LLVMIntPredicate, LLVMLinkage};

use crate::codegen::llvm2::llvm_exts::LLVMExtAttribute::*;
use crate::codegen::llvm2::llvm_exts::*;
use crate::codegen::llvm2::vector::VectorExt;
use crate::codegen::llvm2::{LLVM_VECTOR_WIDTH, SIR_FUNC_CALL_CONV};

use super::{CodeGenExt, FunctionContext, LlvmGenerator};

/// An internal trait for generating parallel For loops.
pub trait ForLoopGenInternal {
    /// Entry point to generating a for loop.
    ///
    /// This is the only function in the trait that should be called -- all other methods are
    /// helpers.
    unsafe fn gen_for_internal(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        output: &Symbol,
        parfor: &ParallelForData,
    ) -> WeldResult<()>;
    /// Generates bounds checking code for the loop and return number of iterations.
    ///
    /// This function ensures that each iterator will only access in-bounds vector elements and
    /// also ensures that each zipped vector has the same number of consumed elements. If these
    /// checks fail, the generated code raises an error.
    unsafe fn gen_bounds_check(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        parfor: &ParallelForData,
    ) -> WeldResult<LLVMValueRef>;
    /// Generates the loop body.
    ///
    /// This generates both the loop control flow and the executing body of the loop.
    unsafe fn gen_loop_body_function(
        &mut self,
        program: &SirProgram,
        func: &SirFunction,
        parfor: &ParallelForData,
    ) -> WeldResult<()>;
    /// Generates a bounds check for the given iterator.
    ///
    /// Returns a value representing the number of iterations the iterator will produce.
    unsafe fn gen_iter_bounds_check(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        iterator: &ParallelForIter,
        pass_block: LLVMBasicBlockRef,
        fail_block: LLVMBasicBlockRef,
    ) -> WeldResult<LLVMValueRef>;
    /// Generates code to check whether the number of iterations in each value is the same.
    ///
    /// If the number of iterations is not the same, the module raises an error and exits.
    unsafe fn gen_check_equal(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        iterations: &[LLVMValueRef],
        pass_block: LLVMBasicBlockRef,
        fail_block: LLVMBasicBlockRef,
    ) -> WeldResult<()>;
    /// Generates code to load potentially zipped elements at index `i` into `e`.
    ///
    /// `e` must be a pointer, and `i` must be a loaded index argument of type `i64`.
    unsafe fn gen_loop_element(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        i: LLVMValueRef,
        e: LLVMValueRef,
        parfor: &ParallelForData,
    ) -> WeldResult<()>;
}

impl ForLoopGenInternal for LlvmGenerator {
    /// Entry point to generating a for loop.
    unsafe fn gen_for_internal(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        output: &Symbol,
        parfor: &ParallelForData,
    ) -> WeldResult<()> {
        let iterations = self.gen_bounds_check(ctx, parfor)?;

        let sir_function = &ctx.sir_program.funcs[parfor.body];
        assert!(sir_function.loop_body);

        self.gen_loop_body_function(ctx.sir_program, sir_function, parfor)?;
        let body_function = self.functions[&parfor.body];

        // The parameters of the body function have symbol names that must exist in the current
        // context.
        let mut arguments = vec![];
        for (symbol, _) in sir_function.params.iter() {
            let value = self.load(ctx.builder, ctx.get_value(symbol)?)?;
            arguments.push(value);
        }
        // The body function has an additional arguement representing the number of iterations.
        arguments.push(iterations);
        // Last argument is always the run handle.
        arguments.push(ctx.get_run());

        // Call the body function, which runs the loop and updates the builder. The updated builder
        // is returned to the current function.
        let builder = LLVMBuildCall(
            ctx.builder,
            body_function,
            arguments.as_mut_ptr(),
            arguments.len() as u32,
            c_str!(""),
        );
        LLVMSetInstructionCallConv(builder, SIR_FUNC_CALL_CONV);
        // XXX what is parfor.builder now...
        LLVMBuildStore(ctx.builder, builder, ctx.get_value(&parfor.builder)?);
        LLVMBuildStore(ctx.builder, builder, ctx.get_value(&output)?);

        Ok(())
    }

    /// Generate runtime bounds checking, which looks as follows:
    ///
    /// passed0 = <check bounds of iterator 0>
    /// if passed: goto next, else: goto fail
    /// next:
    /// passed1 = <check bounds of iterator 1>
    /// ...
    /// fail:
    /// raise error
    ///
    /// The bounds check code positions the `FunctionContext` builder after all bounds checking is
    /// complete.
    unsafe fn gen_bounds_check(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        parfor: &ParallelForData,
    ) -> WeldResult<LLVMValueRef> {
        let mut pass_blocks = vec![];
        for _ in 0..parfor.data.len() {
            pass_blocks.push(LLVMAppendBasicBlockInContext(
                self.context,
                ctx.llvm_function,
                c_str!("bounds.check"),
            ));
        }
        // Jump here if the iterator will cause an array out of bounds error.
        let fail_boundscheck_block =
            LLVMAppendBasicBlockInContext(self.context, ctx.llvm_function, c_str!("bounds.fail"));
        // Jump here if the zipped vectors produce different numbers of iterations.
        let fail_zip_block =
            LLVMAppendBasicBlockInContext(self.context, ctx.llvm_function, c_str!("bounds.fail"));
        // Jump here if all checks pass.
        let pass_all_block =
            LLVMAppendBasicBlockInContext(self.context, ctx.llvm_function, c_str!("bounds.passed"));

        if self.conf.enable_bounds_checks {
            info!("Generating bounds checking code")
        } else {
            info!("Omitting bounds checking code")
        }

        let mut num_iterations = vec![];
        for (iter, pass_block) in parfor.data.iter().zip(pass_blocks) {
            let iterations =
                self.gen_iter_bounds_check(ctx, iter, pass_block, fail_boundscheck_block)?;
            num_iterations.push(iterations);
            LLVMPositionBuilderAtEnd(ctx.builder, pass_block);
        }

        assert!(!num_iterations.is_empty());

        // Make sure each iterator produces the same number of iterations.
        if num_iterations.len() > 1 {
            self.gen_check_equal(ctx, &num_iterations, pass_all_block, fail_zip_block)?;
        } else {
            let _ = LLVMBuildBr(ctx.builder, pass_all_block);
        }

        LLVMPositionBuilderAtEnd(ctx.builder, fail_boundscheck_block);
        let error = self.i64(WeldRuntimeErrno::BadIteratorLength as i64);
        self.intrinsics
            .call_weld_run_set_errno(ctx.builder, ctx.get_run(), error, None);
        LLVMBuildUnreachable(ctx.builder);

        LLVMPositionBuilderAtEnd(ctx.builder, fail_zip_block);
        let error = self.i64(WeldRuntimeErrno::MismatchedZipSize as i64);
        self.intrinsics
            .call_weld_run_set_errno(ctx.builder, ctx.get_run(), error, None);
        LLVMBuildUnreachable(ctx.builder);

        // Bounds check passed - jump to the final block.
        LLVMPositionBuilderAtEnd(ctx.builder, pass_all_block);
        Ok(num_iterations[0])
    }

    /// Generate a loop body function.
    ///
    /// A loop body function has the following layout:
    ///
    /// { builders } FuncName(arg1, arg2, ..., iterations, run):
    /// entry:
    ///     alloca all variables except the local builder.
    ///     alias builder argument with parfor.builder_arg
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
    unsafe fn gen_loop_body_function(
        &mut self,
        program: &SirProgram,
        func: &SirFunction,
        parfor: &ParallelForData,
    ) -> WeldResult<()> {
        // Construct the return type, which is the builder passed into the function.
        let builders: Vec<Type> = func
            .params
            .values()
            .filter(|v| v.is_builder())
            .cloned()
            .collect();

        // Each loop provides a single builder expression (which could be a struct of builders).
        // The loop's output is by definition derived from this builder.
        assert_eq!(builders.len(), 1);
        let weld_ty = &builders[0];

        let mut arg_tys = self.argument_types(func)?;
        // The second-to-last argument is the *total* number of iterations across all threads (in a
        // multi-threaded setting) that this loop will execute for.
        arg_tys.push(self.i64_type());
        let num_iterations_index = (arg_tys.len() - 1) as u32;
        // Last argument is run handle, as always.
        arg_tys.push(self.run_handle_type());

        let ret_ty = self.llvm_type(weld_ty)?;
        let func_ty = LLVMFunctionType(ret_ty, arg_tys.as_mut_ptr(), arg_tys.len() as u32, 0);
        let name = CString::new(format!("f{}_loop", func.id)).unwrap();
        let function = LLVMAddFunction(self.module, name.as_ptr(), func_ty);
        LLVMSetLinkage(function, LLVMLinkage::LLVMPrivateLinkage);

        LLVMExtAddDefaultAttrs(self.context(), function);
        // We always inline this since it will appear only once in the program, so there's no code
        // size cost to doing it.
        //
        // XXX this should only be here in the single threaded setting.
        LLVMExtAddAttrsOnFunction(self.context, function, &[AlwaysInline]);

        self.functions.insert(func.id, function);

        // Create a context for the function.
        let context = &mut FunctionContext::new(self.context, program, func, function);
        // Reference to the parameter storing the max number of iterations.
        let max = LLVMGetParam(context.llvm_function, num_iterations_index);
        // Create the entry basic block, where we define alloca'd variables.
        let entry_bb =
            LLVMAppendBasicBlockInContext(self.context, context.llvm_function, c_str!(""));
        LLVMPositionBuilderAtEnd(context.builder, entry_bb);

        self.gen_allocas(context)?;
        self.gen_store_parameters(context)?;

        // Store the loop induction variable and the builder argument.
        LLVMBuildStore(
            context.builder,
            self.load(context.builder, context.get_value(&parfor.builder)?)?,
            context.get_value(&parfor.builder_arg)?,
        );
        LLVMBuildStore(
            context.builder,
            self.i64(0),
            context.get_value(&parfor.idx_arg)?,
        );

        // Add the SIR function basic blocks.
        self.gen_basic_block_defs(context)?;

        // Add the loop end basic blocks.
        let loop_end_bb =
            LLVMAppendBasicBlockInContext(self.context, context.llvm_function, c_str!("loop.end"));
        let loop_exit_bb =
            LLVMAppendBasicBlockInContext(self.context, context.llvm_function, c_str!("loop.exit"));

        // Check whether we need to loop at all.
        let any_iters_cond = LLVMBuildICmp(
            context.builder,
            LLVMIntPredicate::LLVMIntNE,
            max,
            self.i64(0),
            c_str!(""),
        );
        // First body block of the SIR function.
        let first_body_block = context.blocks[&0];
        LLVMBuildCondBr(
            context.builder,
            any_iters_cond,
            first_body_block,
            loop_exit_bb,
        );

        // Build the loop body.
        LLVMPositionBuilderAtEnd(context.builder, first_body_block);

        // Load the loop element.
        let i = self.load(context.builder, context.get_value(&parfor.idx_arg)?)?;
        let e = context.get_value(&parfor.data_arg)?;
        self.gen_loop_element(context, i, e, parfor)?;

        // Generate the body - this resembles the usual SIR function generation, but we pass a
        // basic block ID to gen_terminator to change the `EndFunction` terminators to a basic
        // block jump to the end of the loop.
        for bb in func.blocks.iter() {
            LLVMPositionBuilderAtEnd(context.builder, context.get_block(bb.id)?);
            for statement in bb.statements.iter() {
                self.gen_statement(context, statement)?;
            }
            let loop_terminator = (loop_end_bb, context.get_value(&parfor.builder_arg)?);
            self.gen_terminator(context, &bb, Some(loop_terminator))?;
        }

        // The EndFunction terminators in the loop body jump to this block.
        LLVMPositionBuilderAtEnd(context.builder, loop_end_bb);

        // Increment the iteration variable i. We don't need to load it again because this block is
        // only reachable from the loop body, which does not mutate i.
        let updated = LLVMBuildNSWAdd(context.builder, i, self.i64(1), c_str!(""));
        LLVMBuildStore(
            context.builder,
            updated,
            context.get_value(&parfor.idx_arg)?,
        );

        // Check whether to continue looping.
        //
        // NOTE: It's important to use `eq` here! LLVM checks for it in its analyses and won't
        // unroll loops without it.
        let finished_cond = LLVMBuildICmp(
            context.builder,
            LLVMIntPredicate::LLVMIntEQ,
            max,
            updated,
            c_str!(""),
        );
        let _ = LLVMBuildCondBr(
            context.builder,
            finished_cond,
            loop_exit_bb,
            first_body_block,
        );

        // The last basic block loads the updated builder and returns it.
        LLVMPositionBuilderAtEnd(context.builder, loop_exit_bb);
        let updated_builder =
            self.load(context.builder, context.get_value(&parfor.builder_arg)?)?;
        LLVMBuildRet(context.builder, updated_builder);
        Ok(())
    }

    unsafe fn gen_loop_element(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        i: LLVMValueRef,
        e: LLVMValueRef,
        parfor: &ParallelForData,
    ) -> WeldResult<()> {
        let mut values = vec![];
        for iter in parfor.data.iter() {
            match iter.kind {
                ScalarIter if iter.start.is_some() => {
                    let start =
                        self.load(ctx.builder, ctx.get_value(iter.start.as_ref().unwrap())?)?;
                    let stride =
                        self.load(ctx.builder, ctx.get_value(iter.stride.as_ref().unwrap())?)?;

                    // Index = (start + stride * i)
                    let tmp = LLVMBuildNSWMul(ctx.builder, stride, i, c_str!(""));
                    let i = LLVMBuildNSWAdd(ctx.builder, start, tmp, c_str!(""));

                    let vector = self.load(ctx.builder, ctx.get_value(&iter.data)?)?;
                    let vector_type = ctx.sir_function.symbol_type(&iter.data)?;
                    let element_pointer = self.gen_at(ctx.builder, vector_type, vector, i)?;
                    let element = self.load(ctx.builder, element_pointer)?;
                    values.push(element);
                }
                ScalarIter => {
                    // Iterates over the full vector: Index = i.
                    let vector = self.load(ctx.builder, ctx.get_value(&iter.data)?)?;
                    let vector_type = ctx.sir_function.symbol_type(&iter.data)?;
                    let element_pointer = self.gen_at(ctx.builder, vector_type, vector, i)?;
                    let element = self.load(ctx.builder, element_pointer)?;
                    values.push(element);
                }
                SimdIter if iter.start.is_some() => unreachable!(),
                SimdIter => {
                    let i = LLVMBuildNSWMul(
                        ctx.builder,
                        i,
                        self.i64(i64::from(LLVM_VECTOR_WIDTH)),
                        c_str!(""),
                    );
                    let vector = self.load(ctx.builder, ctx.get_value(&iter.data)?)?;
                    let vector_type = ctx.sir_function.symbol_type(&iter.data)?;
                    let element_pointer = self.gen_vat(ctx.builder, vector_type, vector, i)?;
                    let element = self.load(ctx.builder, element_pointer)?;
                    values.push(element);
                }
                FringeIter if iter.start.is_some() => unreachable!(),
                FringeIter => {
                    let vector = self.load(ctx.builder, ctx.get_value(&iter.data)?)?;
                    let vector_type = ctx.sir_function.symbol_type(&iter.data)?;
                    let size = self.gen_size(ctx.builder, vector_type, vector)?;

                    // Start = Len(vector) - Len(vector) % VECTOR_WIDTH
                    // Index = start + i
                    let tmp = LLVMBuildSRem(
                        ctx.builder,
                        size,
                        self.i64(i64::from(LLVM_VECTOR_WIDTH)),
                        c_str!(""),
                    );
                    let start = LLVMBuildNSWSub(ctx.builder, size, tmp, c_str!(""));
                    let i = LLVMBuildNSWAdd(ctx.builder, start, i, c_str!(""));

                    let element_pointer = self.gen_at(ctx.builder, vector_type, vector, i)?;
                    let element = self.load(ctx.builder, element_pointer)?;
                    values.push(element);
                }
                RangeIter => {
                    let start =
                        self.load(ctx.builder, ctx.get_value(iter.start.as_ref().unwrap())?)?;
                    let stride =
                        self.load(ctx.builder, ctx.get_value(iter.stride.as_ref().unwrap())?)?;

                    // Index = (start + stride * i)
                    let tmp = LLVMBuildNSWMul(ctx.builder, stride, i, c_str!(""));
                    let i = LLVMBuildNSWAdd(ctx.builder, start, tmp, c_str!(""));
                    values.push(i);
                }
                NdIter => unimplemented!(), // NdIter Load Element
            }
        }

        assert!(!values.is_empty());

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

    unsafe fn gen_iter_bounds_check(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        iter: &ParallelForIter,
        pass_block: LLVMBasicBlockRef,
        fail_block: LLVMBasicBlockRef,
    ) -> WeldResult<LLVMValueRef> {
        let vector = self.load(ctx.builder, ctx.get_value(&iter.data)?)?;
        let vector_type = ctx.sir_function.symbol_type(&iter.data)?;
        let size = self.gen_size(ctx.builder, vector_type, vector)?;
        match iter.kind {
            ScalarIter if iter.start.is_some() => {
                use self::llvm_sys::LLVMIntPredicate::{LLVMIntEQ, LLVMIntSLE, LLVMIntSLT};
                let start = self.load(ctx.builder, ctx.get_value(iter.start.as_ref().unwrap())?)?;
                let stride =
                    self.load(ctx.builder, ctx.get_value(iter.stride.as_ref().unwrap())?)?;
                let end = self.load(ctx.builder, ctx.get_value(iter.end.as_ref().unwrap())?)?;

                let diff = LLVMBuildNSWSub(ctx.builder, end, start, c_str!(""));
                let iterations = LLVMBuildSDiv(ctx.builder, diff, stride, c_str!(""));

                if self.conf.enable_bounds_checks {
                    // Checks required:
                    // start < size
                    // end <= size
                    // start < end
                    // (end - start) % stride == 0
                    // Iterations = (end - start) / stride
                    let start_check =
                        LLVMBuildICmp(ctx.builder, LLVMIntSLT, start, size, c_str!(""));
                    let end_check = LLVMBuildICmp(ctx.builder, LLVMIntSLE, end, size, c_str!(""));
                    let end_start_check =
                        LLVMBuildICmp(ctx.builder, LLVMIntSLT, start, end, c_str!(""));
                    let mod_check = LLVMBuildSRem(ctx.builder, diff, stride, c_str!(""));
                    let mod_check =
                        LLVMBuildICmp(ctx.builder, LLVMIntEQ, mod_check, self.i64(0), c_str!(""));

                    let mut check = LLVMBuildAnd(ctx.builder, start_check, end_check, c_str!(""));
                    check = LLVMBuildAnd(ctx.builder, check, end_start_check, c_str!(""));
                    check = LLVMBuildAnd(ctx.builder, check, mod_check, c_str!(""));
                    let _ = LLVMBuildCondBr(ctx.builder, check, pass_block, fail_block);
                } else {
                    let _ = LLVMBuildBr(ctx.builder, pass_block);
                }

                Ok(iterations)
            }
            ScalarIter => {
                // The number of iterations is the size of the vector. No explicit bounds check is
                // necessary here.
                let _ = LLVMBuildBr(ctx.builder, pass_block);
                Ok(size)
            }
            SimdIter if iter.start.is_some() => unreachable!(),
            SimdIter => {
                let iterations = LLVMBuildSDiv(
                    ctx.builder,
                    size,
                    self.i64(i64::from(LLVM_VECTOR_WIDTH)),
                    c_str!(""),
                );
                let _ = LLVMBuildBr(ctx.builder, pass_block);
                Ok(iterations)
            }
            FringeIter if iter.start.is_some() => unreachable!(),
            FringeIter => {
                let iterations = LLVMBuildSRem(
                    ctx.builder,
                    size,
                    self.i64(i64::from(LLVM_VECTOR_WIDTH)),
                    c_str!(""),
                );
                let _ = LLVMBuildBr(ctx.builder, pass_block);
                Ok(iterations)
            }
            RangeIter => {
                use self::llvm_sys::LLVMIntPredicate::{LLVMIntEQ, LLVMIntSLT};
                let start = self.load(ctx.builder, ctx.get_value(iter.start.as_ref().unwrap())?)?;
                let stride =
                    self.load(ctx.builder, ctx.get_value(iter.stride.as_ref().unwrap())?)?;
                let end = self.load(ctx.builder, ctx.get_value(iter.end.as_ref().unwrap())?)?;
                let diff = LLVMBuildNSWSub(ctx.builder, end, start, c_str!(""));
                let iterations = LLVMBuildSDiv(ctx.builder, diff, stride, c_str!(""));

                if self.conf.enable_bounds_checks {
                    // Checks required:
                    // start < end
                    // (end - start) % stride == 0
                    // Iterations = (end - start) / stride
                    let end_start_check =
                        LLVMBuildICmp(ctx.builder, LLVMIntSLT, start, end, c_str!(""));
                    let mod_check = LLVMBuildSRem(ctx.builder, diff, stride, c_str!(""));
                    let mod_check =
                        LLVMBuildICmp(ctx.builder, LLVMIntEQ, mod_check, self.i64(0), c_str!(""));

                    let check = LLVMBuildAnd(ctx.builder, end_start_check, mod_check, c_str!(""));
                    let _ = LLVMBuildCondBr(ctx.builder, check, pass_block, fail_block);
                } else {
                    let _ = LLVMBuildBr(ctx.builder, pass_block);
                }

                Ok(iterations)
            }
            NdIter => unimplemented!(), // NdIter Compute Bounds Check
        }
    }

    unsafe fn gen_check_equal(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        iterations: &[LLVMValueRef],
        pass_block: LLVMBasicBlockRef,
        fail_block: LLVMBasicBlockRef,
    ) -> WeldResult<()> {
        use self::llvm_sys::LLVMIntPredicate::LLVMIntEQ;
        let mut passed = self.i1(true);
        if self.conf.enable_bounds_checks {
            for value in iterations.iter().skip(1) {
                let check =
                    LLVMBuildICmp(ctx.builder, LLVMIntEQ, iterations[0], *value, c_str!(""));
                passed = LLVMBuildAnd(ctx.builder, passed, check, c_str!(""));
            }
        }

        LLVMBuildCondBr(ctx.builder, passed, pass_block, fail_block);
        Ok(())
    }
}
