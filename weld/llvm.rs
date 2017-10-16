use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::BTreeMap;

use easy_ll;

use common::WeldRuntimeErrno;

use super::ast::*;
use super::ast::Type::*;
use super::ast::LiteralKind::*;
use super::ast::ScalarKind::*;
use super::ast::BuilderKind::*;
use super::code_builder::CodeBuilder;
use super::error::*;
use super::macro_processor;
use super::passes::*;
use super::pretty_print::*;
use super::program::Program;
use super::runtime::*;
use super::sir;
use super::sir::*;
use super::sir::Statement::*;
use super::sir::Terminator::*;
use super::transforms;
use super::type_inference;
use super::util::IdGenerator;
use super::util::WELD_INLINE_LIB;

#[cfg(test)]
use super::parser::*;

static PRELUDE_CODE: &'static str = include_str!("resources/prelude.ll");

/// The default grain size for the parallel runtime.
static DEFAULT_GRAIN_SIZE: i64 = 16384;

/// A wrapper for a struct passed as input to the Weld runtime.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct WeldInputArgs {
    pub input: i64,
    pub nworkers: i32,
    pub mem_limit: i64,
}

/// A wrapper for outputs passed out of the Weld runtime.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct WeldOutputArgs {
    pub output: i64,
    pub run_id: i64,
    pub errno: WeldRuntimeErrno,
}

pub fn apply_opt_passes(expr: &mut TypedExpr, opt_passes: &Vec<Pass>) -> WeldResult<()> {
    for pass in opt_passes {
        pass.transform(expr)?;
        trace!("After {} pass:\n{}", pass.pass_name(), print_typed_expr(&expr));
    }
    Ok(())
}

/// Generate a compiled LLVM module from a program whose body is a function.
pub fn compile_program(program: &Program, opt_passes: &Vec<Pass>, llvm_opt_level: u32, multithreaded: bool)
        -> WeldResult<easy_ll::CompiledModule> {
    let mut expr = macro_processor::process_program(program)?;
    trace!("After macro substitution:\n{}\n", print_typed_expr(&expr));

    let _ = transforms::uniquify(&mut expr)?;
    type_inference::infer_types(&mut expr)?;
    let mut expr = expr.to_typed()?;
    trace!("After type inference:\n{}\n", print_typed_expr(&expr));

    apply_opt_passes(&mut expr, opt_passes)?;

    transforms::uniquify(&mut expr)?;
    debug!("Optimized Weld program:\n{}\n", print_expr(&expr));

    let sir_prog = sir::ast_to_sir(&expr, multithreaded)?;
    debug!("SIR program:\n{}\n", &sir_prog);

    let mut gen = LlvmGenerator::new();
    gen.multithreaded = multithreaded;

    gen.add_function_on_pointers("run", &sir_prog)?;
    let llvm_code = gen.result();
    trace!("LLVM program:\n{}\n", &llvm_code);

    debug!("Started compiling LLVM");
    let module = try!(easy_ll::compile_module(
        &llvm_code,
        llvm_opt_level,
        Some(WELD_INLINE_LIB)));
    debug!("Done compiling LLVM");

    debug!("Started runtime_init call");
    unsafe {
        weld_runtime_init();
    }
    debug!("Done runtime_init call");

    Ok(module)
}

/// Stores whether the code generator has created certain helper functions for a given type.
pub struct HelperState {
    hash_func: bool,
    cmp_func: bool,
    eq_func: bool,
}

impl HelperState {
    pub fn new() -> HelperState {
        HelperState {
            hash_func: false,
            cmp_func: false,
            eq_func: false,
        }
    }
}

/// Generates LLVM code for one or more modules.
pub struct LlvmGenerator {
    /// LLVM type name of the form %s0, %s1, etc for each struct generated.
    struct_names: HashMap<Vec<Type>, String>,
    struct_ids: IdGenerator,

    /// LLVM type name of the form %v0, %v1, etc for each vec generated.
    vec_names: HashMap<Type, String>,
    vec_ids: IdGenerator,

    // LLVM type names for each merger type.
    merger_names: HashMap<Type, String>,
    merger_ids: IdGenerator,

    /// LLVM type name of the form %d0, %d1, etc for each dict generated.
    dict_names: HashMap<Type, String>,
    dict_ids: IdGenerator,

    /// Set of declared CUDFs.
    cudf_names: HashSet<String>,

    /// LLVM type names for various builder types
    bld_names: HashMap<BuilderKind, String>,

    /// A CodeBuilder and ID generator for prelude functions such as type and struct definitions.
    prelude_code: CodeBuilder,
    prelude_var_ids: IdGenerator,

    /// A CodeBuilder for body functions in the module.
    body_code: CodeBuilder,

    /// Helper function state for types.
    type_helpers: HashMap<Type, HelperState>,

    /// Functions we have already visited when generating code.
    visited: HashSet<sir::FunctionId>,

    /// Multithreaded configuration set during compilation. If unset, performs
    /// single-threaded optimizations.
    multithreaded: bool,
}

impl LlvmGenerator {
    pub fn new() -> LlvmGenerator {
        let mut generator = LlvmGenerator {
            struct_names: HashMap::new(),
            struct_ids: IdGenerator::new("%s"),
            vec_names: HashMap::new(),
            vec_ids: IdGenerator::new("%v"),
            merger_names: HashMap::new(),
            merger_ids: IdGenerator::new("%m"),
            dict_names: HashMap::new(),
            dict_ids: IdGenerator::new("%d"),
            cudf_names: HashSet::new(),
            bld_names: HashMap::new(),
            prelude_code: CodeBuilder::new(),
            prelude_var_ids: IdGenerator::new("%p.p"),
            body_code: CodeBuilder::new(),
            visited: HashSet::new(),
            type_helpers: HashMap::new(),
            multithreaded: false,
        };
        generator.prelude_code.add(PRELUDE_CODE);
        generator.prelude_code.add("\n");
        generator
    }

    /// Return all the code generated so far.
    pub fn result(&mut self) -> String {
        format!("; PRELUDE:\n\n{}\n; BODY:\n\n{}", self.prelude_code.result(), self.body_code.result())
    }

    /*********************************************************************************************
    //
    // Helpers for Function Code Generation
    //
    *********************************************************************************************/

    /// Given a set of parameters and a suffix, returns a string used to represent LLVM function
    /// arguments. Each argument has the given suffix. Argument names are sorted by their symbol
    /// name.
    fn get_arg_str(&mut self, params: &HashMap<Symbol, Type>, suffix: &str) -> WeldResult<String> {
        let mut arg_types = String::new();
        let params_sorted: BTreeMap<&Symbol, &Type> = params.iter().collect();
        for (arg, ty) in params_sorted.iter() {
            let arg_str = format!("{} {}{}, ", self.llvm_type(&ty)?, llvm_symbol(&arg), suffix);
            arg_types.push_str(&arg_str);
        }
        arg_types.push_str("%work_t* %cur.work");
        Ok(arg_types)
    }

    /// Generates code to unpack a struct containing a set of arguments with the given symbols and
    /// types. The order of arguments is assumed to be sorted by the symbol name.
    fn gen_unload_arg_struct(&mut self, params: &HashMap<Symbol, Type>, ctx: &mut FunctionContext) -> WeldResult<()> {
        let params_sorted: BTreeMap<&Symbol, &Type> = params.iter().collect();
        let ll_ty = self.llvm_type(&Struct(params_sorted.iter().map(|p| p.1.clone()).cloned().collect()))?;
        let storage_typed = ctx.var_ids.next();
        let storage = ctx.var_ids.next();
        let work_data_ptr = ctx.var_ids.next();
        let work_data = ctx.var_ids.next();
        ctx.code.add(format!("{} = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 0", work_data_ptr));
        ctx.code.add(format!("{} = load i8*, i8** {}", work_data, work_data_ptr));
        ctx.code.add(format!("{} = bitcast i8* {} to {}*", storage_typed, work_data, ll_ty));
        ctx.code.add(format!("{} = load {}, {}* {}", storage, ll_ty, ll_ty, storage_typed));
        for (i, (arg, _)) in params_sorted.iter().enumerate() {
            ctx.code.add(format!("{} = extractvalue {} {}, {}", llvm_symbol(arg), ll_ty, storage, i));
        }
        Ok(())
    }

    /// Generates code to create new pieces for the appender.
    fn gen_create_new_vb_pieces(&mut self, params: &HashMap<Symbol, Type>, ctx: &mut FunctionContext) -> WeldResult<()> {
        let full_task_ptr = ctx.var_ids.next();
        let full_task_int = ctx.var_ids.next();
        let full_task_bit = ctx.var_ids.next();
        ctx.code.add(format!("{} = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 4", full_task_ptr));
        ctx.code.add(format!("{} = load i32, i32* {}", full_task_int, full_task_ptr));
        ctx.code.add(format!("{} = trunc i32 {} to i1", full_task_bit, full_task_int));
        ctx.code.add(format!("br i1 {}, label %new_pieces, label %fn_call", full_task_bit));
        ctx.code.add("new_pieces:");
        let params_sorted: BTreeMap<&Symbol, &Type> = params.iter().collect();
        for (arg, ty) in params_sorted.iter() {
            match **ty {
                Builder(ref bk, _) => {
                    match *bk {
                        Appender(_) => {
                            let bld_ty_str = self.llvm_type(ty)?;
                            let bld_prefix = llvm_prefix(&bld_ty_str);
                            ctx.code.add(format!("call void {}.newPiece({} {}, %work_t* %cur.work)",
                                                 bld_prefix,
                                                 bld_ty_str,
                                                 llvm_symbol(arg)));
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        ctx.code.add("br label %fn_call");
        Ok(())
    }

    /// Generates code to create register-based mergers to be used inside innermost loops.
    fn gen_create_new_merger_regs(&mut self, params: &HashMap<Symbol, Type>, ctx: &mut FunctionContext) -> WeldResult<()> {
        let params_sorted: BTreeMap<&Symbol, &Type> = params.iter().collect();
        for (arg, ty) in params_sorted.iter() {
            match **ty {
                Builder(ref bk, _) => {
                    match *bk {
                        Merger(ref val_ty, ref op) => {
                            let bld_ll_ty = self.llvm_type(ty)?;
                            let bld_ll_sym = llvm_symbol(arg);
                            let bld_ll_reg_sym = format!("{}.reg", bld_ll_sym);
                            let bld_ll_reg_ty = format!("{}.inner", bld_ll_ty);
                            ctx.add_alloca(&bld_ll_reg_sym, &bld_ll_reg_ty)?;
                            let val_ll_ty = self.llvm_type(val_ty)?;
                            let iden_elem = binop_identity(*op, val_ty.as_ref())?;
                            let bld_prefix = llvm_prefix(&bld_ll_ty);
                            ctx.code.add(format!(
                                "call void {}.clearPiece({} {}, {} {})",
                                bld_prefix,
                                bld_ll_ty,
                                bld_ll_reg_sym,
                                val_ll_ty,
                                iden_elem));
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Generates code to store register-based mergers back to their parent global mergers.
    fn gen_store_merger_regs(&mut self, params: &HashMap<Symbol, Type>, ctx: &mut FunctionContext) -> WeldResult<()> {
        let params_sorted: BTreeMap<&Symbol, &Type> = params.iter().collect();
        for (arg, ty) in params_sorted.iter() {
            match **ty {
                Builder(ref bk, _) => {
                    match *bk {
                        Merger(ref val_ty, ref op) => {
                            let bld_ll_ty = self.llvm_type(ty)?;
                            let bld_ll_sym = llvm_symbol(arg);
                            let bld_ll_reg_sym = format!("{}.reg", bld_ll_sym);
                            let val_ll_scalar_ty = self.llvm_type(val_ty)?;
                            let val_ll_simd_ty = self.llvm_type(&val_ty.simd_type()?)?;
                            let bld_prefix = llvm_prefix(&bld_ll_ty);
                            let bld_tmp = self.gen_load_var(&bld_ll_sym, &bld_ll_ty, ctx)?;
                            let bld_ptr_raw = ctx.var_ids.next();
                            let bld_ptr_scalar = ctx.var_ids.next();
                            let bld_ptr_simd = ctx.var_ids.next();
                            let reg_ptr_scalar = ctx.var_ids.next();
                            let reg_ptr_simd = ctx.var_ids.next();
                            ctx.code.add(format!(
                                "{} = call {} {}.getPtrIndexed({} {}, i32 %cur.tid)",
                                bld_ptr_raw,
                                bld_ll_ty,
                                bld_prefix,
                                bld_ll_ty,
                                bld_tmp));
                            ctx.code.add(format!(
                                "{} = call {}* {}.vectorMergePtr({} {})",
                                bld_ptr_simd,
                                val_ll_simd_ty,
                                bld_prefix,
                                bld_ll_ty,
                                bld_ptr_raw));
                            ctx.code.add(format!(
                                "{} = call {}* {}.scalarMergePtr({} {})",
                                bld_ptr_scalar,
                                val_ll_scalar_ty,
                                bld_prefix,
                                bld_ll_ty,
                                bld_ptr_raw));
                            ctx.code.add(format!(
                                "{} = call {}* {}.vectorMergePtr({} {})",
                                reg_ptr_simd,
                                val_ll_simd_ty,
                                bld_prefix,
                                bld_ll_ty,
                                bld_ll_reg_sym));
                            ctx.code.add(format!(
                                "{} = call {}* {}.scalarMergePtr({} {})",
                                reg_ptr_scalar,
                                val_ll_scalar_ty,
                                bld_prefix,
                                bld_ll_ty,
                                bld_ll_reg_sym));
                            let reg_simd = self.gen_load_var(&reg_ptr_simd, &val_ll_simd_ty, ctx)?;
                            let reg_scalar = self.gen_load_var(&reg_ptr_scalar, &val_ll_scalar_ty, ctx)?;
                            self.gen_merge_op(&bld_ptr_simd, &reg_simd, &val_ll_simd_ty, op, &val_ty.simd_type()?, ctx)?;
                            self.gen_merge_op(&bld_ptr_scalar, &reg_scalar, &val_ll_scalar_ty, op, val_ty, ctx)?;
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Generates code which, given the arguments with symbols and types, creates a struct with the
    /// arguments. The order of the strut elements is sorted by the symbol names. Returns the LLVM
    /// name of the i8* pointer pointing to the created struct.
    fn gen_create_arg_struct(&mut self, params: &HashMap<Symbol, Type>, ctx: &mut FunctionContext) -> WeldResult<String> {
        let params_sorted: BTreeMap<&Symbol, &Type> = params.iter().collect();
        let mut prev_ref = String::from("undef");
        let ll_ty = self.llvm_type(&Struct(params_sorted.iter().map(|p| p.1.clone()).cloned().collect()))?
            .to_string();
        for (i, (arg, ty)) in params_sorted.iter().enumerate() {
            let next_ref = ctx.var_ids.next();
            ctx.code.add(format!("{} = insertvalue {} {}, {} {}, {}",
                                 next_ref,
                                 ll_ty,
                                 prev_ref,
                                 self.llvm_type(&ty)?,
                                 llvm_symbol(arg),
                                 i));
            prev_ref.clear();
            prev_ref.push_str(&next_ref);
        }
        let struct_size_ptr = ctx.var_ids.next();
        let struct_size = ctx.var_ids.next();
        let struct_storage = ctx.var_ids.next();
        let struct_storage_typed = ctx.var_ids.next();
        ctx.code.add(format!("{} = getelementptr {}, {}* null, i32 1", struct_size_ptr, ll_ty, ll_ty));
        ctx.code.add(format!("{} = ptrtoint {}* {} to i64", struct_size, ll_ty, struct_size_ptr));
        // we use regular malloc here because this pointer will always be freed by parlib
        ctx.code.add(format!("{} = call i8* @malloc(i64 {})", struct_storage, struct_size));
        ctx.code.add(format!("{} = bitcast i8* {} to {}*", struct_storage_typed, struct_storage, ll_ty));
        ctx.code.add(format!("store {} {}, {}* {}", ll_ty, prev_ref, ll_ty, struct_storage_typed));
        Ok(struct_storage)
    }


    /// Generates code to compute the number of iterations the given loop will execute. Returns an
    /// LLVM register name representing the number of iterations and, if the iterator is a
    /// `FringeIter`, the LLVM register name representing the starting index of the fringe iterator.
    ///
    /// Precedes gen_loop_bounds_check
    fn gen_num_iters_and_fringe_start(&mut self,
                                      par_for: &ParallelForData,
                                      func: &SirFunction,
                                      ctx: &mut FunctionContext) -> WeldResult<(String, Option<String>)> {
        // Use the first data to compute the indexing.
        let first_data = &par_for.data[0].data;
        let data_str = llvm_symbol(&first_data);
        let data_ty_str = self.llvm_type(func.params.get(&first_data).unwrap())?;
        let data_prefix = llvm_prefix(&data_ty_str);

        let num_iters_str = ctx.var_ids.next();
        let mut fringe_start_str = None;
        if par_for.data[0].kind == IterKind::SimdIter || par_for.data[0].kind == IterKind::ScalarIter {
            if par_for.data[0].start.is_none() {
                // set num_iters_str to len(first_data)
                ctx.code.add(format!("{} = call i64 {}.size({} {})",
                num_iters_str,
                data_prefix,
                data_ty_str,
                data_str));
            } else {
                // TODO(shoumik): Don't support non-unit stride right now.
                if par_for.data[0].kind == IterKind::SimdIter {
                    return weld_err!("vector iterator does not support non-unit stride");
                }
                // set num_iters_str to (end - start) / stride
                let start_str = llvm_symbol(&par_for.data[0].start.clone().unwrap());
                let end_str = llvm_symbol(&par_for.data[0].end.clone().unwrap());
                let stride_str = llvm_symbol(&par_for.data[0].stride.clone().unwrap());
                let diff_tmp = ctx.var_ids.next();
                ctx.code.add(format!("{} = sub i64 {}, {}", diff_tmp, end_str, start_str));
                ctx.code.add(format!("{} = udiv i64 {}, {}", num_iters_str, diff_tmp, stride_str));
            }
        } else {
            // FringeIter
            // TODO(shoumik): Don't support non-unit stride right now.
            if par_for.data[0].start.is_some() {
                return weld_err!("fringe iterator does not support non-unit stride");
            }
            let arr_len = ctx.var_ids.next();
            let tmp = ctx.var_ids.next();
            let tmp2 = ctx.var_ids.next();
            let vector_len = format!("{}", llvm_simd_size(func.symbol_type(&first_data)?)?);

            ctx
                .code
                .add(format!("{} = call i64 {}.size({} {})", arr_len, data_prefix, data_ty_str, data_str));

            // Compute the number of iterations:
            // tmp = arr_len / llvm_simd_size
            // tmp2 = tmp * llvm_simd_size
            // num_iters = arr_len - llvm_simd_size
            ctx.code.add(format!("{} = udiv i64 {}, {}", tmp, arr_len, vector_len));
            // tmp2 is also where the iteration for the FringeIter starts.
            ctx.code.add(format!("{} = mul i64 {}, {}", tmp2, tmp, vector_len));
            // Compute the number of iterations.
            ctx.code.add(format!("{} = sub i64 {}, {}", num_iters_str, arr_len, tmp2));

            fringe_start_str = Some(tmp2);
        }
        Ok((String::from(num_iters_str), fringe_start_str))
    }

    /// Generates a bounds check for a parallel for loop by ensuring that the number of iterations
    /// does not cause an out of bounds error with the given start and stride.
    ///
    /// Follows gen_num_iters_and_fringe_start
    /// Precedes gen_invoke_loop_body
    fn gen_loop_bounds_check(&mut self,
                                 fringe_start_str: Option<String>,
                                 num_iters_str: &str,
                                 par_for: &ParallelForData,
                                 func: &SirFunction,
                                 ctx: &mut FunctionContext) -> WeldResult<()> {
        for iter in par_for.data.iter() {
            let (data_ll_ty, data_ll_sym) = self.llvm_type_and_name(func, &iter.data)?;
            let data_prefix = llvm_prefix(&data_ll_ty);

            let data_size_ll_tmp = ctx.var_ids.next();
            ctx.code.add(format!("{} = call i64 {}.size({} {})",
                data_size_ll_tmp, data_prefix, data_ll_ty, data_ll_sym));

            // Obtain the start and stride values.
            let (start_str, stride_str) = if iter.start.is_none() {
                // We already checked to make sure the FringeIter doesn't have a start, etc.
                let start_str = match iter.kind {
                    IterKind::FringeIter => fringe_start_str.as_ref().unwrap().to_string(),
                    _ => String::from("0")
                };
                let stride_str = String::from("1");
                (start_str, stride_str)
            } else {
                (llvm_symbol(iter.start.as_ref().unwrap()), llvm_symbol(iter.stride.as_ref().unwrap()))
            };

            let t0 = ctx.var_ids.next();
            let t1 = ctx.var_ids.next();
            let t2 = ctx.var_ids.next();
            let cond = ctx.var_ids.next();
            let next_bounds_check_label = ctx.var_ids.next();

            // TODO just compare against end here...this computation is redundant.
            // t0 = sub i64 num_iters, 1
            // t1 = mul i64 stride, t0
            // t2 = add i64 t1, start
            // cond = icmp lte i64 t1, size
            // br i1 cond, label %nextCheck, label %checkFailed
            // nextCheck:
            // (loop)
            ctx.code.add(format!("{} = sub i64 {}, 1", t0, num_iters_str));
            ctx.code.add(format!("{} = mul i64 {}, {}", t1, stride_str, t0));
            ctx.code.add(format!("{} = add i64 {}, {}", t2, t1, start_str));
            ctx.code.add(format!("{} = icmp slt i64 {}, {}", cond, t2, data_size_ll_tmp));
            ctx
                .code
                .add(format!("br i1 {}, label {}, label %fn.boundcheckfailed", cond, next_bounds_check_label));
            ctx.code.add(format!("{}:", next_bounds_check_label.replace("%", "")));
        }

        // If we get here, the bounds check passed.
        ctx.code.add(format!("br label %fn.boundcheckpassed"));
        // Handle a bounds check fail.
        ctx.code.add(format!("fn.boundcheckfailed:"));
        let errno = WeldRuntimeErrno::BadIteratorLength;
        let run_id = ctx.var_ids.next();
        ctx.code.add(format!("{} = call i64 @weld_rt_get_run_id()", run_id));
        ctx.code.add(format!("call void @weld_run_set_errno(i64 {}, i64 {})", run_id, errno as i64));
        ctx.code.add(format!("call void @weld_rt_abort_thread()"));
        ctx.code.add(format!("; Unreachable!"));
        ctx.code.add(format!("br label %fn.end"));
        ctx.code.add(format!("fn.boundcheckpassed:"));

        Ok(())
    }

    /// Generates code to either call into the parallel runtime or to call the function which
    /// executes the for loops body directly on the current thread, based on the number of
    /// iterations to be executed.
    ///
    /// Follows gen_loop_bounds_check
    fn gen_invoke_loop_body(&mut self,
                          num_iters_str: &str,
                          par_for: &ParallelForData,
                          sir: &SirProgram,
                          func: &SirFunction,
                          ctx: &mut FunctionContext) -> WeldResult<()> {
        let bound_cmp = ctx.var_ids.next();
        let mut grain_size = DEFAULT_GRAIN_SIZE;
        if par_for.innermost {
            // Determine whether to always call parallel, always call serial, or
            // choose based on the loop's size.
            if par_for.always_use_runtime {
                ctx.code.add(format!("br label %for.par"));
            } else {
                if self.multithreaded {
                    ctx.code.add(format!("{} = icmp ule i64 {}, {}", bound_cmp, num_iters_str, grain_size));
                    ctx.code.add(format!("br i1 {}, label %for.ser, label %for.par", bound_cmp));
                } else {
                    ctx.code.add(format!("br label %for.ser"));
                }
            }
            ctx.code.add(format!("for.ser:"));
            let mut body_arg_types = self.get_arg_str(&func.params, "")?;
            body_arg_types.push_str(format!(", i64 0, i64 {}", num_iters_str).as_str());
            ctx.code.add(format!("call void @f{}({})", func.id, body_arg_types));
            let cont_arg_types = self.get_arg_str(&sir.funcs[par_for.cont].params, "")?;
            ctx.code.add(format!("call void @f{}({})", par_for.cont, cont_arg_types));
            ctx.code.add(format!("br label %fn.end"));
        } else {
            ctx.code.add("br label %for.par");
            grain_size = 1;
        }
        ctx.code.add(format!("for.par:"));
        let body_struct = self.gen_create_arg_struct(&func.params, ctx)?;
        let cont_struct = self.gen_create_arg_struct(&sir.funcs[par_for.cont].params, ctx)?;
        ctx.code.add(format!(
                "call void @weld_rt_start_loop(%work_t* %cur.work, i8* {}, i8* {}, \
                                void (%work_t*)* @f{}_par, void (%work_t*)* @f{}_par, i64 0, \
                                i64 {}, i32 {})",
                                body_struct,
                                cont_struct,
                                func.id,
                                par_for.cont,
                                num_iters_str,
                                grain_size
                                ));
        Ok(())
    }

    /// Generates the first half of the loop iteration code, which computes the index to iterate to
    /// and loads data before passing it to the loop body code.
    fn gen_loop_iteration_start(&mut self,
                     par_for: &ParallelForData,
                     func: &SirFunction,
                     ctx: &mut FunctionContext) -> WeldResult<()> {
        let bld_ty_str = self.llvm_type(func.params.get(&par_for.builder).unwrap())?;
        let bld_param_str = llvm_symbol(&par_for.builder);
        let bld_arg_str = llvm_symbol(&par_for.builder_arg);
        ctx.code.add(format!("store {} {}.in, {}* {}", &bld_ty_str, bld_param_str, &bld_ty_str, bld_arg_str));
        ctx.add_alloca("%cur.idx", "i64")?;
        ctx.code.add("store i64 %lower.idx, i64* %cur.idx");
        ctx.code.add("br label %loop.start");
        ctx.code.add("loop.start:");
        let idx_tmp = self.gen_load_var("%cur.idx", "i64", ctx)?;
        if !par_for.innermost {
            let work_idx_ptr = ctx.var_ids.next();
            ctx.code.add(format!(
                    "{} = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 3",
                    work_idx_ptr
                    ));
            ctx.code.add(format!("store i64 {}, i64* {}", idx_tmp, work_idx_ptr));
        }

        let elem_ty = func.locals.get(&par_for.data_arg).unwrap();

        let idx_cmp = ctx.var_ids.next();

        if par_for.data[0].kind == IterKind::SimdIter {
            let check_with_vec = ctx.var_ids.next();
            let vector_len = format!("{}", llvm_simd_size(&elem_ty)?);
            // Would need to compute stride, etc. here.
            ctx.code.add(format!("{} = add i64 {}, {}", check_with_vec, idx_tmp, vector_len));
            ctx.code.add(format!("{} = icmp ule i64 {}, %upper.idx", idx_cmp, check_with_vec));
        } else {
            ctx.code.add(format!("{} = icmp ult i64 {}, %upper.idx", idx_cmp, idx_tmp));
        }
        ctx.code.add(format!("br i1 {}, label %loop.body, label %loop.end", idx_cmp));
        ctx.code.add("loop.body:");
        let mut prev_ref = String::from("undef");
        let elem_ty_str = self.llvm_type(&elem_ty)?;
        for (i, iter) in par_for.data.iter().enumerate() {
            let data_ty_str = self.llvm_type(func.params.get(&iter.data).unwrap())?;
            let data_str = self.gen_load_var(llvm_symbol(&iter.data).as_str(), &data_ty_str, ctx)?;
            let data_prefix = llvm_prefix(&data_ty_str);
            let inner_elem_tmp_ptr = ctx.var_ids.next();
            let inner_elem_ty_str = if par_for.data.len() == 1 {
                elem_ty_str.clone()
            } else {
                match *elem_ty {
                    Struct(ref v) => self.llvm_type(&v[i])?,
                    _ => weld_err!("Internal error: invalid element type {}", print_type(elem_ty))?,
                }
            };

            let arr_idx = if iter.start.is_some() {
                // TODO(shoumik) implement. This needs to be a gather instead of a
                // sequential load.
                if iter.kind == IterKind::SimdIter {
                    return weld_err!("Unimplemented: vectorized iterators do not support non-unit stride.");
                }
                let offset = ctx.var_ids.next();
                let stride_str = self.gen_load_var(llvm_symbol(&iter.stride.clone().unwrap()).as_str(), "i64", ctx)?;
                let start_str = self.gen_load_var(llvm_symbol(&iter.start.clone().unwrap()).as_str(), "i64", ctx)?;
                ctx.code.add(format!("{} = mul i64 {}, {}", offset, idx_tmp, stride_str));
                let final_idx = ctx.var_ids.next();
                ctx.code.add(format!("{} = add i64 {}, {}", final_idx, start_str, offset));
                final_idx
            } else {
                if iter.kind == IterKind::FringeIter {
                    let vector_len = format!("{}", llvm_simd_size(&elem_ty)?);
                    let tmp = ctx.var_ids.next();
                    let arr_len = ctx.var_ids.next();
                    let offset = ctx.var_ids.next();
                    let final_idx = ctx.var_ids.next();

                    ctx.code.add(format!("{} = call i64 {}.size({} {})",
                    arr_len,
                    data_prefix,
                    &data_ty_str,
                    data_str));

                    ctx.code.add(format!("{} = udiv i64 {}, {}", tmp, arr_len, vector_len));

                    // tmp2 is also where the iteration for the FringeIter starts (the
                    // offset).
                    ctx.code.add(format!("{} = mul i64 {}, {}", offset, tmp, vector_len));

                    // Compute the number of iterations.
                    ctx.code.add(format!("{} = add i64 {}, {}", final_idx, offset, idx_tmp));

                    final_idx
                } else {
                    idx_tmp.clone()
                }
            };

            match iter.kind {
                IterKind::ScalarIter | IterKind::FringeIter => {
                    ctx.code.add(format!("{} = call {}* {}.at({} {}, i64 {})",
                    inner_elem_tmp_ptr,
                    &inner_elem_ty_str,
                    data_prefix,
                    &data_ty_str,
                    data_str,
                    arr_idx));
                }
                IterKind::SimdIter => {
                    ctx.code.add(format!("{} = call {}* {}.vat({} {}, i64 {})",
                    inner_elem_tmp_ptr,
                    &inner_elem_ty_str,
                    data_prefix,
                    &data_ty_str,
                    data_str,
                    arr_idx));
                }
            };
            let inner_elem_tmp = self.gen_load_var(&inner_elem_tmp_ptr, &inner_elem_ty_str, ctx)?;
            if par_for.data.len() == 1 {
                prev_ref.clear();
                prev_ref.push_str(&inner_elem_tmp);
            } else {
                let elem_tmp = ctx.var_ids.next();
                ctx.code.add(format!("{} = insertvalue {} {}, {} {}, {}",
                                     elem_tmp,
                                     elem_ty_str,
                                     prev_ref,
                                     inner_elem_ty_str,
                                     inner_elem_tmp,
                                     i));
                prev_ref.clear();
                prev_ref.push_str(&elem_tmp);
            }
        }
        let elem_str = llvm_symbol(&par_for.data_arg);
        ctx.code.add(format!("store {} {}, {}* {}", &elem_ty_str, prev_ref, &elem_ty_str, elem_str));
        ctx.code.add(format!("store i64 {}, i64* {}", idx_tmp, llvm_symbol(&par_for.idx_arg)));
        Ok(())
    }

    /// Generates the second half of the loop iteration code, which updates the current index and
    /// jumps to the beginning of the loop if more iterations are left.
    fn gen_loop_iteration_end(&mut self,
                              par_for: &ParallelForData,
                              func: &SirFunction,
                              ctx: &mut FunctionContext) -> WeldResult<()> {
        // TODO - should take the minimum vector size of all elements here?
        let vectorized = par_for.data[0].kind == IterKind::SimdIter;
        let fetch_width = if vectorized {
            llvm_simd_size(func.locals.get(&par_for.data_arg).unwrap())?
        } else {
            1
        };

        ctx.code.add("br label %loop.terminator");
        ctx.code.add("loop.terminator:");
        let idx_tmp = self.gen_load_var("%cur.idx", "i64", ctx)?;
        let idx_inc = ctx.var_ids.next();
        ctx.code.add(format!("{} = add i64 {}, {}", idx_inc, idx_tmp, format!("{}", fetch_width)));
        ctx.code.add(format!("store i64 {}, i64* %cur.idx", idx_inc));
        ctx.code.add("br label %loop.start");
        ctx.code.add("loop.end:");

        Ok(())
    }

    /// Generates a header common to each top-level generated function.
    fn gen_function_header(&mut self, arg_types: &str, func: &SirFunction, ctx: &mut FunctionContext) -> WeldResult<()> {
        // Start the entry block by defining the function and storing all its arguments on the
        // stack (this makes them consistent with other local variables). Later, expressions may
        // add more local variables to alloca_code.
        ctx.alloca_code.add(format!("define void @f{}({}) {{", func.id, arg_types));
        ctx.alloca_code.add(format!("fn.entry:"));
        for (arg, ty) in func.params.iter() {
            let arg_str = llvm_symbol(&arg);
            let ty_str = self.llvm_type(&ty)?;
            ctx.add_alloca(&arg_str, &ty_str)?;
            ctx.code.add(format!("store {} {}.in, {}* {}", ty_str, arg_str, ty_str, arg_str));
        }
        for (arg, ty) in func.locals.iter() {
            let arg_str = llvm_symbol(&arg);
            let ty_str = self.llvm_type(&ty)?;
            ctx.add_alloca(&arg_str, &ty_str)?;
        }

        // Get the current thread ID
        ctx.code.add(format!("%cur.tid = call i32 @weld_rt_thread_id()"));

        Ok(())
    }

    /*********************************************************************************************
    //
    // Function Code Generation
    //
    *********************************************************************************************/

    /// Generates a wrapper function for a for loop, which performs a bounds check on the loop data
    /// and then invokes the loop body.
    fn gen_loop_wrapper_function(&mut self,
                                 par_for: &ParallelForData,
                                 sir: &SirProgram,
                                 func: &SirFunction) -> WeldResult<()> {

        let ref mut ctx = FunctionContext::new(false);
        let serial_arg_types = self.get_arg_str(&get_combined_params(sir, &par_for), "")?;
        ctx.code.add(format!("define void @f{}_wrapper({}) {{", func.id, serial_arg_types));
        ctx.code.add(format!("fn.entry:"));

        // Compute the number of iterations and the start point of a fringe iter if there is one.
        let (num_iters_str, fringe_start_str) = self.gen_num_iters_and_fringe_start(&par_for, func, ctx)?;
        // Check if the loops are in-bounds and throw an error if they are not.
        self.gen_loop_bounds_check(fringe_start_str, &num_iters_str, &par_for, func, ctx)?;
        // Invoke the loop body (either by directly calling a function or starting the runtime).
        self.gen_invoke_loop_body(&num_iters_str, &par_for, sir, func, ctx)?;

        ctx.code.add(format!("br label %fn.end"));
        ctx.code.add("fn.end:");
        ctx.code.add("ret void");
        ctx.code.add("}\n\n");
        self.body_code.add(&ctx.code.result());

        Ok(())
    }

    /// Generates the parallel runtime callback function for a loop body. This function calls the
    /// function which executes a loop body after unpacking the lower and upper bound from the work
    /// item.
    fn gen_parallel_runtime_callback_function(&mut self, func: &SirFunction) -> WeldResult<()> {
            let mut ctx = &mut FunctionContext::new(false);
            ctx.code.add(format!("define void @f{}_par(%work_t* %cur.work) {{", func.id));
            ctx.code.add("entry:");
            self.gen_unload_arg_struct(&func.params, &mut ctx)?;
            let lower_bound_ptr = ctx.var_ids.next();
            let lower_bound = ctx.var_ids.next();
            let upper_bound_ptr = ctx.var_ids.next();
            let upper_bound = ctx.var_ids.next();
            ctx.code.add(format!("{} = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 1", lower_bound_ptr));
            ctx.code.add(format!("{} = load i64, i64* {}", lower_bound, lower_bound_ptr));
            ctx.code.add(format!("{} = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 2", upper_bound_ptr));
            ctx.code.add(format!("{} = load i64, i64* {}", upper_bound, upper_bound_ptr));

            let body_arg_types = try!(self.get_arg_str(&func.params, ""));
            self.gen_create_new_vb_pieces(&func.params, &mut ctx)?;
            ctx.code.add("fn_call:");
            ctx.code.add(format!("call void @f{}({}, i64 {}, i64 {})",
                                            func.id,
                                            body_arg_types,
                                            lower_bound,
                                            upper_bound));
            ctx.code.add("ret void");
            ctx.code.add("}\n\n");
            self.body_code.add(&ctx.code.result());
            Ok(())
    }

    /// Generates the continuation function of the given parallel loop.
    fn gen_loop_continuation_function(&mut self,
                                      par_for: &ParallelForData,
                                      sir: &SirProgram) -> WeldResult<()> {
            let mut ctx = &mut FunctionContext::new(false);
            ctx.code.add(format!("define void @f{}_par(%work_t* %cur.work) {{", par_for.cont));
            ctx.code.add("entry:");

            self.gen_unload_arg_struct(&sir.funcs[par_for.cont].params, &mut ctx)?;
            self.gen_create_new_vb_pieces(&sir.funcs[par_for.cont].params, &mut ctx)?;

            ctx.code.add("fn_call:");
            let cont_arg_types = self.get_arg_str(&sir.funcs[par_for.cont].params, "")?;
            ctx.code.add(format!("call void @f{}({})", par_for.cont, cont_arg_types));
            ctx.code.add("ret void");
            ctx.code.add("}\n\n");
            self.body_code.add(&ctx.code.result());
            Ok(())
    }

    /// Generates all the functions required to run a single parallel for loop. This includes the
    /// function containing the loop body, a wrapper function to choose whether to execute the
    /// runtime, a callback which the runtime uses to invoke the loop body, and the loop's
    /// continuation.
    pub fn gen_par_for_functions(&mut self, par_for: &ParallelForData, sir: &SirProgram, func: &SirFunction) -> WeldResult<()> {
        // If this loop has been visited/generated, just return.
        if !self.visited.insert(func.id) {
            return Ok(());
        }

        let mut ctx = &mut FunctionContext::new(par_for.innermost);
        let mut arg_types = self.get_arg_str(&func.params, ".in")?;
        // Add the lower and upper index to the standard function params.
        arg_types.push_str(", i64 %lower.idx, i64 %upper.idx");

        self.gen_function_header(&arg_types, func, ctx)?;
        if par_for.innermost {
            self.gen_create_new_merger_regs(&func.locals, &mut ctx)?;
        }
        // Generate the first part of the loop.
        self.gen_loop_iteration_start(par_for, func, ctx)?;
        // Jump to block 0, which is where the loop body starts.
        ctx.code.add(format!("br label %b.b{}", func.blocks[0].id));
        // Generate an expression for the function body.
        self.gen_function_body(sir, func, ctx)?;
        ctx.code.add("body.end:");
        // Generate the second part of the loop, which jumps back to the first.
        self.gen_loop_iteration_end(par_for, func, ctx)?;
        if par_for.innermost {
            self.gen_store_merger_regs(&func.locals, &mut ctx)?;
        }
        ctx.code.add("ret void");
        ctx.code.add("}\n\n");

        self.body_code.add(&ctx.alloca_code.result());
        self.body_code.add(&ctx.code.result());

        // Generate functions which call the continuation and wrapper functions
        // used by the runtime to call the loop body.
        self.gen_loop_wrapper_function(&par_for, sir, func)?;
        self.gen_parallel_runtime_callback_function(func)?;
        self.gen_loop_continuation_function(&par_for, sir)?;

        Ok(())
    }

    /// Generates code for an SIR function which is not a loop body.
    pub fn gen_top_level_function(&mut self, sir: &SirProgram, func: &SirFunction) -> WeldResult<()> {
        if !self.visited.insert(func.id) {
            return Ok(());
        }

        let mut ctx = &mut FunctionContext::new(false);
        let arg_types = self.get_arg_str(&func.params, ".in")?;

        self.gen_function_header(&arg_types, func, ctx)?;
        // Jump to block 0, which is where the loop body starts.
        ctx.code.add(format!("br label %b.b{}", func.blocks[0].id));
        // Generate an expression for the function body.
        self.gen_function_body(sir, func, ctx)?;
        ctx.code.add("body.end:");
        ctx.code.add("ret void");
        ctx.code.add("}\n\n");

        self.body_code.add(&ctx.alloca_code.result());
        self.body_code.add(&ctx.code.result());

        Ok(())
    }

    /// Add a function to the generated program, passing its parameters and return value through
    /// pointers encoded as i64. This is used for the main entry point function into Weld modules
    /// to pass them arbitrary structures.
    pub fn add_function_on_pointers(&mut self, name: &str, sir: &SirProgram) -> WeldResult<()> {
        // First add the function on raw values, which we'll call from the pointer version.
        self.gen_top_level_function(sir, &sir.funcs[0])?;

        // Generates an entry point.
        let mut par_top_ctx = &mut FunctionContext::new(false);
        par_top_ctx.code.add("define void @f0_par(%work_t* %cur.work) {");
        self.gen_unload_arg_struct(&sir.funcs[0].params, &mut par_top_ctx)?;
        let top_arg_types = self.get_arg_str(&sir.funcs[0].params, "")?;
        par_top_ctx.code.add(format!("call void @f0({})", top_arg_types));
        par_top_ctx.code.add("ret void");
        par_top_ctx.code.add("}\n\n");
        self.body_code.add(&par_top_ctx.code.result());

        // Define a struct with all the argument types as fields
        let args_struct = Struct(sir.top_params.iter().map(|a| a.ty.clone()).collect());
        let args_type = self.llvm_type(&args_struct)?;

        let mut run_ctx = &mut FunctionContext::new(false);

        run_ctx.code.add(format!("define i64 @{}(i64 %r.input) {{", name));
        // Unpack the input, which is always struct defined by the type %input_arg_t in prelude.ll.
        run_ctx.code.add(format!("%r.inp_typed = inttoptr i64 %r.input to %input_arg_t*"));
        run_ctx.code.add(format!("%r.inp_val = load %input_arg_t, %input_arg_t* %r.inp_typed"));
        run_ctx.code.add(format!("%r.args = extractvalue %input_arg_t %r.inp_val, 0"));
        run_ctx.code.add(format!("%r.nworkers = extractvalue %input_arg_t %r.inp_val, 1"));
        run_ctx.code.add(format!("%r.memlimit = extractvalue %input_arg_t %r.inp_val, 2"));
        // Code to load args and call function
        run_ctx.code.add(format!(
            "%r.args_typed = inttoptr i64 %r.args to {args_type}*
             %r.args_val = load {args_type}, {args_type}* %r.args_typed",
            args_type = args_type
        ));

        let mut arg_pos_map: HashMap<Symbol, usize> = HashMap::new();
        for (i, a) in sir.top_params.iter().enumerate() {
            arg_pos_map.insert(a.name.clone(), i);
        }
        for (arg, _) in sir.funcs[0].params.iter() {
            let idx = arg_pos_map.get(arg).unwrap();
            run_ctx.code.add(format!("{} = extractvalue {} %r.args_val, {}", llvm_symbol(arg), args_type, idx));
        }
        let run_struct = self.gen_create_arg_struct(&sir.funcs[0].params, &mut run_ctx)?;

        let rid = run_ctx.var_ids.next();
        let errno = run_ctx.var_ids.next();
        let tmp0 = run_ctx.var_ids.next();
        let tmp1 = run_ctx.var_ids.next();
        let tmp2 = run_ctx.var_ids.next();
        let size_ptr = run_ctx.var_ids.next();
        let size = run_ctx.var_ids.next();
        let bytes = run_ctx.var_ids.next();
        let typed_out_ptr = run_ctx.var_ids.next();
        let final_address = run_ctx.var_ids.next();

        run_ctx.code.add(format!(
            "{rid} = call i64 @weld_run_begin(void (%work_t*)* @f0_par, i8* {run_struct}, i64 %r.memlimit, i32 %r.nworkers)
             %res_ptr = call i8* @weld_run_get_result(i64 {rid})
             %res_address = ptrtoint i8* %res_ptr to i64
             {errno} = call i64 @weld_run_get_errno(i64 {rid})
             {tmp0} = insertvalue %output_arg_t undef, i64 %res_address, 0
             {tmp1} = insertvalue %output_arg_t {tmp0}, i64 {rid}, 1
             {tmp2} = insertvalue %output_arg_t {tmp1}, i64 {errno}, 2
             {size_ptr} = getelementptr %output_arg_t, %output_arg_t* null, i32 1
             {size} = ptrtoint %output_arg_t* {size_ptr} to i64
             {bytes} = call i8* @malloc(i64 {size})
             {typed_out_ptr} = bitcast i8* {bytes} to %output_arg_t*
             store %output_arg_t {tmp2}, %output_arg_t* {typed_out_ptr}
             {final_address} = ptrtoint %output_arg_t* {typed_out_ptr} to i64
             ret i64 {final_address}",
            run_struct = run_struct,
            rid = rid,
            errno = errno,
            tmp0 = tmp0,
            tmp1 = tmp1,
            tmp2 = tmp2,
            size_ptr = size_ptr,
            size = size,
            bytes = bytes,
            typed_out_ptr = typed_out_ptr,
            final_address = final_address
        ));
        run_ctx.code.add("}\n\n");

        self.body_code.add(&run_ctx.code.result());
        Ok(())
    }

    /*********************************************************************************************
    //
    // Utilities
    //
    *********************************************************************************************/

    /// Returns the LLVM type (as a string), and LLVM symbol name (as a string). This function is
    /// only valid for Symbols defined in a SirFunction.
    fn llvm_type_and_name<'a>(&mut self, func: &'a SirFunction, sym: &Symbol) -> WeldResult<(String, String)> {
        let ty = func.symbol_type(sym)?;
        let llvm_ty = self.llvm_type(ty)?;
        let llvm_str = llvm_symbol(sym);
        Ok((llvm_ty, llvm_str))
    }

    /// Return the LLVM type name corresponding to a Weld type.
    fn llvm_type(&mut self, ty: &Type) -> WeldResult<String> {
        Ok(match *ty {
            Scalar(kind) => llvm_scalar_kind(kind).to_string(),
            Simd(kind) => format!("<{} x {}>", llvm_simd_size(&Scalar(kind))?, llvm_scalar_kind(kind)),

            Struct(ref fields) => {
                if self.struct_names.get(fields) == None {
                    self.gen_struct_definition(fields)?
                }
                self.struct_names.get(fields).unwrap().to_string()
            }

            Vector(ref elem) => {
                if self.vec_names.get(elem) == None {
                    self.gen_vector_definition(elem)?
                }
                self.vec_names.get(elem).unwrap().to_string()
            }

            Dict(ref key, ref value) => {
                let elem = Box::new(Struct(vec![*key.clone(), *value.clone()]));
                if self.dict_names.get(&elem) == None {
                    self.gen_dict_definition(key, value)?;
                    // Generate a hash function and equality function for the key.
                    self.gen_hash(key)?;
                    self.gen_eq(key)?;
                }
                self.dict_names.get(&elem).unwrap().to_string()
            }

            Builder(ref bk, _) => {
                if self.bld_names.get(bk) == None {
                    self.gen_builder_definition(bk)?
                }
                self.bld_names.get(bk).unwrap().to_string()
            }

            _ => {
                return weld_err!("Unsupported type {}", print_type(ty))?
            }
        })
    }

    /*********************************************************************************************
    //
    // Routines for Code Generation of Statements and Blocks.
    //
    *********************************************************************************************/


    /// Generates a `cmp` function for `ty` and any nested types it depends on.
    fn gen_cmp(&mut self, ty: &Type) -> WeldResult<()> {
        // If we've already generated a function for this type, return.
        { 
            let helper_state = self.type_helpers.entry(ty.clone()).or_insert(HelperState::new());
            if helper_state.cmp_func {
                return Ok(());
            }
            helper_state.cmp_func = true;
        } // these braces are necessary so the borrow for `helper_state` ends.

        // Make sure the type is generated.
        let _ = self.llvm_type(ty)?;
        match *ty {
            Struct(ref fields) => {
                // Create comparison functions for each of the nested types.
                let mut field_types: Vec<String> = Vec::new();
                for f in fields.iter() {
                    self.gen_cmp(f)?;
                    field_types.push(self.llvm_type(f)?.to_string());
                }
                // Then, create the comparison function for the full struct, which just compares
                // each field.
                let name = self.struct_names.get(fields).unwrap();
                self.prelude_code.add_line(format!(
                        "define i32 {}.cmp({} %a, {} %b) {{", name.replace("%", "@"), name, name));
                let mut label_ids = IdGenerator::new("%l");
                for i in 0..field_types.len() {
                    if let Simd(_) = fields[i] {
                        continue;
                    }
                    let a_field = self.prelude_var_ids.next();
                    let b_field = self.prelude_var_ids.next();
                    let cmp = self.prelude_var_ids.next();
                    let ne = self.prelude_var_ids.next();
                    let field_ty_str = &field_types[i];
                    let ret_label = label_ids.next();
                    let post_label = label_ids.next();
                    let field_prefix_str = format!("@{}", field_ty_str.replace("%", ""));
                    self.prelude_code.add_line(format!("{} = extractvalue {} %a , {}", a_field, name, i));
                    self.prelude_code.add_line(format!("{} = extractvalue {} %b, {}", b_field, name, i));
                    self.prelude_code.add_line(format!("{} = call i32 {}.cmp({} {}, {} {})",
                    cmp,
                    field_prefix_str,
                    field_ty_str,
                    a_field,
                    field_ty_str,
                    b_field));
                    self.prelude_code.add_line(format!("{} = icmp ne i32 {}, 0", ne, cmp));
                    self.prelude_code.add_line(format!("br i1 {}, label {}, label {}", ne, ret_label, post_label));
                    self.prelude_code.add_line(format!("{}:", ret_label.replace("%", "")));
                    self.prelude_code.add_line(format!("ret i32 {}", cmp));
                    self.prelude_code.add_line(format!("{}:", post_label.replace("%", "")));
                }
                self.prelude_code.add_line(format!("ret i32 0"));
                self.prelude_code.add_line(format!("}}"));
                self.prelude_code.add_line(format!(""));
            }
            Scalar(ref scalar_ty) | Simd(ref scalar_ty) => {
                let ll_ty = self.llvm_type(ty)?;
                let ll_prefix = ll_ty.replace("%", "");
                let ll_cmp = if scalar_ty.is_float() {
                    "fcmp"
                } else {
                    "icmp"
                };

                let ll_eq = llvm_eq(*scalar_ty);
                let ll_lt = llvm_lt(*scalar_ty);

                // Booleans are special cased.
                if *scalar_ty != ScalarKind::Bool {
                    self.prelude_code.add_line(format!("
                    define i32 @{ll_prefix}.cmp({ll_ty} %a, {ll_ty} %b) alwaysinline {{
                      %1 = {ll_cmp} {ll_eq} {ll_ty} %a, %b
                      br i1 %1, label %eq, label %ne
                    eq:
                      ret i32 0
                    ne:
                      %2 = {ll_cmp} {ll_lt} {ll_ty} %a, %b
                      %3 = select i1 %2, i32 -1, i32 1
                      ret i32 %3
                  }}",
                    ll_prefix=ll_prefix, ll_ty=ll_ty, ll_cmp=ll_cmp, ll_eq=ll_eq, ll_lt=ll_lt));
                } else {
                    self.prelude_code.add(format!("
                    define i32 @i1.cmp(i1 %a, i1 %b) {{
                      %1 = icmp eq i1 %a, %b
                      br i1 %1, label %eq, label %ne
                    eq:
                      ret i32 0
                    ne:
                      %2 = select i1 %b, i32 -1, i32 1
                      ret i32 %2
                    }}"));
                }
            }
            Dict(_, _) => {
                // Create a dummy comparison function for structs with the builder as a field.
                let dict_name = self.dict_names.get(ty).unwrap();
                self.prelude_code.add(format!("
                define i32 @{NAME}.cmp(%{NAME} %dict1, %{NAME} %dict2) {{
                  ret i32 -1
                }}", NAME=dict_name.replace("%", "")));
            }
            Builder(ref bk, _) => {
                // Create a dummy comparison function for structs with the builder as a field.
                let bld_name = self.bld_names.get(bk).unwrap();
                self.prelude_code.add(format!("
                define i32 @{NAME}.cmp(%{NAME} %bld1, %{NAME} %bld2) {{
                  ret i32 -1
                }}", NAME=bld_name.replace("%", "")));
            }
            Vector(ref elem) => {
                self.gen_cmp(elem)?;
                self.gen_cmp(&Scalar(ScalarKind::I64))?;
                // For vectors of unsigned chars, we can use memcmp, but for anything else we need
                // element-by-element comparison.
                let elem_ty = self.llvm_type(elem)?;
                let elem_prefix = llvm_prefix(&elem_ty);
                let name = self.vec_names.get(elem).unwrap();
                if let Scalar(ScalarKind::U8) = *elem.as_ref() {
                    self.prelude_code.add(format!(
                            include_str!("resources/vector/vector_comparison_memcmp.ll"),
                            NAME=&name.replace("%", "")));
                } else {
                    self.prelude_code.add(format!(
                            include_str!("resources/vector/vector_comparison.ll"),
                            ELEM_PREFIX=&elem_prefix,
                            ELEM=&elem_ty,
                            NAME=&name.replace("%", "")));
                }
                // Set this flag to true as well, since these templates also generate `eq`.
                let mut helper_state = self.type_helpers.get_mut(ty).unwrap();
                helper_state.eq_func = true;
            }
            _ => {
                return weld_err!("Unsupported function `cmp` for type {:?}", ty);
            }
        };
        Ok(())
    }

    /// Generates an `eq` function for `ty` and any nested types it depends on.
    fn gen_eq(&mut self, ty: &Type) -> WeldResult<()> {
        // If we've already generated a function for this type, return.
        { 
            let helper_state = self.type_helpers.entry(ty.clone()).or_insert(HelperState::new());
            if helper_state.eq_func {
                return Ok(());
            }
            helper_state.eq_func = true;
        } // these braces are necessary so the borrow for `helper_state` ends.

        // Make sure the type is generated.
        let _ = self.llvm_type(ty)?;
        match *ty {
            Struct(ref fields) => {
                // Create comparison functions for each of the nested types.
                let mut field_types: Vec<String> = Vec::new();
                for f in fields.iter() {
                    self.gen_eq(f)?;
                    field_types.push(self.llvm_type(f)?.to_string());
                }
                let name = self.struct_names.get(fields).unwrap();
                self.prelude_code.add_line(format!(
                        "define i1 {}.eq({} %a, {} %b) {{", name.replace("%", "@"), name, name));
                let mut label_ids = IdGenerator::new("%l");
                for i in 0..field_types.len() {
                    let a_field = self.prelude_var_ids.next();
                    let b_field = self.prelude_var_ids.next();
                    let this_eq = self.prelude_var_ids.next();
                    let field_ty_str = &field_types[i];
                    let field_prefix_str = format!("@{}", field_ty_str.replace("%", ""));
                    self.prelude_code.add_line(format!("{} = extractvalue {} %a , {}", a_field, name, i));
                    self.prelude_code.add_line(format!("{} = extractvalue {} %b, {}", b_field, name, i));
                    self.prelude_code.add_line(format!("{} = call i1 {}.eq({} {}, {} {})",
                        this_eq,
                        field_prefix_str,
                        field_ty_str,
                        a_field,
                        field_ty_str,
                        b_field));
                    let on_ne = label_ids.next();
                    let on_eq = label_ids.next();
                    self.prelude_code.add_line(format!("br i1 {}, label {}, label {}", this_eq, on_eq, on_ne));
                    self.prelude_code.add_line(format!("{}:", on_ne.replace("%", "")));
                    self.prelude_code.add_line(format!("ret i1 0"));
                    self.prelude_code.add_line(format!("{}:", on_eq.replace("%", "")));
                }
                self.prelude_code.add_line(format!("ret i1 1"));
                self.prelude_code.add_line(format!("}}"));
                self.prelude_code.add_line(format!(""));
            }
            Scalar(ref scalar_ty) | Simd(ref scalar_ty) => {
                let ll_ty = self.llvm_type(ty)?;
                let ll_prefix = ll_ty.replace("%", "");
                let ll_cmp = if scalar_ty.is_float() {
                    "fcmp"
                } else {
                    "icmp"
                };

                let ll_eq = llvm_eq(*scalar_ty);

                self.prelude_code.add_line(format!("
                    define i1 @{ll_prefix}.eq({ll_ty} %a, {ll_ty} %b) alwaysinline {{
                      %1 = {ll_cmp} {ll_eq} {ll_ty} %a, %b
                      ret i1 %1
                    }}",
                    ll_prefix=ll_prefix, ll_ty=ll_ty, ll_cmp=ll_cmp, ll_eq=ll_eq));
            }
            Dict(_, _) => {
                // Create a dummy comparison function for structs with the builder as a field.
                let dict_name = self.dict_names.get(ty).unwrap();
                self.prelude_code.add(format!("
                define i1 @{NAME}.eq(%{NAME} %dict1, %{NAME} %dict2) {{
                  ret i1 0 
                }}", NAME=dict_name.replace("%", "")));
            }
            Builder(ref bk, _) => {
                // Create a dummy comparison function for structs with the builder as a field.
                let bld_name = self.bld_names.get(bk).unwrap();
                self.prelude_code.add(format!("
                define i1 @{NAME}.cmp(%{NAME} %bld1, %{NAME} %bld2) {{
                  ret i1 0 
                }}", NAME=bld_name.replace("%", "")));
            }
            Vector(_) => {
                // The comparison function template generates equality functions.
                self.gen_cmp(ty)?;
            }
            _ => {
                return weld_err!("Unsupported function `eq` for type {:?}", ty);
            }
        };
        Ok(())
    }

    /// Generates a `hash` function for `ty` and any nested types it depends on.
    fn gen_hash(&mut self, ty: &Type) -> WeldResult<()> {
        // If we've already generated a function for this type, return.
        { 
            let helper_state = self.type_helpers.entry(ty.clone()).or_insert(HelperState::new());
            if helper_state.hash_func {
                return Ok(());
            }
            helper_state.hash_func = true;
        } // these braces are necessary so the borrow for `helper_state` ends.

        // Make sure the type is generated.
        let _ = self.llvm_type(ty)?;
        match *ty {
            Struct(ref fields) => {
                // Create comparison functions for each of the nested types.
                let mut field_types: Vec<String> = Vec::new();
                for f in fields.iter() {
                    self.gen_hash(f)?;
                    field_types.push(self.llvm_type(f)?.to_string());
                }
                // Then, create the comparison function for the full struct, which just compares
                // each field.
                let name = self.struct_names.get(fields).unwrap();

                // Generate hash function for the struct.
                self.prelude_code
                    .add_line(format!("define i32 {}.hash({} %value) {{", name.replace("%", "@"), name));
                let mut res = "0".to_string();
                for i in 0..field_types.len() {
                    // TODO(shoumik): hack to prevent incorrect code gen for vectors.
                    if let Simd(_) = fields[i] {
                        continue;
                    }
                    let field = self.prelude_var_ids.next();
                    let hash = self.prelude_var_ids.next();
                    let new_res = self.prelude_var_ids.next();
                    let field_ty_str = &field_types[i];
                    let field_prefix_str = format!("@{}", field_ty_str.replace("%", ""));
                    self.prelude_code.add_line(format!("{} = extractvalue {} %value, {}", field, name, i));
                    self.prelude_code.add_line(format!("{} = call i32 {}.hash({} {})",
                    hash,
                    field_prefix_str,
                    field_ty_str,
                    field));
                    self.prelude_code
                        .add_line(format!("{} = call i32 @hash_combine(i32 {}, i32 {})", new_res, res, hash));
                    res = new_res;
                }
                self.prelude_code.add_line(format!("ret i32 {}", res));
                self.prelude_code.add_line(format!("}}"));
                self.prelude_code.add_line(format!(""));
            }
            Scalar(_) | Simd(_) => {
                // These are pre-generated in the prelude.
            }
            Dict(_, _) => {
                // Create a dummy comparison function for structs with the builder as a field.
                let dict_name = self.dict_names.get(ty).unwrap();
                self.prelude_code.add(format!("
                define i32 @{NAME}.hash(%{NAME} %dict1) {{
                  ret i32 0
                }}", NAME=dict_name.replace("%", "")));
            }
            Builder(ref bk, _) => {
                // Create a dummy comparison function for structs with the builder as a field.
                let bld_name = self.bld_names.get(bk).unwrap();
                self.prelude_code.add(format!("
                define i32 @{NAME}.hash(%{NAME} %bld1) {{
                  ret i32 0
                }}", NAME=bld_name.replace("%", "")));
            }
            Vector(ref elem) => {
                self.gen_hash(elem)?;
                let elem_ty = self.llvm_type(elem)?;
                let elem_prefix = llvm_prefix(&elem_ty);
                let name = self.vec_names.get(elem).unwrap();
                    self.prelude_code.add(format!(
                            include_str!("resources/vector/vector_hash.ll"),
                            ELEM_PREFIX=&elem_prefix,
                            ELEM=&elem_ty,
                            NAME=&name.replace("%", "")));
            }
            _ => {
                return weld_err!("Unsupported function `hash` for type {:?}", ty);
            }
        };
        Ok(())
    }

    /// Generates a struct definition for the given field types.
    fn gen_struct_definition(&mut self, fields: &Vec<Type>) -> WeldResult<()> {
        // Declare the struct in prelude_code
        let name = self.struct_ids.next();
        let mut field_types: Vec<String> = Vec::new();
        for f in fields {
            field_types.push(try!(self.llvm_type(f)).to_string());
        }
        let field_types_str = field_types.join(", ");
        self.prelude_code.add(format!("{} = type {{ {} }}", name, field_types_str));

        // Add it into our map so we remember its name
        self.struct_names.insert(fields.clone(), name);
        Ok(())
    }

    /// Generates a vector definition with the given type.
    fn gen_vector_definition(&mut self, elem: &Type) -> WeldResult<()> {
        let elem_ty = self.llvm_type(elem)?;
        let name = self.vec_ids.next();
        self.vec_names.insert(elem.clone(), name.clone());

        self.prelude_code.add(format!(
            include_str!("resources/vector/vector.ll"),
            ELEM=&elem_ty,
            NAME=&name.replace("%", "")));
        self.prelude_code.add("\n");

        // If the vector contains scalars only, add in SIMD extensions.
        if elem.is_scalar() {
            self.prelude_code.add(format!(
                include_str!("resources/vector/vvector.ll"),
                ELEM=elem_ty,
                NAME=&name.replace("%", ""),
                VECSIZE=&format!("{}", llvm_simd_size(elem)?)));
            self.prelude_code.add("\n");
        }
        Ok(())
    }

    /// Generates a dictionary definition with the given element type, key type `key`, and  value
    /// type `value`.
    fn gen_dict_definition(&mut self, key: &Type, value: &Type) -> WeldResult<()> {
        let elem = Box::new(Struct(vec![key.clone(), value.clone()]));
        let key_ty = self.llvm_type(key)?;
        let value_ty = self.llvm_type(value)?;
        let key_prefix = llvm_prefix(&key_ty);
        let name = self.dict_ids.next();
        self.dict_names.insert(*elem.clone(), name.clone());
        let kv_struct_ty = self.llvm_type(&elem)?;
        let kv_vec = Box::new(Vector(elem.clone()));
        let kv_vec_ty = self.llvm_type(&kv_vec)?;
        let kv_vec_prefix = llvm_prefix(&&kv_vec_ty);

        let dict_def = format!(include_str!("resources/dictionary.ll"),
            NAME=&name.replace("%", ""),
            KEY=&key_ty,
            KEY_PREFIX=&key_prefix,
            VALUE=&value_ty,
            KV_STRUCT=&kv_struct_ty,
            KV_VEC_PREFIX=&kv_vec_prefix,
            KV_VEC=&kv_vec_ty);

        self.prelude_code.add(&dict_def);
        self.prelude_code.add("\n");
        Ok(())
    }

    /// Generates a builder definition with the given type.
    fn gen_builder_definition(&mut self, bk: &BuilderKind) -> WeldResult<()> {
        match *bk {
            Appender(ref t) => {
                let bld_ty = Vector(t.clone());
                let bld_ty_str = self.llvm_type(&bld_ty)?;
                self.bld_names.insert(bk.clone(), format!("{}.bld", bld_ty_str));
            }
            Merger(ref t, _) => {
                if self.merger_names.get(t) == None {
                    let elem_ty = self.llvm_type(t)?;
                    let name = self.merger_ids.next();
                    self.merger_names.insert(*t.clone(), name.clone());

                    let merger_def = format!(include_str!("resources/merger/merger.ll"),
                        ELEM=&elem_ty,
                        VECSIZE=&format!("{}", llvm_simd_size(t)?),
                        NAME=&name.replace("%", ""));

                    self.prelude_code.add(&merger_def);
                    self.prelude_code.add("\n");
                }
                let bld_ty_str = self.merger_names.get(t).unwrap();
                self.bld_names.insert(bk.clone(), format!("{}.bld", bld_ty_str));
            }
            DictMerger(ref kt, ref vt, ref op) => {
                let bld_ty = Dict(kt.clone(), vt.clone());
                let bld_ty_str = self.llvm_type(&bld_ty)?;
                let elem = Box::new(Struct(vec![*kt.clone(), *vt.clone()]));
                let kv_struct_ty = self.llvm_type(&elem)?;
                let key_ty = self.llvm_type(kt)?;
                let value_ty = self.llvm_type(vt)?;
                let kv_vec = Box::new(Vector(elem.clone()));
                let kv_vec_ty = self.llvm_type(&kv_vec)?;
                let kv_vec_prefix = llvm_prefix(&&kv_vec_ty);

                let dictmerger_def = format!(include_str!("resources/dictmerger.ll"),
                    NAME=&bld_ty_str.replace("%", ""),
                    KEY=&key_ty,
                    VALUE=&value_ty,
                    KV_STRUCT=&kv_struct_ty.replace("%", ""),
                    KV_VEC_PREFIX=&kv_vec_prefix,
                    KV_VEC=&kv_vec_ty);

                self.prelude_code.add(&dictmerger_def);
                self.prelude_code.add("\n");

                // Add the merge_op function for this dictmerger
                let mut var_ids = IdGenerator::new("%t");
                let mut merge_op_code = CodeBuilder::new();
                merge_op_code.add(format!(
                    "define {} @{}.bld.merge_op({} %a, {} %b) alwaysinline {{",
                    value_ty, bld_ty_str.replace("%", ""), value_ty, value_ty));
                let res = self.gen_merge_op_on_registers(
                    "%a", "%b", op, vt, &mut var_ids, &mut merge_op_code)?;
                merge_op_code.add(format!("ret {} {}", value_ty, res));
                merge_op_code.add("}");
                self.prelude_code.add_code(&merge_op_code);
                self.prelude_code.add("\n");

                self.bld_names.insert(bk.clone(), format!("{}.bld", bld_ty_str));
            }
            GroupMerger(ref kt, ref vt) => {
                let elem = Box::new(Struct(vec![*kt.clone(), *vt.clone()]));
                let bld_ty = Vector(elem.clone());
                let bld_ty_str = self.llvm_type(&bld_ty)?;
                self.bld_names.insert(bk.clone(), format!("{}.bld", bld_ty_str));
            }
            VecMerger(ref elem, _) => {
                let bld_ty = Vector(elem.clone());
                let bld_ty_str = self.llvm_type(&bld_ty)?;
                self.bld_names.insert(bk.clone(), format!("{}.vm.bld", bld_ty_str));
            }
        }
        Ok(())
    }

    /// Generate code to load a symbol sym with LLVM type ty into a local variable, and return the variable's name.
    fn gen_load_var(&mut self, sym: &str, ty: &str, ctx: &mut FunctionContext) -> WeldResult<String> {
        let var = ctx.var_ids.next();
        // Hacky...but need an aligned load for vectors to prevent strange segfaults.
        let is_vector = ty.contains("<") && ty.contains(">") && ty.contains("x");
        if is_vector {
            ctx.code.add(format!("{} = load {}, {}* {}, align 1", var, ty, ty, sym));
        } else {
            ctx.code.add(format!("{} = load {}, {}* {}", var, ty, ty, sym));
        }
        Ok(var)
    }

    /// Generate code to store data with LLVM type ty into a a target.
    fn gen_store_var(&mut self, data: &str, target: &str, ty: &str, ctx: &mut FunctionContext) {
        ctx.code.add(format!("store {} {}, {}* {}", ty, data, ty, target));
    }

    /// Generates code for a SIMD literal with the given `kind` and `ty`. The result is stored in
    /// `output`.
    fn gen_simd_literal(&mut self,
                               output: &str,
                               kind: &LiteralKind,
                               vec_ty: &Type,
                               ctx: &mut FunctionContext)
                               -> WeldResult<()> {

        let size = llvm_simd_size(vec_ty)?;
        let vec_ty_str = self.llvm_type(vec_ty)?;
        let size_str = format!("{}", size);

        let value_str = llvm_literal(*kind);
        let elem_ty_str = match *kind {
            BoolLiteral(_) => "bool",
            I8Literal(_) => "i8",
            I16Literal(_) => "i16",
            I32Literal(_) => "i32",
            I64Literal(_) => "i64",
            U8Literal(_) => "i8",
            U16Literal(_) => "i16",
            U32Literal(_) => "i32",
            U64Literal(_) => "i64",
            F32Literal(_) => "float",
            F64Literal(_) => "double",
        }.to_string();

        let insert_str = format!("insertelement <{size} x {elem}> $NAME, {elem} {value}, i32 $INDEX",
                                 size=size_str, elem=elem_ty_str, value=value_str);

        let mut prev_name = "undef".to_string();
        for i in 0..size {
            let replaced = insert_str.replace("$NAME", &prev_name);
            let replaced = replaced.replace("$INDEX", format!("{}", i).as_str());
            let name = ctx.var_ids.next().to_string();
            ctx.code.add(format!("{} = {}", name, replaced));
            prev_name = name;
        }

        self.gen_store_var(&prev_name, output, &vec_ty_str, ctx);
        Ok(())
    }

    /// Generate code to perform a binary operation on two registers, returning the name of a new register
    /// that will hold the value. For structs, this will perform the same merge operation on each element
    /// of the struct, regardless of its type. Takes an IdGenerator for variable names.
    fn gen_merge_op_on_registers(&mut self,
                                 arg1: &str,
                                 arg2: &str,
                                 bin_op: &BinOpKind,
                                 arg_ty: &Type,
                                 var_ids: &mut IdGenerator,
                                 code: &mut CodeBuilder)
                                 -> WeldResult<String> {
        let llvm_ty = self.llvm_type(arg_ty)?.to_string();
        let mut res = var_ids.next();

        match *arg_ty {
            Scalar(_) | Simd(_) => {
                code.add(format!("{} = {} {} {}, {}",
                    &res, try!(llvm_binop(*bin_op, arg_ty)), &llvm_ty, arg1, arg2));
            }

            Struct(ref fields) => {
                res = "undef".to_string();
                for (i, ty) in fields.iter().enumerate() {
                    let arg1_elem = var_ids.next();
                    let arg2_elem = var_ids.next();
                    let new_res = var_ids.next();
                    let elem_ty_str = try!(self.llvm_type(ty)).to_string();
                    code.add(format!("{} = extractvalue {} {}, {}", &arg1_elem, &llvm_ty, &arg1, i));
                    code.add(format!("{} = extractvalue {} {}, {}", &arg2_elem, &llvm_ty, &arg2, i));
                    let elem_res = self.gen_merge_op_on_registers(
                        &arg1_elem, &arg2_elem, &bin_op, &ty, var_ids, code)?;
                    code.add(format!("{} = insertvalue {} {}, {} {}, {}",
                        &new_res, &llvm_ty, &res, &elem_ty_str, &elem_res, i));
                    res = new_res;
                }
            }

            _ => return weld_err!("gen_merge_op_on_registers called on invalid type {}", print_type(arg_ty))
        }

        Ok(res)
    }

    /// Given a pointer to a result variable (e.g. from a builder), generates code to merge a value
    /// into it using a binary operation. `result_ptr` is the pointer into which the original value
    /// is read and the updated value will be stored. `merge_value` is the value to merge in, which
    /// should be a value in a register.
    fn gen_merge_op(&mut self,
                    builder_ptr: &str,
                    merge_value: &str,
                    merge_ty_str: &str,
                    bin_op: &BinOpKind,
                    merge_ty: &Type,
                    ctx: &mut FunctionContext) -> WeldResult<()> {
        let builder_value = self.gen_load_var(&builder_ptr, &merge_ty_str, ctx)?;
        let result_reg = self.gen_merge_op_on_registers(
            &builder_value, &merge_value, &bin_op, &merge_ty, &mut ctx.var_ids, &mut ctx.code)?;
        self.gen_store_var(&result_reg, &builder_ptr, &merge_ty_str, ctx);
        Ok(())
    }


    /// Generate code to perform a unary operation on `child` and store the result in `output` (which should
    /// be a location on the stack).
    fn gen_unary_op(&mut self,
                    ctx: &mut FunctionContext,
                    func: &SirFunction,
                    output: &Symbol,
                    child: &Symbol,
                    op_kind: UnaryOpKind)
                    -> WeldResult<()> {
        let (child_ll_ty, child_ll_sym) = self.llvm_type_and_name(func, child)?;
        let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
        let child_ty = func.symbol_type(child)?;

        let child_tmp = self.gen_load_var(&child_ll_sym, &child_ll_ty, ctx)?;

        if let Scalar(ref ty) = *child_ty {
            let res_tmp = ctx.var_ids.next();
            let op_name = llvm_scalar_unaryop(op_kind, ty)?;
            ctx.code.add(format!("{} = call {} {}({} {})", res_tmp, child_ll_ty, op_name, child_ll_ty, child_tmp));
            self.gen_store_var(&res_tmp, &output_ll_sym, &output_ll_ty, ctx);
        }
        else if let Simd(ref ty) = *child_ty {
            let width = llvm_simd_size(child_ty)?;
            // If an intrinsic exists for this SIMD op, use it.
            if let Ok(op_name) = llvm_simd_unaryop(op_kind, ty, width) {
                let res_tmp = ctx.var_ids.next();
                ctx.code.add(format!("{} = call {} {}({} {})", res_tmp, child_ll_ty, op_name, child_ll_ty, child_tmp));
                self.gen_store_var(&res_tmp, &output_ll_sym, &output_ll_ty, ctx);
            } else {
                // Unroll and apply the scalar op, and then pack back into vector
                let scalar_ll_ty = self.llvm_type(&Scalar(*ty))?;
                let op_name = llvm_scalar_unaryop(op_kind, ty)?;
                let mut prev_tmp = "undef".to_string();
                for i in 0..width {
                    let elem_tmp = self.gen_simd_extract(child_ty, &child_tmp, i, ctx)?;
                    let val_tmp = ctx.var_ids.next();
                    ctx.code.add(format!("{} = call {} {}({} {})", val_tmp, scalar_ll_ty, op_name, scalar_ll_ty, elem_tmp));
                    let next = ctx.var_ids.next();
                    ctx.code.add(format!("{} = insertelement {} {}, {} {}, i32 {}",
                                         next, child_ll_ty, prev_tmp, scalar_ll_ty, val_tmp, i));
                    prev_tmp = next;
                }
                self.gen_store_var(&prev_tmp, &output_ll_sym, &output_ll_ty, ctx);
            }
        } else {
            weld_err!("Illegal type {} in {}", print_type(child_ty), op_kind)?;
        }
        Ok(())
    }

    /// Generate code for a function and append it to its FunctionContext.
    fn gen_function_body(&mut self, sir: &SirProgram, func: &SirFunction, ctx: &mut FunctionContext) -> WeldResult<()> {
        for b in func.blocks.iter() {
            ctx.code.add(format!("b.b{}:", b.id));
            for s in b.statements.iter() {
                self.gen_statement(s, func, ctx)?
            }
            self.gen_terminator(&b.terminator, sir, func, ctx)?
        }
        Ok(())
    }

    /// Generate code for a single statement, appending it to the code in a FunctionContext.
    fn gen_statement(&mut self, statement: &Statement, func: &SirFunction, ctx: &mut FunctionContext) -> WeldResult<()> {
        ctx.code.add(format!("; {}", statement));
        match *statement {
            MakeStruct { ref output, ref elems } => {
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                for (index, elem) in elems.iter().enumerate() {
                    let (elem_ll_ty, elem_ll_sym) = self.llvm_type_and_name(func, elem)?;
                    let elem_value = self.gen_load_var(&elem_ll_sym, &elem_ll_ty, ctx)?;
                    let ptr_tmp = ctx.var_ids.next();
                    ctx.code.add(format!("{} = getelementptr inbounds {}, {}* {}, i32 0, i32 {}",
                        ptr_tmp, output_ll_ty, output_ll_ty, output_ll_sym, index));
                    self.gen_store_var(&elem_value, &ptr_tmp, &elem_ll_ty, ctx);
                }
            }

            CUDF { ref output, ref symbol_name, ref args } => {
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                if !self.cudf_names.contains(symbol_name) {
                    let mut arg_tys = vec![];
                    for ref arg in args {
                        arg_tys.push(format!("{}*", self.llvm_type(func.symbol_type(arg)?)?));
                    }
                    arg_tys.push(format!("{}*", &output_ll_ty));
                    let arg_sig = arg_tys.join(", ");
                    self.prelude_code.add(format!("declare void @{}({});", symbol_name, arg_sig));
                    self.cudf_names.insert(symbol_name.clone());
                }

                // Prepare the parameter list for the function
                let mut arg_tys = vec![];
                for ref arg in args {
                    let (arg_ll_ty, arg_ll_sym) = self.llvm_type_and_name(func, arg)?;
                    arg_tys.push(format!("{}* {}", arg_ll_ty, arg_ll_sym));
                }
                arg_tys.push(format!("{}* {}", &output_ll_ty, &output_ll_sym));
                let parameters = arg_tys.join(", ");
                ctx.code.add(format!("call void @{}({})", symbol_name, parameters));
            }

            MakeVector { ref output, ref elems } => {
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                let output_ty = func.symbol_type(output)?;
                // Pull the element type from output instead of elems because elems could be empty.
                let elem_ll_ty = if let Vector(ref elem_ty) = *output_ty {
                    self.llvm_type(elem_ty)?
                } else {
                    unreachable!();
                };
                let output_ll_prefix = output_ll_ty.replace("%", "@");
                let vec = ctx.var_ids.next();
                ctx.code.add(format!("{} = call {} {}.new(i64 {})", vec, output_ll_ty, output_ll_prefix, elems.len()));
                // For each element, pull out the pointer and write a value to it.
                for (i, elem) in elems.iter().enumerate() {
                    let elem_ll_sym = llvm_symbol(&elem);
                    let elem_loaded = self.gen_load_var(&elem_ll_sym, &elem_ll_ty, ctx)?;
                    let ptr = ctx.var_ids.next();
                    ctx.code.add(format!("{} = call {}* {}.at({} {}, i64 {})", ptr, elem_ll_ty, output_ll_prefix, output_ll_ty, vec, i));
                    self.gen_store_var(&elem_loaded, &ptr, &elem_ll_ty, ctx);
                }
                self.gen_store_var(&vec, &output_ll_sym, &output_ll_ty, ctx);
            }

            BinOp { ref output, op, ref left, ref right } => {
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                let ty = func.symbol_type(left)?;
                // Assume the left and right operands have the same type.
                let (ll_ty, left_ll_sym) = self.llvm_type_and_name(func, left)?;
                let right_ll_sym = llvm_symbol(right);
                let left_tmp = self.gen_load_var(&left_ll_sym, &ll_ty, ctx)?;
                let right_tmp = self.gen_load_var(&right_ll_sym, &ll_ty, ctx)?;
                let output_tmp = ctx.var_ids.next();
                match *ty {
                    Scalar(_) | Simd(_) => {
                        ctx.code.add(format!("{} = {} {} {}, {}",
                                             &output_tmp, llvm_binop(op, ty)?, &ll_ty, &left_tmp, &right_tmp));
                        self.gen_store_var(&output_tmp, &output_ll_sym, &output_ll_ty, ctx);
                    }

                    Vector(_) => {
                        // We support BinOps between vectors as long as they're comparison operators
                        let (op_name, value) = llvm_binop_vector(op, ty)?;
                        let tmp = ctx.var_ids.next();
                        let vec_prefix = llvm_prefix(&ll_ty);
                        // Make sure a comparison function exists for this type.
                        self.gen_cmp(ty)?;
                        ctx.code.add(format!("{} = call i32 {}.cmp({} {}, {} {})",
                                                tmp,
                                                vec_prefix,
                                                ll_ty,
                                                left_tmp,
                                                ll_ty,
                                                right_tmp));
                        ctx.code.add(format!("{} = icmp {} i32 {}, {}", output_tmp, op_name, tmp, value));
                        self.gen_store_var(&output_tmp, &output_ll_sym, &output_ll_ty, ctx);
                    }

                    _ => weld_err!("Illegal type {} in BinOp", print_type(ty))?,
                }
            }

            Broadcast { ref output, ref child } => {
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                let (child_ll_ty, child_ll_sym) = self.llvm_type_and_name(func, child)?;
                let child_ty = func.symbol_type(child)?;
                let value = self.gen_load_var(&child_ll_sym, &child_ll_ty, ctx)?;
                let mut prev_name = "undef".to_string();
                for i in 0..llvm_simd_size(child_ty)? {
                    let next = ctx.var_ids.next();
                    ctx.code.add(format!("{} = insertelement {} {}, {} {}, i32 {}",
                                            next, output_ll_ty, prev_name, child_ll_ty, value, i));
                    prev_name = next;
                }
                self.gen_store_var(&prev_name, &output_ll_sym, &output_ll_ty, ctx);
            }

            UnaryOp { ref output, op, ref child, } => {
                self.gen_unary_op(ctx, func, output, child, op)?
            }

            Negate { ref output, ref child } => {
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                let output_ty = func.symbol_type(output)?;
                let zero = match *output_ty {
                    Scalar(F32) | Scalar(F64) => "0.0",
                    _ => "0",
                };
                let op_name = llvm_binop(BinOpKind::Subtract, output_ty)?;
                let child_tmp = self.gen_load_var(&llvm_symbol(child), &output_ll_ty, ctx)?;
                let value = ctx.var_ids.next();
                ctx.code.add(format!("{} = {} {} {}, {}", value, op_name, output_ll_ty, zero, child_tmp));
                self.gen_store_var(&value, &output_ll_sym, &output_ll_ty, ctx);
            }

            Cast { ref output, ref child } => {
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                let (child_ll_ty, child_ll_sym) = self.llvm_type_and_name(func, child)?;
                let output_ty = func.symbol_type(output)?;
                let child_ty = func.symbol_type(child)?;
                if child_ty != output_ty {
                    let op_name = llvm_castop(child_ty, output_ty)?;
                    let child_tmp = self.gen_load_var(&child_ll_sym, &child_ll_ty, ctx)?;
                    let cast_tmp = ctx.var_ids.next();
                    ctx.code.add(format!("{} = {} {} {} to {}",
                                            cast_tmp, op_name, child_ll_ty, child_tmp, output_ll_ty));
                    self.gen_store_var(&cast_tmp, &output_ll_sym, &output_ll_ty, ctx);
                } else {
                    let child_tmp = self.gen_load_var(&child_ll_sym, &child_ll_ty, ctx)?;
                    self.gen_store_var(&child_tmp, &output_ll_sym, &child_ll_ty, ctx);
                }
            }

            Lookup { ref output, ref child, ref index } => {
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                let (child_ll_ty, child_ll_sym) = self.llvm_type_and_name(func, child)?;
                let (index_ll_ty, index_ll_sym) = self.llvm_type_and_name(func, index)?;
                let child_ty = func.symbol_type(child)?;
                match *child_ty {
                    Vector(_) => {
                        let child_prefix = llvm_prefix(&child_ll_ty);
                        let child_tmp = self.gen_load_var(&child_ll_sym, &child_ll_ty, ctx)?;
                        let index_tmp = self.gen_load_var(&index_ll_sym, &index_ll_ty, ctx)?;
                        let res_ptr = ctx.var_ids.next();
                        ctx.code.add(format!("{} = call {}* {}.at({} {}, i64 {})",
                            res_ptr, output_ll_ty, child_prefix, child_ll_ty,child_tmp, index_tmp));
                        let res_tmp = self.gen_load_var(&res_ptr, &output_ll_ty, ctx)?;
                        self.gen_store_var(&res_tmp, &output_ll_sym, &output_ll_ty, ctx);
                    }
                    Dict(_, _) => {
                        let child_prefix = llvm_prefix(&child_ll_ty);
                        let child_tmp = self.gen_load_var(&child_ll_sym, &child_ll_ty, ctx)?;
                        let index_tmp = self.gen_load_var(&index_ll_sym, &index_ll_ty, ctx)?;
                        let slot = ctx.var_ids.next();
                        let res_tmp = ctx.var_ids.next();
                        ctx.code.add(format!("{} = call {}.slot {}.lookup({} {}, {} {})",
                                                slot, child_ll_ty, child_prefix, child_ll_ty, child_tmp, index_ll_ty, index_tmp));
                        ctx.code.add(format!("{} = call {} {}.slot.value({}.slot {})",
                                                res_tmp, output_ll_ty, child_prefix, child_ll_ty, slot));
                        self.gen_store_var(&res_tmp, &output_ll_sym, &output_ll_ty, ctx);
                    }
                    _ => weld_err!("Illegal type {} in Lookup", print_type(child_ty))?,
                }
            }

            KeyExists { ref output, ref child, ref key } => {
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                let (child_ll_ty, child_ll_sym) = self.llvm_type_and_name(func, child)?;
                let (key_ll_ty, key_ll_sym) = self.llvm_type_and_name(func, key)?;
                let child_prefix = llvm_prefix(&child_ll_ty);

                let child_tmp = self.gen_load_var(&child_ll_sym, &child_ll_ty, ctx)?;
                let key_tmp = self.gen_load_var(&key_ll_sym, &key_ll_ty, ctx)?;

                let slot = ctx.var_ids.next();
                let res_tmp = ctx.var_ids.next();
                ctx.code.add(format!("{} = call {}.slot {}.lookup({} {}, {} {})",
                    slot, child_ll_ty, child_prefix, child_ll_ty, child_tmp, key_ll_ty, key_tmp));
                ctx.code.add(format!("{} = call i1 {}.slot.filled({}.slot {})",
                    res_tmp, child_prefix, child_ll_ty, slot));
                self.gen_store_var(&res_tmp, &output_ll_sym, &output_ll_ty, ctx);
            }

            Slice { ref output, ref child, ref index, ref size } => {
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                let (child_ll_ty, child_ll_sym) = self.llvm_type_and_name(func, child)?;
                let (index_ll_ty, index_ll_sym) = self.llvm_type_and_name(func, index)?;
                let (size_ll_ty, size_ll_sym) = self.llvm_type_and_name(func, size)?;
                let vec_prefix = llvm_prefix(&child_ll_ty);
                let child_tmp = self.gen_load_var(&child_ll_sym, &child_ll_ty, ctx)?;
                let index_tmp = self.gen_load_var(&index_ll_sym, &index_ll_ty, ctx)?;
                let size_tmp = self.gen_load_var(&size_ll_sym, &size_ll_ty, ctx)?;
                let res_ptr = ctx.var_ids.next();
                ctx.code.add(format!("{} = call {} {}.slice({} {}, i64 {}, i64{})",
                                                res_ptr,
                                                output_ll_ty,
                                                vec_prefix,
                                                child_ll_ty,
                                                child_tmp,
                                                index_tmp,
                                                size_tmp));
                self.gen_store_var(&res_ptr, &output_ll_sym, &output_ll_ty, ctx);
            }

            Select { ref output, ref cond, ref on_true, ref on_false } => {
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                let (cond_ll_ty, cond_ll_sym) = self.llvm_type_and_name(func, cond)?;
                let (on_true_ll_ty, on_true_ll_sym) = self.llvm_type_and_name(func, on_true)?;
                let (on_false_ll_ty, on_false_ll_sym) = self.llvm_type_and_name(func, on_false)?;
                let cond_tmp = self.gen_load_var(&cond_ll_sym, &cond_ll_ty, ctx)?;
                let on_true_tmp = self.gen_load_var(&on_true_ll_sym, &on_true_ll_ty, ctx)?;
                let on_false_tmp = self.gen_load_var(&on_false_ll_sym, &on_false_ll_ty, ctx)?;
                let tmp = ctx.var_ids.next();
                ctx.code.add(format!("{} = select {} {}, {} {}, {} {}",
                    tmp, cond_ll_ty, cond_tmp, on_true_ll_ty, on_true_tmp, on_false_ll_ty, on_false_tmp));
                self.gen_store_var(&tmp, &output_ll_sym, &output_ll_ty, ctx);
            }

            ToVec { ref output, ref child } => {
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                let (child_ll_ty, child_ll_sym) = self.llvm_type_and_name(func, child)?;
                let dict_prefix = llvm_prefix(&child_ll_ty);
                let child_tmp = self.gen_load_var(&child_ll_sym, &child_ll_ty, ctx)?;
                let res_tmp = ctx.var_ids.next();
                ctx.code.add(format!("{} = call {} {}.tovec({} {})",
                    res_tmp, output_ll_ty, dict_prefix, child_ll_ty, child_tmp));
                self.gen_store_var(&res_tmp, &output_ll_sym, &output_ll_ty, ctx);
            }

            Length { ref output, ref child } => {
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                let (child_ll_ty, child_ll_sym) = self.llvm_type_and_name(func, child)?;
                let vec_prefix = llvm_prefix(&child_ll_ty);
                let child_tmp = self.gen_load_var(&child_ll_sym, &child_ll_ty, ctx)?;
                let res_tmp = ctx.var_ids.next();
                ctx.code.add(format!("{} = call i64 {}.size({} {})", res_tmp, vec_prefix, child_ll_ty, child_tmp));
                self.gen_store_var(&res_tmp, &output_ll_sym, &output_ll_ty, ctx);
            }

            Assign { ref output, ref value } => {
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                let (value_ll_ty, value_ll_sym) = self.llvm_type_and_name(func, value)?;
                let val_tmp = self.gen_load_var(&value_ll_sym, &value_ll_ty, ctx)?;
                self.gen_store_var(&val_tmp, &output_ll_sym, &output_ll_ty, ctx);
            }

            GetField { ref output, ref value, index } => {
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                let (value_ll_ty, value_ll_sym) = self.llvm_type_and_name(func, value)?;
                let ptr_tmp = ctx.var_ids.next();
                ctx.code.add(format!("{} = getelementptr inbounds {}, {}* {}, i32 0, i32 {}",
                    ptr_tmp, value_ll_ty, value_ll_ty, value_ll_sym, index));
                let res_tmp = self.gen_load_var(&ptr_tmp, &output_ll_ty, ctx)?;
                self.gen_store_var(&res_tmp, &output_ll_sym, &output_ll_ty, ctx);
            }

            AssignLiteral { ref output, ref value } => {
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                let output_ty = func.symbol_type(output)?;
                if let Simd(_) = *output_ty {
                    self.gen_simd_literal(&output_ll_sym, value, output_ty, ctx)?;
                } else {
                    let ref value = llvm_literal(*value);
                    self.gen_store_var(value, &output_ll_sym, &output_ll_ty, ctx);
                }
            }

            Merge { ref builder, ref value } => {
                let bld_ty = func.symbol_type(builder)?;
                if let Builder(ref bld_kind, _) = *bld_ty {
                    let val_ty = func.symbol_type(value)?;
                    self.gen_merge(bld_kind, builder, val_ty, value, func, ctx)?;
                } else {
                    return weld_err!("Non builder type {} found in Merge", print_type(bld_ty))
                }
            }

            Res { ref output, ref builder } => {
                let bld_ty = func.symbol_type(builder)?;
                if let Builder(ref bld_kind, _) = *bld_ty {
                    self.gen_result(bld_kind, builder, output, func, ctx)?;
                } else {
                    return weld_err!("Non builder type {} found in Result", print_type(bld_ty))
                }
            }

            NewBuilder { ref output, ref arg, ref ty } => {
                if let Builder(ref bld_kind, ref annotations) = *ty {
                    self.gen_new_builder(bld_kind, annotations, arg, output, func, ctx)?;
                } else {
                    return weld_err!("Non builder type {} found in NewBuilder", print_type(ty))
                }
            }
        }

        Ok(())
    }

    /// Generate code for a Merge instruction, appending it to the given FunctionContext.
    fn gen_merge(&mut self,
                 builder_kind: &BuilderKind,
                 builder: &Symbol,
                 value_ty: &Type,
                 value: &Symbol,
                 func: &SirFunction,
                 ctx: &mut FunctionContext)
                 -> WeldResult<()> {

        let (bld_ll_ty, bld_ll_sym) = self.llvm_type_and_name(func, builder)?;
        let val_ll_ty = self.llvm_type(value_ty)?;
        let val_ll_sym = llvm_symbol(value);
        let bld_prefix = llvm_prefix(&bld_ll_ty);

        // Special case: if the value is a SIMD vector, we need to merge each element separately (because the final
        // builder will still operate on scalars). We have specialized functions to merge all of them together for
        // some builders, such as Merger, but for others we just call the merge operation multiple times.
        if value_ty.is_simd() {
            match *builder_kind {
                Merger(_, ref op) => {
                    let elem_tmp = self.gen_load_var(&val_ll_sym, &val_ll_ty, ctx)?;
                    if !ctx.is_innermost {
                        // For Merger, call the vectorMergePtr function to get a pointer to a vector we can merge into.
                        let bld_tmp = self.gen_load_var(&bld_ll_sym, &bld_ll_ty, ctx)?;
                        let bld_ptr_raw = ctx.var_ids.next();
                        let bld_ptr = ctx.var_ids.next();
                        ctx.code.add(format!(
                            "{} = call {} {}.getPtrIndexed({} {}, i32 %cur.tid)",
                            bld_ptr_raw,
                            bld_ll_ty,
                            bld_prefix,
                            bld_ll_ty,
                            bld_tmp));
                        ctx.code.add(format!(
                            "{} = call {}* {}.vectorMergePtr({} {})",
                            bld_ptr,
                            val_ll_ty,
                            bld_prefix,
                            bld_ll_ty,
                            bld_ptr_raw));
                        self.gen_merge_op(&bld_ptr, &elem_tmp, &val_ll_ty, op, value_ty, ctx)?;
                    } else {
                        let bld_ll_reg_sym = format!("{}.reg", bld_ll_sym);
                        let reg_ptr_simd = ctx.var_ids.next();
                        ctx.code.add(format!(
                            "{} = call {}* {}.vectorMergePtr({} {})",
                            reg_ptr_simd,
                            val_ll_ty,
                            bld_prefix,
                            bld_ll_ty,
                            bld_ll_reg_sym));
                        self.gen_merge_op(&reg_ptr_simd, &elem_tmp, &val_ll_ty, op, value_ty, ctx)?;
                    }
                }

                Appender(ref elem) if elem.is_scalar() => {
                    let bld_tmp = self.gen_load_var(&bld_ll_sym, &bld_ll_ty, ctx)?;
                    let val_tmp = self.gen_load_var(&val_ll_sym, &val_ll_ty, ctx)?;
                    ctx.code.add(format!("call {} {}.vmerge({} {}, {} {}, i32 %cur.tid)",
                    bld_ll_ty,
                    bld_prefix,
                    bld_ll_ty,
                    bld_tmp,
                    val_ll_ty,
                    val_tmp));
                }

                _ => {
                    // For all other builders, extract each value in the vector and merge it separately
                    let value_tmp = self.gen_load_var(&val_ll_sym, &val_ll_ty, ctx)?;
                    let item_ty = value_ty.scalar_type()?;
                    let item_ll_ty = self.llvm_type(&item_ty)?;
                    for i in 0..llvm_simd_size(&item_ty)? {
                        // Extract the item to a register and put it into a stack variable so we can call gen_merge.
                        let item_tmp = self.gen_simd_extract(value_ty, &value_tmp, i, ctx)?;
                        let item_stack = ctx.var_ids.next();
                        ctx.add_alloca(&item_stack, &item_ll_ty)?;
                        let item_stack_sym = Symbol::new(&item_stack.replace("%", ""), 0);
                        self.gen_store_var(&item_tmp, &item_stack, &item_ll_ty, ctx);
                        self.gen_merge(builder_kind, builder, &item_ty, &item_stack_sym, func, ctx)?;
                    }
                }
            }
            return Ok(())
        }

        // For non-vectorized elements, just switch on the builder type. TODO: Use annotations here too.
        match *builder_kind {
            Appender(_) => {
                let bld_tmp = self.gen_load_var(&bld_ll_sym, &bld_ll_ty, ctx)?;
                let val_tmp = self.gen_load_var(&val_ll_sym, &val_ll_ty, ctx)?;
                ctx.code.add(format!("call {} {}.merge({} {}, {} {}, i32 %cur.tid)",
                                        bld_ll_ty,
                                        bld_prefix,
                                        bld_ll_ty,
                                        bld_tmp,
                                        val_ll_ty,
                                        val_tmp));
            }

            DictMerger(_, _, _) => {
                let bld_tmp = self.gen_load_var(&bld_ll_sym, &bld_ll_ty, ctx)?;
                let val_tmp = self.gen_load_var(&val_ll_sym, &val_ll_ty, ctx)?;
                ctx.code.add(format!(
                    "call {} {}.merge({} {}, {} {}, i32 %cur.tid)",
                    bld_ll_ty,
                    bld_prefix,
                    bld_ll_ty,
                    bld_tmp,
                    val_ll_ty,
                    val_tmp));
            }

            GroupMerger(_, _) => {
                let bld_tmp = self.gen_load_var(&bld_ll_sym, &bld_ll_ty, ctx)?;
                let val_tmp = self.gen_load_var(&val_ll_sym, &val_ll_ty, ctx)?;
                ctx.code.add(format!(
                    "call {} {}.merge({} {}, {} {}, i32 %cur.tid)",
                    bld_ll_ty,
                    bld_prefix,
                    bld_ll_ty,
                    bld_tmp,
                    val_ll_ty,
                    val_tmp));
            }

            Merger(ref t, ref op) => {
                let val_tmp = self.gen_load_var(&val_ll_sym, &val_ll_ty, ctx)?;
                if !ctx.is_innermost {
                    let bld_tmp = self.gen_load_var(&bld_ll_sym, &bld_ll_ty, ctx)?;
                    let bld_ptr_raw = ctx.var_ids.next();
                    let bld_ptr = ctx.var_ids.next();
                    ctx.code.add(format!(
                        "{} = call {} {}.getPtrIndexed({} {}, i32 %cur.tid)",
                        bld_ptr_raw,
                        bld_ll_ty,
                        bld_prefix,
                        bld_ll_ty,
                        bld_tmp));
                    ctx.code.add(format!(
                        "{} = call {}* {}.scalarMergePtr({} {})",
                        bld_ptr,
                        val_ll_ty,
                        bld_prefix,
                        bld_ll_ty,
                        bld_ptr_raw));
                    self.gen_merge_op(&bld_ptr, &val_tmp, &val_ll_ty, op, t, ctx)?;
                } else {
                    let bld_ll_reg_sym = format!("{}.reg", bld_ll_sym);
                    let reg_ptr_scalar = ctx.var_ids.next();
                    ctx.code.add(format!(
                        "{} = call {}* {}.scalarMergePtr({} {})",
                        reg_ptr_scalar,
                        val_ll_ty,
                        bld_prefix,
                        bld_ll_ty,
                        bld_ll_reg_sym));
                    self.gen_merge_op(&reg_ptr_scalar, &val_tmp, &val_ll_ty, op, t, ctx)?;
                }
            }

            VecMerger(ref t, ref op) => {
                let elem_ll_ty = self.llvm_type(t)?;
                let bld_tmp = self.gen_load_var(&bld_ll_sym, &bld_ll_ty, ctx)?;
                let val_tmp = self.gen_load_var(&val_ll_sym, &val_ll_ty, ctx)?;
                let index_var = ctx.var_ids.next();
                let elem_var = ctx.var_ids.next();
                ctx.code.add(format!("{} = extractvalue {} {}, 0", index_var, &val_ll_ty, val_tmp));
                ctx.code.add(format!("{} = extractvalue {} {}, 1", elem_var, &val_ll_ty, val_tmp));
                let bld_ptr_raw = ctx.var_ids.next();
                let bld_ptr = ctx.var_ids.next();
                ctx.code.add(format!("{} = call i8* {}.merge_ptr({} {}, i64 {}, i32 %cur.tid)",
                                        bld_ptr_raw,
                                        bld_prefix,
                                        bld_ll_ty,
                                        bld_tmp,
                                        index_var));
                ctx.code.add(format!("{} = bitcast i8* {} to {}*",
                                        bld_ptr,
                                        bld_ptr_raw,
                                        elem_ll_ty));
                self.gen_merge_op(&bld_ptr, &elem_var, &elem_ll_ty, op, t, ctx)?;
            }
        }
        Ok(())
    }

    /// Generate code to compute the result of a `builder` and store it in `output`, appending it to a FunctionContext.
    fn gen_result(&mut self,
                  builder_kind: &BuilderKind,
                  builder: &Symbol,
                  output: &Symbol,
                  func: &SirFunction,
                  ctx: &mut FunctionContext)
                  -> WeldResult<()> {

        let (bld_ll_ty, bld_ll_sym) = self.llvm_type_and_name(func, builder)?;
        let (res_ll_ty, res_ll_sym) = self.llvm_type_and_name(func, output)?;
        let bld_ty = func.symbol_type(builder)?;
        let res_ty = func.symbol_type(output)?;

        match *builder_kind {
            Appender(_) => {
                let bld_prefix = llvm_prefix(&bld_ll_ty);
                let bld_tmp = self.gen_load_var(&bld_ll_sym, &bld_ll_ty, ctx)?;
                let res_tmp = ctx.var_ids.next();
                ctx.code.add(format!("{} = call {} {}.result({} {})",
                                     res_tmp,
                                     res_ll_ty,
                                     bld_prefix,
                                     bld_ll_ty,
                                     bld_tmp));
                self.gen_store_var(&res_tmp, &res_ll_sym, &res_ll_ty, ctx);
            }

            Merger(ref t, ref op) => {
                // Type of element to merge.
                let elem_ty_str = self.llvm_type(t)?;

                // Vector type.
                let ref vec_type = if let Scalar(ref k) = **t {
                    Simd(k.clone())
                } else {
                    return weld_err!("Invalid non-scalar type in merger");
                };

                let elem_vec_ty_str = self.llvm_type(vec_type)?;

                // Builder type.
                let bld_ty_str = self.llvm_type(&bld_ty)?;
                // Prefix of the builder.
                let bld_prefix = llvm_prefix(&bld_ty_str);
                // Result type.
                let res_ty_str = self.llvm_type(&res_ty)?;
                // Temporary builder variable.
                let bld_tmp = self.gen_load_var(llvm_symbol(builder).as_str(), &bld_ty_str, ctx)?;

                // Generate names for all temporaries.
                let t0 = ctx.var_ids.next();
                let scalar_ptr = ctx.var_ids.next();
                let vector_ptr = ctx.var_ids.next();
                let first_scalar = ctx.var_ids.next();
                let first_vector = ctx.var_ids.next();
                let nworkers = ctx.var_ids.next();
                let cond = ctx.var_ids.next();
                let i = ctx.var_ids.next();
                let bld_ptr = ctx.var_ids.next();
                let val_scalar_ptr = ctx.var_ids.next();
                let val_vector_ptr = ctx.var_ids.next();
                let val_scalar = ctx.var_ids.next();
                let val_vector = ctx.var_ids.next();
                let i2 = ctx.var_ids.next();
                let cond2 = ctx.var_ids.next();
                let as_ptr = ctx.var_ids.next();

                // Generate label names without prefix.
                let entry_label = ctx.var_ids.next().replace("%", "");
                let body_label = ctx.var_ids.next().replace("%", "");
                let done_label = ctx.var_ids.next().replace("%", "");

                // state for the vector collapse
                let i_v = ctx.var_ids.next();
                let val_v = ctx.var_ids.next();
                let i2_v = ctx.var_ids.next();
                let cond_v = ctx.var_ids.next();
                let cond2_v = ctx.var_ids.next();
                let final_val_vec = ctx.var_ids.next();
                let scalar_val_2 = ctx.var_ids.next();
                let entry_label_v = ctx.var_ids.next().replace("%", "");
                let body_label_v = ctx.var_ids.next().replace("%", "");
                let done_label_v = ctx.var_ids.next().replace("%", "");
                let vector_width = format!("{}", llvm_simd_size(t)?);

                ctx.code.add(format!(include_str!("resources/merger/merger_result_start.ll"),
                                        t0 = t0,
                                        scalar_ptr = scalar_ptr,
                                        vector_ptr = vector_ptr,
                                        nworkers = nworkers,
                                        first_scalar = first_scalar,
                                        first_vector = first_vector,
                                        bld_tmp = bld_tmp,
                                        cond = cond,
                                        i = i,
                                        bld_ptr = bld_ptr,
                                        val_scalar_ptr = val_scalar_ptr,
                                        val_vector_ptr = val_vector_ptr,
                                        val_scalar = val_scalar,
                                        val_vector = val_vector,
                                        i2 = i2,
                                        elem_ty_str = elem_ty_str,
                                        elem_vec_ty_str = elem_vec_ty_str,
                                        bld_ty_str = bld_ll_ty,
                                        bld_prefix = bld_prefix,
                                        entry = entry_label,
                                        body = body_label,
                                        done = done_label));

                // Add the scalar and vector values to the aggregate result.
                self.gen_merge_op(&scalar_ptr, &val_scalar, &elem_ty_str, op, t, ctx)?;
                self.gen_merge_op(&vector_ptr, &val_vector, &elem_vec_ty_str, op, vec_type, ctx)?;

                ctx.code.add(format!(include_str!("resources/merger/merger_result_end_vectorized_1.ll"),
                        nworkers = nworkers,
                        i=i,
                        i2=i2,
                        cond2=cond2,
                        i_v=i_v,
                        i2_v=i2_v,
                        cond_v=cond_v,
                        res_ty_str=res_ty_str,
                        vector_ptr=vector_ptr,
                        scalar_ptr=scalar_ptr,
                        final_val_vec=final_val_vec,
                        scalar_val_2=scalar_val_2,
                        vector_width=vector_width,
                        elem_vec_ty_str=elem_vec_ty_str,
                        val_v=val_v,
                        body=body_label,
                        done=done_label,
                        entry_v=entry_label_v,
                        body_v=body_label_v,
                        done_v=done_label_v,
                        output=res_ll_sym));

                self.gen_merge_op(&res_ll_sym, &val_v, &res_ty_str, op, t, ctx)?;

                ctx.code.add(format!(include_str!("resources/merger/merger_result_end_vectorized_2.ll"),
                        i_v=i_v,
                        i2_v=i2_v,
                        cond2_v=cond2_v,
                        as_ptr=as_ptr,
                        bld_ty_str=bld_ty_str,
                        bld_tmp=bld_tmp,
                        body_v=body_label_v,
                        vector_width=vector_width,
                        done_v=done_label_v));
            }

            DictMerger(_, _, _) => {
                let bld_ty_str = self.llvm_type(&bld_ty)?;
                let bld_prefix = llvm_prefix(&bld_ty_str);
                let res_ty_str = self.llvm_type(&res_ty)?;
                let bld_tmp =
                    self.gen_load_var(llvm_symbol(builder).as_str(), &bld_ty_str, ctx)?;
                let res_tmp = ctx.var_ids.next();
                ctx.code.add(format!("{} = call {} {}.result({} {})",
                                        res_tmp,
                                        res_ty_str,
                                        bld_prefix,
                                        bld_ty_str,
                                        bld_tmp));

                self.gen_store_var(&res_tmp, &llvm_symbol(output), &res_ty_str, ctx);
            }

            GroupMerger(ref kt, ref vt) => {
                let mut func_gen = IdGenerator::new("%func");
                let function_id = func_gen.next();
                let func_str = llvm_prefix(&&function_id);
                let bld_ty = Dict(kt.clone(), Box::new(Vector(vt.clone())));
                let elem = Box::new(Struct(vec![*kt.clone(), *vt.clone()]));
                let bld_ty_str = self.llvm_type(&bld_ty)?;
                let kv_struct_ty = self.llvm_type(&elem)?;
                let key_ty = self.llvm_type(kt)?;
                let value_ty = self.llvm_type(vt)?;
                let value_vec_ty = self.llvm_type(&Box::new(Vector(vt.clone())))?;
                let kv_vec = Box::new(Vector(elem.clone()));
                let kv_vec_ty = self.llvm_type(&kv_vec)?;
                let kv_vec_builder_ty = format!("{}.bld", &kv_vec_ty);
                let key_prefix = llvm_prefix(&&key_ty);
                let kv_vec_prefix = llvm_prefix(&&kv_vec_ty);
                let value_vec_prefix = llvm_prefix(&&value_vec_ty);
                let dict_prefix = llvm_prefix(&&bld_ty_str);

                // Required for result calls.
                self.gen_eq(kt)?;
                self.gen_cmp(kt)?;

                let groupmerger_def = format!(include_str!("resources/groupbuilder.ll"),
                    NAME=&function_id.replace("%", ""),
                    KEY_PREFIX=&key_prefix,
                    KEY=&key_ty,
                    VALUE_VEC_PREFIX=&value_vec_prefix,
                    VALUE_VEC=&value_vec_ty,
                    VALUE=&value_ty,
                    KV_STRUCT=&kv_struct_ty.replace("%", ""),
                    KV_VEC_PREFIX=&kv_vec_prefix,
                    KV_VEC=&kv_vec_ty,
                    DICT_PREFIX=&dict_prefix,
                    DICT=&bld_ty_str);

                self.prelude_code.add(&groupmerger_def);
                let res_ty_str = self.llvm_type(&res_ty)?;
                let bld_tmp = self.gen_load_var(llvm_symbol(builder).as_str(), &kv_vec_builder_ty, ctx)?;
                let res_tmp = ctx.var_ids.next();
                ctx.code.add(format!("{} = call {} {}({} {})",
                                      res_tmp,
                                      bld_ty_str,
                                      func_str,
                                      kv_vec_builder_ty,
                                      bld_tmp));
                self.gen_store_var(&res_tmp, &llvm_symbol(output), &res_ty_str, ctx);
            }

            VecMerger(ref t, ref op) => {
                // The builder type (special internal type).
                let bld_ty_str = self.llvm_type(&bld_ty)?;
                let bld_prefix = llvm_prefix(&bld_ty_str);
                // The result type (vec[elem_type])
                let res_ty_str = self.llvm_type(&res_ty)?;
                let res_prefix = llvm_prefix(&res_ty_str);
                // The element type
                let elem_ty_str = self.llvm_type(t)?;
                // The builder we operate on.
                let bld_ptr =
                    self.gen_load_var(llvm_symbol(builder).as_str(), &bld_ty_str, ctx)?;

                // Generate names for all temporaries.
                let nworkers = ctx.var_ids.next();
                let t0 = ctx.var_ids.next();
                let typed_ptr = ctx.var_ids.next();
                let first_vec = ctx.var_ids.next();
                let size = ctx.var_ids.next();
                let ret_value = ctx.var_ids.next();
                let cond = ctx.var_ids.next();
                let i = ctx.var_ids.next();
                let vec_ptr = ctx.var_ids.next();
                let cur_vec = ctx.var_ids.next();
                let copy_cond = ctx.var_ids.next();
                let j = ctx.var_ids.next();
                let elem_ptr = ctx.var_ids.next();
                let merge_value = ctx.var_ids.next();
                let merge_ptr = ctx.var_ids.next();
                let j2 = ctx.var_ids.next();
                let copy_cond2 = ctx.var_ids.next();
                let i2 = ctx.var_ids.next();
                let cond2 = ctx.var_ids.next();

                // Generate label names without prefix.
                let entry = ctx.var_ids.next().replace("%", "");
                let body_label = ctx.var_ids.next().replace("%", "");
                let copy_entry_label = ctx.var_ids.next().replace("%", "");
                let copy_body_label = ctx.var_ids.next().replace("%", "");
                let copy_done_label = ctx.var_ids.next().replace("%", "");
                let done_label = ctx.var_ids.next().replace("%", "");
                let raw_ptr = ctx.var_ids.next();

                ctx.code.add(format!(include_str!("resources/vecmerger/vecmerger_result_start.ll"),
                                    nworkers = nworkers,
                                    t0 = t0,
                                    buildPtr = bld_ptr,
                                    resType = res_ty_str,
                                    resPrefix = res_prefix,
                                    elemType = elem_ty_str,
                                    typedPtr = typed_ptr,
                                    firstVec = first_vec,
                                    size = size,
                                    retValue = ret_value,
                                    cond = cond,
                                    i = i,
                                    i2 = i2,
                                    vecPtr = vec_ptr,
                                    curVec = cur_vec,
                                    copyCond = copy_cond,
                                    j = j,
                                    j2 = j2,
                                    elemPtr = elem_ptr,
                                    mergeValue = merge_value,
                                    mergePtr = merge_ptr,
                                    entry = entry,
                                    bodyLabel = body_label,
                                    copyEntryLabel = copy_entry_label,
                                    copyBodyLabel = copy_body_label,
                                    copyDoneLabel = copy_done_label,
                                    doneLabel = done_label,
                                    bldType = bld_ty_str,
                                    bldPrefix = bld_prefix));

                self.gen_merge_op(&merge_ptr, &merge_value, &elem_ty_str, op, t, ctx)?;

                ctx.code.add(format!(include_str!("resources/vecmerger/vecmerger_result_end.ll"),
                                    j2 = j2,
                                    j = j,
                                    copyCond2 = copy_cond2,
                                    size = size,
                                    i2 = i2,
                                    i = i,
                                    cond2 = cond2,
                                    nworkers = nworkers,
                                    resType = res_ty_str,
                                    retValue = ret_value,
                                    copyBodyLabel = copy_body_label,
                                    copyDoneLabel = copy_done_label,
                                    doneLabel = done_label,
                                    bodyLabel = body_label,
                                    rawPtr = raw_ptr,
                                    buildPtr = bld_ptr,
                                    bldType = bld_ty_str,
                                    output = llvm_symbol(output)));
            }
        }

        Ok(())
    }

    /// Generate code for a NewBuilder statement, creating a builder of the given type with a given `arg` and
    /// storing the result in an `output` symbol. Appends the code to a given FunctionContext.
    fn gen_new_builder(&mut self,
                       builder_kind: &BuilderKind,
                       annotations: &Annotations,
                       arg: &Option<Symbol>,
                       output: &Symbol,
                       func: &SirFunction,
                       ctx: &mut FunctionContext)
                       -> WeldResult<()> {
        let bld_ty = func.symbol_type(output)?;
        let bld_ty_str = self.llvm_type(bld_ty)?;
        let bld_prefix = llvm_prefix(&bld_ty_str);

        let mut builder_size = 16;
        if let Some(ref e) = *annotations.size() {
            builder_size = e.clone();
        }

        match *builder_kind {
            Appender(_) => {
                let bld_tmp = ctx.var_ids.next();
                let size_tmp = if let Some(ref sym) = *arg {
                    let (arg_ll_ty, arg_ll_sym) = self.llvm_type_and_name(func, sym)?;
                    self.gen_load_var(&arg_ll_sym, &arg_ll_ty, ctx)?
                } else {
                    format!("{}", builder_size)
                };
                let fixed_size = if let Some(_) = *arg { "1" } else { "0" };

                ctx.code.add(format!(
                    "{} = call {} {}.new(i64 {}, %work_t* \
                                    %cur.work, i32 {})",
                    bld_tmp,
                    bld_ty_str,
                    bld_prefix,
                    size_tmp,
                    fixed_size
                ));
                self.gen_store_var(&bld_tmp, &llvm_symbol(output), &bld_ty_str, ctx);
            }
            Merger(ref elem_ty, ref op) => {
                let elem_type = self.llvm_type(elem_ty)?;
                let bld_tmp = ctx.var_ids.next();
                ctx.code.add(format!("{} = call {} {}.new()", bld_tmp, bld_ty_str, bld_prefix));

                // Generate code to initialize the builder.
                let iden_elem = binop_identity(*op, elem_ty.as_ref())?;
                let init_elem = match *arg {
                    Some(ref s) => {
                        let arg_str = self.gen_load_var(llvm_symbol(s).as_str(), &elem_type, ctx)?;
                        arg_str
                    }
                    _ => iden_elem.clone(),
                };

                let first = ctx.var_ids.next();
                let first_raw = ctx.var_ids.next();
                let nworkers = ctx.var_ids.next();
                let i = ctx.var_ids.next();
                let cur_bld_ptr = ctx.var_ids.next();
                let i2 = ctx.var_ids.next();
                let cond = ctx.var_ids.next();
                let cond2 = ctx.var_ids.next();

                let entry = ctx.var_ids.next().replace("%", "");
                let body = ctx.var_ids.next().replace("%", "");
                let done = ctx.var_ids.next().replace("%", "");

                ctx.code.add(format!(include_str!("resources/merger/init_merger.ll"),
                                        first = first,
                                        first_raw = first_raw,
                                        nworkers = nworkers,
                                        bld_ty_str = bld_ty_str,
                                        bld_prefix = bld_prefix,
                                        init_elem = init_elem,
                                        elem_type = elem_type,
                                        cond = cond,
                                        iden_elem = iden_elem,
                                        bld_inp = bld_tmp,
                                        i = i,
                                        cur_bld_ptr = cur_bld_ptr,
                                        i2 = i2,
                                        cond2 = cond2,
                                        entry = entry,
                                        body = body,
                                        done = done));
                self.gen_store_var(&bld_tmp, &llvm_symbol(output), &bld_ty_str, ctx);
            }
            DictMerger(_, _, _) => {
                let bld_tmp = ctx.var_ids.next();
                ctx.code.add(format!("{} = call {} {}.new(i64 {})",
                                        bld_tmp,
                                        bld_ty_str,
                                        bld_prefix,
                                        builder_size));
                self.gen_store_var(&bld_tmp, &llvm_symbol(output), &bld_ty_str, ctx);
            }
            GroupMerger(_, _) => {
                let bld_tmp = ctx.var_ids.next();
                ctx.code.add(format!(
                    "{} = call {} {}.new(i64 {}, %work_t* \
                                    %cur.work, i32 0)",
                    bld_tmp,
                    bld_ty_str,
                    bld_prefix,
                    builder_size
                ));
                self.gen_store_var(&bld_tmp, &llvm_symbol(output), &bld_ty_str, ctx);
            }
            VecMerger(ref elem, ref op) => {
                if *op != BinOpKind::Add {
                    return weld_err!("VecMerger only supports +");
                }
                match *arg {
                    Some(ref s) => {
                        let arg_ty = self.llvm_type(&Vector(elem.clone()))?;
                        let arg_ty_str = arg_ty.to_string();
                        let arg_str = self.gen_load_var(llvm_symbol(s).as_str(), &arg_ty_str, ctx)?;
                        let bld_tmp = ctx.var_ids.next();
                        ctx.code.add(format!("{} = call {} {}.new({} \
                                                {})",
                                                bld_tmp,
                                                bld_ty_str,
                                                bld_prefix,
                                                arg_ty_str,
                                                arg_str));
                        self.gen_store_var(&bld_tmp, &llvm_symbol(output), &bld_ty_str, ctx);
                    }
                    None => {
                        weld_err!("Internal error: NewBuilder(VecMerger) \
                                    expected argument in LLVM codegen")?
                    }
                }
            }
        }

        Ok(())
    }

    /// Generate code for a basic block's terminator, appending it to the given FunctionContext.
    fn gen_terminator(&mut self,
                      terminator: &Terminator,
                      sir: &SirProgram,
                      func: &SirFunction,
                      ctx: &mut FunctionContext)
                      -> WeldResult<()> {
        ctx.code.add(format!("; {}", terminator));
        match *terminator {
            Branch { ref cond, on_true, on_false } => {
                let cond_tmp = self.gen_load_var(llvm_symbol(cond).as_str(), "i1", ctx)?;
                ctx.code.add(format!("br i1 {}, label %b.b{}, label %b.b{}", cond_tmp, on_true, on_false));
            }

            ParallelFor(ref pf) => {
                // Generate the continuation function
                self.gen_top_level_function(sir, &sir.funcs[pf.cont])?;
                // Generate the functions to execute the loop
                self.gen_par_for_functions(pf, sir, &sir.funcs[pf.body])?;
                // Call into the loop.
                let params = get_combined_params(sir, pf);
                let params_sorted: BTreeMap<&Symbol, &Type> = params.iter().collect();
                let mut arg_types = String::new();
                for (arg, ty) in params_sorted.iter() {
                    let ll_ty = self.llvm_type(&ty)?;
                    let arg_tmp = self.gen_load_var(llvm_symbol(arg).as_str(), &ll_ty, ctx)?;
                    let arg_str = format!("{} {}, ", &ll_ty, arg_tmp);
                    arg_types.push_str(&arg_str);
                }
                arg_types.push_str("%work_t* %cur.work");
                ctx.code.add(format!("call void @f{}_wrapper({})", pf.body, arg_types));
                ctx.code.add("br label %body.end");
            }

            JumpBlock(block) => {
                ctx.code.add(format!("br label %b.b{}", block));
            }

            JumpFunction(func) => {
                self.gen_top_level_function(sir, &sir.funcs[func])?;
                let params_sorted: BTreeMap<&Symbol, &Type> = sir.funcs[func].params.iter().collect();
                let mut arg_types = String::new();
                for (arg, ty) in params_sorted.iter() {
                    let ll_ty = self.llvm_type(&ty)?;
                    let arg_tmp = self.gen_load_var(llvm_symbol(arg).as_str(), &ll_ty, ctx)?;
                    let arg_str = format!("{} {}, ", ll_ty, arg_tmp);
                    arg_types.push_str(&arg_str);
                }
                arg_types.push_str("%work_t* %cur.work");
                ctx.code.add(format!("call void @f{}({})", func, arg_types));
                ctx.code.add("br label %body.end");
            }

            ProgramReturn(ref sym) => {
                let ty = func.symbol_type(sym)?;
                let ty_str = self.llvm_type(ty)?;
                let res_tmp = self.gen_load_var(llvm_symbol(sym).as_str(), &ty_str, ctx)?;
                let elem_size_ptr = ctx.var_ids.next();
                let elem_size = ctx.var_ids.next();
                let elem_storage = ctx.var_ids.next();
                let elem_storage_typed = ctx.var_ids.next();
                let run_id = ctx.var_ids.next();
                ctx.code.add(format!("{} = getelementptr {}, {}* null, i32 1", &elem_size_ptr, &ty_str, &ty_str));
                ctx.code.add(format!("{} = ptrtoint {}* {} to i64", &elem_size, &ty_str, &elem_size_ptr));

                ctx.code.add(format!("{} = call i64 @weld_rt_get_run_id()", run_id));
                ctx.code.add(format!("{} = call i8* @weld_run_malloc(i64 {}, i64 {})",
                                        &elem_storage,
                                        &run_id,
                                        &elem_size));
                ctx.code.add(format!("{} = bitcast i8* {} to {}*", &elem_storage_typed, &elem_storage, &ty_str));
                self.gen_store_var(&res_tmp, &elem_storage_typed, &ty_str, ctx);
                ctx.code.add(format!("call void @weld_rt_set_result(i8* {})", elem_storage));
                ctx.code.add("br label %body.end");
            }

            EndFunction => {
                ctx.code.add("br label %body.end");
            }

            Crash => {
                let errno = WeldRuntimeErrno::Unknown as i64;
                let run_id = ctx.var_ids.next();
                ctx.code.add(format!("call void @weld_run_set_errno(i64 {}, i64 {})", run_id, errno));
            }
        }

        Ok(())
    }

    /// Generate code to extract the index'th element of a SIMD value, returning a register name for it.
    /// This is slightly tricky because the SIMD value may be a struct, in which case we need to recurse into each of
    /// its elements. `simd_reg` must be the name of an LLVM register containing the SIMD value.
    fn gen_simd_extract(&mut self,
                        simd_type: &Type,
                        simd_reg: &str,
                        index: u32,
                        ctx: &mut FunctionContext)
                        -> WeldResult<String> {

        let simd_ll_ty = self.llvm_type(simd_type)?;
        let item_type = simd_type.scalar_type()?;
        let item_ll_ty = self.llvm_type(&item_type)?;

        match *simd_type {
            Simd(_) => {
                // Simple case: extract the element using an extractelement instruction
                let item_reg = ctx.var_ids.next();
                ctx.code.add(format!("{} = extractelement {} {}, i32 {}", item_reg, simd_ll_ty, simd_reg, index));
                Ok(item_reg)
            }

            Struct(ref fields) => {
                // Complex case: build up a struct with each item
                let mut current_struct = "undef".to_string();
                for (i, field_type) in fields.iter().enumerate() {
                    // First, extract the i'th field of our SIMD struct into a register
                    let field_reg = ctx.var_ids.next();
                    ctx.code.add(format!("{} = extractvalue {} {}, {}", field_reg, simd_ll_ty, simd_reg, i));

                    // Call ourselves recursively to extract the right item from this sub-vector
                    let item_reg = self.gen_simd_extract(field_type, &field_reg, index, ctx)?;
                    let item_type = field_type.scalar_type()?;

                    // Insert this into our result struct
                    let new_struct = ctx.var_ids.next();
                    ctx.code.add(format!("{} = insertvalue {} {}, {} {}, {}",
                        new_struct, item_ll_ty, current_struct, self.llvm_type(&item_type)?, item_reg, i));
                    current_struct = new_struct;
                }
                Ok(current_struct)
            }

            _ => weld_err!("Invalid type for gen_simd_extract: {:?}", simd_type)
        }
    }
}

/// Converts an LLVM type string to a prefix for functions operating over the type.
fn llvm_prefix(ty_str: &str) -> String {
    format!("@{}", ty_str.replace("%", ""))
}

/// Returns a vector size for a type. If a Vetor is passed in, returns the vector size of the
/// element type. TODO this just returns 4 right now.
fn llvm_simd_size(_: &Type) -> WeldResult<u32> {
    Ok(4)
}

/// Returns the LLVM name for the `ScalarKind` `k`.
fn llvm_scalar_kind(k: ScalarKind) -> &'static str {
    match k {
        Bool => "i1",
        I8 => "i8",
        I16 => "i16",
        I32 => "i32",
        I64 => "i64",
        U8 => "%u8",
        U16 => "%u16",
        U32 => "%u32",
        U64 => "%u64",
        F32 => "float",
        F64 => "double",
    }
}

/// Returns the LLVM equality function for the `ScalarKind` `k`.
fn llvm_eq(k: ScalarKind) -> &'static str {
    match k {
        F32 | F64 => "oeq",
        _ => "eq",
    }
}


/// Returns the LLVM less than function for the `ScalarKind` `k`.
fn llvm_lt(k: ScalarKind) -> &'static str {
    match k {
        F32 | F64 => "olt",
        Bool | I8 | I16 | I32 | I64 => "slt",
        U8 | U16 | U32 | U64 => "ult",
    }
}

/// Returns an LLVM formatted String for a literal.
fn llvm_literal(k: LiteralKind) -> String {
    match k {
        BoolLiteral(l) => format!("{}", if l { 1 } else { 0 }),
        I8Literal(l) => format!("{}", l),
        I16Literal(l) => format!("{}", l),
        I32Literal(l) =>  format!("{}", l),
        I64Literal(l) => format!("{}", l),
        U8Literal(l) => format!("{}", l),
        U16Literal(l) => format!("{}", l),
        U32Literal(l) => format!("{}", l),
        U64Literal(l) => format!("{}", l),
        F32Literal(l) => format!("{:.30e}", l),
        F64Literal(l) => format!("{:.30e}", l),
    }.to_string()
}

/// Return the LLVM version of a Weld symbol (encoding any special characters for LLVM).
fn llvm_symbol(symbol: &Symbol) -> String {
    if symbol.id == 0 { format!("%{}", symbol.name) } else { format!("%{}.{}", symbol.name, symbol.id) }
}

fn binop_identity(op_kind: BinOpKind, ty: &Type) -> WeldResult<String> {
    use super::ast::BinOpKind::*;
    match (op_kind, ty) {
        (Add, &Scalar(s)) if s.is_integer() => Ok("0".to_string()),
        (Add, &Scalar(s)) if s.is_float() => Ok("0.0".to_string()),

        (Multiply, &Scalar(s)) if s.is_integer() => Ok("1".to_string()),
        (Multiply, &Scalar(s)) if s.is_float() => Ok("1.0".to_string()),

        _ => weld_err!("Unsupported identity for binary op: {} on {}", op_kind, print_type(ty)),
    }
}

/// Return the name of the LLVM instruction for a binary operation on a specific type.
fn llvm_binop(op_kind: BinOpKind, ty: &Type) -> WeldResult<&'static str> {
    use super::ast::BinOpKind::*;
    match ty {
        &Scalar(s) | &Simd(s) => {
            match op_kind {
                Add if s.is_integer() => Ok("add"),
                Add if s.is_float() => Ok("fadd"),

                Subtract if s.is_integer() => Ok("sub"),
                Subtract if s.is_float() => Ok("fsub"),

                Multiply if s.is_integer() => Ok("mul"),
                Multiply if s.is_float() => Ok("fmul"),

                Divide if s.is_signed_integer() => Ok("sdiv"),
                Divide if s.is_unsigned_integer() => Ok("udiv"),
                Divide if s.is_float() => Ok("fdiv"),

                Modulo if s.is_signed_integer() => Ok("srem"),
                Modulo if s.is_unsigned_integer() => Ok("urem"),
                Modulo if s.is_float() => Ok("frem"),

                Equal if s.is_integer() || s.is_bool() => Ok("icmp eq"),
                Equal if s.is_float() => Ok("fcmp oeq"),

                NotEqual if s.is_integer() || s.is_bool() => Ok("icmp ne"),
                NotEqual if s.is_float() => Ok("fcmp one"),

                LessThan if s.is_signed_integer() => Ok("icmp slt"),
                LessThan if s.is_unsigned_integer() => Ok("icmp ult"),
                LessThan if s.is_float() => Ok("fcmp olt"),

                LessThanOrEqual if s.is_signed_integer() => Ok("icmp sle"),
                LessThanOrEqual if s.is_unsigned_integer() => Ok("icmp ule"),
                LessThanOrEqual if s.is_float() => Ok("fcmp ole"),

                GreaterThan if s.is_signed_integer() => Ok("icmp sgt"),
                GreaterThan if s.is_unsigned_integer() => Ok("icmp ugt"),
                GreaterThan if s.is_float() => Ok("fcmp ogt"),

                GreaterThanOrEqual if s.is_signed_integer() => Ok("icmp sge"),
                GreaterThanOrEqual if s.is_unsigned_integer() => Ok("icmp uge"),
                GreaterThanOrEqual if s.is_float() => Ok("fcmp oge"),

                LogicalAnd if s.is_bool() => Ok("and"),
                BitwiseAnd if s.is_integer() || s.is_bool() => Ok("and"),

                LogicalOr if s.is_bool() => Ok("or"),
                BitwiseOr if s.is_integer() || s.is_bool() => Ok("or"),

                Xor if s.is_integer() || s.is_bool() => Ok("xor"),

                _ => return weld_err!("Unsupported binary op: {} on {}", op_kind, print_type(ty))
            }
        }

        _ => return weld_err!("Unsupported binary op: {} on {}", op_kind, print_type(ty))
    }
}

/// Return the name of the scalar LLVM instruction for the given operation and type.
fn llvm_scalar_unaryop(op_kind: UnaryOpKind, ty: &ScalarKind) -> WeldResult<&'static str> {
    match (op_kind, ty) {
        (UnaryOpKind::Log, &F32) => Ok("@llvm.log.f32"),
        (UnaryOpKind::Log, &F64) => Ok("@llvm.log.f64"),

        (UnaryOpKind::Exp, &F32) => Ok("@llvm.exp.f32"),
        (UnaryOpKind::Exp, &F64) => Ok("@llvm.exp.f64"),

        (UnaryOpKind::Sqrt, &F32) => Ok("@llvm.sqrt.f32"),
        (UnaryOpKind::Sqrt, &F64) => Ok("@llvm.sqrt.f64"),

        (UnaryOpKind::Erf, &F32) => Ok("@erff"),
        (UnaryOpKind::Erf, &F64) => Ok("@erf"),

        _ => weld_err!("Unsupported unary op: {} on {}", op_kind, ty),
    }
}

/// Return the name of the SIMD LLVM instruction for the given operation and type.
fn llvm_simd_unaryop(op_kind: UnaryOpKind, ty: &ScalarKind, width: u32) -> WeldResult<&'static str> {
    match (op_kind, ty, width) {
        (UnaryOpKind::Sqrt, &F32, 4) => Ok("@llvm.sqrt.v4f32"),
        (UnaryOpKind::Sqrt, &F32, 8) => Ok("@llvm.sqrt.v8f32"),
        (UnaryOpKind::Sqrt, &F64, 2) => Ok("@llvm.sqrt.v2f64"),
        (UnaryOpKind::Sqrt, &F64, 4) => Ok("@llvm.sqrt.v4f64"),

        (UnaryOpKind::Log, &F32, 4) => Ok("@llvm.log.v4f32"),
        (UnaryOpKind::Log, &F32, 8) => Ok("@llvm.log.v8f32"),
        (UnaryOpKind::Log, &F64, 2) => Ok("@llvm.log.v2f64"),
        (UnaryOpKind::Log, &F64, 4) => Ok("@llvm.log.v4f64"),

        (UnaryOpKind::Exp, &F32, 4) => Ok("@llvm.exp.v4f32"),
        (UnaryOpKind::Exp, &F32, 8) => Ok("@llvm.exp.v8f32"),
        (UnaryOpKind::Exp, &F64, 2) => Ok("@llvm.exp.v2f64"),
        (UnaryOpKind::Exp, &F64, 4) => Ok("@llvm.exp.v4f64"),

        _ => weld_err!("Unsupported unary op: {} on <{} x {}>", op_kind, width, ty),
    }
}

/// Return the name of the LLVM instruction for a binary operation between vectors.
fn llvm_binop_vector(op_kind: BinOpKind, ty: &Type) -> WeldResult<(&'static str, i32)> {
    match op_kind {
        BinOpKind::Equal => Ok(("eq", 0)),
        BinOpKind::NotEqual => Ok(("ne", 0)),
        BinOpKind::LessThan => Ok(("eq", -1)),
        BinOpKind::LessThanOrEqual => Ok(("ne", 1)),
        BinOpKind::GreaterThan => Ok(("eq", 1)),
        BinOpKind::GreaterThanOrEqual => Ok(("ne", -1)),

        _ => weld_err!("Unsupported binary op: {} on {}", op_kind, print_type(ty)),
    }
}

/// Return the name of hte LLVM instruction for a cast operation between specific types.
fn llvm_castop(ty1: &Type, ty2: &Type) -> WeldResult<&'static str> {
    match (ty1, ty2) {
        (&Scalar(s1), &Scalar(s2)) => {
            match (s1, s2) {
                (F32, F64) => Ok("fpext"),
                (F64, F32) => Ok("fptrunc"),

                (F32, _) if s2.is_signed_integer() => Ok("fptosi"),
                (F64, _) if s2.is_signed_integer() => Ok("fptosi"),

                (F32, _) => Ok("fptoui"),
                (F64, _) => Ok("fptoui"),

                (_, F32) if s1.is_signed_integer() => Ok("sitofp"),
                (_, F64) if s1.is_signed_integer() => Ok("sitofp"),

                (_, F32) => Ok("uitofp"),
                (_, F64) => Ok("uitofp"),

                (Bool, _) => Ok("zext"),

                (U8, _) if s2.bits() > 8 => Ok("zext"),
                (U16, _) if s2.bits() > 16 => Ok("zext"),
                (U32, _) if s2.bits() > 32 => Ok("zext"),
                (U64, _) if s2.bits() > 64 => Ok("zext"),

                (_, _) if s2.bits() > s1.bits() => Ok("sext"),

                (_, _) if s2.bits() < s1.bits() => Ok("trunc"),

                (_, _) if s2.bits() == s1.bits() => Ok("bitcast"),

                 _ => weld_err!("Can't cast {} to {}", print_type(ty1), print_type(ty2))
            }
        }

        _ => weld_err!("Can't cast {} to {}", print_type(ty1), print_type(ty2))

    }
}

/// Struct used to track state while generating a function.
struct FunctionContext {
    /// Code section at the start of the function with alloca instructions for local symbols
    alloca_code: CodeBuilder,
    /// Other code in function
    code: CodeBuilder,
    defined_symbols: HashSet<String>,
    var_ids: IdGenerator,
    is_innermost: bool
}

impl FunctionContext {
    fn new(_is_innermost: bool) -> FunctionContext {
        FunctionContext {
            alloca_code: CodeBuilder::new(),
            code: CodeBuilder::new(),
            var_ids: IdGenerator::new("%t.t"),
            defined_symbols: HashSet::new(),
            is_innermost: _is_innermost
        }
    }

    fn add_alloca(&mut self, symbol: &str, ty: &str) -> WeldResult<()> {
        if !self.defined_symbols.insert(symbol.to_string()) {
            weld_err!("Symbol already defined in function: {}", symbol)
        } else {
            self.alloca_code.add(format!("{} = alloca {}", symbol, ty));
            Ok(())
        }
    }
}

fn get_combined_params(sir: &SirProgram, par_for: &ParallelForData) -> HashMap<Symbol, Type> {
    let mut body_params = sir.funcs[par_for.body].params.clone();
    for (arg, ty) in sir.funcs[par_for.cont].params.iter() {
        body_params.insert(arg.clone(), ty.clone());
    }
    body_params
}

#[cfg(test)]
fn predicate_only(code: &str) -> WeldResult<TypedExpr> {
    let mut e = parse_expr(code).unwrap();
    assert!(type_inference::infer_types(&mut e).is_ok());
    let mut typed_e = e.to_typed().unwrap();

    let optstr = ["predicate"];
    let optpass = optstr.iter().map(|x| (*OPTIMIZATION_PASSES.get(x).unwrap()).clone()).collect();

    apply_opt_passes(&mut typed_e, &optpass)?;

    Ok(typed_e)
}

#[test]
fn predicate_iff_annotated() {
    /* Ensure predication is only applied if annotation is present. */

    /* annotation true */
    let code = "|v:vec[i32]| result(for(v, merger[i32,+], |b,i,e| @(predicate:true)if(e>0, merge(b,e), b)))";
    let typed_e = predicate_only(code);
    assert!(typed_e.is_ok());
    let expected = "|v:vec[i32]|result(for(v:vec[i32],merger[i32,+],|b:merger[i32,+],i:i64,e:i32|merge(b:merger[i32,+],select((e:i32>0),e:i32,0))))";
    assert_eq!(print_typed_expr_without_indent(&typed_e.unwrap()).as_str(),
               expected);

    /* annotation false */
    let code = "|v:vec[i32]| result(for(v, merger[i32,+], |b,i,e| @(predicate:false)if(e>0, merge(b,e), b)))";
    let typed_e = predicate_only(code);
    assert!(typed_e.is_ok());
    let expected = "|v:vec[i32]|result(for(v:vec[i32],merger[i32,+],|b:merger[i32,+],i:i64,e:i32|@(predicate:false)if((e:i32>0),merge(b:merger[i32,+],e:i32),b:merger[i32,+])))";
    assert_eq!(print_typed_expr_without_indent(&typed_e.unwrap()).as_str(),
               expected);

    /* annotation missing */
    let code = "|v:vec[i32]| result(for(v, merger[i32,+], |b,i,e| if(e>0, merge(b,e), b)))";
    let typed_e = predicate_only(code);
    assert!(typed_e.is_ok());
    let expected = "|v:vec[i32]|result(for(v:vec[i32],merger[i32,+],|b:merger[i32,+],i:i64,e:i32|if((e:i32>0),merge(b:merger[i32,+],e:i32),b:merger[i32,+])))";
    assert_eq!(print_typed_expr_without_indent(&typed_e.unwrap()).as_str(),
               expected);
}


#[test]
fn predicate_dictmerger() {
    /* Ensure predication is applied to mergers other than Merger. */

    let code = "|v:vec[i32]| result(for(v, dictmerger[{i32, i32},i32,+], |b,i,e| @(predicate:true)if(e>0, merge(b,{{e,e},e*2}), b)))";
    let typed_e = predicate_only(code);
    assert!(typed_e.is_ok());
    let expected = "|v:vec[i32]|result(for(v:vec[i32],dictmerger[{i32,i32},i32,+],|b:dictmerger[{i32,i32},i32,+],i:i64,e:i32|(let k:{{i32,i32},i32}=({{e:i32,e:i32},(e:i32*2)});select((e:i32>0),k:{{i32,i32},i32},{k:{{i32,i32},i32}.$0,0}))))";
    assert_eq!(expected,
               print_typed_expr_without_indent(&typed_e.unwrap()).as_str());
}

#[test]
fn types() {
    let mut gen = LlvmGenerator::new();

    assert_eq!(gen.llvm_type(&Scalar(I8)).unwrap(), "i8");
    assert_eq!(gen.llvm_type(&Scalar(I16)).unwrap(), "i16");
    assert_eq!(gen.llvm_type(&Scalar(I32)).unwrap(), "i32");
    assert_eq!(gen.llvm_type(&Scalar(I64)).unwrap(), "i64");
    assert_eq!(gen.llvm_type(&Scalar(U8)).unwrap(), "%u8");
    assert_eq!(gen.llvm_type(&Scalar(F32)).unwrap(), "float");
    assert_eq!(gen.llvm_type(&Scalar(F64)).unwrap(), "double");
    assert_eq!(gen.llvm_type(&Scalar(Bool)).unwrap(), "i1");

    let struct1 = parse_type("{i32,bool,i32}").unwrap().to_type().unwrap();
    assert_eq!(gen.llvm_type(&struct1).unwrap(), "%s0");
    assert_eq!(gen.llvm_type(&struct1).unwrap(), "%s0"); // Name is reused for same struct

    let struct2 = parse_type("{i32,bool}").unwrap().to_type().unwrap();
    assert_eq!(gen.llvm_type(&struct2).unwrap(), "%s1");
}
