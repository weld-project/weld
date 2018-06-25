use std::collections::HashSet;
use std::collections::BTreeMap;

use super::easy_ll;

extern crate time;
extern crate fnv;
extern crate code_builder;

use time::PreciseTime;
use code_builder::CodeBuilder;

use std::io::Write;
use std::path::PathBuf;
use std::fs::OpenOptions;

use optimizer::*;
use runtime::WeldRuntimeErrno;
use syntax::program::*;

use ast::*;
use ast::Type::*;
use ast::LiteralKind::*;
use ast::ScalarKind::*;
use ast::BuilderKind::*;
use error::*;
use syntax;
use runtime;
use sir;
use sir::*;
use sir::Statement;
use sir::StatementKind::*;
use sir::Terminator::*;
use sir::optimizations;
use util::IdGenerator;
use annotation::*;
use conf::ParsedConf;
use util::stats::CompilationStats;

#[cfg(test)]
use syntax::parser::*;

#[cfg(test)]
use tests::print_typed_expr_without_indent;

/// useful to make the code related to accessing elements from the array less verbose.
#[derive(Clone)]
pub struct VecLLVMInfo {
    pub ty_str: String,
    pub arr_str: String,
    pub prefix: String,
    pub len_str: String,
    pub el_ty_str: String,
}

static PRELUDE_CODE: &'static str = include_str!("resources/prelude.ll");
const WELD_INLINE_LIB: &'static [u8] = include_bytes!("../../../weld_rt/cpp/inline.bc");

/// The default grain size for the parallel runtime.
static DEFAULT_INNER_GRAIN_SIZE: i32 = 16384;
static DEFAULT_OUTER_GRAIN_SIZE: i32 = 4096;

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

/// A compiled module holding the generated LLVM module and some additional
/// information (e.g., the parameter and return types of the module).
pub struct CompiledModule {
    llvm_module: easy_ll::CompiledModule,
    param_types: Vec<Type>,
    return_type: Type,
}

impl CompiledModule {
    pub fn run(&self, data: i64) -> i64 {
        self.llvm_module.run(data)
    }

    /// Returns a mutable reference to the LLVM module.
    pub fn llvm_mut(&mut self) -> &mut easy_ll::CompiledModule {
        &mut self.llvm_module
    }

    /// Returns the parameter types of the module.
    pub fn param_types(&self) -> &Vec<Type> {
        &self.param_types
    }

    /// Returns the return type of the module.
    pub fn return_type(&self) -> &Type {
        &self.return_type
    }
}

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

/// Returns `true` if the target contains a pointer, or false otherwise.
trait HasPointer {
    fn has_pointer(&self) -> bool;
}

impl HasPointer for Type {
    fn has_pointer(&self) -> bool {
        match *self {
            Scalar(_) => false,
            Simd(_) => false,
            Vector(_) => true,
            Dict(_, _) => true,
            Builder(_, _) => true,
            Struct(ref tys) => tys.iter().any(|ref t| t.has_pointer()),
            Function(_, _) => true,
            Unknown => false,
        }
    }
}

/// Generate a compiled LLVM module from a program whose body is a function.
pub fn compile_program(program: &Program, conf: &ParsedConf, stats: &mut CompilationStats)
        -> WeldResult<CompiledModule> {
    let mut expr = syntax::macro_processor::process_program(program)?;
    debug!("After macro substitution:\n{}\n", expr.pretty_print());

    let start = PreciseTime::now();
    expr.uniquify()?;
    let end = PreciseTime::now();

    let mut uniquify_dur = start.to(end);

    let start = PreciseTime::now();
    expr.infer_types()?;
    let end = PreciseTime::now();
    debug!("After type inference:\n{}\n", expr.pretty_print());
    stats.weld_times.push(("Type Inference".to_string(), start.to(end)));

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
    debug!("SIR program:\n{}\n", &sir_prog);
    stats.weld_times.push(("AST to SIR".to_string(), start.to(end)));

    // Optimizations over the SIR.
    // TODO(shoumik): A pass manager like the one used over the AST representation will eventually
    // be useful.
    let start = PreciseTime::now();
    if conf.enable_sir_opt {
        info!("Applying SIR optimizations");
        optimizations::fold_constants::fold_constants(&mut sir_prog)?;
    }
    let end = PreciseTime::now();
    debug!("Optimized SIR program:\n{}\n", &sir_prog);
    stats.weld_times.push(("SIR Optimization".to_string(), start.to(end)));

    let start = PreciseTime::now();
    let mut gen = LlvmGenerator::new();
    gen.multithreaded = conf.support_multithread;
    gen.trace_run = conf.trace_run;

    if !gen.multithreaded {
        info!("Generating code without multithreading support");
    }

    if gen.trace_run {
        info!("Generating code with SIR tracing");
    }

    gen.add_function_on_pointers("run", &sir_prog)?;
    let llvm_code = gen.result();
    let end = PreciseTime::now();
    trace!("LLVM program:\n{}\n", &llvm_code);
    stats.weld_times.push(("LLVM Codegen".to_string(), start.to(end)));

    let ref timestamp = format!("{}", time::now().to_timespec().sec);

    // Dump files if needed. Do this here in case the actual LLVM code gen fails.
    if conf.dump_code.enabled {
        info!("Writing code to directory '{}' with timestamp {}", &conf.dump_code.dir.display(), timestamp);
        write_code(expr.pretty_print().as_ref(), "weld", timestamp, &conf.dump_code.dir);
        write_code(&format!("{}", &sir_prog), "sir", timestamp, &conf.dump_code.dir);
        write_code(&llvm_code, "ll", timestamp, &conf.dump_code.dir);
    }

    debug!("Started compiling LLVM");
    let compiled = try!(easy_ll::compile_module(
        &llvm_code,
        conf.llvm_optimization_level,
        conf.dump_code.enabled,
        Some(WELD_INLINE_LIB)));
    debug!("Done compiling LLVM");

    let module = compiled.module;
    let llvm_times = compiled.timing;
    let llvm_op_code = compiled.code;

    // Add LLVM statistics to the stats.
    for &(ref name, ref time) in llvm_times.times.iter() {
        stats.llvm_times.push((name.clone(), time.clone()));
    }

    debug!("Started runtime_init call");
    let start = PreciseTime::now();
    unsafe {
        runtime::weld_runtime_init();
    }
    let end = PreciseTime::now();
    debug!("Done runtime_init call");
    stats.weld_times.push(("Runtime Init".to_string(), start.to(end)));

    // Dump remaining files if needed.
    if conf.dump_code.enabled {
        let llvm_op_code = llvm_op_code.unwrap();
        // Write the optimized LLVM code and assembly.
        write_code(&llvm_op_code.optimized_llvm, "ll", format!("{}-opt", timestamp).as_ref(), &conf.dump_code.dir);
        write_code(&llvm_op_code.assembly, "S", format!("{}-opt", timestamp).as_ref(), &conf.dump_code.dir);
    }

    if let Function(ref param_tys, ref return_ty) = expr.ty {
        Ok(CompiledModule {
            llvm_module: module,
            param_types: param_tys.clone(),
            return_type: *return_ty.clone(),
        })
    } else {
        unreachable!();
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
    struct_names: fnv::FnvHashMap<Vec<Type>, String>,
    struct_ids: IdGenerator,

    /// LLVM type name of the form %v0, %v1, etc for each vec generated.
    vec_names: fnv::FnvHashMap<Type, String>,
    vec_ids: IdGenerator,
    growable_vec_names: fnv::FnvHashSet<Type>,

    // Key function id generator
    keyfunc_ids: IdGenerator,
    
    // LLVM type names for each merger type.
    merger_names: fnv::FnvHashMap<Type, String>,
    merger_ids: IdGenerator,

    /// LLVM type name of the form %d0, %d1, etc for each dict generated.
    dict_names: fnv::FnvHashMap<Type, String>,
    dict_ids: IdGenerator,

    /// Set of declared CUDFs.
    cudf_names: HashSet<String>,

    /// LLVM type names for various builder types
    bld_names: fnv::FnvHashMap<BuilderKind, String>,

    /// Pointer name for each declared string constant.
    string_names: fnv::FnvHashMap<String, String>,

    serialize_fns: fnv::FnvHashMap<Type, String>,
    deserialize_fns: fnv::FnvHashMap<Type, String>,

    /// A CodeBuilder and ID generator for prelude functions such as type and struct definitions.
    prelude_code: CodeBuilder,
    prelude_var_ids: IdGenerator,

    /// A CodeBuilder for body functions in the module.
    body_code: CodeBuilder,

    /// Helper function state for types.
    type_helpers: fnv::FnvHashMap<Type, HelperState>,

    /// Functions we have already visited when generating code.
    visited: HashSet<sir::FunctionId>,

    /// Multithreaded configuration set during compilation. If unset, performs
    /// single-threaded optimizations.
    multithreaded: bool,

    /// If true, compiles the program so that, at runtime, the generated program
    /// prints each SIR statement before evaluating it.
    trace_run: bool,
}

impl LlvmGenerator {
    pub fn new() -> LlvmGenerator {
        let mut generator = LlvmGenerator {
            struct_names: fnv::FnvHashMap::default(),
            struct_ids: IdGenerator::new("%s"),
            vec_names: fnv::FnvHashMap::default(),
            vec_ids: IdGenerator::new("%v"),
            growable_vec_names: fnv::FnvHashSet::default(),
            keyfunc_ids: IdGenerator::new("%sortkeyfunc"),
            merger_names: fnv::FnvHashMap::default(),
            merger_ids: IdGenerator::new("%m"),
            dict_names: fnv::FnvHashMap::default(),
            dict_ids: IdGenerator::new("%d"),
            serialize_fns: fnv::FnvHashMap::default(),
            deserialize_fns: fnv::FnvHashMap::default(),
            cudf_names: HashSet::new(),
            bld_names: fnv::FnvHashMap::default(),
            string_names: fnv::FnvHashMap::default(),
            prelude_code: CodeBuilder::new(),
            prelude_var_ids: IdGenerator::new("%p.p"),
            body_code: CodeBuilder::new(),
            visited: HashSet::new(),
            type_helpers: fnv::FnvHashMap::default(),
            multithreaded: false,
            trace_run: false
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
    fn get_arg_str(&mut self, params_sorted: &BTreeMap<Symbol, Type>, suffix: &str) -> WeldResult<String> {
        let mut arg_types = String::new();
        for (arg, ty) in params_sorted.iter() {
            let arg_str = format!("{} {}{}, ", self.llvm_type(&ty)?, llvm_symbol(&arg), suffix);
            arg_types.push_str(&arg_str);
        }
        arg_types.push_str("%work_t* %cur.work");
        Ok(arg_types)
    }

    fn gen_store_args(&mut self, params_sorted: &BTreeMap<Symbol, Type>, input_suffix: &str, stored_suffix: &str,
        ctx: &mut FunctionContext) -> WeldResult<()> {
        for (_, (arg, ty)) in params_sorted.iter().enumerate() {
            let ll_ty = self.llvm_type(ty)?;
            let ll_sym = format!("{}{}", llvm_symbol(arg), stored_suffix);
            ctx.add_alloca(&ll_sym, &ll_ty)?;
            ctx.code.add(format!("store {} {}{}, {}* {}", ll_ty, llvm_symbol(arg), input_suffix, ll_ty, ll_sym));
        }
        Ok(())
    }

    fn gen_load_args(&mut self, params_sorted: &BTreeMap<Symbol, Type>, loaded_suffix: &str, stored_suffix: &str,
        ctx: &mut FunctionContext) -> WeldResult<()> {
        for (_, (arg, ty)) in params_sorted.iter().enumerate() {
            let ll_ty = self.llvm_type(ty)?;
            ctx.code.add(format!("{}{} = load {}, {}* {}{}", llvm_symbol(arg), loaded_suffix, ll_ty, ll_ty, llvm_symbol(arg),
                stored_suffix));
        }
        Ok(())
    }

    /// Generates code to unpack a struct containing a set of arguments with the given symbols and
    /// types. The order of arguments is assumed to be sorted by the symbol name.
    fn gen_unload_arg_struct(&mut self, params_sorted: &BTreeMap<Symbol, Type>, suffix: &str, ctx: &mut FunctionContext) -> WeldResult<()> {
        let ref struct_ty = Struct(params_sorted.values().map(|e| e.clone()).collect());
        let arg_struct_ll_ty = self.llvm_type(struct_ty)?;
        let storage_ptr = ctx.var_ids.next();
        let work_data_ptr = ctx.var_ids.next();
        let work_data = ctx.var_ids.next();
        ctx.code.add(format!("{} = getelementptr inbounds %work_t, %work_t* %cur.work, i32 0, i32 0", work_data_ptr));
        ctx.code.add(format!("{} = load i8*, i8** {}", work_data, work_data_ptr));
        ctx.code.add(format!("{} = bitcast i8* {} to {}*", storage_ptr, work_data, arg_struct_ll_ty));
        for (i, (arg, ty)) in params_sorted.iter().enumerate() {
            let ptr_tmp = ctx.var_ids.next();
            let arg_ll_ty = self.llvm_type(ty)?;
            ctx.code.add(format!("{} = getelementptr inbounds {}, {}* {}, i32 0, i32 {}",
                ptr_tmp, arg_struct_ll_ty, arg_struct_ll_ty, storage_ptr, i));
            ctx.code.add(format!("{}{} = load {}, {}* {}", llvm_symbol(arg), suffix, arg_ll_ty, arg_ll_ty, ptr_tmp));
        }
        Ok(())
    }

    fn gen_create_new_vb_pieces_helper(&mut self, sym: &str, ty: &Type, inner_ctx: &mut FunctionContext,
        outer_ctx: &mut FunctionContext) -> WeldResult<(bool)> {
        let mut has_builder = false;
        match *ty {
            Builder(ref bk, _) => {
                match *bk {
                    Appender(_) => {
                        let bld_ty_str = self.llvm_type(ty)?;
                        let bld_prefix = llvm_prefix(&bld_ty_str);
                        inner_ctx.code.add(format!("call void {}.newPiece({} {}, %work_t* %cur.work)",
                                                bld_prefix,
                                                bld_ty_str,
                                                sym));
                        has_builder = true;
                    }
                    _ => {}
                }
            }
            Struct(ref fields) => {
                for (i, f) in fields.iter().enumerate() {
                    let struct_ty = self.llvm_type(ty)?;
                    let bld_sym = outer_ctx.var_ids.next();
                    inner_ctx.code.add(format!("{} = extractvalue {} {}, {}", bld_sym, struct_ty,
                        sym, i));
                    has_builder |= self.gen_create_new_vb_pieces_helper(&bld_sym, f, inner_ctx, outer_ctx)?;
                }
            }
            _ => {}
        }
        Ok(has_builder)
    }

    /// Generates code to create new pieces for the appender.
    fn gen_create_new_vb_pieces(&mut self, params_sorted: &BTreeMap<Symbol, Type>, suffix: &str, ctx: &mut FunctionContext) -> WeldResult<()> {
        let full_task_ptr = ctx.var_ids.next();
        let full_task_int = ctx.var_ids.next();
        let full_task_bit = ctx.var_ids.next();
        ctx.code.add(format!("{} = getelementptr inbounds %work_t, %work_t* %cur.work, i32 0, i32 4", full_task_ptr));
        ctx.code.add(format!("{} = load i32, i32* {}", full_task_int, full_task_ptr));
        ctx.code.add(format!("{} = trunc i32 {} to i1", full_task_bit, full_task_int));
        ctx.code.add(format!("br i1 {}, label %new_pieces, label %fn_call", full_task_bit));
        ctx.code.add("new_pieces:");
        for (arg, ty) in params_sorted.iter() {
            let inner_ctx = &mut FunctionContext::new(false);
            let has_builder = self.gen_create_new_vb_pieces_helper(
                format!("{}{}", llvm_symbol(arg), suffix).as_str(), ty, inner_ctx, ctx)?;
            if has_builder {
                ctx.code.add(&inner_ctx.code.result());
            }
        }
        ctx.code.add("br label %fn_call");
        Ok(())
    }

    fn gen_create_stack_mergers_helper(&mut self, sym: &str, ty: &Type, inner_ctx: &mut FunctionContext,
        outer_ctx: &mut FunctionContext) -> WeldResult<(bool)> {
        let mut has_builder = false;
        match *ty {
            Builder(ref bk, _) => {
                match *bk {
                    Merger(ref val_ty, ref op) => {
                        let bld_ll_ty = self.llvm_type(ty)?;
                        let bld_prefix = llvm_prefix(&bld_ll_ty);
                        let bld_ll_stack_sym = format!("{}.stack", sym);
                        let bld_ll_stack_ty = format!("{}.piece", bld_ll_ty);
                        let val_ll_scalar_ty = self.llvm_type(val_ty)?;
                        let iden_elem = binop_identity(*op, val_ty.as_ref())?;
                        outer_ctx.add_alloca(&bld_ll_stack_sym, &bld_ll_stack_ty)?;
                        inner_ctx.code.add(format!(
                            "call void {}.insertStackPiece({}* {}, {}.piecePtr {})",
                            bld_prefix,
                            bld_ll_ty,
                            sym,
                            bld_ll_ty,
                            bld_ll_stack_sym));
                        inner_ctx.code.add(format!(
                            "call void {}.clearStackPiece({}* {}, {} {})",
                            bld_prefix,
                            bld_ll_ty,
                            sym,
                            val_ll_scalar_ty,
                            iden_elem
                            ));
                        has_builder = true;
                    }
                    _ => {}
                }
            }
            Struct(ref fields) => {
                for (i, f) in fields.iter().enumerate() {
                    let struct_ty = self.llvm_type(ty)?;
                    let bld_sym = outer_ctx.var_ids.next();
                    inner_ctx.code.add(format!("{} = getelementptr {}, {}* {}, i32 0, i32 {}", bld_sym, struct_ty,
                        struct_ty, sym, i));
                    has_builder |= self.gen_create_stack_mergers_helper(&bld_sym, f, inner_ctx, outer_ctx)?;
                }
            }
            _ => {}
        }
        Ok(has_builder)
    }

    /// Generates code to create new stack-based storage for mergers at the start of a serial code sequence.
    fn gen_create_stack_mergers(&mut self, params_sorted: &BTreeMap<Symbol, Type>, ctx: &mut FunctionContext) -> WeldResult<()> {
        for (arg, ty) in params_sorted.iter() {
            let inner_ctx = &mut FunctionContext::new(false);
            let has_builder = self.gen_create_stack_mergers_helper(&llvm_symbol(arg), ty, inner_ctx, ctx)?;
            if has_builder {
                ctx.code.add(&inner_ctx.code.result());
            }
        }
        Ok(())
    }

    /// Generates code to create register-based mergers to be used inside innermost loops.
    fn gen_create_new_merger_regs(&mut self, params_sorted: &BTreeMap<Symbol, Type>, ctx: &mut FunctionContext) -> WeldResult<()> {
        for (arg, ty) in params_sorted.iter() {
            match *ty {
                Builder(ref bk, _) => {
                    match *bk {
                        Merger(ref val_ty, ref op) => {
                            let bld_ll_ty = self.llvm_type(ty)?;
                            let bld_ll_sym = llvm_symbol(arg);
                            let bld_ll_reg_sym = format!("{}.reg", bld_ll_sym);
                            let bld_ll_reg_ty = format!("{}.piece", bld_ll_ty);
                            ctx.add_alloca(&bld_ll_reg_sym, &bld_ll_reg_ty)?;
                            let val_ll_ty = self.llvm_type(val_ty)?;
                            let iden_elem = binop_identity(*op, val_ty.as_ref())?;
                            let bld_prefix = llvm_prefix(&bld_ll_ty);
                            ctx.code.add(format!(
                                "call void {}.clearPiece({}* {}, {} {})",
                                bld_prefix,
                                bld_ll_reg_ty,
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

    /// Generates code to create global mergers when parallel work is about to be spawned.
    fn gen_create_global_mergers(&mut self, params_sorted: &BTreeMap<Symbol, Type>, suffix: &str, ctx: &mut FunctionContext) -> WeldResult<()> {
        for (arg, ty) in params_sorted.iter() {
            match *ty {
                Builder(ref bk, _) => {
                    match *bk {
                        Merger(ref val_ty, ref op) => {
                            let bld_ll_ty = self.llvm_type(ty)?;
                            let bld_ll_sym = llvm_symbol(arg);
                            let val_ll_scalar_ty = self.llvm_type(val_ty)?;
                            let val_ll_simd_ty = self.llvm_type(&val_ty.simd_type()?)?;
                            let iden_elem = binop_identity(*op, val_ty.as_ref())?;
                            let bld_prefix = llvm_prefix(&bld_ll_ty);
                            let bld_ptr_raw = ctx.var_ids.next();
                            let bld_ptr_scalar = ctx.var_ids.next();
                            let bld_ptr_simd = ctx.var_ids.next();
                            let stack_ptr_scalar = ctx.var_ids.next();
                            let stack_ptr_simd = ctx.var_ids.next();
                            ctx.code.add(format!(
                                "call void {}.initGlobalIfNeeded({}* {}{}, {} {})",
                                bld_prefix,
                                bld_ll_ty,
                                bld_ll_sym,
                                suffix,
                                val_ll_scalar_ty,
                                iden_elem));
                            ctx.code.add(format!(
                                "{} = call {}.piecePtr {}.getPtrIndexed({}* {}{}, i32 %cur.tid)",
                                bld_ptr_raw,
                                bld_ll_ty,
                                bld_prefix,
                                bld_ll_ty,
                                bld_ll_sym,
                                suffix));
                            ctx.code.add(format!(
                                "{} = call {}* {}.vectorMergePtrForPiece({}.piecePtr {})",
                                bld_ptr_simd,
                                val_ll_simd_ty,
                                bld_prefix,
                                bld_ll_ty,
                                bld_ptr_raw));
                            ctx.code.add(format!(
                                "{} = call {}* {}.scalarMergePtrForPiece({}.piecePtr {})",
                                bld_ptr_scalar,
                                val_ll_scalar_ty,
                                bld_prefix,
                                bld_ll_ty,
                                bld_ptr_raw));
                            ctx.code.add(format!(
                                "{} = call {}* {}.vectorMergePtrForStackPiece({}* {}{})",
                                stack_ptr_simd,
                                val_ll_simd_ty,
                                bld_prefix,
                                bld_ll_ty,
                                bld_ll_sym,
                                suffix));
                            ctx.code.add(format!(
                                "{} = call {}* {}.scalarMergePtrForStackPiece({}* {}{})",
                                stack_ptr_scalar,
                                val_ll_scalar_ty,
                                bld_prefix,
                                bld_ll_ty,
                                bld_ll_sym,
                                suffix));
                            let stack_simd = self.gen_load_var(&stack_ptr_simd, &val_ll_simd_ty, ctx)?;
                            let stack_scalar = self.gen_load_var(&stack_ptr_scalar, &val_ll_scalar_ty, ctx)?;
                            self.gen_merge_op(&bld_ptr_simd, &stack_simd, &val_ll_simd_ty, op, &val_ty.simd_type()?, ctx)?;
                            self.gen_merge_op(&bld_ptr_scalar, &stack_scalar, &val_ll_scalar_ty, op, val_ty, ctx)?;
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Generates code to store stack-based mergers back to their global counterparts if they exist.
    fn gen_store_stack_mergers(&mut self, params_sorted: &BTreeMap<Symbol, Type>, ctx: &mut FunctionContext) -> WeldResult<()> {
        for (arg, ty) in params_sorted.iter() {
            match *ty {
                Builder(ref bk, _) => {
                    match *bk {
                        Merger(ref val_ty, ref op) => {
                            let next_label = ctx.var_ids.next();
                            let cur_label = ctx.var_ids.next();
                            let bld_ll_ty = self.llvm_type(ty)?;
                            let bld_ll_sym = llvm_symbol(arg);
                            let val_ll_scalar_ty = self.llvm_type(val_ty)?;
                            let val_ll_simd_ty = self.llvm_type(&val_ty.simd_type()?)?;
                            let bld_prefix = llvm_prefix(&bld_ll_ty);
                            let bld_ptr_raw = ctx.var_ids.next();
                            let bld_ptr_scalar = ctx.var_ids.next();
                            let bld_ptr_simd = ctx.var_ids.next();
                            let stack_ptr_scalar = ctx.var_ids.next();
                            let stack_ptr_simd = ctx.var_ids.next();
                            let is_global = ctx.var_ids.next();
                            ctx.code.add(format!(
                                "{} = call i1 {}.isGlobal({}* {})",
                                is_global,
                                bld_prefix,
                                bld_ll_ty,
                                bld_ll_sym));
                            ctx.code.add(format!("br i1 {}, label {}, label {}", is_global, cur_label, next_label));
                            ctx.code.add(format!("{}:", cur_label.replace("%", "")));
                            ctx.code.add(format!(
                                "{} = call {}.piecePtr {}.getPtrIndexed({}* {}, i32 %cur.tid)",
                                bld_ptr_raw,
                                bld_ll_ty,
                                bld_prefix,
                                bld_ll_ty,
                                bld_ll_sym));
                            ctx.code.add(format!(
                                "{} = call {}* {}.vectorMergePtrForPiece({}.piecePtr {})",
                                bld_ptr_simd,
                                val_ll_simd_ty,
                                bld_prefix,
                                bld_ll_ty,
                                bld_ptr_raw));
                            ctx.code.add(format!(
                                "{} = call {}* {}.scalarMergePtrForPiece({}.piecePtr {})",
                                bld_ptr_scalar,
                                val_ll_scalar_ty,
                                bld_prefix,
                                bld_ll_ty,
                                bld_ptr_raw));
                            ctx.code.add(format!(
                                "{} = call {}* {}.vectorMergePtrForStackPiece({}* {})",
                                stack_ptr_simd,
                                val_ll_simd_ty,
                                bld_prefix,
                                bld_ll_ty,
                                bld_ll_sym));
                            ctx.code.add(format!(
                                "{} = call {}* {}.scalarMergePtrForStackPiece({}* {})",
                                stack_ptr_scalar,
                                val_ll_scalar_ty,
                                bld_prefix,
                                bld_ll_ty,
                                bld_ll_sym));
                            let stack_simd = self.gen_load_var(&stack_ptr_simd, &val_ll_simd_ty, ctx)?;
                            let stack_scalar = self.gen_load_var(&stack_ptr_scalar, &val_ll_scalar_ty, ctx)?;
                            self.gen_merge_op(&bld_ptr_simd, &stack_simd, &val_ll_simd_ty, op, &val_ty.simd_type()?, ctx)?;
                            self.gen_merge_op(&bld_ptr_scalar, &stack_scalar, &val_ll_scalar_ty, op, val_ty, ctx)?;
                            ctx.code.add(format!("br label {}", next_label));
                            ctx.code.add(format!("{}:", next_label.replace("%", "")));
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Generates code to store register-based mergers back to their stack-based counterparts.
    fn gen_store_merger_regs(&mut self, params_sorted: &BTreeMap<Symbol, Type>, ctx: &mut FunctionContext) -> WeldResult<()> {
        for (arg, ty) in params_sorted.iter() {
            match *ty {
                Builder(ref bk, _) => {
                    match *bk {
                        Merger(ref val_ty, ref op) => {
                            let bld_ll_ty = self.llvm_type(ty)?;
                            let bld_ll_sym = llvm_symbol(arg);
                            let bld_ll_reg_sym = format!("{}.reg", llvm_symbol(arg));
                            let val_ll_scalar_ty = self.llvm_type(val_ty)?;
                            let val_ll_simd_ty = self.llvm_type(&val_ty.simd_type()?)?;
                            let bld_prefix = llvm_prefix(&bld_ll_ty);
                            let bld_ptr_scalar = ctx.var_ids.next();
                            let bld_ptr_simd = ctx.var_ids.next();
                            let reg_ptr_scalar = ctx.var_ids.next();
                            let reg_ptr_simd = ctx.var_ids.next();
                            ctx.code.add(format!(
                                "{} = call {}* {}.vectorMergePtrForStackPiece({}* {})",
                                bld_ptr_simd,
                                val_ll_simd_ty,
                                bld_prefix,
                                bld_ll_ty,
                                bld_ll_sym));
                            ctx.code.add(format!(
                                "{} = call {}* {}.scalarMergePtrForStackPiece({}* {})",
                                bld_ptr_scalar,
                                val_ll_scalar_ty,
                                bld_prefix,
                                bld_ll_ty,
                                bld_ll_sym));
                            ctx.code.add(format!(
                                "{} = call {}* {}.vectorMergePtrForPiece({}.piecePtr {})",
                                reg_ptr_simd,
                                val_ll_simd_ty,
                                bld_prefix,
                                bld_ll_ty,
                                bld_ll_reg_sym));
                            ctx.code.add(format!(
                                "{} = call {}* {}.scalarMergePtrForPiece({}.piecePtr {})",
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
    fn gen_create_arg_struct(&mut self, params_sorted: &BTreeMap<Symbol, Type>, suffix: &str, ctx: &mut FunctionContext) -> WeldResult<String> {
        let mut prev_ref = String::from("undef");
        let ll_ty = self.llvm_type(&Struct(params_sorted.values().map(|e| e.clone()).collect()))?
            .to_string();
        for (i, (arg, ty)) in params_sorted.iter().enumerate() {
            let next_ref = ctx.var_ids.next();
            ctx.code.add(format!("{} = insertvalue {} {}, {} {}{}, {}",
                                 next_ref,
                                 ll_ty,
                                 prev_ref,
                                 self.llvm_type(&ty)?,
                                 llvm_symbol(arg),
                                 suffix,
                                 i));
            prev_ref.clear();
            prev_ref.push_str(&next_ref);
        }
        let struct_size_ptr = ctx.var_ids.next();
        let struct_size = ctx.var_ids.next();
        let struct_storage = ctx.var_ids.next();
        let struct_storage_typed = ctx.var_ids.next();
        ctx.code.add(format!("{} = getelementptr inbounds {}, {}* null, i32 1", struct_size_ptr, ll_ty, ll_ty));
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
        let first_iter = &par_for.data[0];
        let data_str = llvm_symbol(&first_data);
        let data_ty_str = self.llvm_type(func.params.get(&first_data).unwrap())?;
        let data_prefix = llvm_prefix(&data_ty_str);
        let num_iters_str = ctx.var_ids.next();
        let mut fringe_start_str = None;

        match par_for.data[0].kind {
            IterKind::SimdIter | IterKind::ScalarIter => {
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
                        return compile_err!("vector iterator does not support non-unit stride");
                    }
                    // set num_iters_str to (end - start) / stride
                    let start_str = llvm_symbol(&par_for.data[0].start.clone().unwrap());
                    let end_str = llvm_symbol(&par_for.data[0].end.clone().unwrap());
                    let stride_str = llvm_symbol(&par_for.data[0].stride.clone().unwrap());
                    let diff_tmp = ctx.var_ids.next();
                    ctx.code.add(format!("{} = sub i64 {}, {}", diff_tmp, end_str, start_str));
                    ctx.code.add(format!("{} = udiv i64 {}, {}", num_iters_str, diff_tmp, stride_str));
                }
            },
            IterKind::FringeIter => {
                if par_for.data[0].start.is_some() {
                    return compile_err!("fringe iterator does not support non-unit stride");
                }
                let arr_len = ctx.var_ids.next();
                let tmp = ctx.var_ids.next();
                let vector_len = format!("{}", llvm_simd_size(func.symbol_type(&first_data)?)?);

                ctx.code.add(format!("{} = call i64 {}.size({} {})", arr_len, data_prefix, data_ty_str, data_str));

                // num_iters = arr_len % llvm_simd_size // number of iterations
                // tmp2 = arr_len - num_iters // start index
                ctx.code.add(format!("{} = urem i64 {}, {}", num_iters_str, arr_len, vector_len));
                ctx.code.add(format!("{} = sub nuw nsw i64 {}, {}", tmp, arr_len, num_iters_str));

                // TODO somewhat hacky way to ensure the fringe for fixed-size appender loop writes at the
                // appropriate offset (without this the offset is set to w->cur_idx at the start of the
                // main loop body's continuation)
                let bld_ty = func.symbol_type(&par_for.builder)?;
                if let Builder(ref bk, _) = *bld_ty {
                    match *bk {
                        Appender(_) => {
                            let bld_ty_str = self.llvm_type(bld_ty)?;
                            let bld_prefix = llvm_prefix(&bld_ty_str);
                            ctx.code.add(format!("call void {}.setOffsetIfFixed({} {}, i64 {})",
                                                 bld_prefix,
                                                 bld_ty_str,
                                                 llvm_symbol(&par_for.builder),
                                                 tmp));
                        }
                        _ => {}
                    }
                }
                fringe_start_str = Some(tmp);
            },
            IterKind::RangeIter => {
                // set num_iters_str to (end - start) / stride
                let start_str = llvm_symbol(&par_for.data[0].start.clone().unwrap());
                let end_str = llvm_symbol(&par_for.data[0].end.clone().unwrap());
                let stride_str = llvm_symbol(&par_for.data[0].stride.clone().unwrap());
                let diff_tmp = ctx.var_ids.next();
                ctx.code.add(format!("{} = sub i64 {}, {}", diff_tmp, end_str, start_str));
                ctx.code.add(format!("{} = udiv i64 {}, {}", num_iters_str, diff_tmp, stride_str));
            }
            IterKind::NdIter => {
                /* llvm code for:
                 * end = shape[0]*shape[1]...*shape[n-1] 
                 * num_iters_str = end - start */
                let shape_el_ty = self.llvm_type(&Scalar(I64))?;
                let shape_llvm_info = self.get_array_llvm_info(func, ctx, first_iter.shape.as_ref().unwrap(), 
                                                                shape_el_ty, false)?; 
                let prod_ptr = ctx.var_ids.next();
                ctx.code.add(format!("{} = alloca i64", prod_ptr));
                ctx.code.add(format!("store i64 1, i64* {}", prod_ptr)); 
                let loop_name = "gen_num_iters_loop";
                let (cur_i_ptr, cur_i) = self.add_llvm_for_loop_start(ctx, &loop_name, "0", 
                                                   &shape_llvm_info.len_str, "slt")?;
                let shape_i = self.get_array_idx(ctx, shape_llvm_info, false, &cur_i)?;
                let prod = ctx.var_ids.next();
                ctx.code.add(format!("{} = load i64, i64* {}", prod, prod_ptr));
                let tmp_result = ctx.var_ids.next();
                ctx.code.add(format!("{} = mul i64 {}, {}", tmp_result, prod, shape_i)); 
                /* store back in prod_ptr */
                ctx.code.add(format!("store i64 {}, i64* {}", tmp_result, prod_ptr));
                /* Loop body done, so now update cur_i, and jump back to loop.start. */
                self.add_llvm_for_loop_end(ctx, &loop_name, &cur_i_ptr, &cur_i, "add");
                /* Now, prod_ptr should have the correct value for num_iters_str.
                 * Note: compared to the scalar/simd case, we don't need to consider start/end */
                ctx.code.add(format!("{} = load i64, i64* {}", num_iters_str, prod_ptr));
            }
        }
        Ok((String::from(num_iters_str), fringe_start_str))
    }

    /// Generates a bounds check for a parallel for loop by ensuring that the number of iterations
    /// does not cause an out of bounds error with the given start and stride.
    /// Follows gen_num_iters_and_fringe_start
    /// Precedes gen_invoke_loop_body
    fn gen_loop_bounds_check(&mut self,
                                 fringe_start_str: Option<String>,
                                 num_iters_str: &str,
                                 par_for: &ParallelForData,
                                 func: &SirFunction,
                                 ctx: &mut FunctionContext) -> WeldResult<()> {
        // We require a bounds check if (a) we have more than one iterator, or (b) if the iterator
        // has a user-defined iteration pattern (start, end, stride).
        if par_for.data.len() > 1 || par_for.data[0].start.is_some() {
            for (i, iter) in par_for.data.iter().enumerate() {
                if iter.kind == IterKind::NdIter {
                    /* Note: here we can't use num_iters_str as a proxy for end as the array may not be
                     * contiguous */
                    let data_llvm_info = self.get_array_llvm_info(func, ctx, &iter.data, "".to_string(), false)?;
                    /* Equivalent C code is:
                     * int offset = 0;
                     * for (i = 0; i < len(shape); i++) {
                     *     int max_i = shape[i] - 1;
                     *     int stride_i = strides[i];
                     *     offset += max_i*stride_i;
                     * }
                     * int max_val = start + offset;
                     * cmp max_val, data_size_ll_tmp
                     */
                    let el_ty = self.llvm_type(&Scalar(I64))?;
                    /* both these have same type */
                    let shape_llvm_info = self.get_array_llvm_info(func, ctx, iter.shape.as_ref().unwrap(), el_ty.clone(), false)?;
                    let strides_llvm_info = self.get_array_llvm_info(func, ctx, iter.strides.as_ref().unwrap(), el_ty, false)?;
                    let offset_ptr = ctx.var_ids.next();
                    ctx.code.add(format!("{} = alloca i64", offset_ptr));
                    ctx.code.add(format!("store i64 0, i64* {}", offset_ptr));
                    let loop_name = format!("boundscheck_loop{}", i);
                    let (cur_i_ptr, cur_i) = self.add_llvm_for_loop_start(ctx, &loop_name, "0",
                                               &strides_llvm_info.len_str, "slt")?;
                    let shape_i = self.get_array_idx(ctx, shape_llvm_info, false, &cur_i)?;
                    let strides_i = self.get_array_idx(ctx, strides_llvm_info, false, &cur_i)?;
                    let (tmp_prod, max_i, cur_offset) = (ctx.var_ids.next(), ctx.var_ids.next(), ctx.var_ids.next());
                    ctx.code.add(format!("{} = sub i64 {}, 1", max_i, shape_i));
                    ctx.code.add(format!("{} = mul i64 {}, {}", tmp_prod, max_i, strides_i));
                    ctx.code.add(format!("{} = load i64, i64* {}", cur_offset, offset_ptr));
                    let new_offset = ctx.var_ids.next();
                    ctx.code.add(format!("{} = add i64 {}, {}", new_offset, cur_offset, tmp_prod));
                    ctx.code.add(format!("store i64 {}, i64* {}", new_offset, offset_ptr));
                    /* generic boilerplate to end loop */
                    self.add_llvm_for_loop_end(ctx, &loop_name, &cur_i_ptr, &cur_i, "add");
                    /* start + offset should be the correct limit now */
                    let start_str = llvm_symbol(iter.start.as_ref().unwrap());
                    let (max_iter_str, final_offset) = (ctx.var_ids.next(), ctx.var_ids.next()); 
                    ctx.code.add(format!("{} = load i64, i64* {}", final_offset, offset_ptr));
                    ctx.code.add(format!("{} = add i64 {}, {}", max_iter_str, start_str, final_offset));
                    /* if max_iter_str > data_len, then bad, else all is good */ 
                    let (next_bounds_check_label, cond) = (ctx.var_ids.next(), ctx.var_ids.next());
                    /* Since start = 0, max_iter_str can be at most len(data)-1 */
                    ctx.code.add(format!("{} = icmp slt i64 {}, {}", cond, max_iter_str, data_llvm_info.len_str));
                    ctx.code.add(format!("br i1 {}, label {}, label %fn.boundcheckfailed", cond, next_bounds_check_label));
                    ctx.code.add(format!("{}:", next_bounds_check_label.replace("%", "")));
                } else {

                    let (data_ll_ty, data_ll_sym) = self.llvm_type_and_name(func, &iter.data)?;
                    let data_prefix = llvm_prefix(&data_ll_ty);

                    let mut data_size_ll_tmp = ctx.var_ids.next();
                    ctx.code.add(format!("{} = call i64 {}.size({} {})",
                        data_size_ll_tmp, data_prefix, data_ll_ty, data_ll_sym));

                    // Obtain the start and stride values.
                    let (start_str, end_str, stride_str) = if iter.start.is_none() {
                        // We already checked to make sure the FringeIter doesn't have a start, etc.
                        let start_str = match iter.kind {
                            IterKind::FringeIter => fringe_start_str.as_ref().unwrap().to_string(),
                            _ => String::from("0")
                        };
                        let stride_str = String::from("1");
                        (start_str, None, stride_str)
                    } else {
                        (
                            llvm_symbol(iter.start.as_ref().unwrap()),
                            if iter.kind == IterKind::RangeIter {
                                Some(llvm_symbol(iter.end.as_ref().unwrap()))
                            } else { None },
                            llvm_symbol(iter.stride.as_ref().unwrap())
                        )
                    };

                    let t0 = ctx.var_ids.next();
                    let t1 = ctx.var_ids.next();
                    let t2 = ctx.var_ids.next();
                    let cond = ctx.var_ids.next();
                    let next_bounds_check_label = ctx.var_ids.next();

                    // For range iterators, use the specified end instead of the data size.
                    if let Some(end_str) = end_str {
                        data_size_ll_tmp = end_str;
                    }

                    // TODO just compare against end here...this computation is redundant.
                    // t0 = sub i64 num_iters, 1
                    // t1 = mul i64 stride, t0
                    // t2 = add i64 t1, start
                    // cond = icmp lte i64 t2, size
                    // br i1 cond, label %nextCheck, label %checkFailed
                    // nextCheck:
                    // (loop)
                    ctx.code.add(format!("{} = sub i64 {}, 1", t0, num_iters_str));
                    ctx.code.add(format!("{} = mul i64 {}, {}", t1, stride_str, t0));
                    ctx.code.add(format!("{} = add i64 {}, {}", t2, t1, start_str));
                    ctx.code.add(format!("{} = icmp slt i64 {}, {}", cond, t2, data_size_ll_tmp));
                    ctx.code.add(format!("br i1 {}, label {}, label %fn.boundcheckfailed", cond,
                                           next_bounds_check_label));
                    ctx.code.add(format!("{}:", next_bounds_check_label.replace("%", "")));
                }
            }
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
                          mut ctx: &mut FunctionContext) -> WeldResult<()> {
        let bound_cmp = ctx.var_ids.next();
        let grain_size = match par_for.grain_size {
            Some(size) => size,
            None => {
                if par_for.innermost {
                    DEFAULT_INNER_GRAIN_SIZE
                } else {
                    DEFAULT_OUTER_GRAIN_SIZE
                }
            }
        };
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
            ctx.code.add(format!("call void @f{}({}, i32 %cur.tid)", func.id, body_arg_types));
            let cont_arg_types = self.get_arg_str(&sir.funcs[par_for.cont].params, "")?;
            ctx.code.add(format!("call void @f{}({}, i32 %cur.tid)", par_for.cont, cont_arg_types));
            ctx.code.add(format!("br label %fn.end"));
        } else {
            // at least one task is always created for outer loops
            ctx.code.add("br label %for.par");
        }
        ctx.code.add(format!("for.par:"));
        self.gen_create_global_mergers(&func.params, ".ptr", &mut ctx)?;
        self.gen_create_global_mergers(&sir.funcs[par_for.cont].params, ".ptr", &mut ctx)?;
        self.gen_load_args(&func.params, ".ab", ".ptr", &mut ctx)?;
        self.gen_load_args(&sir.funcs[par_for.cont].params, ".ac", ".ptr", &mut ctx)?;
        let body_struct = self.gen_create_arg_struct(&func.params, ".ab", ctx)?;
        let cont_struct = self.gen_create_arg_struct(&sir.funcs[par_for.cont].params, ".ac", ctx)?;
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

    /// Helper function used at various places. Takes in a symbol of an array (as is passed to
    /// Weld), and based on the func and ctx, loads the value at index 'idx' of array and returns a
    /// string representing the value.
    /// @llvm_info: generated from get_llvm_info(...) call at some point.
    /// @vectorized: call to get the element is slightly different for simd ops.
    fn get_array_idx(&mut self, 
                     ctx: &mut FunctionContext,
                     llvm_info: VecLLVMInfo,
                     vectorized: bool,
                     idx: &String) -> WeldResult<String> {
        let arr_elem_tmp_ptr = ctx.var_ids.next();
        let at = if vectorized {
            "vat"
        } else {
            "at"
        };
        ctx.code.add(format!("{} = call {}* {}.{}({} {}, i64 {})",
                    arr_elem_tmp_ptr, &llvm_info.el_ty_str, llvm_info.prefix,
                    at, &llvm_info.ty_str, llvm_info.arr_str, idx)); 
        // Loading the cur element from the data array.
        Ok(self.gen_load_var(&arr_elem_tmp_ptr, &llvm_info.el_ty_str, ctx)?)
    }

    /// Generates a VecLLVMInfo struct for the given array.
    /// @el_ty_str: The type of elements in the array. This isn't required (ie. can just pass in
    /// "", if you aren't planning to use this field in the future.
    /// @in_func: hacky. If we are within a lower level func, or calling from outside the
    /// func (eg. wrapper) - this seems to change the type of the array (eg. in_func it is v0*,
    /// while !in_func: it is v0) -- and we have different methods to access the correct type in
    /// both scenarios.
    fn get_array_llvm_info(&mut self,
                           func: &SirFunction,
                           ctx: &mut FunctionContext,
                           arr: &Symbol,
                           el_ty_str: String,
                           in_func: bool) -> WeldResult<VecLLVMInfo> {
        let (arr_ty_str, arr_str) = if in_func {
            let arr_ty_str = self.llvm_type(func.params.get(arr).unwrap())?;
            let arr_str = self.gen_load_var(llvm_symbol(arr).as_str(), &arr_ty_str, ctx)?; 
            (arr_ty_str, arr_str)
        } else {
            self.llvm_type_and_name(func, &arr)?
        };
        let arr_prefix = llvm_prefix(&arr_ty_str);
        let len = ctx.var_ids.next();
        ctx.code.add(format!("{} = extractvalue {} {}, 1 ", len, arr_ty_str, arr_str));
        let v = VecLLVMInfo { 
                    ty_str: arr_ty_str,
                    arr_str: arr_str,
                    prefix: arr_prefix,
                    len_str: len,
                    el_ty_str: el_ty_str,
                };
        Ok(v)
    }

    /// Adds generic for loop code for an llvm loop to ctx.
    /// Returns the name of the pointer to, and the variable name of loop variable 'i'.
    fn add_llvm_for_loop_start(&mut self, 
                        ctx: &mut FunctionContext,
                        loop_name: &str,
                        loop_start: &str,
                        end_cmp: &str,
                        cmp_type: &str) -> WeldResult<(String, String)> {
        let cur_i_ptr = ctx.var_ids.next();
        ctx.code.add(format!("{} = alloca i64", cur_i_ptr));
        ctx.code.add(format!("store i64 {}, i64* {}", loop_start, cur_i_ptr));
        ctx.code.add(format!("br label %{}.start", loop_name));
        ctx.code.add(format!("{}.start:", loop_name));
        /* compare cur_i_ptr with end condition at loop start */
        let cur_i = ctx.var_ids.next();
        ctx.code.add(format!("{} = load i64, i64* {}", cur_i, cur_i_ptr));
        let cmp_str = ctx.var_ids.next();
        ctx.code.add(format!("{} = icmp {} i64 {}, {}", cmp_str, cmp_type, cur_i, end_cmp));
        ctx.code.add(format!("br i1 {}, label %{name}.body, label %{name}.end", cmp_str, name=loop_name));
        ctx.code.add(format!("{}.body:", loop_name)); 
        Ok((cur_i_ptr, cur_i))
    }

    /// Adds generic end conditions for the loop - called after add_llvm_for_loop_start, and 
    /// the loop body has been added.
    fn add_llvm_for_loop_end(&mut self, 
                        ctx: &mut FunctionContext,
                        loop_name: &str,
                        cur_i_ptr: &str,
                        cur_i: &str,
                        incr_op: &str) {
        /* cur_i = incr_op cur_i, 1 and jump back to loop start */
        let tmp_cur_i = ctx.var_ids.next();
        ctx.code.add(format!("{} = {} i64 {}, {}", tmp_cur_i, incr_op, cur_i, 1));
        ctx.code.add(format!("store i64 {}, i64* {}", tmp_cur_i, cur_i_ptr));
        ctx.code.add(format!("br label %{}.start", loop_name));
        /* loop.end needs to be added */
        ctx.code.add(format!("{}.end:", loop_name));
    } 

    /// Calculates the next element when performing a non-contiguous iteration. Essentially does
    /// idx = start + dot(counter, strides)
    /// @i: For zipped iters, nditer_next_element may be called for each individual iter.
    /// Since this involves declaring a new loop to do the dot product, we need 'i', to get unique
    /// names for each of the zipped values.
    fn nditer_next_element(&mut self,
                           func: &SirFunction,
                           ctx: &mut FunctionContext,
                           iter: &ParallelForIter,
                           i: String) -> WeldResult<String> {
        let strides_el_ty = self.llvm_type(&Scalar(I64))?;
        let strides_llvm_info = self.get_array_llvm_info(func, ctx, iter.strides.as_ref().unwrap(), 
                                                         strides_el_ty, true)?;
        /* sum += counter[i]*strides[i] loop to find arr_idx of next element. */    
        let sum_ptr = ctx.var_ids.next();
        ctx.code.add(format!("{} = alloca i64", sum_ptr));
        ctx.code.add(format!("store i64 0, i64* {}", sum_ptr));
        let loop_name = format!("next_element_loop{}", i);
        let (cur_i_ptr, cur_i) = self.add_llvm_for_loop_start(ctx, &loop_name, "0",
                                   &strides_llvm_info.len_str, "slt")?;
        /* sum += counter[i]*strides[i] */
        let tmp_sum = ctx.var_ids.next();
        ctx.code.add(format!("{} = load i64, i64* {}", tmp_sum, sum_ptr));
        let strides_i = self.get_array_idx(ctx, strides_llvm_info, false, &cur_i)?;
        let counter_i_ptr = ctx.var_ids.next();
        ctx.code.add(format!("{id} = getelementptr i64, i64* %counter.idx, i64 {idx}", 
                             id=counter_i_ptr, idx=cur_i));
        let counter_i = ctx.var_ids.next();
        ctx.code.add(format!("{} = load i64, i64* {}", counter_i, counter_i_ptr));
        let tmp_prod = ctx.var_ids.next();
        ctx.code.add(format!("{} = mul i64 {}, {}", tmp_prod, counter_i, strides_i));
        let tmp_sum2 = ctx.var_ids.next();
        ctx.code.add(format!("{} = add i64 {}, {}", tmp_sum2, tmp_sum, tmp_prod));
        /* Load the correct value back into sum. Could be more efficient to use the tmp variables,
         * but this stuff should be optimized by llvm anyway (?) */
        ctx.code.add(format!("store i64 {}, i64* {}", tmp_sum2, sum_ptr)); 
        /* Update cur_i_ptr and go back to start */
        self.add_llvm_for_loop_end(ctx, &loop_name, &cur_i_ptr, &cur_i, "add"); 
        /* sum must be the correct offset right now. */
        let offset = ctx.var_ids.next();
        ctx.code.add(format!("{} = load i64, i64* {}", offset, sum_ptr));
        /* next_idx = start + offset */
        let start_str = self.gen_load_var(llvm_symbol(&iter.start.clone().unwrap()).as_str(), "i64", ctx)?; 
        let final_idx = ctx.var_ids.next();
        ctx.code.add(format!("{} = add i64 {}, {}", final_idx, start_str, offset));
        /* final idx into original array that the iteration is on right now. */
        Ok(final_idx)
    }
    
    /// Helper function to check if any of the iters are of kind NdIter, and returns it if found.
    fn check_any_nditer(&mut self,
                        par_for: &ParallelForData) -> Option<ParallelForIter> {
        let mut nditer :Option<ParallelForIter> = None;
        for cur_iter in par_for.data.iter() {
            if cur_iter.kind == IterKind::NdIter {
                nditer = Some(cur_iter.clone());
                break;
            }
        }
        nditer
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
        if par_for.innermost {
            ctx.add_alloca("%cur.idx", "i64")?;
        } else {
            ctx.code.add("%cur.idx = getelementptr inbounds %work_t, %work_t* %cur.work, i32 0, i32 3");
        }
        ctx.code.add("store i64 %lower.idx, i64* %cur.idx");
        let nditer = self.check_any_nditer(par_for);
        if nditer.is_some() {
            let first_iter = nditer.unwrap();
            /* declare a counter == len(shape) */
            let shape_el_ty = self.llvm_type(&Scalar(I64))?;
            let shape_llvm_info = self.get_array_llvm_info(func, ctx, first_iter.shape.as_ref().unwrap(), shape_el_ty, true)?;
            /* dynamically generates an array of len(shape) ints on the stack. There does not seem to
             * be any reason to use malloc here. */
            ctx.code.add(format!("%counter.idx = alloca i64, i64 {}", shape_llvm_info.len_str)); 
            /* Zero it out, maybe use memset instead? */
            let loop_name = "zero_out_counter";
            let (cur_i_ptr, cur_i) = self.add_llvm_for_loop_start(ctx, &loop_name, "0",
                                               &shape_llvm_info.len_str, "slt")?;  
            let tmp_id = ctx.var_ids.next();
            /* counter.idx[cur_i] */
            ctx.code.add(format!("{} = getelementptr i64, i64* %counter.idx, \
                                i64 {}", tmp_id, cur_i));
            ctx.code.add(format!("store i64 0, i64* {}", tmp_id));
            self.add_llvm_for_loop_end(ctx, &loop_name, &cur_i_ptr, &cur_i, "add");
        }
        // Declare loop body as counter etc. have already been initialized.
        ctx.code.add("br label %loop.start");
        ctx.code.add("loop.start:"); 
        /* Loop termination condition. 
         * Keeping it the same in nditer, and ensuring "num_iterations" value is set correctly in
         * gen_num_iters_and_fringe_start.*/
        let idx_tmp = self.gen_load_var("%cur.idx", "i64", ctx)?;  
        let elem_ty = func.locals.get(&par_for.data_arg).unwrap();
        let idx_cmp = ctx.var_ids.next();
        if par_for.data[0].kind == IterKind::SimdIter {
            let check_with_vec = ctx.var_ids.next();
            let vector_len = format!("{}", llvm_simd_size(&elem_ty)?);
            ctx.code.add(format!("{} = add nuw nsw i64 {}, {}", check_with_vec, idx_tmp, vector_len));
            ctx.code.add(format!("{} = icmp ule i64 {}, %upper.idx", idx_cmp, check_with_vec));
        } else {
            ctx.code.add(format!("{} = icmp ult i64 {}, %upper.idx", idx_cmp, idx_tmp));
        }
        /* go to loop body or loop end depending on idx_cmp being T/F */
        ctx.code.add(format!("br i1 {}, label %loop.body, label %loop.end", idx_cmp));
        ctx.code.add("loop.body:");
        let mut prev_ref = String::from("undef");
        let elem_ty_str = self.llvm_type(&elem_ty)?;
        for (i, iter) in par_for.data.iter().enumerate() {
            let inner_elem_ty_str = if par_for.data.len() == 1 {
                elem_ty_str.clone()
            } else {
                match *elem_ty {
                    Struct(ref v) => self.llvm_type(&v[i])?,
                    _ => compile_err!("Internal error: invalid element type {}", elem_ty)?,
                }
            }; 
            let data_llvm_info = self.get_array_llvm_info(func, ctx, &iter.data, inner_elem_ty_str.clone(), true)?;
            /* idx into the original array at iteration %cur.i */
            let arr_idx = if iter.kind == IterKind::NdIter {
                self.nditer_next_element(func, ctx, iter, i.to_string()).unwrap()
            } else if iter.start.is_some() {
                // TODO(shoumik) implement. This needs to be a gather instead of a
                // sequential load.
                if iter.kind == IterKind::SimdIter {
                    return compile_err!("Unimplemented: vectorized iterators do not support non-unit stride.");
                }
                let offset = ctx.var_ids.next();
                let stride_str = self.gen_load_var(llvm_symbol(&iter.stride.clone().unwrap()).as_str(), "i64", ctx)?;
                let start_str = self.gen_load_var(llvm_symbol(&iter.start.clone().unwrap()).as_str(), "i64", ctx)?;
                let final_idx = ctx.var_ids.next();
                ctx.code.add(format!("{} = mul nsw nuw i64 {}, {}", offset, idx_tmp, stride_str));
                ctx.code.add(format!("{} = add nsw nuw i64 {}, {}", final_idx, start_str, offset));
                final_idx
            } else {
                if iter.kind == IterKind::FringeIter {
                    let vector_len = format!("{}", llvm_simd_size(&elem_ty)?);
                    let tmp = ctx.var_ids.next();
                    let arr_len = ctx.var_ids.next();
                    let offset = ctx.var_ids.next();
                    let final_idx = ctx.var_ids.next();
                    ctx.code.add(format!("{} = call i64 {}.size({} {})", arr_len, data_llvm_info.prefix,
                                 data_llvm_info.ty_str,data_llvm_info.arr_str));
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

            let inner_elem_tmp = match iter.kind {
               IterKind::SimdIter => {
                    self.get_array_idx(ctx, data_llvm_info, true, &arr_idx)?
               }
               IterKind::RangeIter => {
                    // Range Iterators always return the type `i64`. Just pass the array
                    // index we would have computed.
                    let mut inner_elem_tmp = ctx.var_ids.next();
                    ctx.code.add(format!("{} = add i64 0, {}", inner_elem_tmp, arr_idx));
                    inner_elem_tmp
               }
               /* General case for ScalarIter, NdIter and FringeIter */
               _ => {
                    self.get_array_idx(ctx, data_llvm_info, false, &arr_idx)?
               }
            };

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
        /* stores prev_ref in place pointed to by elem_str */
        ctx.code.add(format!("store {} {}, {}* {}", &elem_ty_str, prev_ref, &elem_ty_str, elem_str)); 
        /* updating the value of the current index, "i". */
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
        ctx.code.add(format!("{} = add nsw nuw i64 {}, {}", idx_inc, idx_tmp, format!("{}", fetch_width)));
        ctx.code.add(format!("store i64 {}, i64* %cur.idx", idx_inc));
        /* Nditer case: need to update the n-d counter as well. Check if any of the iters are of
         * Nditer kind, and update the counter if they are.
         */
        let nditer = self.check_any_nditer(par_for);
        if nditer.is_some() {
            //let iter = &par_for.data[0];
            let iter = nditer.unwrap();
            // Add the counter incrementing loop here and break to label %loop.start when done.
            let shape_el_ty = self.llvm_type(&Scalar(I64))?;
            let shape_llvm_info = self.get_array_llvm_info(func, ctx, iter.shape.as_ref().unwrap(), shape_el_ty, true)?;
            ctx.code.add(format!("{i} = alloca i64
                                 {tmp0} = sub i64 {counter_len}, 1
                                 store i64 {tmp0}, i64* {i}     
                                 br label %counter_loop.start
                                 counter_loop.start:
                                 {tmp00} = load i64, i64* {i}
                                 {tmp01} = icmp sge i64 {tmp00}, 0
                                 br i1 {tmp01}, label %counter_loop.body, label %loop.start
                                 counter_loop.body:
                                 {tmp1} = load i64, i64* {i}
                                 {tmp2} = getelementptr i64, i64* {counter}, i64 {tmp1} 
                                 {tmp3} = load i64, i64* {tmp2}
                                 {tmp4} = add i64 {tmp3}, 1
                                 store i64 {tmp4}, i64* {tmp2}"
                                 , i="%counter.i", counter_len=shape_llvm_info.len_str, 
                                 counter="%counter.idx", tmp0=ctx.var_ids.next(), tmp00=ctx.var_ids.next(),
                                 tmp01=ctx.var_ids.next(), tmp1="%cur_i", tmp2=ctx.var_ids.next(),
                                 tmp3=ctx.var_ids.next(), tmp4=ctx.var_ids.next()));
            /* Need to break off the long sequence of llvm IR because no easy way to get ith element of shape. */
            let shape_elem_str = self.get_array_idx(ctx, shape_llvm_info, false, &"%cur_i".to_string())?;
            ctx.code.add(format!("{tmp5} = load i64, i64* {i}
                                 {tmp6} = getelementptr i64, i64* {counter}, i64 {tmp5} 
                                 {tmp7} = load i64, i64* {tmp6}
                                 {tmp8} = icmp eq i64 {shape_elem}, {tmp7}
                                 br i1 {tmp8}, label %counter_loop.end, label %loop.start
                                 counter_loop.end:
                                 ; zero-ing out the value because it was equal to shape[i]
                                 store i64 0, i64* {tmp6}
                                 {tmp9} = sub i64 {tmp5}, 1
                                 store i64 {tmp9}, i64* {i}
                                 br label %counter_loop.start", 
                                 shape_elem = shape_elem_str,
                                 counter="%counter.idx", i = "%counter.i",
                                 tmp5 = ctx.var_ids.next(), tmp6 = ctx.var_ids.next(), tmp7 =
                                 ctx.var_ids.next(), tmp8 = ctx.var_ids.next(), tmp9 =
                                 ctx.var_ids.next()));
            ctx.code.add("loop.end:");
        } else { 
            ctx.code.add("br label %loop.start");
            ctx.code.add("loop.end:");
        }
        Ok(())
    }

    /// Generates a header common to each top-level generated function.
    fn gen_function_header(&mut self, arg_types: &str, func: &SirFunction, ctx: &mut FunctionContext) -> WeldResult<()> {
        // Start the entry block by defining the function and storing all its arguments on the
        // stack (this makes them consistent with other local variables). Later, expressions may
        // add more local variables to alloca_code.
        ctx.alloca_code.add(format!("define void @f{}({}, i32 %cur.tid) {{", func.id, arg_types));
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
        let combined_params = get_combined_params(sir, &par_for);
        let serial_arg_types = self.get_arg_str(&combined_params, "")?;
        self.gen_store_args(&combined_params, "", ".ptr", ctx)?;
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
        self.body_code.add(format!("define void @f{}_wrapper({}, i32 %cur.tid) {{", func.id, serial_arg_types));
        self.body_code.add(format!("fn.entry:"));
        self.body_code.add(&ctx.alloca_code.result());
        self.body_code.add(&ctx.code.result());

        Ok(())
    }

    /// Generates the parallel runtime callback function for a loop body. This function calls the
    /// function which executes a loop body after unpacking the lower and upper bound from the work
    /// item.
    fn gen_parallel_runtime_callback_function(&mut self, func: &SirFunction) -> WeldResult<()> {
            let mut ctx = &mut FunctionContext::new(false);
            self.gen_unload_arg_struct(&func.params, ".load", &mut ctx)?;
            self.gen_store_args(&func.params, ".load", "", &mut ctx)?;
            self.gen_create_stack_mergers(&func.params, &mut ctx)?;
            let lower_bound_ptr = ctx.var_ids.next();
            let lower_bound = ctx.var_ids.next();
            let upper_bound_ptr = ctx.var_ids.next();
            let upper_bound = ctx.var_ids.next();
            ctx.code.add(format!("%cur.tid = call i32 @weld_rt_thread_id()"));
            ctx.code.add(format!("{} = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 1", lower_bound_ptr));
            ctx.code.add(format!("{} = load i64, i64* {}", lower_bound, lower_bound_ptr));
            ctx.code.add(format!("{} = getelementptr %work_t, %work_t* %cur.work, i32 0, i32 2", upper_bound_ptr));
            ctx.code.add(format!("{} = load i64, i64* {}", upper_bound, upper_bound_ptr));
            self.gen_load_args(&func.params, ".arg", "", &mut ctx)?;
            let body_arg_types = try!(self.get_arg_str(&func.params, ".arg"));
            self.gen_create_new_vb_pieces(&func.params, ".arg", &mut ctx)?;
            ctx.code.add("fn_call:");
            ctx.code.add(format!("call void @f{}({}, i64 {}, i64 {}, i32 %cur.tid)",
                                            func.id,
                                            body_arg_types,
                                            lower_bound,
                                            upper_bound));
            self.gen_store_stack_mergers(&func.params, &mut ctx)?;
            ctx.code.add("ret void");
            ctx.code.add("}\n\n");
            self.body_code.add(format!("define void @f{}_par(%work_t* %cur.work) {{", func.id));
            self.body_code.add("entry:");
            self.body_code.add(&ctx.alloca_code.result());
            self.body_code.add(&ctx.code.result());
            Ok(())
    }

    /// Generates the continuation function of the given parallel loop.
    fn gen_loop_continuation_function(&mut self,
                                      par_for: &ParallelForData,
                                      sir: &SirProgram) -> WeldResult<()> {
            let mut ctx = &mut FunctionContext::new(false);

            ctx.code.add(format!("%cur.tid = call i32 @weld_rt_thread_id()"));
            self.gen_unload_arg_struct(&sir.funcs[par_for.cont].params, ".load", &mut ctx)?;
            self.gen_store_args(&sir.funcs[par_for.cont].params, ".load", "", &mut ctx)?;
            self.gen_create_stack_mergers(&sir.funcs[par_for.cont].params, &mut ctx)?;
            self.gen_load_args(&sir.funcs[par_for.cont].params, ".arg", "", &mut ctx)?;
            self.gen_create_new_vb_pieces(&sir.funcs[par_for.cont].params, ".arg", &mut ctx)?;

            ctx.code.add("fn_call:");
            let cont_arg_types = self.get_arg_str(&sir.funcs[par_for.cont].params, ".arg")?;
            ctx.code.add(format!("call void @f{}({}, i32 %cur.tid)", par_for.cont, cont_arg_types));
            self.gen_store_stack_mergers(&sir.funcs[par_for.cont].params, &mut ctx)?;
            ctx.code.add("ret void");
            ctx.code.add("}\n\n");
            self.body_code.add(format!("define void @f{}_par(%work_t* %cur.work) {{", par_for.cont));
            self.body_code.add("entry:");
            self.body_code.add(&ctx.alloca_code.result());
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

        let ctx = &mut FunctionContext::new(false);
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
        par_top_ctx.code.add(format!("%cur.tid = call i32 @weld_rt_thread_id()"));
        self.gen_unload_arg_struct(&sir.funcs[0].params, "", &mut par_top_ctx)?;
        let top_arg_types = self.get_arg_str(&sir.funcs[0].params, "")?;
        par_top_ctx.code.add(format!("call void @f0({}, i32 %cur.tid)", top_arg_types));
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

        let mut arg_pos_map: fnv::FnvHashMap<Symbol, usize> = fnv::FnvHashMap::default();
        for (i, a) in sir.top_params.iter().enumerate() {
            arg_pos_map.insert(a.name.clone(), i);
        }
        for (arg, _) in sir.funcs[0].params.iter() {
            let idx = arg_pos_map.get(arg).unwrap();
            run_ctx.code.add(format!("{} = extractvalue {} %r.args_val, {}", llvm_symbol(arg), args_type, idx));
        }
        let run_struct = self.gen_create_arg_struct(&sir.funcs[0].params, "", &mut run_ctx)?;

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
                return compile_err!("Unsupported type {}", ty)?
            }
        })
    }

    /*********************************************************************************************
    //
    // Routines for Code Generation of Statements and Blocks.
    //
    *********************************************************************************************/


    fn gen_minmax(&mut self, ll_ty: &str,
                  op: &BinOpKind,
                  left_tmp: &str,
                  right_tmp: &str,
                  output_tmp: &str,
                  ty: &Type,
                  var_ids: &mut IdGenerator,
                  code: &mut CodeBuilder) -> WeldResult<()> {
        use ast::BinOpKind::*;
        match *ty {
            Scalar(s) | Simd(s) => {
                if s.is_integer() {
                    let sel_tmp = var_ids.next();
                    match *op {
                        Max => {
                            code.add(format!("{} = {} {} {}, {}",
                                                 &sel_tmp,
                                                 llvm_binop(GreaterThan, ty)?,
                                             &ll_ty, &left_tmp, &right_tmp));
                        }
                        Min => {
                            code.add(format!("{} = {} {} {}, {}",
                                                 &sel_tmp,
                                                 llvm_binop(LessThan, ty)?,
                                                 &ll_ty, &left_tmp, &right_tmp));
                        }
                        _ => return compile_err!("Illegal operation using Min/Max generator"),
                    }

                    let sel_type = if ty.is_scalar() 
                                    { "i1".to_string() } 
                                    else { format!("<{} x i1>", llvm_simd_size(ty)?) };
                    code.add(format!("{} = select {} {}, {} {}, {} {}",
                                         &output_tmp, &sel_type, sel_tmp,
                                         self.llvm_type(ty)?, left_tmp,
                                         self.llvm_type(ty)?, right_tmp));
                } else if s.is_float() { /* has one-line intrinsic */
                    let intrinsic = if ty.is_scalar() 
                                    { llvm_binary_intrinsic(*op, &s)? } 
                                    else { llvm_simd_binary_intrinsic(*op, &s, llvm_simd_size(ty)?)? };
                    code.add(format!("{} = call {} {}({} {}, {} {})",
                                         &output_tmp, &ll_ty,
                                         intrinsic,
                                         self.llvm_type(ty)?, &left_tmp,
                                         self.llvm_type(ty)?, &right_tmp));
                }
            }
            _ => compile_err!("Illegal type {} in Min/Max", ty)?,
        }

        Ok(())
    }

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
                let helper_state = self.type_helpers.get_mut(ty).unwrap();
                helper_state.eq_func = true;
            }
            _ => {
                return compile_err!("Unsupported function `cmp` for type {:?}", ty);
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
                return compile_err!("Unsupported function `eq` for type {:?}", ty);
            }
        };
        let ll_ty = self.llvm_type(ty)?;
        let ll_prefix = llvm_prefix(&ll_ty);
        let eq_on_pointers = format!(include_str!("resources/eq_on_pointers.ll"),
            TYPE=&ll_ty,
            TYPE_PREFIX=&ll_prefix);
        self.prelude_code.add(&eq_on_pointers);
        self.prelude_code.add("\n");
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
                match *elem.as_ref() {
                    Scalar(ScalarKind::U8) | Scalar(ScalarKind::I8) => {
                        self.prelude_code.add(format!(
                                include_str!("resources/vector/veci8_hash.ll"),
                                ELEM=&elem_ty,
                                NAME=&name.replace("%", "")));
                    }
                    _ => {
                        self.prelude_code.add(format!(
                                include_str!("resources/vector/vector_hash.ll"),
                                ELEM_PREFIX=&elem_prefix,
                                ELEM=&elem_ty,
                                NAME=&name.replace("%", "")));
                    }
                }
            }
            _ => {
                return compile_err!("Unsupported function `hash` for type {:?}", ty);
            }
        };
        Ok(())
    }

    fn escape_str(&self, string: &str) -> String {
        string.replace("\\", "\\\\").replace("\"", "\\\"")
    }

    /// Retrieve the stored pointer for a String constant or create one if it doesn't exist.
    fn get_string_ptr(&mut self, string: &str) -> WeldResult<String> {
        if self.string_names.get(string) == None {
            self.gen_string_definition(string)?;
        }
        Ok(self.string_names.get(string).unwrap().to_string())
    }

    /// Generates a global pointer for a String constant.
    fn gen_string_definition(&mut self, string: &str) -> WeldResult<()> {
        if !(string.is_ascii()) {
            return compile_err!("Weld strings must be valid ASCII");
        }

        let global = self.prelude_var_ids.next().replace("%", "@");
        let text = self.escape_str(string);
        let len = text.len();
        self.prelude_code.add(format!(
            "{} = private unnamed_addr constant [{} x i8] c\"{}\"",
            global, len, text));
        self.string_names.insert(string.to_string(), global);
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

    fn string_literal(&mut self, string: &str, vec_ty: &str, ctx: &mut FunctionContext) -> WeldResult<String> {
        let global = self.get_string_ptr(string).unwrap();
        let len = self.escape_str(string).len();
        let local = ctx.var_ids.next();
        ctx.code.add(format!(
            "{} = getelementptr [{} x i8], [{} x i8]* {}, i32 0, i32 0",
            local, len, len, global));
        let tmp_vec = ctx.var_ids.next();
        ctx.code.add(format!(
            "{} = insertvalue {} undef, i8* {}, 0",
            tmp_vec, vec_ty, local));
        let tmp_vec2 = ctx.var_ids.next();
        ctx.code.add(format!(
            "{} = insertvalue {} {}, i64 {}, 1",
            tmp_vec2, vec_ty, tmp_vec, len));
        Ok(tmp_vec2)
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

        let dict_def = format!(include_str!("resources/dictionary/dictionary.ll"),
            NAME=&name.replace("%", ""),
            KEY=&key_ty,
            KEY_PREFIX=&key_prefix,
            VALUE=&value_ty,
            KV_STRUCT=&kv_struct_ty,
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
                let key_prefix = llvm_prefix(&key_ty);
                let value_ty = self.llvm_type(vt)?;

                let dictmerger_def = format!(include_str!("resources/dictionary/dictmerger.ll"),
                    NAME=&bld_ty_str.replace("%", ""),
                    KEY=&key_ty,
                    KEY_PREFIX=&key_prefix,
                    VALUE=&value_ty,
                    KV_STRUCT=&kv_struct_ty.replace("%", ""));

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
                let key_ty = self.llvm_type(kt)?;
                let key_prefix = llvm_prefix(&key_ty);
                let value_ty = self.llvm_type(vt)?;
                let kv_struct = Box::new(Struct(vec![*kt.clone(), *vt.clone()]));
                let kv_struct_ty = self.llvm_type(&kv_struct)?;
                let vec = Box::new(Vector(vt.clone()));
                let bld = Dict(kt.clone(), vec);
                let bld_ty = self.llvm_type(&bld)?;

                let groupmerger_def = format!(include_str!("resources/dictionary/groupbuilder.ll"),
                    NAME=&bld_ty.replace("%", ""),
                    KEY=&key_ty,
                    KEY_PREFIX=&key_prefix,
                    VALUE=&value_ty,
                    KV_STRUCT=&kv_struct_ty.replace("%", ""));

                self.prelude_code.add(&groupmerger_def);
                self.prelude_code.add("\n");
                self.bld_names.insert(bk.clone(), format!("{}.gbld", bld_ty));
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

        let value_str = llvm_literal((*kind).clone()).unwrap();
        let elem_ty_str = match *kind {
            BoolLiteral(_) => "i1",
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
            StringLiteral(_) => {
                return compile_err!("Cannot create SIMD StringLiteral");
            }
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
        use ast::BinOpKind::*;
        let llvm_ty = self.llvm_type(arg_ty)?.to_string();
        let mut res = var_ids.next();

        match *arg_ty {
            Scalar(_) | Simd(_) => {
                match *bin_op {
                    Max | Min => {
                        assert!(self.gen_minmax(&llvm_ty, bin_op, arg1, arg2, &res, arg_ty, var_ids, code).is_ok());
                    }
                    _ => {
                        code.add(format!("{} = {} {} {}, {}",
                            &res, try!(llvm_binop(*bin_op, arg_ty)), &llvm_ty, arg1, arg2));
                    }
                }
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

            _ => return compile_err!("gen_merge_op_on_registers called on invalid type {}", arg_ty)
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

    /// Generates a serialization function for each type.
    fn gen_serialize_helper(&mut self,
                            buffer_ll_ty: &str,
                            buffer_ll_prefix: &str,
                            expr_ll_ty: &str,
                            expr_ll_prefix: &str,
                            expr_ty: &Type,
                            func: &SirFunction,
                            ctx: &mut FunctionContext) -> WeldResult<String> {

        // If we already generated a serialization call, return it.
        if let Some(ref serialize_fn) = self.serialize_fns.get(expr_ty) {
            return Ok(String::from(serialize_fn.as_ref()))
        }

        let serialize_fn = format!("{}.serialize", expr_ll_prefix);

        match *expr_ty {
            Unknown => {
                return compile_err!("Unexpected Unknown type in code gen");
            }
            Scalar(_) | Struct(_) if !expr_ty.has_pointer() => {
                // These are primitive pointer-less values that we can store directly into the
                // buffer.
                let mut serialize_code = CodeBuilder::new();
                serialize_code.add(format!("define {}.growable {}({}.growable %buf, {} %data) alwaysinline {{",
                buffer_ll_ty,
                serialize_fn,
                buffer_ll_ty,
                expr_ll_ty));

                // Get size of the type.
                serialize_code.add(format!("%sizePtr = getelementptr {}, {}* null, i32 1", expr_ll_ty, expr_ll_ty));
                serialize_code.add(format!("%size = ptrtoint {}* %sizePtr to i64", expr_ll_ty));

                // Resize the buffer to fit and get the pointer to write at.
                serialize_code.add(format!("%tmp = call {}.growable @{}.growable.resize_to_fit({}.growable %buf, i64 %size)",
                buffer_ll_ty,
                buffer_ll_prefix,
                buffer_ll_ty));
                serialize_code.add(format!("%tmp2 = call i8* @{}.growable.last({}.growable %tmp)",
                buffer_ll_prefix,
                buffer_ll_ty));

                // Store the final value and return the new buffer.
                serialize_code.add(format!("%tmp3 = bitcast i8* %tmp2 to {}*", expr_ll_ty));
                serialize_code.add(format!("store {} %data, {}* %tmp3", &expr_ll_ty, &expr_ll_ty));
                serialize_code.add(format!("%tmp4 = call {}.growable @{}.growable.extend({}.growable %tmp, i64 %size)",
                buffer_ll_ty,
                buffer_ll_prefix,
                buffer_ll_ty));
                serialize_code.add(format!("ret {}.growable %tmp4", buffer_ll_ty));
                serialize_code.add("}");
                self.prelude_code.add_code(&serialize_code);
                self.prelude_code.add("\n");

            }
            Vector(ref elem_ty) if elem_ty.has_pointer() => {
                // Serialized as an i64 length, followed by each `length` serialized elements.
                let ref elem_ll_ty = self.llvm_type(elem_ty)?;
                let elem_serialize = self.gen_serialize_helper(buffer_ll_ty,
                                                               buffer_ll_prefix,
                                                               elem_ll_ty,
                                                               &llvm_prefix(elem_ll_ty),
                                                               elem_ty,
                                                               func,
                                                               ctx)?;

                self.prelude_code.add(format!(include_str!("resources/vector/serialize_with_pointers.ll"),
                BUFNAME=buffer_ll_ty.replace("%", ""),
                NAME=expr_ll_ty.replace("%", ""),
                ELEM=elem_ll_ty,
                ELEM_SERIALIZE=elem_serialize));
            }
            Vector(ref elem_ty) => {
                // Serialized as an i64 length, followed by each `length` serialized elements.
                // The elements are not pointer-based types, so it is safe to perform a memcpy.
                let ref elem_ll_ty = self.llvm_type(elem_ty)?;
                self.prelude_code.add(format!(include_str!("resources/vector/serialize_without_pointers.ll"),
                BUFNAME=buffer_ll_ty.replace("%", ""),
                NAME=expr_ll_ty.replace("%", ""),
                ELEM=elem_ll_ty));
            }
            Dict(ref key, ref value) if !key.has_pointer() && !value.has_pointer() => {
                // Dictionaries are serialized as <8-byte length (in # of Key/value pairs)>
                // followed by packed {key, value} pairs. This case handles dictionaries where
                // the key and value do not have pointers. The following case handles pointers by
                // calling serialize on the key and value.
                self.prelude_code.add(format!(include_str!("resources/dictionary/serialize_dictionary.ll"),
                    NAME=expr_ll_ty.replace("%", ""),
                    BUFNAME=buffer_ll_ty.replace("%", ""),
                    HAS_POINTER=0,
                    KEY_SERIALIZE_ON_PTR="null", 
                    VAL_SERIALIZE_ON_PTR="null"
                    ));
            }
            Dict(ref key, ref value) => {
                // Dictionaries are serialized as <8-byte length (in # of Key/value pairs)>
                // followed by packed {key, value} pairs. This case handles dictionaries where
                // the key and value do not have pointers. The following case handles pointers by
                // calling serialize on the key and value.
                let ref key_ll_ty = self.llvm_type(key)?;
                let _ = self.gen_serialize_helper(buffer_ll_ty,
                                                  buffer_ll_prefix,
                                                  key_ll_ty,
                                                  &llvm_prefix(key_ll_ty),
                                                  key,
                                                  func,
                                                  ctx)?;

                let ref value_ll_ty = self.llvm_type(value)?;
                let _ = self.gen_serialize_helper(buffer_ll_ty,
                                                  buffer_ll_prefix,
                                                  value_ll_ty,
                                                  &llvm_prefix(value_ll_ty),
                                                  value,
                                                  func,
                                                  ctx)?;

                self.prelude_code.add(format!(include_str!("resources/dictionary/serialize_dictionary.ll"),
                    NAME=expr_ll_ty.replace("%", ""),
                    BUFNAME=buffer_ll_ty.replace("%", ""),
                    HAS_POINTER=1,
                    KEY_SERIALIZE_ON_PTR=format!("{}.serialize_on_pointers", llvm_prefix(key_ll_ty)), 
                    VAL_SERIALIZE_ON_PTR=format!("{}.serialize_on_pointers", llvm_prefix(value_ll_ty))
                    ));
            }
            Struct(ref tys) => {
                // Serialized as each struct element serialized in order. This version handles
                // struct members with pointers, and generates a serialization function for each
                // struct member.
                let mut serialize_code = CodeBuilder::new();
                serialize_code.add(format!("define {}.growable {}({}.growable %buf, {} %data) alwaysinline {{",
                buffer_ll_ty,
                serialize_fn,
                buffer_ll_ty,
                expr_ll_ty));

                let mut prev_gvec = "%buf".to_string();
                for (i, elem_ty) in tys.iter().enumerate() {
                    let ref elem_ll_ty = self.llvm_type(elem_ty)?;
                    let elem_serialize = self.gen_serialize_helper(buffer_ll_ty,
                                                                   buffer_ll_prefix,
                                                                   elem_ll_ty,
                                                                   &llvm_prefix(elem_ll_ty),
                                                                   elem_ty,
                                                                   func,
                                                                   ctx)?;

                    let tmp = ctx.var_ids.next();
                    let next_gvec = ctx.var_ids.next();
                    serialize_code.add(format!("{} = extractvalue {} %data, {}", tmp, &expr_ll_ty, i));
                    serialize_code.add(format!("{} = call {}.growable {}({}.growable {}, {} {})",
                    next_gvec,
                    buffer_ll_ty, 
                    elem_serialize,
                    buffer_ll_ty,
                    prev_gvec,
                    elem_ll_ty,
                    tmp));
                    prev_gvec = next_gvec;
                }
                serialize_code.add(format!("ret {}.growable {}", buffer_ll_ty, prev_gvec));
                serialize_code.add("}");

                self.prelude_code.add_code(&serialize_code);
                self.prelude_code.add("\n");
            }
            Simd(_) | Builder(_, _) | Function(_, _) => {
                // Non-serializable types.
                return compile_err!("Cannot serialize type {:?}", expr_ty);
            }
            // Covered by the first case since scalars never have pointers.
            Scalar(_) => unreachable!(),
        }

        // Generate the serialize function on pointers.
        self.prelude_code.add(format!(include_str!("resources/serialize_on_pointers.ll"),
        TYPE=expr_ll_ty,
        TYPE_PREFIX=&llvm_prefix(expr_ll_ty),
        BUFNAME=buffer_ll_ty.replace("%", ""),
        SERIALIZE=serialize_fn));

        self.serialize_fns.insert(expr_ty.clone(), serialize_fn.clone());
        Ok(serialize_fn)
    }

    /// Generates a serialization function for each type.
    fn gen_deserialize_helper(&mut self,
                            output_ll_ty: &str,
                            output_ll_prefix: &str,
                            buffer_ll_ty: &str,
                            buffer_ll_prefix: &str,
                            output_ty: &Type,
                            func: &SirFunction,
                            ctx: &mut FunctionContext) -> WeldResult<String> {

        // If we already generated a serialization call, return it.
        if let Some(ref deserialize_fn) = self.deserialize_fns.get(output_ty) {
            return Ok(String::from(deserialize_fn.as_ref()))
        }
        let deserialize_fn = format!("{}.deserialize", output_ll_prefix);

        match *output_ty {
            Unknown => {
                return compile_err!("Unexpected Unknown type in code gen");
            }
            Scalar(_) | Struct(_) if !output_ty.has_pointer() => {
                let mut deserialize_code = CodeBuilder::new();
                deserialize_code.add(format!("define i64 {}({} %buf, i64 %offset, {}* %resPtr) alwaysinline {{",
                deserialize_fn,
                buffer_ll_ty,
                output_ll_ty));

                // Get size of the type.
                deserialize_code.add(format!("%sizePtr = getelementptr {}, {}* null, i32 1", output_ll_ty, output_ll_ty));
                deserialize_code.add(format!("%size = ptrtoint {}* %sizePtr to i64", output_ll_ty));

                // Get the pointer to the value from the serialized buffer, load it, and copy it to
                // the result pointer.
                deserialize_code.add(format!("%dataPtrRaw = call i8* {}.at({} %buf, i64 %offset)", buffer_ll_prefix, buffer_ll_ty));
                deserialize_code.add(format!("%dataPtr = bitcast i8* %dataPtrRaw to {}*", output_ll_ty));
                deserialize_code.add(format!("%dataTmp = load {}, {}* %dataPtr", output_ll_ty, output_ll_ty));
                deserialize_code.add(format!("store {} %dataTmp, {}* %resPtr", output_ll_ty, output_ll_ty));

                // Increment the offset and return the new offset.
                deserialize_code.add(format!("%result = add i64 %offset, %size"));
                deserialize_code.add("ret i64 %result");
                deserialize_code.add("}");
                self.prelude_code.add_code(&deserialize_code);
                self.prelude_code.add("\n");

            }
            Vector(ref elem_ty) if elem_ty.has_pointer() => {
                let ref elem_ll_ty = self.llvm_type(elem_ty)?;
                let _ = self.gen_deserialize_helper(elem_ll_ty,
                                                    &llvm_prefix(elem_ll_ty),
                                                    buffer_ll_ty,
                                                    buffer_ll_prefix,
                                                    elem_ty,
                                                    func,
                                                    ctx)?;
                self.prelude_code.add(format!(include_str!("resources/vector/deserialize_with_pointers.ll"),
                BUFNAME=buffer_ll_ty,
                BUF_PREFIX=buffer_ll_prefix,
                NAME=output_ll_ty.replace("%", ""),
                ELEM=elem_ll_ty,
                ELEM_PREFIX=&llvm_prefix(elem_ll_ty)
                ));
            }
            Vector(ref elem_ty) => {
                let elem_ll_ty = self.llvm_type(elem_ty)?;

                self.prelude_code.add(format!(include_str!("resources/vector/deserialize_without_pointers.ll"),
                BUFNAME=buffer_ll_ty,
                BUF_PREFIX=buffer_ll_prefix,
                NAME=output_ll_ty.replace("%", ""),
                ELEM=elem_ll_ty));
            }
            Dict(ref key, ref value) => {
                // For dictionaries, the deserialization path for keys and values with and without
                // pointers is the same.
                let ref key_ll_ty = self.llvm_type(key)?;
                let ref key_ll_prefix = llvm_prefix(key_ll_ty);
                let _ = self.gen_deserialize_helper(key_ll_ty,
                                                    key_ll_prefix,
                                                    buffer_ll_ty,
                                                    buffer_ll_prefix,
                                                    key,
                                                    func,
                                                    ctx)?;
                let ref val_ll_ty = self.llvm_type(value)?;
                let ref val_ll_prefix = llvm_prefix(val_ll_ty);
                let _ = self.gen_deserialize_helper(val_ll_ty,
                                                    val_ll_prefix,
                                                    buffer_ll_ty,
                                                    buffer_ll_prefix,
                                                    value,
                                                    func,
                                                    ctx)?;

                self.prelude_code.add(format!(
                        include_str!("resources/dictionary/deserialize_dictionary.ll"),
                        NAME=output_ll_ty.replace("%", ""),
                        KEY=key_ll_ty,
                        KEY_PREFIX=key_ll_prefix,
                        VALUE=val_ll_ty,
                        VALUE_PREFIX=val_ll_prefix,
                        BUFNAME=buffer_ll_ty,
                        BUF_PREFIX=buffer_ll_prefix));
            }
            Struct(ref tys) => {
                // This is a struct with pointers, so we need to go through each element and decode
                // it.
                let mut deserialize_code = CodeBuilder::new();
                let mut var_ids = IdGenerator::new("%t.t");
                deserialize_code.add(format!("define i64 {}({} %buf, i64 %offset, {}* %resPtr) {{",
                deserialize_fn,
                buffer_ll_ty,
                output_ll_ty));

                deserialize_code.add(format!("%dataPtrRaw = call i8* {}.at({} %buf, i64 %offset)", buffer_ll_prefix, buffer_ll_ty));
                deserialize_code.add(format!("%dataPtr = bitcast i8* %dataPtrRaw to {}*", output_ll_ty));

                let mut offset = "%offset".to_string();
                for (i, elem_ty) in tys.iter().enumerate() {
                    let ref elem_ll_ty = self.llvm_type(elem_ty)?;
                    let _ = self.gen_deserialize_helper(elem_ll_ty,
                                                        &llvm_prefix(elem_ll_ty),
                                                        buffer_ll_ty,
                                                        buffer_ll_prefix,
                                                        elem_ty,
                                                        func,
                                                        ctx)?;

                    // Get the pointer to the correct field in the struct.
                    let res_ptr_tmp = var_ids.next();
                    deserialize_code.add(format!("{} = getelementptr inbounds {}, {}* %resPtr, i32 0, i32 {}",
                                                 res_ptr_tmp, output_ll_ty, output_ll_ty, i));

                    // Deserialize directly into the pointer.
                    let next_offset = var_ids.next();
                    deserialize_code.add(format!("{} = call i64 {}.deserialize({} %buf, i64 {}, {}* {})", 
                                                 next_offset,
                                                 &llvm_prefix(elem_ll_ty),
                                                 buffer_ll_ty,
                                                 offset,
                                                 elem_ll_ty,
                                                 res_ptr_tmp));
                    offset = next_offset;

                }
                deserialize_code.add(format!("ret i64 {}", offset));
                deserialize_code.add("}");
                self.prelude_code.add_code(&deserialize_code);
                self.prelude_code.add("\n");

            }
            Simd(_) | Builder(_, _) | Function(_, _) => {
                // Non-deserializable types.
                return compile_err!("Cannot deserialize to type {:?}", output_ty);
            }
            // Covered by the first case since scalars never have pointers.
            Scalar(_) => unreachable!(),
        }

        self.deserialize_fns.insert(output_ty.clone(), deserialize_fn.clone());
        Ok(deserialize_fn)
    }

    /// Generates deserialization code for `expr`, which is a vec[i8] that is converted to `ty`.
    fn gen_deserialize(&mut self,
                     expr: &Symbol,
                     output: &Symbol,
                     func: &SirFunction,
                     ctx: &mut FunctionContext) -> WeldResult<()> {

        let (expr_ll_ty, expr_ll_sym) = self.llvm_type_and_name(func, expr)?;
        let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;

        let expr_ty = func.symbol_type(expr)?;
        if *expr_ty != Vector(Box::new(Scalar(ScalarKind::I8))) {
            return compile_err!("codegen error: input of deserialize is not vec[i8]");
        }

        let output_ty = func.symbol_type(output)?;

        let expr_ll_prefix = llvm_prefix(&expr_ll_ty);
        let output_ll_prefix = llvm_prefix(&output_ll_ty);

        let _ = self.gen_deserialize_helper(&output_ll_ty,
                                            &output_ll_prefix,
                                            &expr_ll_ty,
                                            &expr_ll_prefix,
                                            output_ty,
                                            func,
                                            ctx)?;

        let expr_tmp = self.gen_load_var(&expr_ll_sym, &expr_ll_ty, ctx)?;
        let bytes_tmp = ctx.var_ids.next();
        ctx.code.add(format!("{} = call i64 {}.deserialize({} {}, i64 0, {}* {})", 
                                     bytes_tmp,
                                     output_ll_prefix,
                                     expr_ll_ty,
                                     expr_tmp,
                                     output_ll_ty,
                                     output_ll_sym));

        // Check if all the bytes were consumed, and abort if not.
        let array_size = ctx.var_ids.next();
        let cond = ctx.var_ids.next();
        let continue_label = ctx.var_ids.next();
        let abort_label = ctx.var_ids.next();
        let run_id = ctx.var_ids.next();
        let errno = WeldRuntimeErrno::DeserializationError;

        ctx.code.add("; Check to ensure that the full vector was consume during deserialization.");
        ctx.code.add(format!("{} = call i64 {}.size({} {})", array_size, expr_ll_prefix, expr_ll_ty, expr_tmp));
        ctx.code.add(format!("{} = icmp ne i64 {}, {}", cond, bytes_tmp, array_size)); 
        ctx.code.add(format!("br i1 {}, label {}, label {}", cond, abort_label, continue_label));
        ctx.code.add(format!("{}:", abort_label.replace("%", "")));
        ctx.code.add(format!("{} = call i64 @weld_rt_get_run_id()", run_id));
        ctx.code.add(format!("call void @weld_run_set_errno(i64 {}, i64 {})", run_id, errno as i64));
        ctx.code.add(format!("call void @weld_rt_abort_thread()"));
        ctx.code.add(format!("; Unreachable!"));
        ctx.code.add(format!("br label {}", continue_label));
        ctx.code.add(format!("{}:", continue_label.replace("%", "")));
        Ok(())
    }

    /// Generates serialization code for `expr`, which converts it into a flat vec[i8].
    fn gen_serialize(&mut self,
                     expr: &Symbol,
                     output: &Symbol,
                     func: &SirFunction,
                     ctx: &mut FunctionContext) -> WeldResult<()> {

        let (expr_ll_ty, expr_ll_sym) = self.llvm_type_and_name(func, expr)?;
        let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;

        let output_ty = func.symbol_type(output)?;
        if *output_ty != Vector(Box::new(Scalar(ScalarKind::I8))) {
            return compile_err!("codegen error: output of serialize is not vec[i8]");
        }

        // Generate a growable vec[i8] if it doesn't exist.
        if !self.growable_vec_names.contains(&output_ty) {
            self.growable_vec_names.insert(output_ty.clone());
            let elem_ll_ty = self.llvm_type(&Scalar(ScalarKind::I8))?;
            self.prelude_code.add(format!(
                    include_str!("resources/vector/growable_vector.ll"),
                    ELEM=elem_ll_ty,
                    NAME=&output_ll_ty.replace("%", "")));
            self.prelude_code.add("\n");
        }

        let expr_ty = func.symbol_type(expr)?;
        let output_ll_prefix = output_ll_ty.replace("%", "");

        let serialize_fn = self.gen_serialize_helper(&output_ll_ty,
                                                     &output_ll_prefix,
                                                     &expr_ll_ty,
                                                     &llvm_prefix(&expr_ll_ty),
                                                     expr_ty,
                                                     func,
                                                     ctx)?;

        let expr_tmp = self.gen_load_var(&expr_ll_sym, &expr_ll_ty, ctx)?;

        let buf_tmp = ctx.var_ids.next();
        ctx.code.add(format!("{} = call {}.growable @{}.growable.new(i64 1024)",
        buf_tmp, output_ll_ty, output_ll_prefix));

        let result = ctx.var_ids.next();
        ctx.code.add(format!("{} = call {}.growable {}({}.growable {}, {} {})",
        result,
        output_ll_ty,
        serialize_fn,
        output_ll_ty,
        buf_tmp,
        expr_ll_ty,
        expr_tmp));

        let output_tmp = ctx.var_ids.next();
        ctx.code.add(format!("{} = call {} @{}.growable.tovec({}.growable {})",
        output_tmp, output_ll_ty, output_ll_prefix, output_ll_ty, result));

        self.gen_store_var(&output_tmp, &output_ll_sym, &output_ll_ty, ctx);
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
            compile_err!("Illegal type {} in {}", child_ty, op_kind)?;
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
        let ref output = statement.output.clone().unwrap_or(Symbol::new("unused", 0));
        ctx.code.add(format!("; {}", statement));
        if self.trace_run {
            self.gen_puts(&format!("  {}", statement), ctx);
        }
        match statement.kind {
            MakeStruct(ref elems) => {
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

            CUDF { ref symbol_name, ref args } => {
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

            MakeVector(ref elems) => {
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

            BinOp { op, ref left, ref right } => {
                use ast::BinOpKind::*;
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                let ty = func.symbol_type(left)?;
                // Assume the left and right operands have the same type.
                let (ll_ty, left_ll_sym) = self.llvm_type_and_name(func, left)?;
                let right_ll_sym = llvm_symbol(right);
                let left_tmp = self.gen_load_var(&left_ll_sym, &ll_ty, ctx)?;
                let right_tmp = self.gen_load_var(&right_ll_sym, &ll_ty, ctx)?;
                let output_tmp = ctx.var_ids.next();
                match *ty {
                    Scalar(s) | Simd(s) => {
                        match op {
                            // Special-case max and min, which don't have int intrinsics
                            Max | Min => {
                                self.gen_minmax(&ll_ty.as_str(), &op,
                                                &left_tmp.as_str(),
                                                &right_tmp.as_str(),
                                                &output_tmp.as_str(),
                                                ty, &mut ctx.var_ids, &mut ctx.code)?;
                            }
                            // Pow only supported for floating point.
                            Pow if ty.is_scalar() && s.is_float() => {
                                    ctx.code.add(format!("{} = call {} {}({} {}, {} {})",
                                    &output_tmp, &ll_ty,
                                    llvm_binary_intrinsic(op, &s)?,
                                    self.llvm_type(ty)?, &left_tmp,
                                    self.llvm_type(ty)?, &right_tmp));
                            }
                            Pow if ty.is_simd() && s.is_float() => {
                                // Unroll and apply the scalar op, and then pack back into vector
                                let scalar_ll_ty = self.llvm_type(&Scalar(s))?;
                                let simd_ll_ty = self.llvm_type(ty)?;
                                let mut prev_tmp = "undef".to_string();
                                let width = llvm_simd_size(&Scalar(s))?;
                                for i in 0..width {
                                    let left_elem_tmp = self.gen_simd_extract(ty, &left_tmp, i, ctx)?;
                                    let right_elem_tmp = self.gen_simd_extract(ty, &right_tmp, i, ctx)?;
                                    let val_tmp = ctx.var_ids.next();
                                     ctx.code.add(format!("{} = call {} {}({} {}, {} {})",
                                                                &val_tmp,
                                                                &scalar_ll_ty,
                                                                llvm_binary_intrinsic(op, &s)?,
                                                                scalar_ll_ty,
                                                                &left_elem_tmp,
                                                                scalar_ll_ty,
                                                                &right_elem_tmp));
                                    let next = if i == width - 1 {
                                        output_tmp.clone()
                                    } else {
                                        ctx.var_ids.next()
                                    };
                                    ctx.code.add(format!("{} = insertelement {} {}, {} {}, i32 {}",
                                                         next, simd_ll_ty, prev_tmp, scalar_ll_ty, val_tmp, i));
                                    prev_tmp = next;
                                }
                            }
                            _ => {
                                ctx.code.add(format!("{} = {} {} {}, {}",
                                                     &output_tmp, llvm_binop(op, ty)?, &ll_ty, &left_tmp, &right_tmp));
                            }
                        }
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

                    _ => compile_err!("Illegal type {} in BinOp", ty)?,
                }
            }

            Broadcast(ref child) => {
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

            Serialize(ref child) => {
                self.gen_serialize(child, output, func, ctx)?;
            }

            Deserialize(ref child) => {
                self.gen_deserialize(child, output, func, ctx)?;
            }

            UnaryOp { op, ref child, } => {
                self.gen_unary_op(ctx, func, output, child, op)?
            }

            Negate(ref child) => {
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

            Cast(ref child, _) => {
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

            Lookup { ref child, ref index } => {
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
                    _ => compile_err!("Illegal type {} in Lookup", child_ty)?,
                }
            }

            KeyExists { ref child, ref key } => {
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

            Slice { ref child, ref index, ref size } => {
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

            Sort { ref child, ref keyfunc } => {
                // keyfunc is an actual SirFunction
                let out_ty = func.symbol_type(output)?;
                let function_id = self.keyfunc_ids.next();
                let func_str = &function_id.replace("%", "");
                match *out_ty {
                    Vector(ref elem_ty) => {
                        let elem_ll_ty = self.llvm_type(elem_ty)?;
                        let mut str_args = String::from("");
                        let param_syms: Vec<_> = keyfunc.params.keys().collect();
                        if param_syms.len() == 1 {
                            let param_ty = keyfunc.symbol_type(param_syms[0])?;
                            if *param_ty == **elem_ty {
                                let (param_ll_ty, param_ll_sym) = self.llvm_type_and_name(keyfunc, param_syms[0])?;
                                str_args.push_str(&format!("{}* {}", param_ll_ty, param_ll_sym));
                            } else {
                                return compile_err!("Type mismatch: vector:{ } and sort key parameter:{ }",
                                                 &**elem_ty, &*param_ty);
                            }
                        }

                        // Check that type of key is scalar
                        let key_ll_sym;
                        let mut key_ll_ty;
                        let last_block = keyfunc.blocks.last().unwrap();
                        if let Terminator::ProgramReturn(ref key_sym) = last_block.terminator {
                            let key_ty = keyfunc.symbol_type(key_sym)?;
                            match *key_ty{
                                Scalar(_) | Vector(_) => {
                                    self.gen_cmp(key_ty)?;
                                    let (key_ll_tyc, key_ll_symc) = self.llvm_type_and_name(keyfunc, key_sym)?;
                                    key_ll_ty = key_ll_tyc;
                                    if let Scalar(_) = *key_ty {
                                        key_ll_ty = key_ll_ty.replace("%", "");
                                    }
                                    key_ll_sym = key_ll_symc;
                                    // Generate prelude code
                                    let keyfunc_ctx = &mut FunctionContext::new(false);
                                    keyfunc_ctx.alloca_code.add(format!("fn.entry:"));
                                    for (arg, ty) in keyfunc.locals.iter() {
                                        let arg_str = llvm_symbol(&arg);
                                        let ty_str = self.llvm_type(&ty)?;
                                        keyfunc_ctx.add_alloca(&arg_str, &ty_str)?;
                                    }
                                    keyfunc_ctx.code.add(format!("br label %b.b{}", keyfunc.blocks[0].id));
                                    let end_block_index = keyfunc.blocks.len() - 1;
                                    for (i, b) in keyfunc.blocks.iter().enumerate() {
                                        keyfunc_ctx.code.add(format!("b.b{}:", b.id));
                                        for s in b.statements.iter() {
                                            self.gen_statement(s, keyfunc, keyfunc_ctx)?
                                        }
                                        if i != end_block_index {
                                            if let Terminator::Branch { ref cond, on_true, on_false} = b.terminator {
                                                keyfunc_ctx.code.add(format!("; {}", b.terminator));
                                                if self.trace_run {
                                                    self.gen_puts(&format!("  {}", b.terminator), ctx);
                                                }
                                                let cond_tmp = self.gen_load_var(llvm_symbol(cond).as_str(), "i1", keyfunc_ctx)?;
                                                keyfunc_ctx.code.add(format!("br i1 {}, label %b.b{}, label %b.b{}", cond_tmp, on_true, on_false));
                                            } else if let Terminator::JumpBlock(block) = b.terminator {
                                                keyfunc_ctx.code.add(format!("; {}", b.terminator));
                                                if self.trace_run {
                                                    self.gen_puts(&format!("  {}", b.terminator), ctx);
                                                }
                                                keyfunc_ctx.code.add(format!("br label %b.b{}", block));
                                            } else if let Terminator::ProgramReturn(_) = b.terminator {
                                            } else {
                                                return compile_err!("Can't have terminator other than Branch or JumpBlock in sort key function");
                                            }
                                        }
                                    }
                                    // Add key function and sort prelude code
                                    let name = self.vec_names.get(elem_ty).unwrap();
                                    self.prelude_code.add(format!(
                                        "define {} @{}({}) alwaysinline {{",
                                        key_ll_ty,
                                        func_str,
                                        str_args));
                                    self.prelude_code.add(keyfunc_ctx.alloca_code.result());
                                    self.prelude_code.add(keyfunc_ctx.code.result());
                                    self.prelude_code.add(format!("{}.ret = load {}, {}* {}",
                                                                  key_ll_sym, key_ll_ty, key_ll_ty, key_ll_sym));
                                    self.prelude_code.add(format!("ret {} {}.ret\n}}", key_ll_ty, key_ll_sym));
                                    self.prelude_code.add(format!(
                                        include_str!("resources/vector/vector_sort.ll"),
                                        ELEM=&elem_ll_ty,
                                        KEY=&key_ll_ty,
                                        RAWKEY=&key_ll_ty.replace("%", ""),
                                        FUNC=&func_str,
                                        NAME=&name.replace("%", "")));
                                }
                                _=> { return compile_err!("Sort key function must have scalar or vector return type");}
                            }
                        } else {
                            return compile_err!("Sort key Function must have return type");
                        }

                    }
                    _ => {
                        return compile_err!("Unsupported function `sort` for type {:?}", out_ty);
                    }
                }

                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                let (child_ll_ty, child_ll_sym) = self.llvm_type_and_name(func, child)?;
                let vec_prefix = llvm_prefix(&child_ll_ty);
                let child_tmp = self.gen_load_var(&child_ll_sym, &child_ll_ty, ctx)?;
                let res_ptr = ctx.var_ids.next();
                ctx.code.add(format!("{} = call {} {}.{}.sort({} {})",
                                     res_ptr,
                                     output_ll_ty,
                                     vec_prefix,
                                     func_str,
                                     child_ll_ty,
                                     child_tmp));
                self.gen_store_var(&res_ptr, &output_ll_sym, &output_ll_ty, ctx);
            }

            Select { ref cond, ref on_true, ref on_false } => {
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

            ToVec(ref child) => {
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                let (child_ll_ty, child_ll_sym) = self.llvm_type_and_name(func, child)?;
                let dict_prefix = llvm_prefix(&child_ll_ty);
                let child_tmp = self.gen_load_var(&child_ll_sym, &child_ll_ty, ctx)?;
                let res_tmp = ctx.var_ids.next();
                ctx.code.add(format!("{} = call {} {}.tovec({} {})",
                    res_tmp, output_ll_ty, dict_prefix, child_ll_ty, child_tmp));
                self.gen_store_var(&res_tmp, &output_ll_sym, &output_ll_ty, ctx);
            }

            Length(ref child) => {
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                let (child_ll_ty, child_ll_sym) = self.llvm_type_and_name(func, child)?;
                let vec_prefix = llvm_prefix(&child_ll_ty);
                let child_tmp = self.gen_load_var(&child_ll_sym, &child_ll_ty, ctx)?;
                let res_tmp = ctx.var_ids.next();
                ctx.code.add(format!("{} = call i64 {}.size({} {})", res_tmp, vec_prefix, child_ll_ty, child_tmp));
                self.gen_store_var(&res_tmp, &output_ll_sym, &output_ll_ty, ctx);
            }

            Assign(ref value) => {
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                let (value_ll_ty, value_ll_sym) = self.llvm_type_and_name(func, value)?;
                let val_tmp = self.gen_load_var(&value_ll_sym, &value_ll_ty, ctx)?;
                self.gen_store_var(&val_tmp, &output_ll_sym, &output_ll_ty, ctx);
            }

            GetField { ref value, index } => {
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                let (value_ll_ty, value_ll_sym) = self.llvm_type_and_name(func, value)?;
                let ptr_tmp = ctx.var_ids.next();
                ctx.code.add(format!("{} = getelementptr inbounds {}, {}* {}, i32 0, i32 {}",
                    ptr_tmp, value_ll_ty, value_ll_ty, value_ll_sym, index));
                let res_tmp = self.gen_load_var(&ptr_tmp, &output_ll_ty, ctx)?;
                self.gen_store_var(&res_tmp, &output_ll_sym, &output_ll_ty, ctx);
            }

            AssignLiteral(ref value) => {
                let (output_ll_ty, output_ll_sym) = self.llvm_type_and_name(func, output)?;
                let output_ty = func.symbol_type(output)?;
                if let Simd(_) = *output_ty {
                    self.gen_simd_literal(&output_ll_sym, value, output_ty, ctx)?;
                } else {
                    match *value {
                        StringLiteral(ref string) => {
                            let ref value = self.string_literal(string, &output_ll_ty, ctx).unwrap();
                            self.gen_store_var(value.as_str(), &output_ll_sym, &output_ll_ty, ctx);
                        }
                        _ => {
                            let ref value = llvm_literal((*value).clone()).unwrap();
                            self.gen_store_var(value, &output_ll_sym, &output_ll_ty, ctx);
                        }
                    }
                }
            }

            Merge { ref builder, ref value } => {
                let bld_ty = func.symbol_type(builder)?;
                if let Builder(ref bld_kind, _) = *bld_ty {
                    let val_ty = func.symbol_type(value)?;
                    self.gen_merge(bld_kind, builder, val_ty, value, func, ctx)?;
                } else {
                    return compile_err!("Non builder type {} found in Merge", bld_ty)
                }
            }

            Res(ref builder) => {
                let bld_ty = func.symbol_type(builder)?;
                if let Builder(ref bld_kind, _) = *bld_ty {
                    self.gen_result(bld_kind, builder, output, func, ctx)?;
                } else {
                    return compile_err!("Non builder type {} found in Result", bld_ty)
                }
            }

            NewBuilder { ref arg, ref ty } => {
                if let Builder(ref bld_kind, ref annotations) = *ty {
                    self.gen_new_builder(bld_kind, annotations, arg, output, func, ctx)?;
                } else {
                    return compile_err!("Non builder type {} found in NewBuilder", ty)
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
                    let merge_ptr = ctx.var_ids.next();
                    if !ctx.is_innermost {
                        ctx.code.add(format!(
                            "{} = call {}* {}.vectorMergePtrForStackPiece({}* {})",
                            merge_ptr,
                            val_ll_ty,
                            bld_prefix,
                            bld_ll_ty,
                            bld_ll_sym));
                    } else {
                        ctx.code.add(format!(
                            "{} = call {}* {}.vectorMergePtrForPiece({}.piecePtr {}.reg)",
                            merge_ptr,
                            val_ll_ty,
                            bld_prefix,
                            bld_ll_ty,
                            bld_ll_sym));
                    }
                    self.gen_merge_op(&merge_ptr, &elem_tmp, &val_ll_ty, op, value_ty, ctx)?;
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

            DictMerger(_, _, _) | GroupMerger(_, _) => {
                let bld_tmp = self.gen_load_var(&bld_ll_sym, &bld_ll_ty, ctx)?;
                let val_tmp = self.gen_load_var(&val_ll_sym, &val_ll_ty, ctx)?;
                ctx.code.add(format!(
                    "call {} {}.merge({} {}, {} {})",
                    bld_ll_ty,
                    bld_prefix,
                    bld_ll_ty,
                    bld_tmp,
                    val_ll_ty,
                    val_tmp));
            }

            Merger(ref t, ref op) => {
                let val_tmp = self.gen_load_var(&val_ll_sym, &val_ll_ty, ctx)?;
                let merge_ptr = ctx.var_ids.next();
                if !ctx.is_innermost {
                    ctx.code.add(format!(
                        "{} = call {}* {}.scalarMergePtrForStackPiece({}* {})",
                        merge_ptr,
                        val_ll_ty,
                        bld_prefix,
                        bld_ll_ty,
                        bld_ll_sym));
                } else {
                    ctx.code.add(format!(
                        "{} = call {}* {}.scalarMergePtrForPiece({}.piecePtr {}.reg)",
                        merge_ptr,
                        val_ll_ty,
                        bld_prefix,
                        bld_ll_ty,
                        bld_ll_sym));
                }
                self.gen_merge_op(&merge_ptr, &val_tmp, &val_ll_ty, op, t, ctx)?;
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
                    return compile_err!("Invalid non-scalar type in merger");
                };

                let elem_vec_ty_str = self.llvm_type(vec_type)?;

                // Builder type.
                let bld_ty_str = self.llvm_type(&bld_ty)?;
                // Prefix of the builder.
                let bld_prefix = llvm_prefix(&bld_ty_str);
                // Result type.
                let res_ty_str = self.llvm_type(&res_ty)?;
                let bld_ll_sym = llvm_symbol(builder);

                // Generate names for all temporaries.
                let is_global = ctx.var_ids.next();
                let scalar_ptr = ctx.var_ids.next();
                let vector_ptr = ctx.var_ids.next();
                let nworkers = ctx.var_ids.next();
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
                                        is_global = is_global,
                                        scalar_ptr = scalar_ptr,
                                        vector_ptr = vector_ptr,
                                        nworkers = nworkers,
                                        bld_sym = bld_ll_sym,
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
                        bld_sym=bld_ll_sym,
                        bld_prefix=bld_prefix,
                        body_v=body_label_v,
                        vector_width=vector_width,
                        done_v=done_label_v));
            }

            DictMerger(_, _, _) | GroupMerger(_, _) => {
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
        if let Some(ref e) = annotations.size() {
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
                // Generate code to initialize the builder.
                let iden_elem = binop_identity(*op, elem_ty.as_ref())?;
                let init_elem = match *arg {
                    Some(ref s) => {
                        let arg_str = self.gen_load_var(llvm_symbol(s).as_str(), &elem_type, ctx)?;
                        arg_str
                    }
                    _ => iden_elem.clone(),
                };
                let bld_tmp_stack = format!("{}.stack", bld_tmp);
                let bld_stack_ty_str = format!("{}.piece", bld_ty_str);
                ctx.add_alloca(&bld_tmp_stack, &bld_stack_ty_str)?;
                ctx.code.add(format!("{} = call {} {}.new({} {}, {} {}, {}* {})", bld_tmp, bld_ty_str,
                    bld_prefix, elem_type, iden_elem, elem_type, init_elem, bld_stack_ty_str, bld_tmp_stack));
                self.gen_store_var(&bld_tmp, &llvm_symbol(output), &bld_ty_str, ctx);
            }
            DictMerger(_, _, _) | GroupMerger(_, _) => {
                let bld_tmp = ctx.var_ids.next();
                let max_local_bytes = if let Some(ref sym) = *arg {
                    let (arg_ll_ty, arg_ll_sym) = self.llvm_type_and_name(func, sym)?;
                    self.gen_load_var(&arg_ll_sym, &arg_ll_ty, ctx)?
                } else {
                    format!("{}", 100000000)
                };
                ctx.code.add(format!("{} = call {} {}.new(i64 {}, i64 {})",
                                        bld_tmp,
                                        bld_ty_str,
                                        bld_prefix,
                                        builder_size,
                                        max_local_bytes));
                self.gen_store_var(&bld_tmp, &llvm_symbol(output), &bld_ty_str, ctx);
            }
            VecMerger(ref elem, _) => {
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
                        compile_err!("Internal error: NewBuilder(VecMerger) \
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
        if self.trace_run {
            self.gen_puts(&format!("  {}", terminator), ctx);
        }
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
                let params_sorted = get_combined_params(sir, pf);
                let mut arg_types = String::new();
                for (arg, ty) in params_sorted.iter() {
                    let ll_ty = self.llvm_type(&ty)?;
                    let arg_tmp = self.gen_load_var(llvm_symbol(arg).as_str(), &ll_ty, ctx)?;
                    let arg_str = format!("{} {}, ", &ll_ty, arg_tmp);
                    arg_types.push_str(&arg_str);
                }
                arg_types.push_str("%work_t* %cur.work");
                ctx.code.add(format!("call void @f{}_wrapper({}, i32 %cur.tid)", pf.body, arg_types));
                ctx.code.add("br label %body.end");
            }

            JumpBlock(block) => {
                ctx.code.add(format!("br label %b.b{}", block));
            }

            JumpFunction(func) => {
                self.gen_top_level_function(sir, &sir.funcs[func])?;
                let ref params_sorted = sir.funcs[func].params;
                let mut arg_types = String::new();
                for (arg, ty) in params_sorted.iter() {
                    let ll_ty = self.llvm_type(&ty)?;
                    let arg_tmp = self.gen_load_var(llvm_symbol(arg).as_str(), &ll_ty, ctx)?;
                    let arg_str = format!("{} {}, ", ll_ty, arg_tmp);
                    arg_types.push_str(&arg_str);
                }
                arg_types.push_str("%work_t* %cur.work");
                ctx.code.add(format!("call void @f{}({}, i32 %cur.tid)", func, arg_types));
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

            _ => compile_err!("Invalid type for gen_simd_extract: {:?}", simd_type)
        }
    }

    /// Generate a puts() call to print text at runtime.
    /// Note that unlike StringLiteral constants, gen_puts generates a null-terminated string.
    fn gen_puts(&mut self, text: &str, ctx: &mut FunctionContext) {
        let global = self.prelude_var_ids.next().replace("%", "@");
        let text = text.replace("\\", "\\\\").replace("\"", "\\\"");
        let len = text.len() + 1;
        self.prelude_code.add(format!(
            "{} = private unnamed_addr constant [{} x i8] c\"{}\\00\"",
            global, len, text));
        let local = ctx.var_ids.next();
        ctx.code.add(format!(
            "{} = getelementptr [{} x i8], [{} x i8]* {}, i32 0, i32 0",
            local, len, len, global));
        ctx.code.add(format!("call i32 @puts(i8* {})", local));
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
fn llvm_literal(k: LiteralKind) -> WeldResult<String> {
    let res = match k {
        BoolLiteral(l) => format!("{}", if l { 1 } else { 0 }),
        I8Literal(l) => format!("{}", l),
        I16Literal(l) => format!("{}", l),
        I32Literal(l) =>  format!("{}", l),
        I64Literal(l) => format!("{}", l),
        U8Literal(l) => format!("{}", l),
        U16Literal(l) => format!("{}", l),
        U32Literal(l) => format!("{}", l),
        U64Literal(l) => format!("{}", l),
        F32Literal(l) => format!("{:.30e}", f32::from_bits(l)),
        F64Literal(l) => format!("{:.30e}", f64::from_bits(l)),
        StringLiteral(_) => {
            return compile_err!("String literal must be declared as global constant in LLVM");
        }
    }.to_string();
    Ok(res)
}

/// Return the LLVM version of a Weld symbol (encoding any special characters for LLVM).
fn llvm_symbol(symbol: &Symbol) -> String {
    if symbol.id == 0 { format!("%{}", symbol.name) } else { format!("%{}.{}", symbol.name, symbol.id) }
}

fn binop_identity(op_kind: BinOpKind, ty: &Type) -> WeldResult<String> {
    use ast::BinOpKind::*;
    match (op_kind, ty) {
        (Add, &Scalar(s)) if s.is_integer() => Ok("0".to_string()),
        (Add, &Scalar(s)) if s.is_float() => Ok("0.0".to_string()),

        (Multiply, &Scalar(s)) if s.is_integer() => Ok("1".to_string()),
        (Multiply, &Scalar(s)) if s.is_float() => Ok("1.0".to_string()),

        (Min, &Scalar(s)) => match s {
            I8  => Ok(::std::i8::MAX.to_string()),
            I16 => Ok(::std::i16::MAX.to_string()),
            I32 => Ok(::std::i32::MAX.to_string()),
            I64 => Ok(::std::i64::MAX.to_string()),
            U8  => Ok(::std::u8::MAX.to_string()),
            U16 => Ok(::std::u16::MAX.to_string()),
            U32 => Ok(::std::u32::MAX.to_string()),
            U64 => Ok(::std::u64::MAX.to_string()),
            F32 => Ok("0x7FF0000000000000".to_string()), // inf 
            F64 => Ok("0x7FF0000000000000".to_string()), // inf
            _ => compile_err!("Unsupported identity for binary op: {} on {}", op_kind, ty),
        },

        (Max, &Scalar(s)) => match s {
            I8  => Ok(::std::i8::MIN.to_string()),
            I16 => Ok(::std::i16::MIN.to_string()),
            I32 => Ok(::std::i32::MIN.to_string()),
            I64 => Ok(::std::i64::MIN.to_string()),
            U8  => Ok(::std::u8::MIN.to_string()),
            U16 => Ok(::std::u16::MIN.to_string()),
            U32 => Ok(::std::u32::MIN.to_string()),
            U64 => Ok(::std::u64::MIN.to_string()),
            F32 => Ok("0xFFF0000000000000".to_string()), // -inf
            F64 => Ok("0xFFF0000000000000".to_string()), // -inf
            _ => compile_err!("Unsupported identity for binary op: {} on {}", op_kind, ty),
        },

        _ => compile_err!("Unsupported identity for binary op: {} on {}", op_kind, ty),
    }
}

/// Return the name of the LLVM instruction for a binary operation on a specific type.
fn llvm_binop(op_kind: BinOpKind, ty: &Type) -> WeldResult<&'static str> {
    use ast::BinOpKind::*;
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

                _ => return compile_err!("Unsupported binary op: {} on {}", op_kind, ty)
            }
        }

        _ => return compile_err!("Unsupported binary op: {} on {}", op_kind, ty)
    }
}

/// Return LLVM intrinsic for float max/min.
fn llvm_binary_intrinsic(op_kind: BinOpKind, ty: &ScalarKind) -> WeldResult<&'static str> {
    match (op_kind, ty) {
        (BinOpKind::Min, &F32) => Ok("@llvm.minnum.f32"),
        (BinOpKind::Min, &F64) => Ok("@llvm.minnum.f64"),

        (BinOpKind::Max, &F32) => Ok("@llvm.maxnum.f32"),
        (BinOpKind::Max, &F64) => Ok("@llvm.maxnum.f64"),

        (BinOpKind::Pow, &F32) => Ok("@llvm.pow.f32"),
        (BinOpKind::Pow, &F64) => Ok("@llvm.pow.f64"),

        _ => compile_err!("Unsupported binary op: {} on {}", op_kind, ty),
    }
}

/// Return LLVM intrinsic for simd float max/min
fn llvm_simd_binary_intrinsic(op_kind: BinOpKind, ty: &ScalarKind, width: u32) -> WeldResult<&'static str> {
    match (op_kind, ty, width) {
        (BinOpKind::Min, &F32, 4) => Ok("@llvm.minnum.v4f32"),
        (BinOpKind::Min, &F32, 8) => Ok("@llvm.minnum.v8f32"),
        (BinOpKind::Min, &F64, 2) => Ok("@llvm.minnum.v2f64"),
        (BinOpKind::Min, &F64, 4) => Ok("@llvm.minnum.v4f64"),

        (BinOpKind::Max, &F32, 4) => Ok("@llvm.maxnum.v4f32"),
        (BinOpKind::Max, &F32, 8) => Ok("@llvm.maxnum.v8f32"),
        (BinOpKind::Max, &F64, 2) => Ok("@llvm.maxnum.v2f64"),
        (BinOpKind::Max, &F64, 4) => Ok("@llvm.maxnum.v4f64"),

        _ => compile_err!("Unsupported binnary op: {} on <{} x {}>", op_kind, width, ty),
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

        (UnaryOpKind::Sin, &F32) => Ok("@llvm.sin.f32"),
        (UnaryOpKind::Sin, &F64) => Ok("@llvm.sin.f64"),

        (UnaryOpKind::Cos, &F32) => Ok("@llvm.cos.f32"),
        (UnaryOpKind::Cos, &F64) => Ok("@llvm.cos.f64"),

        (UnaryOpKind::Tan, &F32) => Ok("@tanf"),
        (UnaryOpKind::Tan, &F64) => Ok("@tan"),

        (UnaryOpKind::ASin, &F32) => Ok("@asinf"),
        (UnaryOpKind::ASin, &F64) => Ok("@asin"),
        (UnaryOpKind::ACos, &F32) => Ok("@acosf"),
        (UnaryOpKind::ACos, &F64) => Ok("@acos"),
        (UnaryOpKind::ATan, &F32) => Ok("@atanf"),
        (UnaryOpKind::ATan, &F64) => Ok("@atan"),

        (UnaryOpKind::Sinh, &F32) => Ok("@sinhf"),
        (UnaryOpKind::Sinh, &F64) => Ok("@sinh"),
        (UnaryOpKind::Cosh, &F32) => Ok("@coshf"),
        (UnaryOpKind::Cosh, &F64) => Ok("@cosh"),
        (UnaryOpKind::Tanh, &F32) => Ok("@tanhf"),
        (UnaryOpKind::Tanh, &F64) => Ok("@tanh"),



        (UnaryOpKind::Erf, &F32) => Ok("@erff"),
        (UnaryOpKind::Erf, &F64) => Ok("@erf"),

        _ => compile_err!("Unsupported unary op: {} on {}", op_kind, ty),
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

        _ => compile_err!("Unsupported unary op: {} on <{} x {}>", op_kind, width, ty),
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

        _ => compile_err!("Unsupported binary op: {} on {}", op_kind, ty),
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

                 _ => compile_err!("Can't cast {} to {}", ty1, ty2)
            }
        }

        _ => compile_err!("Can't cast {} to {}", ty1, ty2)

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
            compile_err!("Symbol already defined in function: {}", symbol)
        } else {
            self.alloca_code.add(format!("{} = alloca {}", symbol, ty));
            Ok(())
        }
    }
}

fn get_combined_params(sir: &SirProgram, par_for: &ParallelForData) -> BTreeMap<Symbol, Type> {
    let mut body_params = sir.funcs[par_for.body].params.clone();
    for (arg, ty) in sir.funcs[par_for.cont].params.iter() {
        body_params.insert(arg.clone(), ty.clone());
    }
    body_params
}

#[cfg(test)]
fn predicate_only(code: &str) -> WeldResult<Expr> {
    let mut e = parse_expr(code).unwrap();
    assert!(e.infer_types().is_ok());
    let mut typed_e = e;

    let optstr = ["predicate"];
    let optpass = optstr.iter().map(|x| (*OPTIMIZATION_PASSES.get(x).unwrap()).clone()).collect();

    apply_opt_passes(&mut typed_e, &optpass, &mut CompilationStats::new(), false)?;

    Ok(typed_e)
}

#[test]
fn simple_predicate() {
    /* Ensure that the simple_predicate transform works. */
    let code = "|v1:vec[i32],v2:vec[bool]| result(for(zip(v1, v2), merger[i32,+], |b,i,e| merge(b, @(predicate:true) if(e.$1, e.$0, 0))))";
    let typed_e = predicate_only(code);
    assert!(typed_e.is_ok());
    let expected = "|v1:vec[i32],v2:vec[bool]|result(for(zip(v1:vec[i32],v2:vec[bool]),merger[i32,+],|b:merger[i32,+],i:i64,e:{i32,bool}|merge(b:merger[i32,+],select(e.$1,e.$0,0))))";
    assert_eq!(print_typed_expr_without_indent(&typed_e.unwrap()).as_str(),
               expected);

    /* Ensure that the simple_predicate transform doesn't change the program if on_true or on_false contains a builder expression. */
    let code = "|v1:vec[i32],v2:vec[bool]| result(for(v2, appender, |b,i,e| merge(b, @(predicate:true) if(e, result(for(v1, merger[i32,+], |b2,i2,e2| merge(b2,e2))), 0))))";
    let typed_e = predicate_only(code);
    assert!(typed_e.is_ok());
    let expected = "|v1:vec[i32],v2:vec[bool]|result(for(v2:vec[bool],appender[i32],|b:appender[i32],i:i64,e:bool|merge(b:appender[i32],@(predicate:true)if(e:bool,result(for(v1:vec[i32],merger[i32,+],|b2:merger[i32,+],i2:i64,e2:i32|merge(b2:merger[i32,+],e2:i32))),0))))";
    assert_eq!(print_typed_expr_without_indent(&typed_e.unwrap()).as_str(),
               expected);
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
    let expected = "|v:vec[i32]|result(for(v:vec[i32],merger[i32,+],|b:merger[i32,+],i:i64,e:i32|if((e:i32>0),merge(b:merger[i32,+],e:i32),b:merger[i32,+])))";
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
    let expected = "|v:vec[i32]|result(for(v:vec[i32],dictmerger[{i32,i32},i32,+],|b:dictmerger[{i32,i32},i32,+],i:i64,e:i32|(let k:{{i32,i32},i32}=({{e:i32,e:i32},(e:i32*2)});select((e:i32>0),k:{{i32,i32},i32},{k.$0,0}))))";
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

    let struct1 = parse_type("{i32,bool,i32}").unwrap();
    assert_eq!(gen.llvm_type(&struct1).unwrap(), "%s0");
    assert_eq!(gen.llvm_type(&struct1).unwrap(), "%s0"); // Name is reused for same struct

    let struct2 = parse_type("{i32,bool}").unwrap();
    assert_eq!(gen.llvm_type(&struct2).unwrap(), "%s1");
}
