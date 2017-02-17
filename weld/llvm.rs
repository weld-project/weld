use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::BTreeMap;

use easy_ll;

use super::WeldRuntimeErrno;

use super::ast::*;
use super::ast::Type::*;
use super::ast::LiteralKind::*;
use super::ast::ScalarKind::*;
use super::ast::BuilderKind::*;
use super::code_builder::CodeBuilder;
use super::error::*;
use super::macro_processor;
use super::pretty_print::*;
use super::program::Program;
use super::sir;
use super::sir::*;
use super::sir::Statement::*;
use super::sir::Terminator::*;
use super::transforms;
use super::type_inference;
use super::util::IdGenerator;

#[cfg(test)]
use super::parser::*;
#[cfg(test)]
use super::weld_run_free;
#[cfg(test)]
use super::weld_rt_malloc;
#[cfg(test)]
use super::weld_rt_realloc;
#[cfg(test)]
use super::weld_rt_free;
#[cfg(test)]
use super::weld_rt_get_errno;
#[cfg(test)]
use super::weld_rt_set_errno;

static PRELUDE_CODE: &'static str = include_str!("resources/prelude.ll");
static VECTOR_CODE: &'static str = include_str!("resources/vector.ll");
static MERGER_CODE: &'static str = include_str!("resources/merger.ll");
static DICTIONARY_CODE: &'static str = include_str!("resources/dictionary.ll");
static DICTMERGER_CODE: &'static str = include_str!("resources/dictmerger.ll");

/// Generates LLVM code for one or more modules.
pub struct LlvmGenerator {
    /// Track a unique name of the form %s0, %s1, etc for each struct generated.
    struct_names: HashMap<Vec<Type>, String>,
    struct_ids: IdGenerator,

    /// Track a unique name of the form %v0, %v1, etc for each vec generated.
    vec_names: HashMap<Type, String>,
    vec_ids: IdGenerator,

    merger_names: HashMap<(Type, BinOpKind), String>,
    merger_ids: IdGenerator,

    /// Tracks a unique name of the form %d0, %d1, etc for each dict generated.
    dict_names: HashMap<Type, String>,
    dict_ids: IdGenerator,

    /// TODO This is unnecessary but satisfies the compiler for now.
    bld_names: HashMap<BuilderKind, String>,

    /// A CodeBuilder for prelude functions such as type and struct definitions.
    prelude_code: CodeBuilder,
    prelude_var_ids: IdGenerator,

    /// A CodeBuilder for body functions in the module.
    body_code: CodeBuilder,
    visited: HashSet<sir::FunctionId>,
}

/// A wrapper for a struct passed as input to the Weld runtime.
pub struct WeldInputArgs {
    pub input: i64,
    pub nworkers: i32,
    pub run_id: i64,
}

fn get_combined_params(sir: &SirProgram, par_for: &ParallelForData) -> HashMap<Symbol, Type> {
    let mut body_params = sir.funcs[par_for.body].params.clone();
    for (arg, ty) in sir.funcs[par_for.cont].params.iter() {
        body_params.insert(arg.clone(), ty.clone());
    }
    body_params
}

fn get_sym_ty<'a>(func: &'a SirFunction, sym: &Symbol) -> WeldResult<&'a Type> {
    if func.locals.get(sym).is_some() {
        Ok(func.locals.get(sym).unwrap())
    } else if func.params.get(sym).is_some() {
        Ok(func.params.get(sym).unwrap())
    } else {
        weld_err!("Can't find symbol {}#{}", sym.name, sym.id)
    }
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
            bld_names: HashMap::new(),
            prelude_code: CodeBuilder::new(),
            prelude_var_ids: IdGenerator::new("%p.p"),
            body_code: CodeBuilder::new(),
            visited: HashSet::new(),
        };
        generator.prelude_code.add(PRELUDE_CODE);
        generator.prelude_code.add("\n");
        generator
    }

    /// Return all the code generated so far.
    pub fn result(&mut self) -> String {
        format!("; PRELUDE:\n\n{}\n; BODY:\n\n{}",
                self.prelude_code.result(),
                self.body_code.result())
    }

    fn get_arg_str(&mut self, params: &HashMap<Symbol, Type>, suffix: &str) -> WeldResult<String> {
        let mut arg_types = String::new();
        let params_sorted: BTreeMap<&Symbol, &Type> = params.iter().collect();
        for (arg, ty) in params_sorted.iter() {
            let arg_str = format!("{} {}{}, ",
                                  try!(self.llvm_type(&ty)),
                                  llvm_symbol(&arg),
                                  suffix);
            arg_types.push_str(&arg_str);
        }
        arg_types.push_str("%work_t* %cur.work");
        Ok(arg_types)
    }

    fn unload_arg_struct(&mut self,
                         params: &HashMap<Symbol, Type>,
                         ctx: &mut FunctionContext)
                         -> WeldResult<()> {
        let params_sorted: BTreeMap<&Symbol, &Type> = params.iter().collect();
        let ll_ty = try!(self.llvm_type(
            &Struct(params_sorted.iter().map(|p| p.1.clone()).cloned().collect())));
        let storage_typed = ctx.var_ids.next();
        let storage = ctx.var_ids.next();
        let work_data_ptr = ctx.var_ids.next();
        let work_data = ctx.var_ids.next();
        ctx.code.add(format!("{} = getelementptr %work_t* %cur.work, i32 0, i32 0",
                             work_data_ptr));
        ctx.code.add(format!("{} = load i8** {}", work_data, work_data_ptr));
        ctx.code.add(format!("{} = bitcast i8* {} to {}*",
                             storage_typed,
                             work_data,
                             ll_ty));
        ctx.code.add(format!("{} = load {}* {}", storage, ll_ty, storage_typed));
        for (i, (arg, _)) in params_sorted.iter().enumerate() {
            ctx.code.add(format!("{} = extractvalue {} {}, {}",
                                 llvm_symbol(arg),
                                 ll_ty,
                                 storage,
                                 i));
        }
        Ok(())
    }

    fn get_arg_struct(&mut self,
                      params: &HashMap<Symbol, Type>,
                      ctx: &mut FunctionContext)
                      -> WeldResult<String> {
        let params_sorted: BTreeMap<&Symbol, &Type> = params.iter().collect();
        let mut prev_ref = String::from("undef");
        let ll_ty = try!(self.llvm_type(
            &Struct(params_sorted.iter().map(|p| p.1.clone()).cloned().collect())))
            .to_string();
        for (i, (arg, ty)) in params_sorted.iter().enumerate() {
            let next_ref = ctx.var_ids.next();
            ctx.code.add(format!("{} = insertvalue {} {}, {} {}, {}",
                                 next_ref,
                                 ll_ty,
                                 prev_ref,
                                 try!(self.llvm_type(&ty)),
                                 llvm_symbol(arg),
                                 i));
            prev_ref.clear();
            prev_ref.push_str(&next_ref);
        }
        let struct_size_ptr = ctx.var_ids.next();
        let struct_size = ctx.var_ids.next();
        let struct_storage = ctx.var_ids.next();
        let struct_storage_typed = ctx.var_ids.next();
        let run_id = ctx.var_ids.next();
        ctx.code.add(format!("{} = getelementptr {}* null, i32 1", struct_size_ptr, ll_ty));
        ctx.code.add(format!("{} = ptrtoint {}* {} to i64",
                             struct_size,
                             ll_ty,
                             struct_size_ptr));
        ctx.code.add(format!("{} = call i64 @get_runid()", run_id));
        ctx.code
            .add(format!("{} = call i8* @weld_rt_malloc(i64 {}, i64 {})",
                         struct_storage,
                         run_id,
                         struct_size));
        ctx.code.add(format!("{} = bitcast i8* {} to {}*",
                             struct_storage_typed,
                             struct_storage,
                             ll_ty));
        ctx.code.add(format!("store {} {}, {}* {}",
                             ll_ty,
                             prev_ref,
                             ll_ty,
                             struct_storage_typed));
        Ok(struct_storage)
    }

    /// Add a function to the generated program.
    pub fn add_function(&mut self,
                        sir: &SirProgram,
                        func: &SirFunction,
                        // non-None only if func is loop body
                        containing_loop: Option<ParallelForData>)
                        -> WeldResult<()> {
        if !self.visited.insert(func.id) {
            return Ok(());
        }
        {
            let mut ctx = &mut FunctionContext::new();
            let mut arg_types = try!(self.get_arg_str(&func.params, ".in"));
            if containing_loop.is_some() {
                arg_types.push_str(", i64 %lower.idx, i64 %upper.idx");
            }

            // Start the entry block by defining the function and storing all its arguments on the
            // stack (this makes them consistent with other local variables). Later, expressions may
            // add more local variables to alloca_code.
            ctx.alloca_code.add(format!("define void @f{}({}) {{", func.id, arg_types));
            ctx.alloca_code.add(format!("fn.entry:"));
            for (arg, ty) in func.params.iter() {
                let arg_str = llvm_symbol(&arg);
                let ty_str = try!(self.llvm_type(&ty)).to_string();
                try!(ctx.add_alloca(&arg_str, &ty_str));
                ctx.code.add(format!("store {} {}.in, {}* {}", ty_str, arg_str, ty_str, arg_str));
            }
            for (arg, ty) in func.locals.iter() {
                let arg_str = llvm_symbol(&arg);
                let ty_str = try!(self.llvm_type(&ty)).to_string();
                try!(ctx.add_alloca(&arg_str, &ty_str));
            }

            if containing_loop.is_some() {
                let par_for = containing_loop.clone().unwrap();
                let bld_ty_str = try!(self.llvm_type(func.params.get(&par_for.builder).unwrap()))
                    .to_string();
                let bld_param_str = llvm_symbol(&par_for.builder);
                let bld_arg_str = llvm_symbol(&par_for.builder_arg);
                ctx.code.add(format!("store {} {}.in, {}* {}",
                                     &bld_ty_str,
                                     bld_param_str,
                                     &bld_ty_str,
                                     bld_arg_str));
                try!(ctx.add_alloca("%cur.idx", "i64"));
                ctx.code.add("store i64 %lower.idx, i64* %cur.idx");
                ctx.code.add("br label %loop.start");
                ctx.code.add("loop.start:");
                let idx_tmp = try!(self.load_var("%cur.idx", "i64", ctx));
                let idx_cmp = ctx.var_ids.next();
                ctx.code.add(format!("{} = icmp ult i64 {}, %upper.idx", idx_cmp, idx_tmp));
                ctx.code.add(format!("br i1 {}, label %loop.body, label %loop.end", idx_cmp));
                ctx.code.add("loop.body:");
                let mut prev_ref = String::from("undef");
                let elem_ty = func.locals.get(&par_for.data_arg).unwrap();
                let elem_ty_str = try!(self.llvm_type(&elem_ty)).to_string();
                for (i, iter) in par_for.data.iter().enumerate() {
                    let data_ty_str = try!(self.llvm_type(func.params.get(&iter.data).unwrap()))
                        .to_string();
                    let data_str =
                        try!(self.load_var(llvm_symbol(&iter.data).as_str(), &data_ty_str, ctx));
                    let data_prefix = format!("@{}", data_ty_str.replace("%", ""));
                    let inner_elem_tmp_ptr = ctx.var_ids.next();
                    let inner_elem_ty_str = if par_for.data.len() == 1 {
                        elem_ty_str.clone()
                    } else {
                        match *elem_ty {
                            Struct(ref v) => try!(self.llvm_type(&v[i])).to_string(),
                            _ => {
                                weld_err!("Internal error: invalid element type {}",
                                          print_type(elem_ty))?
                            }
                        }
                    };
                    let arr_idx = if iter.start.is_some() {
                        let offset = ctx.var_ids.next();
                        let stride_str = try!(self.load_var(
                                llvm_symbol(&iter.stride.clone().unwrap()).as_str(), "i64", ctx));
                        let start_str = try!(self.load_var(
                                llvm_symbol(&iter.start.clone().unwrap()).as_str(), "i64", ctx));
                        ctx.code.add(format!("{} = mul i64 {}, {}", offset, idx_tmp, stride_str));
                        let final_idx = ctx.var_ids.next();
                        ctx.code.add(format!("{} = add i64 {}, {}", final_idx, start_str, offset));
                        final_idx
                    } else {
                        idx_tmp.clone()
                    };
                    ctx.code.add(format!("{} = call {}* {}.at({} {}, i64 {})",
                                         inner_elem_tmp_ptr,
                                         &inner_elem_ty_str,
                                         data_prefix,
                                         &data_ty_str,
                                         data_str,
                                         arr_idx));
                    let inner_elem_tmp =
                        try!(self.load_var(&inner_elem_tmp_ptr, &inner_elem_ty_str, ctx));
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
                ctx.code.add(format!("store {} {}, {}* {}",
                                     &elem_ty_str,
                                     prev_ref,
                                     &elem_ty_str,
                                     elem_str));
                ctx.code.add(format!("store i64 {}, i64* {}",
                                     idx_tmp,
                                     llvm_symbol(&par_for.idx_arg)));
            }

            ctx.code.add(format!("br label %b.b{}", func.blocks[0].id));
            // Generate an expression for the function body.
            try!(self.gen_func(sir, func, ctx));
            ctx.code.add("body.end:");
            if containing_loop.is_some() {
                ctx.code.add("br label %loop.terminator");
                ctx.code.add("loop.terminator:");
                let idx_tmp = try!(self.load_var("%cur.idx", "i64", ctx));
                let idx_inc = ctx.var_ids.next();
                ctx.code.add(format!("{} = add i64 {}, 1", idx_inc, idx_tmp));
                ctx.code.add(format!("store i64 {}, i64* %cur.idx", idx_inc));
                ctx.code.add("br label %loop.start");
                ctx.code.add("loop.end:");
            }
            ctx.code.add("ret void");
            ctx.code.add("}\n\n");

            self.body_code.add(&ctx.alloca_code.result());
            self.body_code.add(&ctx.code.result());
        }

        if containing_loop.is_some() {
            let par_for = containing_loop.clone().unwrap();
            {
                let mut wrap_ctx = &mut FunctionContext::new();
                let serial_arg_types =
                    try!(self.get_arg_str(&get_combined_params(sir, &par_for), ""));
                wrap_ctx.code
                    .add(format!("define void @f{}_wrapper({}) {{", func.id, serial_arg_types));
                wrap_ctx.code.add(format!("fn.entry:"));

                // Use the first data to compute the indexing.
                let first_data = &par_for.data[0].data;
                let data_str = llvm_symbol(&first_data);
                let data_ty_str = try!(self.llvm_type(func.params.get(&first_data).unwrap()))
                    .to_string();
                let data_prefix = format!("@{}", data_ty_str.replace("%", ""));

                let num_iters_str = wrap_ctx.var_ids.next();
                if par_for.data[0].start.is_none() {
                    // set num_iters_str to len(first_data)
                    wrap_ctx.code.add(format!("{} = call i64 {}.size({} {})",
                                              num_iters_str,
                                              data_prefix,
                                              data_ty_str,
                                              data_str));
                } else {
                    // set num_iters_str to (end - start) / stride
                    let start_str = llvm_symbol(&par_for.data[0].start.clone().unwrap());
                    let end_str = llvm_symbol(&par_for.data[0].end.clone().unwrap());
                    let stride_str = llvm_symbol(&par_for.data[0].stride.clone().unwrap());
                    let diff_tmp = wrap_ctx.var_ids.next();
                    let vector_len = wrap_ctx.var_ids.next();
                    wrap_ctx.code.add(format!("{} = call i64 {}.size({} {})",
                                              vector_len,
                                              data_prefix,
                                              data_ty_str,
                                              data_str));
                    wrap_ctx.code
                        .add(format!("{} = sub i64 {}, {}", diff_tmp, end_str, start_str));
                    wrap_ctx.code
                        .add(format!("{} = udiv i64 {}, {}", num_iters_str, diff_tmp, stride_str));
                }

                // Perform a bounds check on each of the data items before launching the loop
                for iter in par_for.data.iter() {
                    // Vector LLVM information for the current iter.
                    let data_str = llvm_symbol(&iter.data);
                    let data_ty_str = try!(self.llvm_type(func.params.get(&iter.data).unwrap()))
                        .to_string();
                    let data_prefix = format!("@{}", data_ty_str.replace("%", ""));

                    let vec_size_str = wrap_ctx.var_ids.next();
                    wrap_ctx.code.add(format!("{} = call i64 {}.size({} {})",
                                              vec_size_str,
                                              data_prefix,
                                              data_ty_str,
                                              data_str));

                    let (start_str, end_str, stride_str) = if iter.start.is_none() {
                        let start_str = "0".to_string();
                        let stride_str = "1".to_string();
                        (start_str, vec_size_str.clone(), stride_str)
                    } else {
                        (llvm_symbol(iter.start.as_ref().unwrap()),
                         llvm_symbol(iter.end.as_ref().unwrap()),
                         llvm_symbol(iter.stride.as_ref().unwrap()))
                    };

                    let t0 = wrap_ctx.var_ids.next();
                    let t1 = wrap_ctx.var_ids.next();
                    let t2 = wrap_ctx.var_ids.next();
                    let cond = wrap_ctx.var_ids.next();
                    let next_bounds_check_label = wrap_ctx.var_ids.next();

                    // t0 = mul i64 stride, num_iters
                    // t1 = add i64 t0, start
                    // cond = icmp lte i64 t1, size
                    // br i1 cond, label %nextCheck, label %checkFailed
                    // nextCheck:
                    // (loop)
                    wrap_ctx.code
                        .add(format!("{} = mul i64 {}, {}", t0, stride_str, num_iters_str));
                    wrap_ctx.code
                        .add(format!("{} = add i64 {}, {}", t1, t0, start_str));
                    wrap_ctx.code
                        .add(format!("{} = icmp ule i64 {}, {}", cond, t1, vec_size_str));
                    wrap_ctx.code
                        .add(format!("br i1 {}, label {}, label %fn.boundcheckfailed",
                                     cond,
                                     next_bounds_check_label));
                    wrap_ctx.code.add(format!("{}:", next_bounds_check_label.replace("%", "")));
                }
                // If we get here, the bounds check passed.
                wrap_ctx.code.add(format!("br label %fn.boundcheckpassed"));
                // Handle a bounds check fail.
                wrap_ctx.code.add(format!("fn.boundcheckfailed:"));
                let errno = WeldRuntimeErrno::BadIteratorLength;
                let run_id = wrap_ctx.var_ids.next();
                wrap_ctx.code.add(format!("{} = call i64 @get_runid()", run_id));
                wrap_ctx.code.add(format!("call void @weld_rt_set_errno(i64 {}, i64 {})",
                                          run_id,
                                          errno as i64));
                wrap_ctx.code.add(format!("call void @weld_abort_thread()"));
                wrap_ctx.code.add(format!("; Unreachable!"));
                wrap_ctx.code.add(format!("br label %fn.end"));

                // TODO make a smarter decision on whether to call serial here (+ context-dependent
                // grain size)
                wrap_ctx.code.add(format!("fn.boundcheckpassed:"));
                let bound_cmp = wrap_ctx.var_ids.next();
                wrap_ctx.code.add(format!("{} = icmp ult i64 {}, 1024", bound_cmp, num_iters_str));
                wrap_ctx.code.add(format!("br i1 {}, label %for.ser, label %for.par", bound_cmp));
                wrap_ctx.code.add(format!("for.ser:"));
                let mut body_arg_types = try!(self.get_arg_str(&func.params, ""));
                body_arg_types.push_str(format!(", i64 0, i64 {}", num_iters_str).as_str());
                wrap_ctx.code.add(format!("call void @f{}({})", func.id, body_arg_types));
                let cont_arg_types = try!(self.get_arg_str(&sir.funcs[par_for.cont].params, ""));
                wrap_ctx.code.add(format!("call void @f{}({})", par_for.cont, cont_arg_types));
                wrap_ctx.code.add(format!("br label %fn.end"));
                wrap_ctx.code.add(format!("for.par:"));
                let body_struct = try!(self.get_arg_struct(&func.params, &mut wrap_ctx));
                let cont_struct =
                    try!(self.get_arg_struct(&sir.funcs[par_for.cont].params, &mut wrap_ctx));
                wrap_ctx.code
                    .add(format!("call void @pl_start_loop(%work_t* %cur.work, i8* {}, i8* {}, \
                                  void (%work_t*)* @f{}_par, void (%work_t*)* @f{}_par, i64 0, \
                                  i64 {})",
                                 body_struct,
                                 cont_struct,
                                 func.id,
                                 par_for.cont,
                                 num_iters_str));
                wrap_ctx.code.add(format!("br label %fn.end"));
                wrap_ctx.code.add("fn.end:");
                wrap_ctx.code.add("ret void");
                wrap_ctx.code.add("}\n\n");
                self.body_code.add(&wrap_ctx.code.result());
            }
            {
                let mut par_body_ctx = &mut FunctionContext::new();
                par_body_ctx.code
                    .add(format!("define void @f{}_par(%work_t* %cur.work) {{", func.id));
                try!(self.unload_arg_struct(&func.params, &mut par_body_ctx));
                let lower_bound_ptr = par_body_ctx.var_ids.next();
                let lower_bound = par_body_ctx.var_ids.next();
                let upper_bound_ptr = par_body_ctx.var_ids.next();
                let upper_bound = par_body_ctx.var_ids.next();
                par_body_ctx.code
                    .add(format!("{} = getelementptr %work_t* %cur.work, i32 0, i32 1",
                                 lower_bound_ptr));
                par_body_ctx.code.add(format!("{} = load i64* {}", lower_bound, lower_bound_ptr));
                par_body_ctx.code
                    .add(format!("{} = getelementptr %work_t* %cur.work, i32 0, i32 2",
                                 upper_bound_ptr));
                par_body_ctx.code.add(format!("{} = load i64* {}", upper_bound, upper_bound_ptr));
                let body_arg_types = try!(self.get_arg_str(&func.params, ""));
                par_body_ctx.code.add(format!("call void @f{}({}, i64 {}, i64 {})",
                                              func.id,
                                              body_arg_types,
                                              lower_bound,
                                              upper_bound));
                par_body_ctx.code.add("ret void");
                par_body_ctx.code.add("}\n\n");
                self.body_code.add(&par_body_ctx.code.result());
            }
            {
                let mut par_cont_ctx = &mut FunctionContext::new();
                par_cont_ctx.code
                    .add(format!("define void @f{}_par(%work_t* %cur.work) {{", par_for.cont));
                try!(self.unload_arg_struct(&sir.funcs[par_for.cont].params, &mut par_cont_ctx));
                let cont_arg_types = try!(self.get_arg_str(&sir.funcs[par_for.cont].params, ""));
                par_cont_ctx.code.add(format!("call void @f{}({})", par_for.cont, cont_arg_types));
                par_cont_ctx.code.add("ret void");
                par_cont_ctx.code.add("}\n\n");
                self.body_code.add(&par_cont_ctx.code.result());
            }
        }
        Ok(())
    }

    /// Add a function to the generated program, passing its parameters and return value through
    /// pointers encoded as i64. This is used for the main entry point function into Weld modules
    /// to pass them arbitrary structures.
    pub fn add_function_on_pointers(&mut self, name: &str, sir: &SirProgram) -> WeldResult<()> {
        // First add the function on raw values, which we'll call from the pointer version.
        try!(self.add_function(sir, &sir.funcs[0], None));

        // Define a struct with all the argument types as fields
        let args_struct = Struct(sir.top_params.iter().map(|a| a.ty.clone()).collect());
        let args_type = try!(self.llvm_type(&args_struct)).to_string();

        let mut code = &mut CodeBuilder::new();

        code.add(format!("define i64 @{}(i64 %input) {{", name));
        // Unpack the input, which is always struct defined by the type %input_arg_t in prelude.ll.
        code.add(format!("%inp_typed = inttoptr i64 %input to %input_arg_t*"));
        code.add(format!("%inp_val = load %input_arg_t* %inp_typed"));
        code.add(format!("%args = extractvalue %input_arg_t %inp_val, 0"));
        code.add(format!("%nworkers = extractvalue %input_arg_t %inp_val, 1"));
        code.add(format!("%rid = extractvalue %input_arg_t %inp_val, 2"));
        code.add(format!("call void @set_nworkers(i32 %nworkers)"));
        code.add(format!("call void @set_runid(i64 %rid)"));
        // Code to load args and call function
        code.add(format!("%args_typed = inttoptr i64 %args to {args_type}*
             \
                          %args_val = load {args_type}* %args_typed",
                         args_type = args_type));

        let mut arg_pos_map: HashMap<Symbol, usize> = HashMap::new();
        for (i, a) in sir.top_params.iter().enumerate() {
            arg_pos_map.insert(a.name.clone(), i);
        }
        let mut arg_decls = String::new();
        let params_sorted: BTreeMap<&Symbol, &Type> = sir.funcs[0].params.iter().collect();
        for (arg, ty) in params_sorted.iter() {
            let idx = arg_pos_map.get(arg).unwrap();
            code.add(format!("%arg{} = extractvalue {} %args_val, {}",
                             idx,
                             args_type,
                             idx));
            arg_decls.push_str(format!("{} %arg{}, ", try!(self.llvm_type(&ty)), idx).as_str());
        }
        arg_decls.push_str("%work_t* null");
        code.add(format!("call \
                          void @f0({arg_list})
             %res_ptr = call i8* @get_result()
             %res_address = ptrtoint i8* %res_ptr to i64
             \
                          ret i64 %res_address",
                         arg_list = arg_decls));
        code.add(format!("}}\n\n"));

        self.body_code.add_code(code);
        Ok(())
    }

    /// Return the LLVM type name corresponding to a Weld type.
    fn llvm_type(&mut self, ty: &Type) -> WeldResult<&str> {
        match *ty {
            Scalar(Bool) => Ok("i1"),
            Scalar(I8) => Ok("i8"),
            Scalar(I32) => Ok("i32"),
            Scalar(I64) => Ok("i64"),
            Scalar(F32) => Ok("float"),
            Scalar(F64) => Ok("double"),

            Struct(ref fields) => {
                if self.struct_names.get(fields) == None {
                    // Declare the struct in prelude_code
                    let name = self.struct_ids.next();
                    let mut field_types: Vec<String> = Vec::new();
                    for f in fields {
                        field_types.push(try!(self.llvm_type(f)).to_string());
                    }
                    let field_types_str = field_types.join(", ");
                    self.prelude_code.add(format!("{} = type {{ {} }}", name, field_types_str));

                    // Generate hash function for the struct.
                    self.prelude_code.add_line(format!("define i64 {}.hash({} %value) {{",
                                                       name.replace("%", "@"),
                                                       name));
                    let mut res = "0".to_string();
                    for i in 0..field_types.len() {
                        let field = self.prelude_var_ids.next();
                        let hash = self.prelude_var_ids.next();
                        let new_res = self.prelude_var_ids.next();
                        let field_ty_str = &field_types[i];
                        let field_prefix_str = format!("@{}", field_ty_str.replace("%", ""));
                        self.prelude_code
                            .add_line(format!("{} = extractvalue {} %value, {}", field, name, i));
                        self.prelude_code.add_line(format!("{} = call i64 {}.hash({} {})",
                                                           hash,
                                                           field_prefix_str,
                                                           field_ty_str,
                                                           field));
                        self.prelude_code
                            .add_line(format!("{} = call i64 @hash_combine(i64 {}, i64 {})",
                                              new_res,
                                              res,
                                              hash));
                        res = new_res;
                    }
                    self.prelude_code.add_line(format!("ret i64 {}", res));
                    self.prelude_code.add_line(format!("}}"));
                    self.prelude_code.add_line(format!(""));

                    self.prelude_code.add_line(format!("define i32 {}.cmp({} %a, {} %b) {{",
                                                       name.replace("%", "@"),
                                                       name,
                                                       name));
                    let mut label_ids = IdGenerator::new("%l");
                    for i in 0..field_types.len() {
                        let a_field = self.prelude_var_ids.next();
                        let b_field = self.prelude_var_ids.next();
                        let cmp = self.prelude_var_ids.next();
                        let ne = self.prelude_var_ids.next();
                        let field_ty_str = &field_types[i];
                        let ret_label = label_ids.next();
                        let post_label = label_ids.next();
                        let field_prefix_str = format!("@{}", field_ty_str.replace("%", ""));
                        self.prelude_code
                            .add_line(format!("{} = extractvalue {} %a , {}", a_field, name, i));
                        self.prelude_code
                            .add_line(format!("{} = extractvalue {} %b, {}", b_field, name, i));
                        self.prelude_code.add_line(format!("{} = call i32 {}.cmp({} {}, {} {})",
                                                           cmp,
                                                           field_prefix_str,
                                                           field_ty_str,
                                                           a_field,
                                                           field_ty_str,
                                                           b_field));
                        self.prelude_code.add_line(format!("{} = icmp ne i32 {}, 0", ne, cmp));
                        self.prelude_code.add_line(format!("br i1 {}, label {}, label {}",
                                                           ne,
                                                           ret_label,
                                                           post_label));
                        self.prelude_code.add_line(format!("{}:", ret_label.replace("%", "")));
                        self.prelude_code.add_line(format!("ret i32 {}", cmp));
                        self.prelude_code.add_line(format!("{}:", post_label.replace("%", "")));
                    }
                    self.prelude_code.add_line(format!("ret i32 0"));
                    self.prelude_code.add_line(format!("}}"));
                    self.prelude_code.add_line(format!(""));

                    // Add it into our map so we remember its name
                    self.struct_names.insert(fields.clone(), name);
                }
                Ok(self.struct_names.get(fields).unwrap())
            }

            Vector(ref elem) => {
                if self.vec_names.get(elem) == None {
                    let elem_ty = try!(self.llvm_type(elem)).to_string();
                    let elem_prefix = format!("@{}", elem_ty.replace("%", ""));
                    let name = self.vec_ids.next();
                    self.vec_names.insert(*elem.clone(), name.clone());
                    let prefix_replaced = VECTOR_CODE.replace("$ELEM_PREFIX", &elem_prefix);
                    let elem_replaced = prefix_replaced.replace("$ELEM", &elem_ty);
                    let name_replaced = elem_replaced.replace("$NAME", &name.replace("%", ""));
                    self.prelude_code.add(&name_replaced);
                    self.prelude_code.add("\n");
                }
                Ok(self.vec_names.get(elem).unwrap())
            }

            Dict(ref key, ref value) => {
                let elem = Box::new(Struct(vec![*key.clone(), *value.clone()]));
                if self.dict_names.get(&elem) == None {
                    let key_ty = try!(self.llvm_type(key)).to_string();
                    let value_ty = try!(self.llvm_type(value)).to_string();
                    let key_prefix = format!("@{}", key_ty.replace("%", ""));
                    let name = self.dict_ids.next();
                    self.dict_names.insert(*elem.clone(), name.clone());
                    let kv_struct_ty = try!(self.llvm_type(&elem)).to_string();
                    let kv_vec = Box::new(Vector(elem.clone()));
                    let kv_vec_ty = try!(self.llvm_type(&kv_vec)).to_string();
                    let kv_vec_prefix = format!("@{}", &kv_vec_ty.replace("%", ""));
                    let key_prefix_replaced = DICTIONARY_CODE.replace("$KEY_PREFIX", &key_prefix);
                    let name_replaced =
                        key_prefix_replaced.replace("$NAME", &name.replace("%", ""));
                    let key_ty_replaced = name_replaced.replace("$KEY", &key_ty);
                    let value_ty_replaced = key_ty_replaced.replace("$VALUE", &value_ty);
                    let kv_struct_replaced = value_ty_replaced.replace("$KV_STRUCT", &kv_struct_ty);
                    let kv_vec_prefix_replaced =
                        kv_struct_replaced.replace("$KV_VEC_PREFIX", &kv_vec_prefix);
                    let kv_vec_ty_replaced = kv_vec_prefix_replaced.replace("$KV_VEC", &kv_vec_ty);
                    self.prelude_code.add(&kv_vec_ty_replaced);
                    self.prelude_code.add("\n");
                }
                Ok(self.dict_names.get(&elem).unwrap())
            }

            Builder(ref bk) => {
                if self.bld_names.get(bk) == None {
                    match *bk {
                        Appender(ref t) => {
                            let bld_ty = Vector(t.clone());
                            let bld_ty_str = try!(self.llvm_type(&bld_ty)).to_string();
                            self.bld_names.insert(bk.clone(), format!("{}.bld", bld_ty_str));
                        }
                        Merger(ref t, ref op) => {
                            if self.merger_names.get(&(*t.clone(), *op)) == None {
                                let elem_ty = self.llvm_type(t)?.to_string();
                                let elem_prefix = format!("@{}", elem_ty.replace("%", ""));
                                let name = self.merger_ids.next();
                                self.merger_names
                                    .insert((*t.clone(), op.clone()), name.clone());
                                let prefix_replaced =
                                    MERGER_CODE.replace("$ELEM_PREFIX", &elem_prefix);
                                let elem_replaced = prefix_replaced.replace("$ELEM", &elem_ty);
                                let name_replaced =
                                    elem_replaced.replace("$NAME", &name.replace("%", ""));
                                let binop_replaced =
                                    name_replaced.replace("$OP", &llvm_binop(*op, t)?);
                                self.prelude_code.add(&binop_replaced);
                                self.prelude_code.add("\n");
                            }
                            let bld_ty_str = self.merger_names.get(&(*t.clone(), *op)).unwrap();
                            self.bld_names
                                .insert(bk.clone(), format!("{}.bld", bld_ty_str));
                        }
                        DictMerger(ref kt, ref vt, ref op) => {
                            let bld_ty = Dict(kt.clone(), vt.clone());
                            let bld_ty_str = try!(self.llvm_type(&bld_ty)).to_string();
                            let elem = Box::new(Struct(vec![*kt.clone(), *vt.clone()]));
                            let kv_struct_ty = try!(self.llvm_type(&elem)).to_string();
                            let key_ty = try!(self.llvm_type(kt)).to_string();
                            let value_ty = try!(self.llvm_type(vt)).to_string();
                            let name_replaced =
                                DICTMERGER_CODE.replace("$NAME", &bld_ty_str.replace("%", ""));
                            let key_ty_replaced = name_replaced.replace("$KEY", &key_ty);
                            let value_ty_replaced = key_ty_replaced.replace("$VALUE", &value_ty);
                            let kv_struct_replaced = value_ty_replaced.replace("$KV_STRUCT", &kv_struct_ty.replace("%", ""));
                            let op_replaced =
                                kv_struct_replaced.replace("$OP", &llvm_binop(*op, vt)?);
                            self.prelude_code.add(&op_replaced);
                            self.prelude_code.add("\n");
                            self.bld_names.insert(bk.clone(), format!("{}.bld", bld_ty_str));
                        }
                    }
                }
                Ok(self.bld_names.get(bk).unwrap())
            }

            _ => weld_err!("Unsupported type {}", print_type(ty))?,
        }
    }

    fn load_var(&mut self, sym: &str, ty: &str, ctx: &mut FunctionContext) -> WeldResult<String> {
        let var = ctx.var_ids.next();
        ctx.code.add(format!("{} = load {}* {}", var, ty, sym));
        Ok(var)
    }

    /// Add an expression to a CodeBuilder, possibly generating prelude code earlier, and return
    /// a string that can be used to represent its result later (e.g. %var if introducing a local
    /// variable or an integer constant otherwise).
    fn gen_func(&mut self,
                sir: &SirProgram,
                func: &SirFunction,
                ctx: &mut FunctionContext)
                -> WeldResult<String> {
        for b in func.blocks.iter() {
            ctx.code.add(format!("b.b{}:", b.id));
            for s in b.statements.iter() {
                match *s {
                    BinOp { ref output, op, ref ty, ref left, ref right } => {
                        let op_name = try!(llvm_binop(op, ty));
                        let ll_ty = try!(self.llvm_type(ty)).to_string();
                        let left_tmp = try!(self.load_var(llvm_symbol(left).as_str(), &ll_ty, ctx));
                        let right_tmp = try!(self.load_var(llvm_symbol(right).as_str(),
                            &ll_ty, ctx));
                        let bin_tmp = ctx.var_ids.next();
                        ctx.code.add(format!("{} = {} {} {}, {}",
                                             bin_tmp,
                                             op_name,
                                             ll_ty,
                                             left_tmp,
                                             right_tmp));
                        let out_ty = try!(get_sym_ty(func, output));
                        let out_ty_str = try!(self.llvm_type(&out_ty)).to_string();
                        ctx.code.add(format!("store {} {}, {}* {}",
                                             out_ty_str,
                                             bin_tmp,
                                             out_ty_str,
                                             llvm_symbol(output)));
                    }
                    Cast { ref output, ref old_ty, ref new_ty, ref child } => {
                        if old_ty != new_ty {
                            let op_name = try!(llvm_castop(&old_ty, &new_ty));
                            let old_ll_ty = try!(self.llvm_type(&old_ty)).to_string();
                            let new_ll_ty = try!(self.llvm_type(&new_ty)).to_string();
                            let child_tmp = try!(self.load_var(llvm_symbol(child).as_str(),
                                &old_ll_ty, ctx));
                            let cast_tmp = ctx.var_ids.next();
                            ctx.code.add(format!("{} = {} {} {} to {}",
                                                 cast_tmp,
                                                 op_name,
                                                 old_ll_ty,
                                                 child_tmp,
                                                 new_ll_ty));
                            let out_ty = try!(get_sym_ty(func, output));
                            let out_ty_str = try!(self.llvm_type(&out_ty)).to_string();
                            ctx.code.add(format!("store {} {}, {}* {}",
                                                 out_ty_str,
                                                 cast_tmp,
                                                 out_ty_str,
                                                 llvm_symbol(output)));
                        }
                    }
                    Lookup { ref output, ref child, ref index } => {
                        let child_ty = try!(get_sym_ty(func, child));
                        match *child_ty {
                            Vector(_) => {
                                let child_ll_ty = try!(self.llvm_type(&child_ty)).to_string();
                                let output_ty = try!(get_sym_ty(func, output));
                                let output_ll_ty = try!(self.llvm_type(&output_ty)).to_string();
                                let vec_ll_ty = try!(self.llvm_type(&child_ty)).to_string();
                                let vec_prefix = format!("@{}", vec_ll_ty.replace("%", ""));
                                let child_tmp = try!(self.load_var(llvm_symbol(child).as_str(),
                                    &child_ll_ty, ctx));
                                let index_tmp = try!(self.load_var(llvm_symbol(index).as_str(),
                                    "i64", ctx));
                                let res_ptr = ctx.var_ids.next();
                                let res_tmp = ctx.var_ids.next();
                                ctx.code.add(format!("{} = call {}* {}.at({} {}, i64 {})",
                                                     res_ptr,
                                                     output_ll_ty,
                                                     vec_prefix,
                                                     vec_ll_ty,
                                                     child_tmp,
                                                     index_tmp));
                                ctx.code.add(format!("{} = load {}* {}",
                                                     res_tmp,
                                                     output_ll_ty,
                                                     res_ptr));
                                ctx.code.add(format!("store {} {}, {}* {}",
                                                     output_ll_ty,
                                                     res_tmp,
                                                     output_ll_ty,
                                                     llvm_symbol(output)));
                            }
                            Dict(_, _) => {
                                let child_ll_ty = try!(self.llvm_type(&child_ty)).to_string();
                                let output_ty = try!(get_sym_ty(func, output));
                                let output_ll_ty = try!(self.llvm_type(&output_ty)).to_string();
                                let dict_ll_ty = try!(self.llvm_type(&child_ty)).to_string();
                                let index_ty = try!(get_sym_ty(func, index));
                                let index_ll_ty = try!(self.llvm_type(&index_ty)).to_string();
                                let dict_prefix = format!("@{}", dict_ll_ty.replace("%", ""));
                                let child_tmp = try!(self.load_var(llvm_symbol(child).as_str(),
                                    &child_ll_ty, ctx));
                                let index_tmp = try!(self.load_var(llvm_symbol(index).as_str(),
                                    &index_ll_ty, ctx));
                                let slot = ctx.var_ids.next();
                                let res_tmp = ctx.var_ids.next();
                                ctx.code.add(format!("{} = call {}.slot {}.lookup({} {}, {} {})",
                                                     slot,
                                                     dict_ll_ty,
                                                     dict_prefix,
                                                     dict_ll_ty,
                                                     child_tmp,
                                                     index_ll_ty,
                                                     index_tmp));
                                ctx.code.add(format!("{} = call {} {}.slot.value({}.slot {})",
                                                     res_tmp,
                                                     output_ll_ty,
                                                     dict_prefix,
                                                     dict_ll_ty,
                                                     slot));
                                ctx.code.add(format!("store {} {}, {}* {}",
                                                     output_ll_ty,
                                                     res_tmp,
                                                     output_ll_ty,
                                                     llvm_symbol(output)));
                            }
                            _ => weld_err!("Illegal type {} in Lookup", print_type(child_ty))?,
                        }
                    }
                    ToVec { ref output, ref old_ty, ref new_ty, ref child } => {
                        let old_ll_ty = try!(self.llvm_type(&old_ty)).to_string();
                        let new_ll_ty = try!(self.llvm_type(&new_ty)).to_string();
                        let dict_prefix = format!("@{}", old_ll_ty.replace("%", ""));
                        let child_tmp = try!(self.load_var(llvm_symbol(child).as_str(),
                            &old_ll_ty, ctx));
                        let res_tmp = ctx.var_ids.next();
                        ctx.code.add(format!("{} = call {} {}.tovec({} {})",
                                             res_tmp,
                                             new_ll_ty,
                                             dict_prefix,
                                             old_ll_ty,
                                             child_tmp));
                        let out_ty = try!(get_sym_ty(func, output));
                        let out_ty_str = try!(self.llvm_type(&out_ty)).to_string();
                        ctx.code.add(format!("store {} {}, {}* {}",
                                             out_ty_str,
                                             res_tmp,
                                             out_ty_str,
                                             llvm_symbol(output)));
                    }
                    Length { ref output, ref child } => {
                        let child_ty = try!(get_sym_ty(func, child));
                        let child_ll_ty = try!(self.llvm_type(&child_ty)).to_string();
                        let vec_prefix = format!("@{}", child_ll_ty.replace("%", ""));
                        let child_tmp = try!(self.load_var(llvm_symbol(child).as_str(),
                                                           &child_ll_ty, ctx));
                        let res_tmp = ctx.var_ids.next();
                        ctx.code.add(format!("{} = call i64 {}.size({} {})",
                                             res_tmp,
                                             vec_prefix,
                                             child_ll_ty,
                                             child_tmp));
                        let out_ty = try!(get_sym_ty(func, output));
                        let out_ty_str = try!(self.llvm_type(&out_ty)).to_string();
                        ctx.code.add(format!("store {} {}, {}* {}",
                                             out_ty_str,
                                             res_tmp,
                                             out_ty_str,
                                             llvm_symbol(output)));
                    }
                    Assign { ref output, ref value } => {
                        let ty = try!(get_sym_ty(func, output));
                        let ll_ty = try!(self.llvm_type(&ty)).to_string();
                        let val_tmp = try!(self.load_var(llvm_symbol(value).as_str(), &ll_ty, ctx));
                        ctx.code.add(format!("store {} {}, {}* {}",
                                             ll_ty,
                                             val_tmp,
                                             ll_ty,
                                             llvm_symbol(output)));
                    }
                    GetField { ref output, ref value, index } => {
                        let struct_ty = try!(self.llvm_type(try!(get_sym_ty(func, value))))
                            .to_string();
                        let field_ty = try!(self.llvm_type(try!(get_sym_ty(func, output))))
                            .to_string();
                        let struct_tmp = try!(self.load_var(llvm_symbol(value).as_str(),
                            &struct_ty, ctx));
                        let res_tmp = ctx.var_ids.next();
                        ctx.code.add(format!("{} = extractvalue {} {}, {}",
                                             res_tmp,
                                             struct_ty,
                                             struct_tmp,
                                             index));
                        ctx.code.add(format!("store {} {}, {}* {}",
                                             field_ty,
                                             res_tmp,
                                             field_ty,
                                             llvm_symbol(output)));
                    }
                    AssignLiteral { ref output, ref value } => {
                        match *value {
                            BoolLiteral(l) => {
                                ctx.code.add(format!("store i1 {}, i1* {}",
                                                     if l { 1 } else { 0 },
                                                     llvm_symbol(output)))
                            }
                            I8Literal(l) => {
                                ctx.code.add(format!("store i8 {}, i8* {}", l, llvm_symbol(output)))
                            }
                            I32Literal(l) => {
                                ctx.code
                                    .add(format!("store i32 {}, i32* {}", l, llvm_symbol(output)))
                            }
                            I64Literal(l) => {
                                ctx.code
                                    .add(format!("store i64 {}, i64* {}", l, llvm_symbol(output)))
                            }
                            F32Literal(l) => {
                                ctx.code.add(format!("store float {:.3}, float* {}",
                                                     l,
                                                     llvm_symbol(output)))
                            }
                            F64Literal(l) => {
                                ctx.code.add(format!("store double {:.3}, double* {}",
                                                     l,
                                                     llvm_symbol(output)))
                            }
                        }
                    }
                    Merge { ref builder, ref value } => {
                        let bld_ty = try!(get_sym_ty(func, builder));
                        match *bld_ty {
                            Builder(ref bk) => {
                                match *bk {
                                    Appender(ref t) => {
                                        let bld_ty_str = try!(self.llvm_type(&bld_ty)).to_string();
                                        let bld_prefix = format!("@{}",
                                                                 bld_ty_str.replace("%", ""));
                                        let bld_tmp =
                                            try!(self.load_var(llvm_symbol(builder).as_str(),
                                                               &bld_ty_str,
                                                               ctx));
                                        let elem_ty_str = try!(self.llvm_type(t)).to_string();
                                        let elem_tmp =
                                            try!(self.load_var(llvm_symbol(value).as_str(),
                                                               &elem_ty_str,
                                                               ctx));
                                        ctx.code.add(format!("call {} {}.merge({} {}, {} {})",
                                                             bld_ty_str,
                                                             bld_prefix,
                                                             bld_ty_str,
                                                             bld_tmp,
                                                             elem_ty_str,
                                                             elem_tmp));
                                    }
                                    DictMerger(ref kt, ref vt, _) => {
                                        let bld_ty_str = try!(self.llvm_type(&bld_ty)).to_string();
                                        let bld_prefix = format!("@{}",
                                                                 bld_ty_str.replace("%", ""));
                                        let bld_tmp =
                                            try!(self.load_var(llvm_symbol(builder).as_str(),
                                                               &bld_ty_str,
                                                               ctx));
                                        let elem_ty = Struct(vec![*kt.clone(), *vt.clone()]);
                                        let elem_ty_str = try!(self.llvm_type(&elem_ty))
                                            .to_string();
                                        let elem_tmp =
                                            try!(self.load_var(llvm_symbol(value).as_str(),
                                                               &elem_ty_str,
                                                               ctx));
                                        ctx.code.add(format!("call {} {}.merge({} {}, {} {})",
                                                             bld_ty_str,
                                                             bld_prefix,
                                                             bld_ty_str,
                                                             bld_tmp,
                                                             elem_ty_str,
                                                             elem_tmp));
                                    }
                                    Merger(ref t, _) => {
                                        let bld_ty_str = self.llvm_type(&bld_ty)?.to_string();
                                        let bld_prefix = format!("@{}",
                                                                 bld_ty_str.replace("%", ""));
                                        let elem_ty_str = self.llvm_type(t)?.to_string();
                                        let bld_tmp = self.load_var(llvm_symbol(builder).as_str(),
                                                      &bld_ty_str,
                                                      ctx)?;
                                        let elem_tmp = self.load_var(llvm_symbol(value).as_str(),
                                                      &elem_ty_str,
                                                      ctx)?;
                                        // TODO(shoumik) Template for Merger
                                        ctx.code.add(format!("call {} {}.merge({} {}, {} {})",
                                                             bld_ty_str,
                                                             bld_prefix,
                                                             bld_ty_str,
                                                             bld_tmp,
                                                             elem_ty_str,
                                                             elem_tmp));

                                    }
                                }
                            }
                            _ => {
                                weld_err!("Non builder type {} found in DoMerge",
                                          print_type(bld_ty))?
                            }
                        }
                    }
                    Res { ref output, ref builder } => {
                        let bld_ty = try!(get_sym_ty(func, builder));
                        let res_ty = try!(get_sym_ty(func, output));
                        match *bld_ty {
                            Builder(ref bk) => {
                                match *bk {
                                    Appender(_) => {
                                        let bld_ty_str = try!(self.llvm_type(&bld_ty)).to_string();
                                        let bld_prefix = format!("@{}",
                                                                 bld_ty_str.replace("%", ""));
                                        let res_ty_str = try!(self.llvm_type(&res_ty)).to_string();
                                        let bld_tmp =
                                            try!(self.load_var(llvm_symbol(builder).as_str(),
                                                               &bld_ty_str,
                                                               ctx));
                                        let res_tmp = ctx.var_ids.next();
                                        ctx.code.add(format!("{} = call {} {}.result({} {})",
                                                             res_tmp,
                                                             res_ty_str,
                                                             bld_prefix,
                                                             bld_ty_str,
                                                             bld_tmp));
                                        ctx.code.add(format!("store {} {}, {}* {}",
                                                             res_ty_str,
                                                             res_tmp,
                                                             res_ty_str,
                                                             llvm_symbol(output)));
                                    }
                                    Merger(_, _) => {
                                        let bld_ty_str = try!(self.llvm_type(&bld_ty)).to_string();
                                        let bld_prefix = format!("@{}",
                                                                 bld_ty_str.replace("%", ""));
                                        let res_ty_str = try!(self.llvm_type(&res_ty)).to_string();
                                        let bld_tmp =
                                            try!(self.load_var(llvm_symbol(builder).as_str(),
                                                               &bld_ty_str,
                                                               ctx));
                                        let res_tmp = ctx.var_ids.next();
                                        ctx.code.add(format!("{} = call {} {}.result({} {})",
                                                             res_tmp,
                                                             res_ty_str,
                                                             bld_prefix,
                                                             bld_ty_str,
                                                             bld_tmp));
                                        ctx.code.add(format!("store {} {}, {}* {}",
                                                             res_ty_str,
                                                             res_tmp,
                                                             res_ty_str,
                                                             llvm_symbol(output)));
                                    }
                                    DictMerger(_, _, _) => {
                                        let bld_ty_str = try!(self.llvm_type(&bld_ty)).to_string();
                                        let bld_prefix = format!("@{}",
                                                                 bld_ty_str.replace("%", ""));
                                        let res_ty_str = try!(self.llvm_type(&res_ty)).to_string();
                                        let bld_tmp =
                                            try!(self.load_var(llvm_symbol(builder).as_str(),
                                                               &bld_ty_str,
                                                               ctx));
                                        let res_tmp = ctx.var_ids.next();
                                        ctx.code.add(format!("{} = call {} {}.result({} {})",
                                                             res_tmp,
                                                             res_ty_str,
                                                             bld_prefix,
                                                             bld_ty_str,
                                                             bld_tmp));
                                        ctx.code.add(format!("store {} {}, {}* {}",
                                                             res_ty_str,
                                                             res_tmp,
                                                             res_ty_str,
                                                             llvm_symbol(output)));
                                    }
                                }
                            }
                            _ => {
                                weld_err!("Non builder type {} found in GetResult",
                                          print_type(bld_ty))?
                            }
                        }
                    }
                    NewBuilder { ref output, ref ty } => {
                        match *ty {
                            Builder(ref bk) => {
                                match *bk {
                                    Appender(_) => {
                                        let bld_ty_str = try!(self.llvm_type(ty));
                                        let bld_prefix = format!("@{}",
                                                                 bld_ty_str.replace("%", ""));
                                        let bld_tmp = ctx.var_ids.next();
                                        ctx.code.add(format!("{} = call {} {}.new(i64 16)",
                                                             bld_tmp,
                                                             bld_ty_str,
                                                             bld_prefix));
                                        ctx.code.add(format!("store {} {}, {}* {}",
                                                             bld_ty_str,
                                                             bld_tmp,
                                                             bld_ty_str,
                                                             llvm_symbol(output)));
                                    }
                                    Merger(_, _) => {
                                        let bld_ty_str = try!(self.llvm_type(ty));
                                        let bld_prefix = format!("@{}",
                                                                 bld_ty_str.replace("%", ""));
                                        let bld_tmp = ctx.var_ids.next();
                                        ctx.code.add(format!("{} = call {} {}.new()",
                                                             bld_tmp,
                                                             bld_ty_str,
                                                             bld_prefix));
                                        ctx.code.add(format!("store {} {}, {}* {}",
                                                             bld_ty_str,
                                                             bld_tmp,
                                                             bld_ty_str,
                                                             llvm_symbol(output)));
                                    }
                                    DictMerger(_, _, _) => {
                                        let bld_ty_str = try!(self.llvm_type(ty));
                                        let bld_prefix = format!("@{}",
                                                                 bld_ty_str.replace("%", ""));
                                        let bld_tmp = ctx.var_ids.next();
                                        ctx.code.add(format!("{} = call {} {}.new(i64 16)",
                                                             bld_tmp,
                                                             bld_ty_str,
                                                             bld_prefix));
                                        ctx.code.add(format!("store {} {}, {}* {}",
                                                             bld_ty_str,
                                                             bld_tmp,
                                                             bld_ty_str,
                                                             llvm_symbol(output)));
                                    }
                                }
                            }
                            _ => {
                                weld_err!("Non builder type {} found in CreateResult",
                                          print_type(ty))?
                            }
                        }
                    }
                }
            }
            match b.terminator {
                Branch { ref cond, on_true, on_false } => {
                    let cond_tmp = try!(self.load_var(llvm_symbol(cond).as_str(), "i1", ctx));
                    ctx.code.add(format!("br i1 {}, label %b.b{}, label %b.b{}",
                                         cond_tmp,
                                         on_true,
                                         on_false));
                }
                ParallelFor(ref pf) => {
                    try!(self.add_function(sir, &sir.funcs[pf.cont], None));
                    try!(self.add_function(sir, &sir.funcs[pf.body], Some(pf.clone())));
                    // TODO add parallel wrapper call
                    let params = get_combined_params(sir, pf);
                    let params_sorted: BTreeMap<&Symbol, &Type> = params.iter().collect();
                    let mut arg_types = String::new();
                    for (arg, ty) in params_sorted.iter() {
                        let ll_ty = try!(self.llvm_type(&ty)).to_string();
                        let arg_tmp = try!(self.load_var(llvm_symbol(arg).as_str(), &ll_ty, ctx));
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
                    try!(self.add_function(sir, &sir.funcs[func], None));
                    let params_sorted: BTreeMap<&Symbol, &Type> =
                        sir.funcs[func].params.iter().collect();
                    let mut arg_types = String::new();
                    for (arg, ty) in params_sorted.iter() {
                        let ll_ty = try!(self.llvm_type(&ty)).to_string();
                        let arg_tmp = try!(self.load_var(llvm_symbol(arg).as_str(), &ll_ty, ctx));
                        let arg_str = format!("{} {}, ", ll_ty, arg_tmp);
                        arg_types.push_str(&arg_str);
                    }
                    arg_types.push_str("%work_t* %cur.work");
                    ctx.code.add(format!("call void @f{}({})", func, arg_types));
                    ctx.code.add("br label %body.end");
                }
                ProgramReturn(ref sym) => {
                    let ty = try!(get_sym_ty(func, sym));
                    let ty_str = try!(self.llvm_type(ty)).to_string();
                    let res_tmp = try!(self.load_var(llvm_symbol(sym).as_str(), &ty_str, ctx));
                    let elem_size_ptr = ctx.var_ids.next();
                    let elem_size = ctx.var_ids.next();
                    let elem_storage = ctx.var_ids.next();
                    let elem_storage_typed = ctx.var_ids.next();
                    let run_id = ctx.var_ids.next();
                    ctx.code.add(format!("{} = getelementptr {}* null, i32 1",
                                         &elem_size_ptr,
                                         &ty_str));
                    ctx.code.add(format!("{} = ptrtoint {}* {} to i64",
                                         &elem_size,
                                         &ty_str,
                                         &elem_size_ptr));

                    ctx.code.add(format!("{} = call i64 @get_runid()", run_id));
                    ctx.code
                        .add(format!("{} = call i8* @weld_rt_malloc(i64 {}, i64 {})",
                                     &elem_storage,
                                     &run_id,
                                     &elem_size));
                    ctx.code.add(format!("{} = bitcast i8* {} to {}*",
                                         &elem_storage_typed,
                                         &elem_storage,
                                         &ty_str));
                    ctx.code.add(format!("store {} {}, {}* {}",
                                         &ty_str,
                                         res_tmp,
                                         &ty_str,
                                         &elem_storage_typed));
                    ctx.code.add(format!("call void @set_result(i8* {})", elem_storage));
                    ctx.code.add("br label %body.end");
                }
                EndFunction => {
                    ctx.code.add("br label %body.end");
                }
                Crash => {
                    let errno = WeldRuntimeErrno::Unknown as i64;
                    let run_id = ctx.var_ids.next();
                    ctx.code.add(format!("call void @weld_rt_set_errno(i64 {}, i64 {})",
                                         run_id,
                                         errno));
                }
            }
        }
        Ok(format!(""))
    }
}

/// Return the LLVM version of a Weld symbol (encoding any special characters for LLVM).
fn llvm_symbol(symbol: &Symbol) -> String {
    if symbol.id == 0 {
        format!("%{}", symbol.name)
    } else {
        format!("%{}.{}", symbol.name, symbol.id)
    }
}



/// Return the name of the LLVM instruction for a binary operation on a specific type.
fn llvm_binop(op_kind: BinOpKind, ty: &Type) -> WeldResult<&'static str> {
    match (op_kind, ty) {
        (BinOpKind::Add, &Scalar(I32)) => Ok("add"),
        (BinOpKind::Add, &Scalar(I64)) => Ok("add"),
        (BinOpKind::Add, &Scalar(F32)) => Ok("fadd"),
        (BinOpKind::Add, &Scalar(F64)) => Ok("fadd"),

        (BinOpKind::Subtract, &Scalar(I32)) => Ok("sub"),
        (BinOpKind::Subtract, &Scalar(I64)) => Ok("sub"),
        (BinOpKind::Subtract, &Scalar(F32)) => Ok("fsub"),
        (BinOpKind::Subtract, &Scalar(F64)) => Ok("fsub"),

        (BinOpKind::Multiply, &Scalar(I32)) => Ok("mul"),
        (BinOpKind::Multiply, &Scalar(I64)) => Ok("mul"),
        (BinOpKind::Multiply, &Scalar(F32)) => Ok("fmul"),
        (BinOpKind::Multiply, &Scalar(F64)) => Ok("fmul"),

        (BinOpKind::Divide, &Scalar(I32)) => Ok("sdiv"),
        (BinOpKind::Divide, &Scalar(I64)) => Ok("sdiv"),
        (BinOpKind::Divide, &Scalar(F32)) => Ok("fdiv"),
        (BinOpKind::Divide, &Scalar(F64)) => Ok("fdiv"),

        (BinOpKind::Equal, &Scalar(Bool)) => Ok("icmp eq"),
        (BinOpKind::Equal, &Scalar(I32)) => Ok("icmp eq"),
        (BinOpKind::Equal, &Scalar(I64)) => Ok("icmp eq"),
        (BinOpKind::Equal, &Scalar(F32)) => Ok("fcmp oeq"),
        (BinOpKind::Equal, &Scalar(F64)) => Ok("fcmp oeq"),

        (BinOpKind::NotEqual, &Scalar(Bool)) => Ok("icmp ne"),
        (BinOpKind::NotEqual, &Scalar(I32)) => Ok("icmp ne"),
        (BinOpKind::NotEqual, &Scalar(I64)) => Ok("icmp ne"),
        (BinOpKind::NotEqual, &Scalar(F32)) => Ok("fcmp one"),
        (BinOpKind::NotEqual, &Scalar(F64)) => Ok("fcmp one"),

        (BinOpKind::LessThan, &Scalar(I32)) => Ok("icmp slt"),
        (BinOpKind::LessThan, &Scalar(I64)) => Ok("icmp slt"),
        (BinOpKind::LessThan, &Scalar(F32)) => Ok("fcmp olt"),
        (BinOpKind::LessThan, &Scalar(F64)) => Ok("fcmp olt"),

        (BinOpKind::LessThanOrEqual, &Scalar(I32)) => Ok("icmp sle"),
        (BinOpKind::LessThanOrEqual, &Scalar(I64)) => Ok("icmp sle"),
        (BinOpKind::LessThanOrEqual, &Scalar(F32)) => Ok("fcmp ole"),
        (BinOpKind::LessThanOrEqual, &Scalar(F64)) => Ok("fcmp ole"),

        (BinOpKind::GreaterThan, &Scalar(I32)) => Ok("icmp sgt"),
        (BinOpKind::GreaterThan, &Scalar(I64)) => Ok("icmp sgt"),
        (BinOpKind::GreaterThan, &Scalar(F32)) => Ok("fcmp ogt"),
        (BinOpKind::GreaterThan, &Scalar(F64)) => Ok("fcmp ogt"),

        (BinOpKind::GreaterThanOrEqual, &Scalar(I32)) => Ok("icmp sge"),
        (BinOpKind::GreaterThanOrEqual, &Scalar(I64)) => Ok("icmp sge"),
        (BinOpKind::GreaterThanOrEqual, &Scalar(F32)) => Ok("fcmp oge"),
        (BinOpKind::GreaterThanOrEqual, &Scalar(F64)) => Ok("fcmp oge"),

        (BinOpKind::LogicalAnd, &Scalar(Bool)) => Ok("and"),
        (BinOpKind::LogicalOr, &Scalar(Bool)) => Ok("or"),

        _ => weld_err!("Unsupported binary op: {} on {}", op_kind, print_type(ty)),
    }
}

/// Return the name of hte LLVM instruction for a cast operation between specific types.
fn llvm_castop(ty1: &Type, ty2: &Type) -> WeldResult<&'static str> {
    match (ty1, ty2) {
        (&Scalar(F64), &Scalar(Bool)) => Ok("fptoui"),
        (&Scalar(F32), &Scalar(Bool)) => Ok("fptoui"),
        (&Scalar(Bool), &Scalar(F64)) => Ok("uitofp"),
        (&Scalar(Bool), &Scalar(F32)) => Ok("uitofp"),
        (&Scalar(F64), &Scalar(F32)) => Ok("fptrunc"),
        (&Scalar(F32), &Scalar(F64)) => Ok("fpext"),
        (&Scalar(F64), _) => Ok("fptosi"),
        (&Scalar(F32), _) => Ok("fptosi"),
        (_, &Scalar(F64)) => Ok("sitofp"),
        (_, &Scalar(F32)) => Ok("sitofp"),
        (&Scalar(Bool), _) => Ok("zext"),
        (_, &Scalar(I64)) => Ok("sext"),
        _ => Ok("trunc"),
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
}

impl FunctionContext {
    fn new() -> FunctionContext {
        FunctionContext {
            alloca_code: CodeBuilder::new(),
            code: CodeBuilder::new(),
            var_ids: IdGenerator::new("%t.t"),
            defined_symbols: HashSet::new(),
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

/// Generate a compiled LLVM module from a program whose body is a function.
pub fn compile_program(program: &Program) -> WeldResult<easy_ll::CompiledModule> {
    let mut expr = try!(macro_processor::process_program(program));
    transforms::uniquify(&mut expr);
    try!(type_inference::infer_types(&mut expr));
    let mut expr = try!(expr.to_typed());
    transforms::inline_apply(&mut expr);
    transforms::inline_let(&mut expr);
    transforms::inline_zips(&mut expr);
    transforms::fuse_loops_horizontal(&mut expr);
    transforms::fuse_loops_vertical(&mut expr);
    let sir_prog = try!(sir::ast_to_sir(&expr));
    let mut gen = LlvmGenerator::new();
    try!(gen.add_function_on_pointers("run", &sir_prog));
    Ok(try!(easy_ll::compile_module(&gen.result())))
}

#[test]
fn types() {
    let mut gen = LlvmGenerator::new();

    assert_eq!(gen.llvm_type(&Scalar(I32)).unwrap(), "i32");
    assert_eq!(gen.llvm_type(&Scalar(I64)).unwrap(), "i64");
    assert_eq!(gen.llvm_type(&Scalar(F32)).unwrap(), "float");
    assert_eq!(gen.llvm_type(&Scalar(F64)).unwrap(), "double");
    assert_eq!(gen.llvm_type(&Scalar(I8)).unwrap(), "i8");
    assert_eq!(gen.llvm_type(&Scalar(Bool)).unwrap(), "i1");

    let struct1 = parse_type("{i32,bool,i32}").unwrap().to_type().unwrap();
    assert_eq!(gen.llvm_type(&struct1).unwrap(), "%s0");
    assert_eq!(gen.llvm_type(&struct1).unwrap(), "%s0"); // Name is reused for same struct

    let struct2 = parse_type("{i32,bool}").unwrap().to_type().unwrap();
    assert_eq!(gen.llvm_type(&struct2).unwrap(), "%s1");
}

#[test]
fn runtime_functions() {
    weld_rt_free(0, weld_rt_realloc(0, weld_rt_malloc(0, 16), 32));
    weld_rt_set_errno(-1, WeldRuntimeErrno::Success);
    weld_rt_get_errno(-1);
}

// #[test]
// fn basic_program() {
// let code = "|| 40 + 2";
//
// let module = compile_program(&parse_program(code).unwrap()).unwrap();
// let inp = Box::new(WeldInputArgs {
// input: 0,
// nworkers: 1,
// run_id: 0,
// });
// let ptr = Box::into_raw(inp) as i64;
//
// let result = module.run(ptr) as *const i32;
// let result = unsafe { *result };
// assert_eq!(result, 42);
// weld_run_free(-1);
// }
//
// #[test]
// fn f64_cast() {
// let code = "|| f64(40 + 2)";
// let module = compile_program(&parse_program(code).unwrap()).unwrap();
//
// let inp = Box::new(WeldInputArgs {
// input: 0,
// nworkers: 1,
// run_id: 0,
// });
// let ptr = Box::into_raw(inp) as i64;
//
// let result = module.run(ptr) as *const f64;
// let result = unsafe { *result };
// assert_eq!(result, 42.0);
// weld_run_free(-1);
// }
//
// #[test]
// fn i32_cast() {
// let code = "|| i32(0.251 * 4.0)";
// let module = compile_program(&parse_program(code).unwrap()).unwrap();
// let inp = Box::new(WeldInputArgs {
// input: 0,
// nworkers: 1,
// run_id: 0,
// });
// let ptr = Box::into_raw(inp) as i64;
//
// let result = module.run(ptr) as *const i32;
// let result = unsafe { *result };
// assert_eq!(result, 1);
// weld_run_free(-1);
// }
//
// #[test]
// fn program_with_args() {
// let code = "|x:i32| 40 + x";
// let module = compile_program(&parse_program(code).unwrap()).unwrap();
// let input: i32 = 2;
// let inp = Box::new(WeldInputArgs {
// input: &input as *const i32 as i64,
// nworkers: 1,
// run_id: 0,
// });
// let ptr = Box::into_raw(inp) as i64;
//
// let result = module.run(ptr) as *const i32;
// let result = unsafe { *result };
// assert_eq!(result, 42);
// weld_run_free(-1);
// }
//
// #[test]
// fn let_statement() {
// let code = "|x:i32| let y = 40 + x; y + 2";
// let module = compile_program(&parse_program(code).unwrap()).unwrap();
// let input: i32 = 2;
// let inp = Box::new(WeldInputArgs {
// input: &input as *const i32 as i64,
// nworkers: 1,
// run_id: 0,
// });
// let ptr = Box::into_raw(inp) as i64;
//
// let result = module.run(ptr) as *const i32;
// let result = unsafe { *result };
// assert_eq!(result, 44);
// weld_run_free(-1);
// }
//
// #[test]
// fn if_statement() {
// let code = "|x:i32| if(true, 3, 4)";
// let module = compile_program(&parse_program(code).unwrap()).unwrap();
// let input: i32 = 2;
// let inp = Box::new(WeldInputArgs {
// input: &input as *const i32 as i64,
// nworkers: 1,
// run_id: 0,
// });
//
// let ptr = Box::into_raw(inp) as i64;
// let result = module.run(ptr) as *const i32;
// let result = unsafe { *result };
// assert_eq!(result, 3);
// weld_run_free(-1);
// }
//
// #[test]
// fn comparison() {
// let code = "|x:i32| if(x>10, x, 10)";
// let module = compile_program(&parse_program(code).unwrap()).unwrap();
// let mut input: i32 = 2;
// let inp = Box::new(WeldInputArgs {
// input: &input as *const i32 as i64,
// nworkers: 1,
// run_id: 0,
// });
// let ptr = Box::into_raw(inp) as i64;
//
// let result = module.run(ptr) as *const i32;
// let result = unsafe { *result };
// assert_eq!(result, 10);
// weld_run_free(-1);
//
// input = 20;
// let result = module.run(ptr) as *const i32;
// let result = unsafe { *result };
// assert_eq!(result, 20);
// weld_run_free(-1);
// }
//
// #[test]
// fn simple_vector_lookup() {
// #[derive(Clone)]
// #[allow(dead_code)]
// struct Vec {
// data: *const i32,
// len: i64,
// }
// #[allow(dead_code)]
// struct Args {
// x: Vec,
// }
//
// let code = "|x:vec[i32]| lookup(x, 3L)";
// let module = compile_program(&parse_program(code).unwrap()).unwrap();
// let input = [1, 2, 4, 5];
// let args = Args {
// x: Vec {
// data: &input as *const i32,
// len: 4,
// },
// };
//
// let inp = Box::new(WeldInputArgs {
// input: &args as *const Args as i64,
// nworkers: 1,
// run_id: 0,
// });
// let ptr = Box::into_raw(inp) as i64;
//
// let result_raw = module.run(ptr) as *const i32;
// let result = unsafe { (*result_raw).clone() };
// let output = input[3];
// assert_eq!(output, result);
// weld_run_free(-1);
// }
//
// #[test]
// fn simple_for_appender_loop() {
// #[derive(Clone)]
// #[allow(dead_code)]
// struct Vec {
// data: *const i32,
// len: i64,
// }
// #[allow(dead_code)]
// struct Args {
// x: Vec,
// a: i32,
// }
//
// let code = "|x:vec[i32], a:i32| let b=a+1; map(x, |e| e+b)";
// let module = compile_program(&parse_program(code).unwrap()).unwrap();
// let input = [1, 2];
// let args = Args {
// x: Vec {
// data: &input as *const i32,
// len: 2,
// },
// a: 1,
// };
//
// let inp = Box::new(WeldInputArgs {
// input: &args as *const Args as i64,
// nworkers: 1,
// run_id: 0,
// });
// let ptr = Box::into_raw(inp) as i64;
//
// let result_raw = module.run(ptr) as *const Vec;
// let result = unsafe { (*result_raw).clone() };
// let output = [3, 4];
// for i in 0..(result.len as isize) {
// assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
// }
// weld_run_free(-1);
// }
//
// #[test]
// fn simple_for_merger_loop() {
// #[derive(Clone)]
// #[allow(dead_code)]
// struct Vec {
// data: *const i32,
// len: i64,
// }
// #[allow(dead_code)]
// struct Args {
// x: Vec,
// a: i32,
// }
//
// let code = "|x:vec[i32], a:i32| result(for(x, merger[i32,+], |b,i,e| merge(b, e+a)))";
// let module = compile_program(&parse_program(code).unwrap()).unwrap();
// let input = [1, 2, 3, 4, 5];
// let args = Args {
// x: Vec {
// data: &input as *const i32,
// len: 5,
// },
// a: 1,
// };
//
// let inp = Box::new(WeldInputArgs {
// input: &args as *const Args as i64,
// nworkers: 1,
// run_id: 0,
// });
// let ptr = Box::into_raw(inp) as i64;
//
// let result_raw = module.run(ptr) as *const i32;
// let result = unsafe { (*result_raw).clone() };
// let output = 20;
// assert_eq!(result, output);
// weld_run_free(-1);
// }
//
// #[test]
// fn simple_for_dictmerger_loop() {
// #[derive(Clone)]
// #[allow(dead_code)]
// struct Vec {
// data: *const i32,
// len: i64,
// }
// #[allow(dead_code)]
// struct Pair {
// ele1: i32,
// ele2: i32,
// }
// #[derive(Clone)]
// #[allow(dead_code)]
// struct VecPrime {
// data: *const Pair,
// len: i64,
// }
// #[allow(dead_code)]
// struct Args {
// x: Vec,
// y: Vec,
// }
//
// let code = "|x:vec[i32], y:vec[i32]| tovec(result(for(zip(x,y), dictmerger[i32,i32,+], \
// |b,i,e| merge(b, e))))";
// let module = compile_program(&parse_program(code).unwrap()).unwrap();
// let keys = [1, 2, 2, 1, 3];
// let values = [2, 3, 4, 2, 1];
// let args = Args {
// x: Vec {
// data: &keys as *const i32,
// len: 5,
// },
// y: Vec {
// data: &values as *const i32,
// len: 5,
// },
// };
//
// let inp = Box::new(WeldInputArgs {
// input: &args as *const Args as i64,
// nworkers: 1,
// run_id: 0,
// });
// let ptr = Box::into_raw(inp) as i64;
//
// let result_raw = module.run(ptr) as *const VecPrime;
// let result = unsafe { (*result_raw).clone() };
// let output_keys = [1, 2, 3];
// let output_values = [4, 7, 1];
// for i in 0..(output_keys.len() as isize) {
// let mut success = false;
// let key = unsafe { (*result.data.offset(i)).ele1 };
// let value = unsafe { (*result.data.offset(i)).ele2 };
// for j in 0..(output_keys.len()) {
// if output_keys[j] == key {
// if output_values[j] == value {
// success = true;
// }
// }
// }
// assert_eq!(success, true);
// }
// assert_eq!(result.len, output_keys.len() as i64);
// weld_run_free(-1);
// }
//
// #[test]
// fn simple_dict_lookup() {
// #[derive(Clone)]
// #[allow(dead_code)]
// struct Vec {
// data: *const i32,
// len: i64,
// }
// #[allow(dead_code)]
// struct Args {
// x: Vec,
// y: Vec,
// }
//
// let code = "|x:vec[i32], y:vec[i32]| let a = result(for(zip(x,y), dictmerger[i32,i32,+], \
// |b,i,e| merge(b, e))); lookup(a, 1)";
// let module = compile_program(&parse_program(code).unwrap()).unwrap();
// let keys = [1, 2, 2, 1, 3];
// let values = [2, 3, 4, 2, 1];
// let args = Args {
// x: Vec {
// data: &keys as *const i32,
// len: 5,
// },
// y: Vec {
// data: &values as *const i32,
// len: 5,
// },
// };
//
// let inp = Box::new(WeldInputArgs {
// input: &args as *const Args as i64,
// nworkers: 1,
// run_id: 0,
// });
// let ptr = Box::into_raw(inp) as i64;
//
// let result_raw = module.run(ptr) as *const i32;
// let result = unsafe { (*result_raw).clone() };
// let output = 4;
// assert_eq!(output, result);
// weld_run_free(-1);
// }
//
// #[test]
// fn simple_length() {
// #[derive(Clone)]
// #[allow(dead_code)]
// struct Vec {
// data: *const i32,
// len: i64,
// }
// #[allow(dead_code)]
// struct Args {
// x: Vec,
// }
//
// let code = "|x:vec[i32]| len(x)";
// let module = compile_program(&parse_program(code).unwrap()).unwrap();
// let data = [2, 3, 4, 2, 1];
// let args = Args {
// x: Vec {
// data: &data as *const i32,
// len: 5,
// },
// };
//
// let inp = Box::new(WeldInputArgs {
// input: &args as *const Args as i64,
// nworkers: 1,
// run_id: 0,
// });
// let ptr = Box::into_raw(inp) as i64;
//
// let result_raw = module.run(ptr) as *const i32;
// let result = unsafe { (*result_raw).clone() };
// let output = 5;
// assert_eq!(output, result);
// weld_run_free(-1);
// }
//
// #[test]
// fn filter_length() {
// #[derive(Clone)]
// #[allow(dead_code)]
// struct Vec {
// data: *const i32,
// len: i64,
// }
// #[allow(dead_code)]
// struct Args {
// x: Vec,
// }
//
// let code = "|x:vec[i32]| len(filter(x, |i| i < 4 && i > 1))";
// let module = compile_program(&parse_program(code).unwrap()).unwrap();
// let data = [2, 3, 4, 2, 1];
// let args = Args {
// x: Vec {
// data: &data as *const i32,
// len: 5,
// },
// };
//
// let inp = Box::new(WeldInputArgs {
// input: &args as *const Args as i64,
// nworkers: 1,
// run_id: 0,
// });
// let ptr = Box::into_raw(inp) as i64;
//
// let result_raw = module.run(ptr) as *const i32;
// let result = unsafe { (*result_raw).clone() };
// let output = 3;
// assert_eq!(output, result);
// weld_run_free(-1);
// }
//
// #[test]
// fn flat_map_length() {
// #[derive(Clone)]
// #[allow(dead_code)]
// struct Vec {
// data: *const i32,
// len: i64,
// }
// #[allow(dead_code)]
// struct Args {
// x: Vec,
// }
//
// let code = "|x:vec[i32]| len(flatten(map(x, |i:i32| x)))";
// let module = compile_program(&parse_program(code).unwrap()).unwrap();
//
// let data = [2, 3, 4, 2, 1];
// let args = Args {
// x: Vec {
// data: &data as *const i32,
// len: 5,
// },
// };
//
// let inp = Box::new(WeldInputArgs {
// input: &args as *const Args as i64,
// nworkers: 1,
// run_id: 0,
// });
// let ptr = Box::into_raw(inp) as i64;
//
// let result_raw = module.run(ptr) as *const i32;
// let result = unsafe { (*result_raw).clone() };
// let output = 25;
// assert_eq!(output, result);
// weld_run_free(-1);
// }
//
// #[test]
// fn if_for_loop() {
// #[derive(Clone)]
// #[allow(dead_code)]
// struct Vec {
// data: *const i32,
// len: i64,
// }
// #[allow(dead_code)]
// struct Args {
// x: Vec,
// a: i32,
// }
//
// let code = "|x:vec[i32], a:i32| if(a > 5, map(x, |e| e+1), map(x, |e| e+2))";
// let module = compile_program(&parse_program(code).unwrap()).unwrap();
// let input = [1, 2];
//
// let args = Args {
// x: Vec {
// data: &input as *const i32,
// len: 2,
// },
// a: 1,
// };
//
// let inp = Box::new(WeldInputArgs {
// input: &args as *const Args as i64,
// nworkers: 1,
// run_id: 0,
// });
// let ptr = Box::into_raw(inp) as i64;
//
// let result_raw = module.run(ptr) as *const Vec;
// let result = unsafe { (*result_raw).clone() };
// let output = [3, 4];
// for i in 0..(result.len as isize) {
// assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
// }
// weld_run_free(-1);
//
// let args = Args {
// x: Vec {
// data: &input as *const i32,
// len: 2,
// },
// a: 6,
// };
// let inp = Box::new(WeldInputArgs {
// input: &args as *const Args as i64,
// nworkers: 1,
// run_id: 0,
// });
// let ptr = Box::into_raw(inp) as i64;
//
// let result_raw = module.run(ptr) as *const Vec;
// let result = unsafe { (*result_raw).clone() };
// let output = [2, 3];
// for i in 0..(result.len as isize) {
// assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
// }
// weld_run_free(-1);
// }
//
// #[test]
// fn map_zip_loop() {
// #[derive(Clone)]
// #[allow(dead_code)]
// struct Vec {
// data: *const i32,
// len: i64,
// }
// #[allow(dead_code)]
// struct Args {
// x: Vec,
// y: Vec,
// }
//
// let code = "|x:vec[i32], y:vec[i32]| map(zip(x,y), |e| e.$0 + e.$1)";
// let module = compile_program(&parse_program(code).unwrap()).unwrap();
// let x = [1, 2, 3, 4];
// let y = [5, 6, 7, 8];
// let args = Args {
// x: Vec {
// data: &x as *const i32,
// len: 4,
// },
// y: Vec {
// data: &y as *const i32,
// len: 2,
// },
// };
//
// let inp = Box::new(WeldInputArgs {
// input: &args as *const Args as i64,
// nworkers: 1,
// run_id: 0,
// });
// let ptr = Box::into_raw(inp) as i64;
//
// let result_raw = module.run(ptr) as *const Vec;
// let result = unsafe { (*result_raw).clone() };
// let output = [6, 8, 10, 12];
// for i in 0..(result.len as isize) {
// assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
// }
// weld_run_free(-1);
// }
//
// #[test]
// fn iters_for_loop() {
// #[derive(Clone)]
// #[allow(dead_code)]
// struct Vec {
// data: *const i32,
// len: i64,
// }
// #[allow(dead_code)]
// struct Args {
// x: Vec,
// y: Vec,
// }
//
// let code = "|x:vec[i32], y:vec[i32]| result(for(zip(iter(x,0L,4L,2L), y), appender, |b,i,e| \
// merge(b,e.$0+e.$1)))";
// let module = compile_program(&parse_program(code).unwrap()).unwrap();
// let x = [1, 2, 3, 4];
// let y = [5, 6];
// let args = Args {
// x: Vec {
// data: &x as *const i32,
// len: 4,
// },
// y: Vec {
// data: &y as *const i32,
// len: 2,
// },
// };
//
// let inp = Box::new(WeldInputArgs {
// input: &args as *const Args as i64,
// nworkers: 1,
// run_id: 0,
// });
// let ptr = Box::into_raw(inp) as i64;
//
// let result_raw = module.run(ptr) as *const Vec;
// let result = unsafe { (*result_raw).clone() };
// let output = [6, 9];
// for i in 0..(result.len as isize) {
// assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
// }
// weld_run_free(-1);
// }
//
// #[test]
// fn serial_parlib_test() {
// #[derive(Clone)]
// #[allow(dead_code)]
// struct WeldVec {
// data: *const i32,
// len: i64,
// }
// let code = "|x:vec[i32]| result(for(x, merger[i32,+], |b,i,e| merge(b, e)))";
// let module = compile_program(&parse_program(code).unwrap()).unwrap();
// let size: i32 = 10000;
// let input: Vec<i32> = vec![1; size as usize];
// let args = WeldVec {
// data: input.as_ptr() as *const i32,
// len: size as i64,
// };
//
// let inp = Box::new(WeldInputArgs {
// input: &args as *const WeldVec as i64,
// nworkers: 1,
// run_id: 0,
// });
// let ptr = Box::into_raw(inp) as i64;
//
// let result_raw = module.run(ptr) as *const i32;
// let result = unsafe { (*result_raw).clone() };
// assert_eq!(result, size);
// weld_run_free(-1);
// }
//

#[test]
fn iters_outofbounds_error_test() {
    use std::ptr;
    use super::WeldRuntimeErrno;

    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }
    #[allow(dead_code)]
    struct Args {
        x: Vec,
    }

    let code = "|x:vec[i32]| result(for(iter(x,0L,20000L,1L), appender, |b,i,e| \
                merge(b,e+1)))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let x = [4; 10000 as usize];
    let args = Args {
        x: Vec {
            data: &x as *const i32,
            len: 10000,
        },
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const Args as i64,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const Vec;

    // Get the error back for the run ID we used.
    let errno = weld_rt_get_errno(0);
    assert_eq!(errno, WeldRuntimeErrno::BadIteratorLength);
    assert_eq!(result_raw, ptr::null_mut());
    weld_run_free(-1);
}
