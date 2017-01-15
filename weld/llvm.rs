use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::BTreeMap;

use easy_ll;

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
use super::type_inference;
use super::util::IdGenerator;

#[cfg(test)] use super::parser::*;

static PRELUDE_CODE: &'static str = include_str!("resources/prelude.ll");
static VECTOR_CODE: &'static str = include_str!("resources/vector.ll");

/// Generates LLVM code for one or more modules.
pub struct LlvmGenerator {
    /// Track a unique name of the form %s0, %s1, etc for each struct generated.
    struct_names: HashMap<Vec<Type>, String>,
    struct_ids: IdGenerator,

    /// Track a unique name of the form %v0, %v1, etc for each vec generated.
    vec_names: HashMap<Type, String>,
    vec_ids: IdGenerator,

    /// TODO This is unnecessary but satisfies the compiler for now.
    bld_names: HashMap<BuilderKind, String>,

    /// A CodeBuilder for prelude functions such as type and struct definitions.
    prelude_code: CodeBuilder,

    /// A CodeBuilder for body functions in the module.
    body_code: CodeBuilder,
    visited: HashSet<sir::FunctionId>
}

fn get_combined_params(sir: &SirProgram, par_for: &ParallelForData)
    -> HashMap<Symbol, Type> {
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
            bld_names: HashMap::new(),
            prelude_code: CodeBuilder::new(),
            body_code: CodeBuilder::new(),
            visited: HashSet::new()
        };
        generator.prelude_code.add(PRELUDE_CODE);
        generator.prelude_code.add("\n");
        generator
    }

    /// Return all the code generated so far.
    pub fn result(&mut self) -> String {
        format!("; PRELUDE:\n\n{}\n; BODY:\n\n{}", self.prelude_code.result(), self.body_code.result())
    }

    fn get_arg_str(&mut self, params: &HashMap<Symbol, Type>, suffix: &str) -> WeldResult<String> {
        let mut arg_types = String::new();
        let params_sorted: BTreeMap<&Symbol, &Type> = params.iter().collect();
        for (arg, ty) in params_sorted.iter() {
            let arg_str = format!("{} {}{}, ", try!(self.llvm_type(&ty)), llvm_symbol(&arg), suffix);
            arg_types.push_str(&arg_str);
        }
        arg_types.push_str("%work_t* %cur.work");
        Ok(arg_types)
    }

    /// Add a function to the generated program.
    pub fn add_function(
        &mut self,
        sir: &SirProgram,
        func: &SirFunction,
        // only non-None if func is a ParallelFor body
        containing_loop: Option<ParallelForData>
    ) -> WeldResult<()> {
        if !self.visited.insert(func.id) {
            return Ok(());
        }
        let mut ctx = &mut FunctionContext::new();
        let mut arg_types = try!(self.get_arg_str(&func.params, ".in"));
        if containing_loop.is_some() {
            arg_types.push_str(", i64 %lower.idx, i64 %upper.idx");
        }

        // Start the entry block by defining the function and storing all its arguments on the
        // stack (this makes them consistent with other local variables). Later, expressions may
        // add more local variables to alloca_code.
        ctx.alloca_code.add(format!("define void @f{}({}) {{", func.id, arg_types));
        ctx.alloca_code.add(format!("entry:"));
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
            let bld_ty_str = try!(self.llvm_type(func.params.get(&par_for.builder).unwrap())).to_string();
            let bld_param_str = llvm_symbol(&par_for.builder);
            let bld_arg_str = llvm_symbol(&par_for.builder_arg);
            ctx.code.add(format!("store {} {}.in, {}* {}", &bld_ty_str, bld_param_str, &bld_ty_str,
                bld_arg_str));
            try!(ctx.add_alloca("%cur.idx", "i64"));
            ctx.code.add("store i64 %lower.idx, i64* %cur.idx");
            ctx.code.add("br label %loop_start");
            ctx.code.add("loop_start:");
            let idx_tmp = try!(self.load_var("%cur.idx", "i64", ctx));
            let idx_cmp = ctx.var_ids.next();
            ctx.code.add(format!("{} = icmp ult i64 {}, %upper.idx", idx_cmp, idx_tmp));
            ctx.code.add(format!("br i1 {}, label %loop_body, label %loop_end", idx_cmp));
            ctx.code.add("loop_body:");
            // TODO support loop data that is not a single vector
            let data_ty_str = try!(self.llvm_type(func.params.get(&par_for.data).unwrap())).to_string();
            let data_str = try!(self.load_var(llvm_symbol(&par_for.data).as_str(), &data_ty_str, ctx));
            let data_prefix = format!("@{}", data_ty_str.replace("%", ""));
            let elem_str = llvm_symbol(&par_for.data_arg);
            let elem_ty_str = try!(self.llvm_type(func.locals.get(&par_for.data_arg).unwrap())).to_string();
            let elem_tmp_ptr = ctx.var_ids.next();
            ctx.code.add(format!("{} = call {}* {}.at({} {}, i64 {})", elem_tmp_ptr, &elem_ty_str,
                data_prefix, &data_ty_str, data_str, idx_tmp));
            let elem_tmp = try!(self.load_var(&elem_tmp_ptr, &elem_ty_str, ctx));
            ctx.code.add(format!("store {} {}, {}* {}", &elem_ty_str, elem_tmp, &elem_ty_str, elem_str));
        }

        ctx.code.add(format!("br label %b{}", func.blocks[0].id));
        // Generate an expression for the function body.
        try!(self.gen_func(sir, func, ctx));
        if containing_loop.is_some() {
            ctx.code.add("br label %loop_terminator");
            ctx.code.add("loop_terminator:");
            let idx_tmp = try!(self.load_var("%cur.idx", "i64", ctx));
            let idx_inc = ctx.var_ids.next();
            ctx.code.add(format!("{} = add i64 {}, 1", idx_inc, idx_tmp));
            ctx.code.add(format!("store i64 {}, i64* %cur.idx", idx_inc));
            ctx.code.add("br label %loop_start");
            ctx.code.add("loop_end:");
        }
        ctx.code.add("ret void");
        ctx.code.add("}\n\n");

        self.body_code.add(&ctx.alloca_code.result());
        self.body_code.add(&ctx.code.result());

        if containing_loop.is_some() {
            // TODO add parallel wrappers for loop bodies and continuations
            let par_for = containing_loop.clone().unwrap();
            if par_for.body == func.id {
                let mut serial_ctx = &mut FunctionContext::new();
                let serial_arg_types = try!(self.get_arg_str(&get_combined_params(sir, &par_for), ""));
                serial_ctx.code.add(format!("define void @f{}_ser_wrapper({}) {{", func.id,
                    serial_arg_types));
                let data_str = llvm_symbol(&par_for.data);
                let data_ty_str = try!(self.llvm_type(func.params.get(&par_for.data).unwrap())).to_string();
                let data_prefix = format!("@{}", data_ty_str.replace("%", ""));
                let vec_size = serial_ctx.var_ids.next();
                serial_ctx.code.add(format!("{} = call i64 {}.size({} {})", vec_size, data_prefix,
                    data_ty_str, data_str));
                let mut body_arg_types = try!(self.get_arg_str(&func.params, ""));
                body_arg_types.push_str(format!(", i64 0, i64 {}", vec_size).as_str());
                serial_ctx.code.add(format!("call void @f{}({})", func.id, body_arg_types));
                let cont_arg_types = try!(self.get_arg_str(&sir.funcs[par_for.cont].params, ""));
                serial_ctx.code.add(format!("call void @f{}({})", par_for.cont, cont_arg_types));
                serial_ctx.code.add("ret void");
                serial_ctx.code.add("}\n\n");
                self.body_code.add(&serial_ctx.code.result());
            }
        }
        Ok(())
    }

    /// Add a function to the generated program, passing its parameters and return value through
    /// pointers encoded as i64. This is used for the main entry point function into Weld modules
    /// to pass them arbitrary structures.
    pub fn add_function_on_pointers(
        &mut self,
        name: &str,
        sir: &SirProgram
    ) -> WeldResult<()> {
        // First add the function on raw values, which we'll call from the pointer version.
        try!(self.add_function(sir, &sir.funcs[0], None));

        // Define a struct with all the argument types as fields
        let args_struct = Struct(sir.top_params.iter().map(|a| a.ty.clone()).collect());
        let args_type = try!(self.llvm_type(&args_struct)).to_string();

        let mut code = &mut CodeBuilder::new();

        code.add(format!("define i64 @{}(i64 %args) {{", name));

        // Code to load args and call function
        code.add(format!(
            "%args_typed = inttoptr i64 %args to {args_type}*
             %args_val = load {args_type}* %args_typed",
            args_type = args_type
        ));
        
        let mut arg_pos_map: HashMap<Symbol, usize> = HashMap::new();
        for (i, a) in sir.top_params.iter().enumerate() {
            arg_pos_map.insert(a.name.clone(), i);    
        }
        let mut arg_decls = String::new();
        let params_sorted: BTreeMap<&Symbol, &Type> = sir.funcs[0].params.iter().collect();
        for (arg, ty) in params_sorted.iter() {
            let idx = arg_pos_map.get(arg).unwrap();
            code.add(format!("%arg{} = extractvalue {} %args_val, {}", idx, args_type, idx));
            arg_decls.push_str(format!("{} %arg{}, ", try!(self.llvm_type(&ty)), idx).as_str());
        }
        arg_decls.push_str("%work_t* %cur.work");
        code.add(format!(
            "%cur.work.size.ptr = getelementptr %work_t* null, i32 1
             %cur.work.size = ptrtoint %work_t* %cur.work.size.ptr to i64 
             %cur.work.raw = call i8* @malloc(i64 %cur.work.size)
             %cur.work = bitcast i8* %cur.work.raw to %work_t*
             call void @f0({arg_list})
             %res_ptr_ptr = getelementptr %work_t* %cur.work, i32 0, i32 0
             %res_ptr = load i8** %res_ptr_ptr
             %res_address = ptrtoint i8* %res_ptr to i64
             ret i64 %res_address",
            arg_list = arg_decls
        ));
        code.add(format!("}}\n\n"));

        self.body_code.add_code(code);
        Ok(())
    }

    /// Return the LLVM type name corresponding to a Weld type.
    fn llvm_type(&mut self, ty: &Type) -> WeldResult<&str> {
        match *ty {
            Scalar(Bool) => Ok("i1"),
            Scalar(I32) => Ok("i32"),
            Scalar(I64) => Ok("i64"),
            Scalar(F32) => Ok("f32"),
            Scalar(F64) => Ok("f64"),

            Struct(ref fields) => {
                if self.struct_names.get(fields) == None {
                    // Declare the struct in prelude_code
                    let name = self.struct_ids.next();
                    let mut field_types: Vec<String> = Vec::new();
                    for f in fields {
                        field_types.push(try!(self.llvm_type(f)).to_string());
                    }
                    let field_types = field_types.join(", ");
                    self.prelude_code.add(format!("{} = type {{ {} }}", &name, &field_types));
                    // Add it into our map so we remember its name
                    self.struct_names.insert(fields.clone(), name);
                }
                Ok(self.struct_names.get(fields).unwrap())
            },

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
            },

            Builder(ref bk) => {
                if self.bld_names.get(bk) == None {
                    match *bk {
                        Appender(ref t) => {
                            let bld_ty = Vector(t.clone());
                            let bld_ty_str = try!(self.llvm_type(&bld_ty)).to_string();
                            self.bld_names.insert(bk.clone(), format!("{}.bld", &bld_ty_str));
                        },
                        _ => weld_err!("Unsupported builder type {} in llvm_type", print_type(ty))?
                    }
                }
                Ok(self.bld_names.get(bk).unwrap())
            }

            _ => weld_err!("Unsupported type {}", print_type(ty))?
        }
    }

    fn load_var(
        &mut self,
        sym: &str,
        ty: &str,
        ctx: &mut FunctionContext
    ) -> WeldResult<String> {
        let var = ctx.var_ids.next();
        ctx.code.add(format!("{} = load {}* {}", var, ty, sym));
        Ok(var)
    }

    /// Add an expression to a CodeBuilder, possibly generating prelude code earlier, and return
    /// a string that can be used to represent its result later (e.g. %var if introducing a local
    /// variable or an integer constant otherwise).
    fn gen_func(
        &mut self,
        sir: &SirProgram,
        func: &SirFunction,
        ctx: &mut FunctionContext
    ) -> WeldResult<String> {
        for b in func.blocks.iter() {
            ctx.code.add(format!("b{}:", b.id));
            for s in b.statements.iter() {
                match *s {
                    AssignBinOp(ref output, op, ref ty, ref left, ref right) => {
                        let op_name = try!(llvm_binop(op, ty));
                        let ll_ty = try!(self.llvm_type(ty)).to_string();
                        let left_tmp = try!(self.load_var(llvm_symbol(left).as_str(),
                            &ll_ty, ctx));
                        let right_tmp = try!(self.load_var(llvm_symbol(right).as_str(),
                            &ll_ty, ctx));
                        let bin_tmp = ctx.var_ids.next();
                        ctx.code.add(format!("{} = {} {} {}, {}",
                            bin_tmp, op_name, &ll_ty, left_tmp, right_tmp));
                        ctx.code.add(format!("store {} {}, {}* {}", ll_ty, bin_tmp, ll_ty, llvm_symbol(output)));
                    },
                    Assign(ref out, ref value) => {
                        let ty = try!(get_sym_ty(func, out));
                        let ll_ty = try!(self.llvm_type(&ty)).to_string();
                        let val_tmp = try!(self.load_var(llvm_symbol(value).as_str(), &ll_ty, ctx));
                        ctx.code.add(format!("store {} {}, {}* {}", &ll_ty, val_tmp, &ll_ty, llvm_symbol(out)));
                    },
                    AssignLiteral(ref out, ref lit) => {
                        match *lit {
                            BoolLiteral(l) => ctx.code.add(format!("store i1 {}, i1* {}",
                                if l { 1 } else { 0 }, llvm_symbol(out))),
                            I32Literal(l) => ctx.code.add(format!("store i32 {}, i32* {}",
                                l, llvm_symbol(out))),
                            I64Literal(l) => ctx.code.add(format!("store i64 {}, i64* {}",
                                l, llvm_symbol(out))),
                            F32Literal(l) => ctx.code.add(format!("store f32 {}, f32* {}",
                                l, llvm_symbol(out))),
                            F64Literal(l) => ctx.code.add(format!("store f64 {}, f64* {}",
                                l, llvm_symbol(out)))
                        }
                    },
                    DoMerge(ref bld, ref elem) => {
                        let bld_ty = try!(get_sym_ty(func, bld));
                        match *bld_ty {
                            Builder(ref bk) => {
                                match *bk {
                                    Appender(ref t) => {
                                        let bld_ty_str = try!(self.llvm_type(&bld_ty)).to_string();
                                        let bld_prefix = format!("@{}", bld_ty_str.replace("%", ""));
                                        let elem_ty_str = try!(self.llvm_type(t)).to_string();
                                        let bld_tmp = try!(self.load_var(llvm_symbol(bld).as_str(), &bld_ty_str,
                                            ctx));
                                        let elem_tmp = try!(self.load_var(llvm_symbol(elem).as_str(), &elem_ty_str,
                                            ctx));
                                        ctx.code.add(format!("call {} {}.merge({} {}, {} {})", &bld_ty_str,
                                            bld_prefix, &bld_ty_str, bld_tmp, &elem_ty_str, elem_tmp));
                                    },
                                    _ => weld_err!("Unsupported builder type {} in DoMerge", print_type(bld_ty))?
                                }
                            },
                            _ => weld_err!("Non builder type {} found in DoMerge", print_type(bld_ty))?
                        }
                    },
                    GetResult(ref out, ref value) => {
                        let bld_ty = try!(get_sym_ty(func, value));
                        let res_ty = try!(get_sym_ty(func, out));
                        match *bld_ty {
                            Builder(ref bk) => {
                                match *bk {
                                    Appender(_) => {
                                        let bld_ty_str = try!(self.llvm_type(&bld_ty)).to_string();
                                        let bld_prefix = format!("@{}", bld_ty_str.replace("%", ""));
                                        let res_ty_str = try!(self.llvm_type(&res_ty)).to_string();
                                        let bld_tmp = try!(self.load_var(llvm_symbol(value).as_str(), &bld_ty_str,
                                            ctx));
                                        let res_tmp = ctx.var_ids.next();
                                        ctx.code.add(format!("{} = call {} {}.result({} {})", &res_tmp,
                                            &res_ty_str, bld_prefix, &bld_ty_str, bld_tmp));
                                        ctx.code.add(format!("store {} {}, {}* {}", &res_ty_str,
                                            &res_tmp, &res_ty_str, llvm_symbol(out)));
                                    },
                                    _ => weld_err!("Unsupported builder type {} in GetResult", print_type(bld_ty))?
                                }
                            },
                            _ => weld_err!("Non builder type {} found in GetResult", print_type(bld_ty))?
                        }
                    },
                    CreateBuilder(ref out, ref ty) => {
                        match *ty {
                            Builder(ref bk) => {
                                match *bk {
                                    Appender(_) => {
                                        let bld_ty_str = try!(self.llvm_type(ty));
                                        let bld_prefix = format!("@{}", bld_ty_str.replace("%", ""));
                                        let bld_tmp = ctx.var_ids.next();
                                        ctx.code.add(format!("{} = call {} {}.new(i64 16)", bld_tmp,
                                            bld_ty_str, bld_prefix));
                                        ctx.code.add(format!("store {} {}, {}* {}", bld_ty_str,
                                            bld_tmp, bld_ty_str, llvm_symbol(out)));
                                    },
                                    _ => weld_err!("Unsupported builder type {} in CreateBuilder", print_type(ty))?
                                }
                            },
                            _ => weld_err!("Non builder type {} found in CreateResult", print_type(ty))?
                        }
                    }
                }
            }
            match b.terminator {
                Branch(ref cond, on_true, on_false) => {
                    let cond_tmp = try!(self.load_var(llvm_symbol(cond).as_str(), "i1", ctx));
                    ctx.code.add(format!("br i1 {}, label %b{}, label %b{}", cond_tmp, on_true, on_false));
                },
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
                    ctx.code.add(format!("call void @f{}_ser_wrapper({})", pf.body, arg_types));
                },
                JumpBlock(block) => {
                    ctx.code.add(format!("br label %b{}", block));
                },
                JumpFunction(func) => {
                    try!(self.add_function(sir, &sir.funcs[func], None));
                    let params_sorted: BTreeMap<&Symbol, &Type> = sir.funcs[func].params.iter().collect();
                    let mut arg_types = String::new();
                    for (arg, ty) in params_sorted.iter() {
                        let ll_ty = try!(self.llvm_type(&ty)).to_string();
                        let arg_tmp = try!(self.load_var(llvm_symbol(arg).as_str(), &ll_ty, ctx));
                        let arg_str = format!("{} {}, ", &ll_ty, arg_tmp);
                        arg_types.push_str(&arg_str);
                    }
                    arg_types.push_str("%work_t* %cur.work");
                    ctx.code.add(format!("call void @f{}({})", func, arg_types));
                },
                ProgramReturn(ref sym) => {
                    let ty = try!(get_sym_ty(func, sym));
                    let ty_str = try!(self.llvm_type(ty)).to_string();
                    let res_tmp = try!(self.load_var(llvm_symbol(sym).as_str(), &ty_str, ctx));
                    let elem_size_ptr = ctx.var_ids.next();
                    let elem_size = ctx.var_ids.next();
                    let elem_storage = ctx.var_ids.next();
                    let elem_storage_typed = ctx.var_ids.next();
                    let work_res_ptr = ctx.var_ids.next();
                    ctx.code.add(format!("{} = getelementptr {}* null, i32 1", &elem_size_ptr, &ty_str));
                    ctx.code.add(format!("{} = ptrtoint {}* {} to i64", &elem_size, &ty_str, &elem_size_ptr));
                    ctx.code.add(format!("{} = call i8* @malloc(i64 {})", &elem_storage, &elem_size));
                    ctx.code.add(format!("{} = bitcast i8* {} to {}*", &elem_storage_typed,
                        &elem_storage, &ty_str));
                    ctx.code.add(format!("store {} {}, {}* {}", &ty_str, res_tmp, &ty_str, &elem_storage_typed));
                    ctx.code.add(format!("{} = getelementptr %work_t* %cur.work, i32 0, i32 0", &work_res_ptr));
                    ctx.code.add(format!("store i8* {}, i8** {}", &elem_storage, &work_res_ptr));
                },
                EndFunction => {},
                Crash => { /* TODO do something else here? */ }
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

        (BinOpKind::Equal, &Scalar(I32)) => Ok("icmp eq"),
        (BinOpKind::Equal, &Scalar(I64)) => Ok("icmp eq"),
        (BinOpKind::Equal, &Scalar(F32)) => Ok("fcmp oeq"),
        (BinOpKind::Equal, &Scalar(F64)) => Ok("fcmp oeq"),

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

        _ => weld_err!("Unsupported binary op: {} on {}", op_kind, print_type(ty))
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
    if_ids: IdGenerator,
}

impl FunctionContext {
    fn new() -> FunctionContext {
        FunctionContext {
            alloca_code: CodeBuilder::new(),
            code: CodeBuilder::new(),
            var_ids: IdGenerator::new("%t."),
            if_ids: IdGenerator::new("if"),
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
    try!(type_inference::infer_types(&mut expr));
    let expr = try!(expr.to_typed());
    let sir_prog = try!(sir::ast_to_sir(&expr));
    let mut gen = LlvmGenerator::new();
    try!(gen.add_function_on_pointers("run", &sir_prog));
    println!("{}", gen.result());
    Ok(try!(easy_ll::compile_module(&gen.result())))
}

#[test]
fn types() {
    let mut gen = LlvmGenerator::new();

    assert_eq!(gen.llvm_type(&Scalar(I32)).unwrap(), "i32");
    assert_eq!(gen.llvm_type(&Scalar(I64)).unwrap(), "i64");
    assert_eq!(gen.llvm_type(&Scalar(F32)).unwrap(), "f32");
    assert_eq!(gen.llvm_type(&Scalar(F64)).unwrap(), "f64");
    assert_eq!(gen.llvm_type(&Scalar(Bool)).unwrap(), "i1");

    let struct1 = parse_type("{i32,bool,i32}").unwrap().to_type().unwrap();
    assert_eq!(gen.llvm_type(&struct1).unwrap(), "%s0");
    assert_eq!(gen.llvm_type(&struct1).unwrap(), "%s0");   // Name is reused for same struct

    let struct2 = parse_type("{i32,bool}").unwrap().to_type().unwrap();
    assert_eq!(gen.llvm_type(&struct2).unwrap(), "%s1");
}

#[test]
fn basic_program() {
    let code = "|| 40 + 2";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let result = module.run(0) as *const i32;
    let result = unsafe { *result };
    assert_eq!(result, 42);
    // TODO: Free result
}

#[test]
fn program_with_args() {
    let code = "|x:i32| 40 + x";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let input: i32 = 2;
    let result = module.run(&input as *const i32 as i64) as *const i32;
    let result = unsafe { *result };
    assert_eq!(result, 42);
    // TODO: Free result
}

#[test]
fn let_statement() {
    let code = "|x:i32| let y = 40 + x; y + 2";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let input: i32 = 2;
    let result = module.run(&input as *const i32 as i64) as *const i32;
    let result = unsafe { *result };
    assert_eq!(result, 44);
    // TODO: Free result
}

#[test]
fn if_statement() {
    let code = "|x:i32| if(true, 3, 4)";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let input: i32 = 2;
    let result = module.run(&input as *const i32 as i64) as *const i32;
    let result = unsafe { *result };
    assert_eq!(result, 3);
    // TODO: Free result
}

#[test]
fn comparison() {
    let code = "|x:i32| if(x>10, x, 10)";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let input: i32 = 2;
    let result = module.run(&input as *const i32 as i64) as *const i32;
    let result = unsafe { *result };
    assert_eq!(result, 10);
    // TODO: Free result
    let input: i32 = 20;
    let result = module.run(&input as *const i32 as i64) as *const i32;
    let result = unsafe { *result };
    assert_eq!(result, 20);
    // TODO: Free result
}
