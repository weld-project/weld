use std::collections::HashMap;
use std::collections::HashSet;

use easy_ll;

use super::ast::*;
use super::ast::Type::*;
use super::ast::ExprKind::*;
use super::ast::ScalarKind::*;
use super::code_builder::CodeBuilder;
use super::error::*;
use super::macro_processor;
use super::pretty_print::*;
use super::program::Program;
use super::type_inference;
use super::util::IdGenerator;

#[cfg(test)] use super::parser::*;

static PRELUDE_CODE: &'static str = include_str!("resources/prelude.ll");

/// Generates LLVM code for one or more modules.
pub struct LlvmGenerator {
    /// Track a unique name of the form %s0, %s1, etc for each struct generated.
    struct_names: HashMap<Vec<Type>, String>,
    struct_ids: IdGenerator,

    /// Track a unique name of the form %v0, %v1, etc for each vec generated.
    vec_names: HashMap<Type, String>,
    vec_ids: IdGenerator,

    /// A CodeBuilder for prelude functions such as type and struct definitions.
    prelude_code: CodeBuilder,

    /// A CodeBuilder for body functions in the module.
    body_code: CodeBuilder,
}

impl LlvmGenerator {
    pub fn new() -> LlvmGenerator {
        let mut generator = LlvmGenerator {
            struct_names: HashMap::new(),
            struct_ids: IdGenerator::new("%s"),
            vec_names: HashMap::new(),
            vec_ids: IdGenerator::new("%v"),
            prelude_code: CodeBuilder::new(),
            body_code: CodeBuilder::new(),
        };
        generator.prelude_code.add(PRELUDE_CODE);
        generator.prelude_code.add("\n");
        generator
    }

    /// Return all the code generated so far.
    pub fn result(&mut self) -> String {
        format!("; PRELUDE:\n\n{}\n; BODY:\n\n{}", self.prelude_code.result(), self.body_code.result())
    }

    /// Add a function to the generated program.
    pub fn add_function(
        &mut self,
        name: &str,
        args: &Vec<TypedParameter>,
        body: &TypedExpr
    ) -> WeldResult<()> {
        let mut ctx = &mut FunctionContext::new();
        let mut arg_types = String::new();
        for (i, arg) in args.iter().enumerate() {
            let arg = format!("{} {}.in", try!(self.llvm_type(&arg.ty)), llvm_symbol(&arg.name));
            arg_types.push_str(&arg);
            if i < args.len() - 1 {
                arg_types.push_str(", ");
            }
        }
        let res_type = try!(self.llvm_type(&body.ty)).to_string();

        // Start the entry block by defining the function and storing all its arguments on the
        // stack (this makes them consistent with other local variables). Later, expressions may
        // add more local variables to alloca_code.
        ctx.alloca_code.add(format!("define {} @{}({}) {{", res_type, name, arg_types));
        ctx.alloca_code.add(format!("entry:"));
        for arg in args {
            let name = llvm_symbol(&arg.name);
            let ty = try!(self.llvm_type(&arg.ty)).to_string();
            try!(ctx.add_alloca(&name, &ty));
            ctx.code.add(format!("store {} {}.in, {}* {}", ty, name, ty, name));
        }

        // Generate an expression for the function body.
        let res_var = try!(self.gen_expr(&body, ctx));
        ctx.code.add(format!("ret {} {}", res_type, res_var));
        ctx.code.add(format!("}}\n\n"));

        self.body_code.add(&ctx.alloca_code.result());
        self.body_code.add(&ctx.code.result());
        Ok(())
    }

    /// Add a function to the generated program, passing its parameters and return value through
    /// pointers encoded as i64. This is used for the main entry point function into Weld modules
    /// to pass them arbitrary structures.
    pub fn add_function_on_pointers(
        &mut self,
        name: &str,
        args: &Vec<TypedParameter>,
        body: &TypedExpr
    ) -> WeldResult<()> {
        // First add the function on raw values, which we'll call from the pointer version.
        let raw_function_name = format!("{}.raw", name);
        try!(self.add_function(&raw_function_name, args, body));

        // Define a struct with all the argument types as fields
        let args_struct = Struct(args.iter().map(|a| a.ty.clone()).collect());
        let args_type = try!(self.llvm_type(&args_struct)).to_string();

        let res_type = try!(self.llvm_type(&body.ty)).to_string();
        let mut code = &mut CodeBuilder::new();

        code.add(format!("define i64 @{}(i64 %args) {{", name));

        // Code to allocate a result structure
        code.add(format!(
            "%res_size_ptr = getelementptr {res_type}* null, i32 1
             %res_size = ptrtoint {res_type}* %res_size_ptr to i64
             %res_bytes = call i8* @malloc(i64 %res_size)
             %res_typed = bitcast i8* %res_bytes to {res_type}*",
            res_type = res_type
        ));

        // Code to load args and call function
        code.add(format!(
            "%args_typed = inttoptr i64 %args to {args_type}*
             %args_val = load {args_type}* %args_typed",
            args_type = args_type
        ));
        let mut arg_decls: Vec<String> = Vec::new();
        for (i, arg) in args.iter().enumerate() {
            code.add(format!("%arg{} = extractvalue {} %args_val, {}", i, args_type, i));
            arg_decls.push(format!("{} %arg{}", try!(self.llvm_type(&arg.ty)), i));
        }
        code.add(format!(
            "%res_val = call {res_type} @{raw_function_name}({arg_list})
             store {res_type} %res_val, {res_type}* %res_typed
             %res_address = ptrtoint {res_type}* %res_typed to i64
             ret i64 %res_address",
            res_type = res_type,
            raw_function_name = raw_function_name,
            arg_list = arg_decls.join(", ")
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
            }

            Vector(ref elem) => {
                if self.vec_names.get(elem) == None {
                    // TODO: declare the vector's struct and helper functions in self.prelude_code
                    self.vec_names.insert(*elem.clone(), self.vec_ids.next());
                }
                Ok(self.vec_names.get(elem).unwrap())
            }

            _ => weld_err!("Unsupported type {}", print_type(ty))
        }
    }

    /// Add an expression to a CodeBuilder, possibly generating prelude code earlier, and return
    /// a string that can be used to represent its result later (e.g. %var if introducing a local
    /// variable or an integer constant otherwise).
    fn gen_expr(
        &mut self,
        expr: &TypedExpr,
        ctx: &mut FunctionContext
    ) -> WeldResult<String> {
        match expr.kind {
            I32Literal(value) => Ok(format!("{}", value)),
            I64Literal(value) => Ok(format!("{}", value)),
            F32Literal(value) => Ok(format!("{}", value)),
            F64Literal(value) => Ok(format!("{}", value)),
            BoolLiteral(value) => Ok(format!("{}", if value {1} else {0})),

            Ident(ref symbol) => {
                let var = ctx.var_ids.next();
                ctx.code.add(format!("{} = load {}* {}",
                    var, try!(self.llvm_type(&expr.ty)), llvm_symbol(symbol)));
                Ok(var)
            },

            BinOp(kind, ref left, ref right) => {
                let op_name = try!(llvm_binop(kind, &left.ty));
                let left_var = try!(self.gen_expr(left, ctx));
                let right_var = try!(self.gen_expr(right, ctx));
                let var = ctx.var_ids.next();
                ctx.code.add(format!("{} = {} {} {}, {}",
                    var, op_name, try!(self.llvm_type(&left.ty)), left_var, right_var));
                Ok(var)
            },

            Let(ref name, ref value, ref body) => {
                let value_var = try!(self.gen_expr(value, ctx));
                let name = llvm_symbol(name);
                let ty = try!(self.llvm_type(&value.ty)).to_string();
                try!(ctx.add_alloca(&name, &ty));
                ctx.code.add(format!("store {} {}, {}* {}", ty, value_var, ty, name));
                self.gen_expr(body, ctx)
            },

            If(ref cond, ref on_true, ref on_false) => {
                let cond_var = try!(self.gen_expr(cond, ctx));
                let id = ctx.if_ids.next();
                let true_label = format!("{}.true", id);
                let false_label = format!("{}.false", id);
                let end_true_label = format!("{}.true.end", id);
                let end_false_label = format!("{}.false.end", id);
                let end_label = format!("{}.end", id);

                ctx.code.add(format!("br i1 {}, label %{}, label %{}",
                    cond_var, true_label, false_label));
                ctx.code.add(format!("{}:", true_label));
                let true_var = try!(self.gen_expr(on_true, ctx));
                ctx.code.add(format!("br label %{}", end_true_label));
                ctx.code.add(format!("{}:", end_true_label));
                ctx.code.add(format!("br label %{}", end_label));

                ctx.code.add(format!("{}:", false_label));
                let false_var = try!(self.gen_expr(on_false, ctx));
                ctx.code.add(format!("br label %{}", end_false_label));
                ctx.code.add(format!("{}:", end_false_label));
                ctx.code.add(format!("br label %{}", end_label));

                ctx.code.add(format!("{}:", end_label));
                let var = ctx.var_ids.next();
                let ty = try!(self.llvm_type(&expr.ty)).to_string();
                ctx.code.add(format!("{} = phi {} [{}, %{}], [{}, %{}]",
                    var, ty, true_var, end_true_label, false_var, end_false_label));
                Ok(var)
            },

            _ => weld_err!("Unsupported expression: {}", print_expr(expr))
        }
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
            var_ids: IdGenerator::new("%"),
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
    match expr.kind {
        Lambda(ref params, ref body) => {
            let mut gen = LlvmGenerator::new();
            try!(gen.add_function_on_pointers("run", params, body));
            println!("{}", gen.result());
            Ok(try!(easy_ll::compile_module(&gen.result())))
        },
        _ => weld_err!("Expression passed to compile_function must be a Lambda")
    }
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
