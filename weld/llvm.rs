use std::collections::HashMap;

use easy_ll::*;

use super::ast::*;
use super::ast::Type::*;
use super::ast::ExprKind::*;
use super::ast::ScalarKind::*;
use super::code_builder::CodeBuilder;
use super::error::*;
use super::pretty_print::*;
use super::util::IdGenerator;

#[cfg(test)] use super::parser::*;

/// Generates LLVM code for one or more modules.
pub struct LlvmGenerator {
    /// Track a unique name of the form %s0, %s1, etc for each struct generated.
    struct_names: HashMap<Vec<Type>, String>,
    struct_ids: IdGenerator,

    /// Track a unique name of the form %v0, %v1, etc for each vec generated.
    vec_names: HashMap<Type, String>,
    vec_ids: IdGenerator,

    var_ids: IdGenerator,

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
            var_ids: IdGenerator::new("%t"),
            prelude_code: CodeBuilder::new(),
            body_code: CodeBuilder::new(),
        };
        generator.prelude_code.add(include_str!("resources/prelude.ll"));
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
        let mut code = &mut CodeBuilder::new();
        let mut arg_types = String::new();
        for (i, arg) in args.iter().enumerate() {
            arg_types.push_str(try!(self.llvm_type(&arg.ty)));
            if i < args.len() - 1 {
                arg_types.push_str(", ");
            }
        }
        let res_type = try!(self.llvm_type(&body.ty)).to_string();

        code.add(format!("define {} @{}({}) {{", res_type, name, arg_types));
        let res_var = try!(self.gen_expr(&body, code));
        code.add(format!("ret {} {}", res_type, res_var));
        code.add(format!("}}\n\n"));

        self.body_code.add_code(code);
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
    fn gen_expr(&mut self, expr: &TypedExpr, code: &mut CodeBuilder) -> WeldResult<String> {
        match expr.kind {
            I32Literal(value) => Ok(format!("{}", value)),
            I64Literal(value) => Ok(format!("{}", value)),
            F32Literal(value) => Ok(format!("{}", value)),
            F64Literal(value) => Ok(format!("{}", value)),
            BoolLiteral(value) => Ok(format!("{}", if value {1} else {0})),

            _ => weld_err!("Unsupported expression: {}", print_expr(expr))
        }
    }
}

/// Generate a compiled LLVM module from a typed expression representing a function.
pub fn generate(_: &TypedExpr) -> WeldResult<CompiledModule> {
    weld_err!("Not implemented yet")
}

#[test]
fn types() {
    let mut ctx = LlvmGenerator::new();

    assert_eq!(ctx.llvm_type(&Scalar(I32)).unwrap(), "i32");
    assert_eq!(ctx.llvm_type(&Scalar(I64)).unwrap(), "i64");
    assert_eq!(ctx.llvm_type(&Scalar(F32)).unwrap(), "f32");
    assert_eq!(ctx.llvm_type(&Scalar(F64)).unwrap(), "f64");
    assert_eq!(ctx.llvm_type(&Scalar(Bool)).unwrap(), "i1");

    let struct1 = parse_type("{i32,bool,i32}").unwrap().to_type().unwrap();
    assert_eq!(ctx.llvm_type(&struct1).unwrap(), "%s0");
    assert_eq!(ctx.llvm_type(&struct1).unwrap(), "%s0");   // Name is reused for same struct

    let struct2 = parse_type("{i32,bool}").unwrap().to_type().unwrap();
    assert_eq!(ctx.llvm_type(&struct2).unwrap(), "%s1");
}
