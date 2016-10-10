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
        LlvmGenerator {
            struct_names: HashMap::new(),
            struct_ids: IdGenerator::new("%s"),
            vec_names: HashMap::new(),
            vec_ids: IdGenerator::new("%v"),
            var_ids: IdGenerator::new("%t"),
            prelude_code: CodeBuilder::new(),
            body_code: CodeBuilder::new(),
        }
    }

    /// Return all the code generated so far.
    pub fn result(&mut self) -> String {
        format!("; PRELUDE:\n{}\n; BODY:\n{}", self.prelude_code.result(), self.body_code.result())
    }

    pub fn add_function(
        &mut self,
        name: &str,
        args: &Vec<TypedParameter>,
        body: &TypedExpr
    ) -> WeldResult<()> {
        let mut code = CodeBuilder::new();

        let mut arg_types = String::new();
        for (i, arg) in args.iter().enumerate() {
            arg_types.push_str(try!(self.llvm_type(&arg.ty)));
            if i < args.len() - 1 {
                arg_types.push_str(", ");
            }
        }

        let result_type = String::from(try!(self.llvm_type(&body.ty)));

        code.add(&format!("define {} @{}({}) {{", result_type, name, arg_types));
        let result = try!(self.gen_expr(&body, &mut code));
        code.add(&format!("ret {} {}", result_type, result));
        code.add("}");

        self.body_code.add_code(&code);
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
                    self.struct_names.insert(fields.clone(), self.struct_ids.next());
                }
                Ok(self.struct_names.get(fields).unwrap())
            }

            Vector(ref elem) => {
                if self.vec_names.get(elem) == None {
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
