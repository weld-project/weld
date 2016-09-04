use easy_ll::*;

use super::ast::*;
use super::ast::Type::*;
use super::ast::ScalarKind::*;
use super::pretty_print::print_type;
use super::error::*;

/// Structure holding the state of code generation for a module, used to keep track of things
/// like the code builder, unique IDs, structure names, etc.
pub struct GeneratorContext;

pub fn generate(_: TypedExpr) -> WeldResult<CompiledModule> {
    weld_err!("Not implemented yet")
}

impl GeneratorContext {
    pub fn llvm_type(&mut self, t: &Type) -> WeldResult<String> {
        match *t {
            Scalar(I32) => Ok("i32".to_string()),
            Scalar(Bool) => Ok("i1".to_string()),
            _ => weld_err!("Unsupported type {}", print_type(t))
        }
    }
}

#[test]
fn types() {
    let mut ctx = GeneratorContext;
    assert_eq!(ctx.llvm_type(&Scalar(I32)).unwrap(), "i32");
    assert_eq!(ctx.llvm_type(&Scalar(Bool)).unwrap(), "i1");

    //let weld_type = parse_type("{i32,bool,i32}").unwrap().to_type().unwrap();
    //assert_eq!(ctx.llvm_type(&weld_type).unwrap(), "%s1");
}
