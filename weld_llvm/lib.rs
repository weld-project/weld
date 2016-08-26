extern crate easy_ll;
extern crate weld_ast;
extern crate weld_parser;
#[macro_use] extern crate weld_error;

#[cfg(test)] mod tests;

use easy_ll::*;
use weld_ast::*;
use weld_ast::Type::*;
use weld_ast::ScalarKind::*;
use weld_ast::pretty_print::print_type;
use weld_error::*;

mod code_builder;

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
