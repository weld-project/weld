extern crate easy_ll;
extern crate weld_ast;
#[macro_use] extern crate weld_error;
extern crate weld_parser;

use easy_ll::*;

use weld_ast::*;
use weld_error::*;

pub fn generate(expr: TypedExpr) -> WeldResult<CompiledModule> {
    weld_err!("Not implementd yet")
}