extern crate lalrpop_util;

extern crate weld_ast;
#[macro_use] extern crate weld_error;

mod grammar;

pub use grammar::parse_expr;
pub use grammar::parse_program;
pub use grammar::parse_macros;

use std::result::Result;
use lalrpop_util::ParseError;
use weld_ast::partial_types::PartialType;

pub fn parse_type<'input>(input: &'input str)
        -> Result<PartialType, ParseError<usize,(usize, &'input str), &'static str>> {
    grammar::parse_Type(input).map(|b| *b)
}