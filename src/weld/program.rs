//! A program as parsed by our grammar. In reality all programs are turned into a single
//! expression after inlining functions, so this is only a compile-time construct for now.

use std::vec::Vec;

use super::ast::Symbol;
use super::partial_types::*;

#[derive(Debug)]
pub struct Program {
    pub macros: Vec<Macro>,
    /// Program body -- this will likely be a Lambda, but not always.
    pub body: PartialExpr
}

/// A macro we will substitute at compile time.
#[derive(Debug)]
pub struct Macro {
    pub name: Symbol,
    pub parameters: Vec<Symbol>,
    pub body: PartialExpr
}