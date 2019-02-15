//! A program as parsed by our grammar.
//!
//! In reality all programs are turned into a single expression after inlining functions, so this
//! is only a compile-time construct for now. Eventually, it may make more sense to pass around
//! programs through the compiler rather than just expressions, so we can store more information
//! about the expression (e.g., debug info about type aliases and macros)

use std::vec::Vec;

use ast::{Expr, Symbol, Type};

#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub macros: Vec<Macro>,
    pub type_aliases: Vec<TypeAlias>,
    /// Program body -- this will likely be a Lambda, but not always.
    pub body: Expr,
}
/// A type alias we will substitute at compile time.
#[derive(Clone, Debug, PartialEq)]
pub struct TypeAlias {
    pub name: String,
    pub ty: Type,
}

/// A macro we will substitute at compile time.
#[derive(Clone, Debug, PartialEq)]
pub struct Macro {
    pub name: Symbol,
    pub parameters: Vec<Symbol>,
    pub body: Expr,
}
