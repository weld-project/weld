//! Defines the Weld abstract syntax tree.
//!
//! Most of Weld's optimizations occur over the AST, which doubles as a "High-Level IR." The AST
//! captures the expressions in Weld using a tree data structure.

pub use self::ast::*;

// Various convinience methods on the AST.
pub use self::builder::NewExpr;
pub use self::cmp::CompareIgnoringSymbols;
pub use self::hash::HashIgnoringSymbols;
pub use self::pretty_print::{PrettyPrint, PrettyPrintConfig};
pub use self::type_inference::InferTypes;
pub use self::uniquify::Uniquify;

pub mod constructors;
pub mod prelude;

mod ast;
mod builder;
mod cmp;
mod hash;
mod pretty_print;
mod type_inference;
mod uniquify;
