//! Defines the Weld Abstract Syntax Tree.
//!
//! Most of Weld's optimizations occur over the AST, which doubles as a "High-Level IR." The AST
//! captures the expressions in Weld using a tree data structure.

pub use self::ast::*;

pub mod uniquify;
pub mod pretty_print;
pub mod expr_hash;

mod ast;
