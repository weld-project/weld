//! Defines the Weld Abstract Syntax Tree.
//!
//! Most of Weld's optimizations occur over the AST, which doubles as a "High-Level IR." The AST
//! captures the expressions in Weld using a tree data structure.
//!
//! This module also contains a number of traits implemented solely for Weld expression trees:
//!
//! * The `hash::ExprHash` trait provides fast symbol-agnostic comparison of ASTs,
//! * The `pretty_print::PrettyPrint` trait provides pretty printing of ASTs,
//! * The `type_inference::InferTypes` trait provides type inference over ASTs, and
//! * The `uniquify::Uniquify` trait assigns unique symbol names to identifiers in the AST.

pub use self::ast::*;

pub mod constructors;
pub mod hash;
pub mod pretty_print;
pub mod type_inference;
pub mod uniquify;

mod ast;
