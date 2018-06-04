//! Optimizer over the Weld AST.
//!
//! This module manages optimizations over the Weld AST. Optimizations are represented as
//! transformations in the `transforms` module, which convert one AST to another using rule-based
//! pattern matching. The module provides a pass interface that groups related transforms into a
//! pass, and also provides utilities for applying passes until a fix point (that is, until the
//! pass stops modifying the AST).

pub use self::passes::*;

pub mod transforms;
mod passes;
