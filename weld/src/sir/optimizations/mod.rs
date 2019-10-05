//! Optimizations over the sequential IR (SIR).
//!
//! These optimizations simplify the SIR in order to generate more efficient code.
//! Some optimizations are easier to express over the SIR than on the AST (e.g.,
//! constant folding).

pub mod fold_constants;
pub mod simplify_assignments;
