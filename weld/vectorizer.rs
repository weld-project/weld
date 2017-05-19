//! Vectorizes expressions in the AST.
//!
//! This transform marks expressions as vectorizable by converting expressions of type `Scalar`
//! into expressions of type `Vectorized`. It also modifies loops and builders to accept vector
//! arguments instead of scalar arguments.

use super::ast::*;
use super::ast::ExprKind::*;
use super::ast::Type::*;
use super::error::*;

/// Vectorize an expression.
pub fn vectorize(expr: &mut Expr<Type>) -> WeldResult<()> {
    // Step 1. Check each loop to see if it can be vectorized.
    // The complex case here is nested loops; I'll implement simple loops with fringing first.
    //
    // How can loops be "partially" vectorized? We can break a loop into multiple parts, or blocks.
    // There are two challenges with this:
    // 1) We need a node to pull out individual vector elements and then perform operations over
    //    them.
    // 2) We need to determine when to compose these elements back into a vector once this is done.
    //
    // For example, a loop which is vectorizable but has a single dictionary lookup can unravel the
    // unvectorizable part.
    //
    // Some cases to consider:
    //
    // B = merger[+]
    // for x in V:
    //      merge(B, x)
    //
    // B = appender
    // for x in V
    //      merge(B, x + 1)
    expr.transform(&mut |ref mut e| {
        if let BinOp { .. } = e.kind {
            if let Scalar(kind) = e.ty {
                e.ty = VectorizedScalar(kind);
            }
        } else if let Literal(_) = e.kind {
            if let Scalar(kind) = e.ty {
                e.ty = VectorizedScalar(kind);
            }
        }
        None
    });
    Ok(())
}
