//! Sets annotations on various expressions.

use ast::*;
use ast::ExprKind::*;

/// Forces parallelization of for loops inside an `Iterate`. In the current LLVM code generator,
/// this prevents for loop continuations from making recursive calls to the beginning of the
/// iteration, which can cause stack exhaustion.
pub fn force_iterate_parallel_fors(expr: &mut Expr) {
    expr.transform_and_continue(&mut |ref mut e| {
        match e.kind {
             Iterate { .. } => {
                e.transform_and_continue(&mut |ref mut e| {
                   match e.kind {
                        For { .. } => {
                            e.annotations.set_always_use_runtime(true);
                            (None, true)
                        }
                        _ => (None, true)
                   }
                });
                (None, false)
            }
            _ => {
                (None, true)
            }
        }
    });
}
