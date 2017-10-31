//! Implements size inference for `For` loops.

use ast::*;
use ast::ExprKind::*;
use ast::Type::*;
use ast::BuilderKind::*;
use exprs;

/// Infers the size of an `Appender` in a `For` loop.
pub fn infer_size(expr: &mut Expr<Type>) {
    expr.transform_and_continue_res(&mut |ref mut expr| {
        if let For { ref iters, ref builder, ref func } = expr.kind {
            // This constraint prevents copying expensive iters.
            if let Ident(_) = iters[0].data.kind {
                if let NewBuilder(None) = builder.kind {
                    if let Builder(Appender(ref ek), _) = builder.ty {
                        if let Lambda {ref params, ref body } = func.kind {
                            let ref builder_symbol = params[0].name.clone();
                            if simple_merge(builder_symbol, body) {
                                // Compute the inferred length based on the iter.
                                let length = if let Some(ref start) = iters[0].start {
                                    let e = exprs::binop_expr(BinOpKind::Subtract,
                                                              *iters[0].end.as_ref().unwrap().clone(),
                                                              *start.clone())?;
                                    exprs::binop_expr(BinOpKind::Divide, e, *iters[0].stride.as_ref().unwrap().clone())?
                                } else {
                                    exprs::length_expr(*iters[0].data.clone())?
                                };

                                let new_loop = exprs::for_expr(
                                    iters.clone(),
                                    exprs::newbuilder_expr(Appender(ek.clone()), Some(length))?,
                                    func.as_ref().clone(),
                                    false)?;
                                return Ok((Some(new_loop), false));
                            }
                        }
                    }
                }
            }
        }
        Ok((None, true))
    })
}

/// Checks that `expr` performs only one `Merge` per control path - this guarantees
/// that the loop containing `expr`'s size can be inferred.
fn simple_merge(sym: &Symbol, expr: &Expr<Type>) -> bool {
    match expr.kind {
        Merge { ref builder, ref value } => {
            if let Ident(ref s) = builder.kind {
                if s == sym {
                    return !value.contains_symbol(sym);
                }
            }
            return false;
        }
        If { ref cond, ref on_true, ref on_false } => {
            !cond.contains_symbol(sym) && simple_merge(sym, on_true) &&
                simple_merge(sym, on_false)
        }
        _ => false,
    }
}
