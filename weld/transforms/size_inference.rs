//! Implements size inference for `For` loops.

use ast::*;
use ast::ExprKind::*;
use ast::Type::*;
use ast::BuilderKind::*;
use exprs;
use annotations::*;
use util::SymbolGenerator;

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
                                let length = if iters[0].kind == IterKind::NdIter {
                                    /* To calculate the length for nditer, we will use the shapes
                                     * parameter. We want to generate an expression for the
                                     * following weld code:
                                     *     for(shapes, merger[i64, *], |b, i, e| merge(b, e));
                                     */
                                    let builder = exprs::newbuilder_expr(Merger(Box::new(Scalar(ScalarKind::I64)), BinOpKind::Multiply), None)?;
                                    let builder_type = Builder(Merger(Box::new(Scalar(ScalarKind::I64)), BinOpKind::Multiply), Annotations::new());
                                    /* Used for generating the symbols for the merge_params */
                                    let mut sym_gen = SymbolGenerator::from_expression(expr);
                                    /* builder type for the merger builder */
                                    let merge_params = vec![
                                        Parameter{ty: builder_type.clone(), name: sym_gen.new_symbol("b")},
                                        Parameter{ty: Scalar(ScalarKind::I64), name: sym_gen.new_symbol("i")},
                                        /* we know the element will be an i64 already */
                                        Parameter{ty: Scalar(ScalarKind::I64), name: sym_gen.new_symbol("e")},
                                    ];
                                    /* Need to get an expression for the 'e' parameter */
                                    let elem_iden = exprs::ident_expr(params[2].name.clone(), params[2].ty.clone()).unwrap();
                                    let func = exprs::lambda_expr(merge_params, exprs::merge_expr(builder.clone(), elem_iden)?)?;
                                    /* Need to generate an Iter object for the shapes field so it can be passed to for_expr */
                                    let shapes_iter = Iter {
                                            data: iters[0].shapes.clone().unwrap(), // cloning out of shapes to avoid owning errors.
                                            start: None,
                                            end: None,
                                            stride: None,
                                            kind: IterKind::ScalarIter,
                                            shapes: None,
                                            strides: None,
                                            };
                                    let shapes_vec = vec![shapes_iter];
                                    exprs::for_expr(shapes_vec, builder, func, false)?
                                } else if let Some(ref start) = iters[0].start {
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
