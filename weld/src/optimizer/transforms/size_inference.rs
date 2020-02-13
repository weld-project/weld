//! Implements size inference for `For` loops.

use crate::ast::BuilderKind::*;
use crate::ast::ExprKind::*;
use crate::ast::Type::*;
use crate::ast::*;
use crate::util::SymbolGenerator;

struct NewAppender<'a> {
    elem_type: &'a Type,
}

impl<'a> NewAppender<'a> {
    fn extract(expr: &'a Expr) -> Option<NewAppender<'a>> {
        if let NewBuilder(_) = expr.kind {
            if let Builder(Appender(ref elem_type), _) = expr.ty {
                return Some(NewAppender { elem_type });
            }
        }
        None
    }
}

fn newbuilder_with_size(builder: &Expr, length: Expr) -> Option<Expr> {
    if let Some(na) = NewAppender::extract(builder) {
        let bk = Appender(Box::new(na.elem_type.clone()));
        //let inferred_size = length_expr(vector.clone()).unwrap();
        return Some(Expr::new_new_builder(bk, Some(length)).unwrap());
    }
    if let MakeStruct { ref elems } = builder.kind {
        if elems.iter().all(|ref e| NewAppender::extract(e).is_some()) {
            let newbuilders = elems
                .iter()
                .map(|ref e| newbuilder_with_size(e, length.clone()).unwrap())
                .collect();
            return Some(Expr::new_make_struct(newbuilders).unwrap());
        }
    }
    None
}

fn func_has_simple_merge(expr: &Expr) -> bool {
    if let Lambda {
        ref params,
        ref body,
    } = expr.kind
    {
        simple_merge(&params[0].name, body)
    } else {
        false
    }
}

/// Infers the size of an `Appender` in a `For` loop.
pub fn infer_size(expr: &mut Expr) {
    expr.transform_up(&mut |ref mut expr| {
        let mut sym_gen = SymbolGenerator::from_expression(&expr);
        if let For { ref mut iters, ref mut builder, ref mut func } = expr.kind {
            /* Without this condition, the transform_up calls seems to end up in an infinite recursive loop */
            if let NewBuilder(None) = builder.kind {
                if !iters.is_empty() && func_has_simple_merge(func) {
                    if let Ident(_) = iters[0].data.kind {
                        /* Need the data_sym var after the length has been determined to update the
                         * expression. SymbolGenerator seems to borrow expression immutably, so we can't borrow it mutably before.*/
                        let data_sym = sym_gen.new_symbol("data1");
                        let (length, data_expr) = if iters.iter().all(|ref iter| iter.start.is_none()) {
                            /* In this scenario, to determine the length of the appender, we use:
                             *      length = len(iters[0].data);
                             * Here, an issue is if iters[0].data is a more complicated expression,
                             * then we will be re-evaluating it in order to get to its length.
                             * Thus, we do something like:
                             *      let data1 = iters[0].data;
                             *      for(data1, len(data1), ....do appender stuff)
                             * Note: to extract the let statements, let_expr needs the expression
                             * in which the let statement is being defined. So first, we need to
                             * define new_loop -- but to correctly define new loop, we require the
                             * length -- thus we extract the let statements only at the end after
                             * new loop has been defined.
                             */
                            let data_expr = Expr::new_ident(data_sym.clone(), iters[0].data.ty.clone()).unwrap();
                            (Expr::new_length(data_expr.clone()).unwrap(), Some(data_expr))
                        //} else if iters[0].kind == IterKind::RangeIter || iters[1].kind == IterKind::RangeIter {
                        } else if iters.iter().any(|ref iter| iter.kind == IterKind::RangeIter) {
                            // FIXME: pari - temporary fix?
                            return None;
                        } else if iters.iter().any(|ref iter| iter.kind == IterKind::ScalarIter && iter.start.is_some()) {
                             /* For all other iter types - ScalarIter, NdIter etc. which also specify
                             start-end-strides variables. In this case, we use:
                                length = (end - start) / strides;
                             In this case, we don't need data_expr, because we don't use any
                             potentially complicated expressions to calculate length - thus we don't
                             need to take it out in a let statement */
                            // HACK: Get index of first explicit iterator.
                            // Only need the length of one of the iterator, since each iterator will produce
                            // the same number of elements.
                            let mut i = iters.len();
                            for (index, iter) in iters.iter().enumerate() {
                                if iter.start.is_some() {
                                    i = index;
                                }
                            }
                            assert_ne!(i, iters.len());
                            let e = Expr::new_bin_op(BinOpKind::Subtract, *iters[i].end.as_ref().unwrap().clone(),
                                                   *iters[i].start.as_ref().unwrap().clone());
                            let length = Expr::new_bin_op(BinOpKind::Divide, e.unwrap(), *iters[i].stride.as_ref().unwrap().clone());
                            (length.unwrap(), None)
                        } else {
                            // FIXME(shoumik): NDIter uses strides and shapes instead of
                            // stride/start/end, so need to handle it differently...
                            return None;
                        };

                        if let Some(newbuilder) = newbuilder_with_size(builder, length) {
                            match data_expr {
                                None => {
                                    /* here, we do not change anything in the expression besides the
                                     * builder - so we don't need to create a new expression to return.
                                     * Since we have a mutable reference to the builder, we can change
                                     * it directly. */
                                    *builder = Box::new(newbuilder);
                                }
                                Some(val) => {
                                    /* Here, we need to modify the expression itself - because we need
                                     * to add a let statement. It doesn't seem possible to just modify
                                     * it directly, as we did with the builder, so we create and return
                                     * a new expression -- which transform_up will replace expr with */
                                    let orig_data = iters[0].data.clone();
                                    iters[0].data = Box::new(val);
                                    let mut new_loop = Expr::new_for(iters.clone(), newbuilder,
                                    func.as_ref().clone()).unwrap();
                                    new_loop = Expr::new_let(data_sym, *orig_data, new_loop).unwrap();
                                    /* returning from the lambda function we passed to transform_up */
                                    return Some(new_loop);
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    });
}

/// Checks that `expr` performs only one `Merge` per control path - this guarantees
/// that the loop containing `expr`'s size can be inferred.
fn simple_merge(sym: &Symbol, expr: &Expr) -> bool {
    match expr.kind {
        Merge {
            ref builder,
            ref value,
        } => {
            if let Ident(ref s) = builder.kind {
                if s == sym {
                    return !value.contains_symbol(sym);
                }
            }
            false
        }
        If {
            ref cond,
            ref on_true,
            ref on_false,
        } => {
            !cond.contains_symbol(sym) && simple_merge(sym, on_true) && simple_merge(sym, on_false)
        }
        _ => false,
    }
}
