//! Vectorizes expressions in the AST.
//!
//! This transform marks expressions as vectorizable by converting expressions of type `Scalar`
//! into expressions of type `Vectorized`. It also modifies loops and builders to accept vector
//! arguments instead of scalar arguments.

use super::ast::*;
use super::ast::ExprKind::*;
use super::ast::Type::*;
use super::error::*;
use super::util::SymbolGenerator;

use std::collections::HashSet;

use super::exprs;

/// Vectorizes a type.
fn vectorized_type(ty: &Type) -> Type {
    if let Scalar(kind) = *ty {
        Vectorized(kind)
    } else {
        ty.clone()
    }
}

/// Returns `true` if this is a set of iterators we can vectorize, `false` otherwise.
/// 
/// We can vectorize an iterator if all of its iterators consume the entire collection.
fn vectorizable_iters(iters: &Vec<Iter<Type>>) -> bool {
    // Since we don't handle vectorizing Zips yet.
    if iters.len() > 1 {
        return false;
    }
    for ref iter in iters {
        if iter.start.is_some() || iter.end.is_some() || iter.stride.is_some() {
            return false;
        }
    }
    true
}

/// Vectorizes the expression by changing it's type if the expression is a scalar.
fn vectorize_expr(e: &mut Expr<Type>) {
    match e.kind {
        Literal(_) => {
            e.ty = vectorized_type(&e.ty);
        }
        Ident(_) => {
            e.ty = vectorized_type(&e.ty);
        }
        BinOp { .. } => {
            e.ty = vectorized_type(&e.ty);
        }
        _ => {},
    }
}


/// Checks basic vectorizability for a loop - this is a strong check which ensure that the only
/// expressions which appear in a function body are arithmetic, identifiers, literals,and Let
/// statements, and builder merges.
fn vectorizable(for_loop: &Expr<Type>) -> bool {
    if let For { ref iters, builder: ref init_builder, ref func } = for_loop.kind {
        // Check if the iterators are consumed.
        if vectorizable_iters(&iters) {
            // Check if the builder is newly initialized.
            if let NewBuilder(_) = init_builder.kind {
                // Check the builder.
                if let Builder(ref bk, _) = init_builder.ty {
                    match *bk {
                        BuilderKind::Merger(ref ty, _) => {
                            if let Scalar(_) = **ty {} else { return false; }
                        }
                        _ => {
                            return false;
                        }
                    };
                }

                if let Lambda { ref params, ref body } = func.kind {
                    let mut passed = true;

                    // Identifiers defined within the loop.
                    let mut defined_in_loop = HashSet::new();
                    for param in params.iter() {
                        defined_in_loop.insert(param.name.clone());
                    }

                    body.traverse(&mut |f| {
                        match f.kind {
                            Literal(_) => {},
                            Ident(_) => {},
                            BinOp{ .. } => {},
                            Let{ ref name, .. } => {
                                defined_in_loop.insert(name.clone()); 
                            },
                            Merge{ .. } => {},
                            _ => {
                                passed = false;
                            }
                        }
                    });

                    // Check if the index is used anywhere to do any kind of random access,
                    // index computation, etc.
                    let index_iden = exprs::ident_expr(params[1].name.clone(), params[1].ty.clone()).unwrap();
                    if body.contains(&index_iden) {
                        passed = false;
                    }

                    // If the data in the vector is not a Scalar, we can't vectorize it.
                    if let Scalar(_) = params[2].ty {} else {
                        passed = false;
                    }

                    // Check if there are identifiers defined outside the loop. If so, we need to
                    // broadcast them to vectorize them. We ignore this case for now and just bail!
                    body.traverse(&mut |e| {
                        match e.kind {
                            Ident(ref name) if !defined_in_loop.contains(name) => {
                                passed = false;
                            }
                            _ => {}
                        }
                    });

                    return passed;
                }
            }
        }
    }
    return false;
}

/// Vectorize an expression.
pub fn vectorize(expr: &mut Expr<Type>) -> WeldResult<()> {
    let mut vectorized = false;
    expr.transform_and_continue_res(&mut |ref mut expr| {
        //  The Res is a stricter-than-necessary check, but prevents us from having to check nested
        //  loops for now.
        if let Res { builder: ref for_loop } = expr.kind {
            if vectorizable(for_loop) {
                if let For { ref iters, builder: ref init_builder, ref func } = for_loop.kind {
                    if let NewBuilder(_) = init_builder.kind {
                        if let Lambda { ref params, ref body } = func.kind {
                            // This is the vectorized body.
                            let mut vectorized_body = body.clone();
                            vectorized_body.transform_and_continue(&mut |ref mut e| {
                                vectorize_expr(e);
                                (None, true)
                            });

                            let mut vectorized_params = params.clone();
                            vectorized_params[2].ty = vectorized_type(&vectorized_params[2].ty);

                            let vec_func = exprs::lambda_expr(vectorized_params, *vectorized_body)?;

                            // Pull out the iter into a let statement. This lets us repeat the
                            // iter via an identifier in the vectorized loop. Here, we just
                            // create the identifiers which refer to the data items.
                            let mut sym_gen = SymbolGenerator::from_expression(expr);

                            let data_names = iters.iter().map(|_| {
                                sym_gen.new_symbol("a")
                            })
                            .collect::<Vec<_>>();

                            // Iterators for the vectorized loop.
                            let mut vec_iters = vec![];
                            for (e, n) in iters.iter().zip(&data_names) {
                                vec_iters.push(
                                    Iter {
                                        data: Box::new(exprs::ident_expr(n.clone(), e.data.ty.clone())?),
                                        start: e.start.clone(),
                                        end: e.end.clone(),
                                        stride: e.stride.clone(),
                                        kind: IterKind::VectorIter,
                                    });
                            }

                            // Iterators for the fringe loop. This is the same set of iterators, but with the
                            // IteratorKind changed to Fringe.
                            let fringe_iters = vec_iters.iter_mut().map(|i| {
                                let mut i = i.clone();
                                i.kind = IterKind::FringeIter;
                                i
                            }).collect();

                            let vectorized_loop = exprs::for_expr(vec_iters, *init_builder.clone(), vec_func, true)?;
                            let scalar_loop = exprs::for_expr(fringe_iters, vectorized_loop, *func.clone(), false)?;
                            let result = exprs::result_expr(scalar_loop)?;

                            let mut prev_expr = result;
                            for (iter, name) in iters.iter().zip(data_names).rev() {
                                prev_expr = exprs::let_expr(name.clone(), *iter.data.clone(), prev_expr)?;
                            }

                            vectorized = true;
                            return Ok((Some(prev_expr), false));
                        }
                    }
                }
            }
        }
        // Check other expressions.
        Ok((None, true))
    });

    if vectorized {
        Ok(())
    } else {
        weld_err!("could not vectorize expression")
    }
}
