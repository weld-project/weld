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
    for ref iter in iters {
        if iter.start.is_some() || iter.end.is_some() || iter.stride.is_some() {
            return false;
        }
    }
    true
}

/// Vectorizes the expression by changing it's type if the expression is a scalar.
fn vectorize_expr(e: &mut Expr<Type>) -> () {
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

/// Vectorize an expression.
pub fn vectorize(expr: &mut Expr<Type>) -> WeldResult<()> {
    let mut vectorized = false;
    expr.transform_and_continue_res(&mut |ref mut expr| {
        //  The Res is a stricter-than-necessary check, but prevents us from having to check nested
        //  loops for now.
        if let Res { builder: ref for_loop } = expr.kind {
            if let For { ref iters, builder: ref init_builder, ref func } = for_loop.kind {
                if vectorizable_iters(&iters) {
                    if let NewBuilder(_) = init_builder.kind {
                        if let Lambda { ref params, ref body } = func.kind {
                            // This is the vectorized body.
                            let mut vectorized_body = body.clone();
                            vectorized_body.transform_and_continue(&mut |ref mut e| {
                                // TODO this obviously needs to be stricter...
                                vectorize_expr(e);
                                (None, true)
                            });

                            let mut vectorized_params = params.clone();
                            vectorized_params[2].ty = vectorized_type(&vectorized_params[2].ty);

                            println!("lambda expr");
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
                            println!("iters expr");
                                vec_iters.push(
                                    Iter {
                                        data: Box::new(exprs::ident_expr(n.clone(), e.data.ty.clone())?),
                                        start: e.start.clone(),
                                        end: e.end.clone(),
                                        stride: e.stride.clone(),
                                    });
                            }

                            // Iterators for the fringe loop. This is the same set of iterators, but with the
                            // IteratorKind changed to Fringe.
                            let fringe_iters = vec_iters.clone();

                            println!("for expr");
                            let vectorized_loop = exprs::for_expr(vec_iters, *init_builder.clone(), vec_func, true);
                            if vectorized_loop.is_err() {
                                println!("{}", vectorized_loop.unwrap_err());
                                panic!("lol");
                            }
                            let vectorized_loop = vectorized_loop.unwrap();
                            println!("for expr");
                            let scalar_loop = exprs::for_expr(fringe_iters, vectorized_loop, *func.clone(), false)?;
                            println!("resultt expr");
                            let result = exprs::result_expr(scalar_loop)?;

                            let mut prev_expr = result;
                            for (iter, name) in iters.iter().zip(data_names).rev() {
                            println!("let expr");
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
