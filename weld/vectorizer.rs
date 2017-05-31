//! Vectorizes expressions in the AST.
//!
//! This transform marks expressions as vectorizable by converting expressions of type `Scalar`
//! into expressions of type `Vectorized`. It also modifies loops and builders to accept vector
//! arguments instead of scalar arguments.

use super::ast::*;
use super::ast::ExprKind::*;
use super::ast::Type::*;
use super::error::*;

/// Vectorizes a type.
fn vectorized_type(ty: &Type) -> Type {
    if let Scalar(kind) = *ty {
        return VectorizedScalar(kind);
    } else if let Builder(ref kind) = *ty {
        let vectorized_kind = match *kind {
            BuilderKind::Merger(ref t, ref op) => {
                BuilderKind::Merger(Box::new(vectorized_type(t)), *op)
            }
            // TODO(shoumik): need to throw an error here...
            _ => panic!("unsupported type"),
        };
        return VectorizedBuilder(vectorized_kind);
    }
    ty.clone()
}

/// Returns `true` if the type is a vectorizable builder and `false` otherwise.
fn vectorizable_builder(ty: &Type) -> bool {
    match *ty {
        Builder(BuilderKind::Merger(_, _)) => true,
        _ => false,
    }
}

/// Returns `true` if this is a set of iterators we can vectorize, `false` otherwise.
fn vectorizable_iters(iters: &Vec<Iter<Type>>) -> bool {
    for ref iter in iters {
        if iter.start.is_some() || iter.end.is_some() || iter.stride.is_some() {
            return false;
        }
    }
    true
}

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
    //      merge(B, x)
    expr.transform_and_continue(&mut |ref mut expr| {
        //  The Res is a stricter-than-necessary check, but prevents us from having to check nested
        //  loops for now.
        if let Res { builder: ref for_loop } = expr.kind {
            if let For { ref iters, builder: ref init_builder, ref func } = for_loop.kind {
                if vectorizable_iters(&iters) {
                    // TODO(shoumik): Check NewBuilder type - just support Merger.
                    if let NewBuilder(_) = init_builder.kind {
                        if vectorizable_builder(&init_builder.ty) {
                            if let Lambda { ref params, ref body } = func.kind {
                                let mut vectorized_body = body.clone();
                                vectorized_body.transform_and_continue(&mut |ref mut e| {
                            let vectorized = match e.kind {
                                // TODO - for readability, might want to factor this out into a
                                // function.
                                Literal(_) => {
                                    e.ty = vectorized_type(&e.ty);
                                    None
                                }
                                Merge { .. } => {
                                    e.ty = vectorized_type(&e.ty);
                                    // Do something fancy here -- need to check the type of the
                                    // merge, and the second argument, and handle it accordingly.
                                    // For now, let's just support Merger.
                                    None
                                }
                                Ident(_) => {
                                    e.ty = vectorized_type(&e.ty);
                                    None
                                }
                                BinOp { .. } => {
                                    e.ty = vectorized_type(&e.ty);
                                    None
                                }
                                _ => None,
                            };
                            return (vectorized, true);
                        });

                                // Replace the loop with a vectorized version.
                                // TODO(shoumik): add a fringe loop!

                                let vectorized_builder = Expr {
                                    kind: init_builder.kind.clone(),
                                    ty: vectorized_type(&init_builder.ty),
                                };

                                let mut vectorized_params = params.iter()
                                    .map(|ref p| {
                                        Parameter {
                                            name: p.name.clone(),
                                            ty: vectorized_type(&p.ty),
                                        }
                                    })
                                    .collect::<Vec<_>>();
                                // Don't vectorize the index...
                                vectorized_params[1] = params[1].clone();

                                let vectorized_func_ty = Function(vectorized_params.iter()
                                                                      .map(|ref p| p.ty.clone())
                                                                      .collect(),
                                                                  Box::new(vectorized_body.ty
                                                                      .clone()));
                                let vectorized_func = Expr {
                                    kind: Lambda {
                                        params: vectorized_params.clone(),
                                        body: vectorized_body,
                                    },
                                    ty: vectorized_func_ty,
                                };

                                let vectorized_loop = Expr {
                                    kind: For {
                                        iters: iters.clone(),
                                        builder: Box::new(vectorized_builder),
                                        func: Box::new(vectorized_func),
                                    },
                                    ty: vectorized_type(&for_loop.ty),
                                };

                                return (Some(Expr {
                                            kind: Res { builder: Box::new(vectorized_loop) },
                                            ty: expr.ty.clone(),
                                        }),
                                        false);
                            }
                        }
                    }
                }
            }
        }
        // Check other expressions.
        (None, true)
    });
    Ok(())
}
