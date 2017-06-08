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

/// Vectorizes a type.
fn vectorized_type(ty: &Type) -> Type {
    if let Scalar(kind) = *ty {
        return VectorizedScalar(kind);
    } else if let Builder(ref kind, _) = *ty {
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
        Builder(BuilderKind::Merger(_, _), _) => true,
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

/// Returns (len(vec_expr) / VECWIDTH) * VECWIDTH
fn vector_end_index(vec_expr: &Expr<Type>) -> Expr<Type> {
    let len_expr = Expr {
        kind: Length { data: Box::new(vec_expr.clone()) },
        ty: Scalar(ScalarKind::I64),
        annotations: Annotations::new(),
    };

    let vecwidth_expr = Expr {
        kind: CompileTimeConstant(CompileTimeConstantKind::VectorWidth),
        ty: Scalar(ScalarKind::I64),
        annotations: Annotations::new(),
    };

    let div_expr = Expr {
        kind: BinOp {
            kind: BinOpKind::Divide,
            left: Box::new(len_expr.clone()),
            right: Box::new(vecwidth_expr.clone()),
        },
        ty: Scalar(ScalarKind::I64),
        annotations: Annotations::new(),
    };

    Expr {
        kind: BinOp {
            kind: BinOpKind::Multiply,
            left: Box::new(div_expr.clone()),
            right: Box::new(vecwidth_expr.clone()),
        },
        ty: Scalar(ScalarKind::I64),
        annotations: Annotations::new(),
    }
}

/// Vectorize an expression.
pub fn vectorize(expr: &mut Expr<Type>) -> WeldResult<()> {
    let mut vectorized = false;
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
                                        // TODO abort if not vectorizable?
                                        _ => None,
                                    };
                                    return (vectorized, true);
                                });

                                // Replace the loop with a vectorized version.
                                let vectorized_builder = Expr {
                                    kind: init_builder.kind.clone(),
                                    ty: vectorized_type(&init_builder.ty),
                                    annotations: Annotations::new(),
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
                                    annotations: Annotations::new(),
                                };

                                // Pull out the iter into a let statement. This lets us repeat the
                                // iter via an identifier in the vectorized loop. Here, we just
                                // create the identifiers which refer to the data items.
                                let mut sym_gen = SymbolGenerator::from_expression(expr);
                                let iter_names = iters.iter()
                                    .map(|_| {
                                        sym_gen.new_symbol("iterarg")
                                    })
                                .collect::<Vec<_>>();

                                let iter_idens = iters.iter().zip(&iter_names)
                                    .map(|(e, n)| {
                                        Box::new(Expr {
                                            kind: Ident(n.clone()),
                                            ty: e.data.ty.clone(),
                                            annotations: Annotations::new(),
                                        })
                                    })
                                .collect::<Vec<_>>();

                                let iters_vector_loop = iter_idens.iter()
                                    .map(|i| {
                                        Iter {
                                            data: Box::new(*i.clone()),
                                            start: Some(Box::new(Expr {
                                                kind: Literal(LiteralKind::I64Literal(0)),
                                                ty: Scalar(ScalarKind::I64),
                                                annotations: Annotations::new(),
                                            })),
                                            end: Some(Box::new(vector_end_index(i))),
                                            stride: Some(Box::new(Expr {
                                                kind: CompileTimeConstant(CompileTimeConstantKind::VectorWidth),
                                                ty: Scalar(ScalarKind::I64),
                                                annotations: Annotations::new(),
                                            })),
                                        }
                                    })
                                .collect::<Vec<_>>();

                                let iters_fringe_loop = iter_idens.iter()
                                    .map(|i| {
                                        Iter {
                                            data: Box::new(*i.clone()),
                                            start: Some(Box::new(vector_end_index(i))),
                                            end: Some(Box::new(Expr {
                                                kind: Length { data: Box::new(*i.clone()) },
                                                ty: Scalar(ScalarKind::I64),
                                                annotations: Annotations::new(),
                                            })),
                                            stride: Some(Box::new(Expr {
                                                kind: Literal(LiteralKind::I64Literal(1)),
                                                ty: Scalar(ScalarKind::I64),
                                                annotations: Annotations::new(),
                                            })),
                                        }
                                    })
                                .collect::<Vec<_>>();

                                let vectorized_loop = Expr {
                                    kind: For {
                                        iters: iters_vector_loop,
                                        builder: Box::new(vectorized_builder),
                                        func: Box::new(vectorized_func),
                                    },
                                    ty: vectorized_type(&for_loop.ty),
                                    annotations: Annotations::new(),
                                };

                                // The vectorized result, which we pass as the initializer for the
                                // serial loop.
                                let vectorized_result = Expr {
                                    kind: Res { builder: Box::new(vectorized_loop) },
                                    ty: expr.ty.clone(),
                                    annotations: Annotations::new(),
                                };

                                let scalar_loop = Expr {
                                    kind: For {
                                        iters: iters_fringe_loop,
                                        builder: Box::new(Expr {
                                            kind: NewBuilder(Some(Box::new(vectorized_result))),
                                            ty: for_loop.ty.clone(),
                                            annotations: Annotations::new(),
                                        }),
                                        func: func.clone(),
                                    },
                                    ty: for_loop.ty.clone(),
                                    annotations: Annotations::new(),
                                };

                                let final_loop = Expr {
                                    kind: Res { builder: Box::new(scalar_loop) },
                                    ty: expr.ty.clone(),
                                    annotations: Annotations::new(),
                                };

                                let mut prev_expr = final_loop;
                                for (iter, name) in iters.iter().zip(iter_names).rev() {
                                    let ty = prev_expr.ty.clone();
                                    prev_expr = Expr {
                                        kind: Let { name: name.clone(), value: iter.data.clone(), body: Box::new(prev_expr) },
                                        ty: ty,
                                        annotations: Annotations::new(),
                                    };
                                }

                                vectorized = true;
                                return (Some(prev_expr), false);
                            }
                        }
                    }
                }
            }
        }
        // Check other expressions.
        (None, true)
    });

    if vectorized {
        Ok(())
    } else {
        weld_err!("could not vectorize expression")
    }
}
