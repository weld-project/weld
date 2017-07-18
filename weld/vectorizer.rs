//! Vectorizes expressions in the AST.
//!
//! This transform marks expressions as vectorizable by converting expressions of type `Scalar`
//! into expressions of type `Simd`. It also modifies loops and builders to accept vector
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
        Simd(kind)
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
        if let Vector(ref elem_ty) = iter.data.ty {
            if let Scalar(_) = *elem_ty.as_ref() {}
            else {
                return false;
            }
        }
        if iter.kind != IterKind::ScalarIter {
            return false;
        }
    }
    true
}

/// Vectorizes the expression by changing its type if the expression is a scalar.
fn vectorize_expr(e: &mut Expr<Type>, broadcast_idens: &HashSet<Symbol>) -> WeldResult<bool> {
    let mut new_expr = None;
    let mut cont = true;

    match e.kind {
        Literal(_) => {
            e.ty = vectorized_type(&e.ty);
        }
        Ident(ref name) => {
            if let Scalar(_) = e.ty {
                //  The identifier is a scalar defined outside the loop body, so we need to broadcast
                //  it into a vector.
                if broadcast_idens.contains(&name) {
                    // Don't continue if we replace this expression.
                    new_expr = Some(exprs::broadcast_expr(e.clone())?);
                    cont = false;
                } else {
                    e.ty = vectorized_type(&e.ty);
                }
            } else if let Struct(ref mut field_tys) = e.ty {
                for ty in field_tys.iter_mut() {
                    *ty = vectorized_type(&ty);
                }
            }
        }
        GetField { .. } => {
            e.ty = vectorized_type(&e.ty);
        }
        BinOp { .. } => {
            e.ty = vectorized_type(&e.ty);
        }
        Select { .. } => {
            e.ty = vectorized_type(&e.ty);
        }
        // Predication for a value merged into a merger. This pattern checks for if(cond, merge(b, e), b).
        If { ref cond, ref on_true, ref on_false } => {
            if let Merge { ref builder, ref value } = on_true.kind {
                if let Ident(ref name) = on_false.kind {
                    if let Ident(ref name2) = builder.kind {
                        if name == name2 {
                            if let Builder(ref bk, _) = builder.ty {
                                if let BuilderKind::Merger(ref ty, ref op) = *bk {
                                    if let Scalar(ref sk) = *ty.as_ref() {
                                        let identity = match *op {
                                            BinOpKind::Add => {
                                                match *sk {
                                                    ScalarKind::I8 => exprs::literal_expr(LiteralKind::I8Literal(0))?,
                                                    ScalarKind::I32 => exprs::literal_expr(LiteralKind::I32Literal(0))?,
                                                    ScalarKind::I64 => exprs::literal_expr(LiteralKind::I64Literal(0))?,
                                                    ScalarKind::F32 => exprs::literal_expr(LiteralKind::F32Literal(0.0))?,
                                                    ScalarKind::F64 => exprs::literal_expr(LiteralKind::F64Literal(0.0))?,
                                                    _ => {
                                                        return weld_err!("Predication not supported");
                                                    }
                                                }
                                            }
                                            BinOpKind::Multiply => {
                                                match *sk {
                                                    ScalarKind::I8 => exprs::literal_expr(LiteralKind::I8Literal(1))?,
                                                    ScalarKind::I32 => exprs::literal_expr(LiteralKind::I32Literal(1))?,
                                                    ScalarKind::I64 => exprs::literal_expr(LiteralKind::I64Literal(1))?,
                                                    ScalarKind::F32 => exprs::literal_expr(LiteralKind::F32Literal(1.0))?,
                                                    ScalarKind::F64 => exprs::literal_expr(LiteralKind::F64Literal(1.0))?,
                                                    _ => {
                                                        return weld_err!("Predication not supported");
                                                    }
                                                }
                                            }
                                            _ => {
                                                return weld_err!("Merger type not vectorizable.");
                                            }
                                        };
                                        // Change if(cond, merge(b, e), b) => 
                                        // merge(b, select(cond, e, identity).
                                        let mut expr = exprs::merge_expr(*builder.clone(), exprs::select_expr(*cond.clone(), *value.clone(), identity)?)?;
                                        expr.transform_and_continue(&mut |ref mut e| {
                                            let cont = vectorize_expr(e, broadcast_idens).unwrap();
                                            (None, cont)
                                        });
                                        new_expr = Some(expr);
                                        // We already vectorized this subexpression.
                                        cont = false;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        _ => {},
    }

    if new_expr.is_some() {
        *e = new_expr.unwrap();
    }

    Ok(cont)
}


/// Checks basic vectorizability for a loop - this is a strong check which ensure that the only
/// expressions which appear in a function body are vectorizable expressions (see
/// `docs/vectorization.md` for details) 
fn vectorizable(for_loop: &Expr<Type>) -> WeldResult<HashSet<Symbol>> {
    if let For { ref iters, builder: ref init_builder, ref func } = for_loop.kind {
        // Check if the iterators are consumed.
        if vectorizable_iters(&iters) {
            // Check if the builder is newly initialized.
            if let NewBuilder(_) = init_builder.kind {
                // Check the builder.
                if let Builder(ref bk, _) = init_builder.ty {
                    match *bk {
                        BuilderKind::Merger(ref ty, _) => {
                            if let Scalar(_) = **ty {} else { return weld_err!("Unsupported builder"); }
                        }
                        _ => {
                            return weld_err!("Unsupported builder");
                        }
                    };
                }

                // Check the loop function.
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
                            Ident(ref name) => {
                                if f.ty == params[1].ty && *name == params[1].name {
                                    // Used an index expression in the loop body.
                                    passed = false;
                                }
                            },
                            BinOp{ .. } => {},
                            Let{ ref name, .. } => {
                                defined_in_loop.insert(name.clone()); 
                            },
                            // GetField is allowed if the only fields we fetch are the ones from
                            // the argument (in case the input was Zipped). At this point, it is
                            // already guaranteed that each input vector is a scalar.
                            GetField { ref expr, .. } => {
                                let mut getfield_passed = false;
                                let ref elem_param = params[2];
                                if let Ident(ref name) = expr.kind {
                                    if elem_param.name == *name {
                                       getfield_passed = true; 
                                    }
                                }
                                passed = getfield_passed;
                            }
                            Merge{ .. } => {},
                            If { ref on_true, ref on_false, .. } => {
                                let mut can_predicate = false;
                                if let Merge { ref builder, .. } = on_true.kind {
                                    if let Ident(ref name) = on_false.kind {
                                        if let Ident(ref name2) = builder.kind {
                                            if name == name2 {
                                                if let Builder(ref bk, _) = builder.ty {
                                                    if let BuilderKind::Merger(ref ty, _) = *bk {
                                                        if let Scalar(_) = *ty.as_ref() {
                                                            can_predicate = true;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                passed = can_predicate;
                            }
                            _ => {
                                passed = false;
                            }
                        }
                    });

                    if !passed {
                        return weld_err!("Unsupported pattern");
                    }

                    // If the data in the vector is not a Scalar, we can't vectorize it.
                    let mut check_arg_ty = false;
                    if let Scalar(_) = params[2].ty {
                        check_arg_ty = true; 
                    } 
                    else if let Struct(ref field_tys) = params[2].ty {
                        if field_tys.iter().all(|t| match *t {
                            Scalar(_) => true,
                            _ => false
                        }) {
                            check_arg_ty = true;
                        }
                    } 
                    
                    if !check_arg_ty {
                        return weld_err!("Unsupported type");
                    }

                    let mut idens = HashSet::new();

                    // Check if there are identifiers defined outside the loop. If so, we need to
                    // broadcast them to vectorize them.
                    let mut passed = true;
                    body.traverse(&mut |e| {
                        match e.kind {
                            Ident(ref name) if !defined_in_loop.contains(name) => {
                                if let Scalar(_) = e.ty {
                                    idens.insert(name.clone());
                                } else {
                                    passed = false;
                                }
                            }
                            _ => {}
                        }
                    });

                    if !passed {
                        return weld_err!("Unsupporte pattern: non-scalar identifier that must be broadcast");
                    }
                    return Ok(idens);
                }
            }
        }
    }
    return weld_err!("Unsupported pattern");
}

/// Vectorize an expression.
pub fn vectorize(expr: &mut Expr<Type>) {
    let mut vectorized = false;
    expr.transform_and_continue_res(&mut |ref mut expr| {
        //  The Res is a stricter-than-necessary check, but prevents us from having to check nested
        //  loops for now.
        if let Res { builder: ref for_loop } = expr.kind {
            let ref broadcast_idens = vectorizable(for_loop)?;
            if let For { ref iters, builder: ref init_builder, ref func } = for_loop.kind {
                if let NewBuilder(_) = init_builder.kind {
                    if let Lambda { ref params, ref body } = func.kind {
                        // This is the vectorized body.
                        let mut vectorized_body = body.clone();
                        vectorized_body.transform_and_continue(&mut |ref mut e| {
                            let cont = vectorize_expr(e, broadcast_idens).unwrap();
                            (None, cont)
                        });

                        let mut vectorized_params = params.clone();

                        let new_ty = if let Scalar(_) = vectorized_params[2].ty {
                            vectorized_type(&vectorized_params[2].ty)
                        } else if let Struct(ref field_tys) = vectorized_params[2].ty {
                            Struct(field_tys.iter().map(|ref t| vectorized_type(t)).collect())
                        } else {
                            unreachable!();
                        };
                        vectorized_params[2].ty = new_ty;

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
                                    kind: IterKind::SimdIter,
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
        // Check other expressions.
        Ok((None, true))
    });
}

