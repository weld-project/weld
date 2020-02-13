//! Vectorizes expressions in the AST.
//!
//! This transform marks expressions as vectorizable by converting expressions of type `Scalar`
//! into expressions of type `Simd`. It also modifies loops and builders to accept vector
//! arguments instead of scalar arguments.

use std::collections::HashSet;

use crate::ast::ExprKind::*;
use crate::ast::Type::*;
use crate::ast::*;
use crate::error::*;
use crate::util::SymbolGenerator;

#[cfg(test)]
use crate::tests::*;

/// Checks whether an annotation specifies predication.
pub trait ShouldPredicate {
    fn should_predicate(&self) -> bool;
}

impl ShouldPredicate for Expr {
    fn should_predicate(&self) -> bool {
        // If the annotation says we should predicate, always do it.
        if let Some(ref value) = self.annotations.get("predicate") {
            return value.to_lowercase() == "true";
        }
        false
    }
}

/// Vectorize an expression.
pub fn vectorize(expr: &mut Expr) {
    let mut vectorized = false;
    // Used to create the identifiers which refer to the data items. These identifiers are
    // used to pull out the iter into a let statement. This lets us repeat the iter via an
    // identifier in the vectorized loop later. Declaring this before any transformations so
    // there is no clash of variable names.
    let mut sym_gen = SymbolGenerator::from_expression(expr);

    expr.transform_and_continue_res(&mut |ref mut expr| {
        if let Some(ref broadcast_idens) = vectorizable(expr) {
            info!("Vectorizing For loop!");
            if let For {
                ref iters,
                builder: ref init_builder,
                ref func,
            } = expr.kind
            {
                if let Lambda {
                    ref params,
                    ref body,
                } = func.kind
                {
                    // This is the vectorized body.
                    let mut vectorized_body = body.clone();
                    vectorized_body.transform_and_continue(&mut |ref mut e| {
                        let cont = vectorize_expr(e, broadcast_idens).unwrap();
                        (None, cont)
                    });

                    let mut vectorized_params = params.clone();
                    vectorized_params[2].ty = vectorized_params[2].ty.simd_type()?;

                    let vec_func = Expr::new_lambda(vectorized_params, *vectorized_body)?;

                    let data_names = iters
                        .iter()
                        .map(|_| sym_gen.new_symbol("a"))
                        .collect::<Vec<_>>();

                    // Iterators for the vectorized loop.
                    let mut vec_iters = vec![];
                    for (e, n) in iters.iter().zip(&data_names) {
                        vec_iters.push(Iter {
                            data: Box::new(Expr::new_ident(n.clone(), e.data.ty.clone())?),
                            start: e.start.clone(),
                            end: e.end.clone(),
                            stride: e.stride.clone(),
                            kind: IterKind::SimdIter,
                            shape: e.shape.clone(),
                            strides: e.strides.clone(),
                        });
                    }

                    // Iterators for the fringe loop. This is the same set of iterators, but with the
                    // IteratorKind changed to Fringe.
                    let fringe_iters = vec_iters
                        .iter_mut()
                        .map(|i| {
                            let mut i = i.clone();
                            i.kind = IterKind::FringeIter;
                            i
                        })
                        .collect();

                    let vectorized_loop =
                        Expr::new_for(vec_iters, *init_builder.clone(), vec_func)?;
                    let scalar_loop = Expr::new_for(fringe_iters, vectorized_loop, *func.clone())?;
                    let mut prev_expr = scalar_loop;
                    for (iter, name) in iters.iter().zip(data_names).rev() {
                        prev_expr = Expr::new_let(name.clone(), *iter.data.clone(), prev_expr)?;
                    }

                    vectorized = true;
                    return Ok((Some(prev_expr), false));
                }
            }
        }
        // Check other expressions.
        Ok((None, true))
    });
}

/// Predicate an `If` expression by checking for if(cond, merge(b, e), b) and transforms it to
/// merge(b, select(cond, e,identity)).
pub fn predicate_merge_expr(e: &mut Expr) {
    e.transform_and_continue_res(&mut |ref mut e| {
        if !e.should_predicate() {
            return Ok((None, true));
        }

        // Predication for a value merged into a merger. This pattern checks for if(cond, merge(b, e), b).
        if let If {
            ref cond,
            ref on_true,
            ref on_false,
        } = e.kind
        {
            if let Merge {
                ref builder,
                ref value,
            } = on_true.kind
            {
                if let Ident(ref name) = on_false.kind {
                    if let Ident(ref name2) = builder.kind {
                        if name == name2 {
                            if let Builder(ref bk, _) = builder.ty {
                                // Merge in the identity element if the predicate fails (effectively merging in nothing)
                                let (ty, op) = match *bk {
                                    BuilderKind::Merger(ref ty, ref op) => (ty, op),
                                    BuilderKind::DictMerger(_, ref ty2, ref op) => (ty2, op),
                                    BuilderKind::VecMerger(ref ty, ref op) => (ty, op),
                                    _ => {
                                        return Ok((None, true));
                                    }
                                };

                                let identity = get_id_element(ty.as_ref(), *op)?;
                                match identity {
                                    Some(x) => {
                                        match *bk {
                                            BuilderKind::Merger(_, _)
                                            | BuilderKind::VecMerger(_, _) => {
                                                /* Change if(cond, merge(b, e), b) =>
                                                merge(b, select(cond, e, identity). */
                                                let expr = Expr::new_merge(
                                                    *builder.clone(),
                                                    Expr::new_select(
                                                        *cond.clone(),
                                                        *value.clone(),
                                                        x,
                                                    )?,
                                                )?;
                                                return Ok((Some(expr), true));
                                            }
                                            BuilderKind::DictMerger(_, _, _) => {
                                                /* For dictmerger, need to match identity element
                                                back to the key. */
                                                let sel_expr = make_select_for_kv(
                                                    *cond.clone(),
                                                    *value.clone(),
                                                    x,
                                                )?;
                                                return Ok((sel_expr, true));
                                            }
                                            _ => {
                                                return Ok((None, true));
                                            }
                                        }
                                    }
                                    None => {
                                        return Ok((None, true));
                                    }
                                };
                            }
                        }
                    }
                }
            }
        }
        Ok((None, true))
    });
}

fn is_simple(e: &Expr) -> bool {
    match e.kind {
        Ident(_) | Literal(_) => true,
        GetField { ref expr, .. } => is_simple(expr),
        _ => false,
    }
}

/// Predicate an `If` expression by checking for if(cond, scalar1, scalar2) and transforms it to select(cond, scalar1, scalar2).
///
/// Since this predicates only simple "zero-cost" expressions, it's always done.
pub fn predicate_simple_expr(e: &mut Expr) {
    e.transform_and_continue_res(&mut |ref mut e| {
        // This pattern checks for if(cond, scalar1, scalar2).
        if let If {
            ref cond,
            ref on_true,
            ref on_false,
        } = e.kind
        {
            // Check if any sub-expression has a builder; if so bail out in order to not break linearity.
            let mut safe = true;
            on_true.traverse(&mut |ref sub_expr| {
                if sub_expr.kind.is_builder_expr() {
                    safe = false;
                }
            });
            on_false.traverse(&mut |ref sub_expr| {
                if sub_expr.kind.is_builder_expr() {
                    safe = false;
                }
            });
            if !safe {
                return Ok((None, true));
            }

            // Make sure the expression is "simple": for now, that's literals, getfields, and
            // identifiers
            if !(is_simple(on_true) && is_simple(on_false)) {
                return Ok((None, true));
            }

            if let Scalar(_) = on_true.ty {
                if let Scalar(_) = on_false.ty {
                    let expr =
                        Expr::new_select(*cond.clone(), *on_true.clone(), *on_false.clone())?;
                    return Ok((Some(expr), true));
                }
            }
        }
        Ok((None, true))
    });
}

/// Returns `true` if this is a set of iterators we can vectorize, `false` otherwise.
///
/// We can vectorize an iterator if all of its iterators consume the entire collection.
fn vectorizable_iters(iters: &[Iter]) -> bool {
    iters.iter().all(|ref iter| {
        iter.start.is_none()
            && iter.end.is_none()
            && iter.stride.is_none()
            && match iter.data.ty {
                Vector(ref elem) if elem.is_scalar() => true,
                _ => false,
            }
    })
}

/// Vectorizes an expression in-place, also changing its type if needed.
fn vectorize_expr(e: &mut Expr, broadcast_idens: &HashSet<Symbol>) -> WeldResult<bool> {
    let mut new_expr = None;
    let mut cont = true;

    match e.kind {
        Literal(_) => {
            e.ty = e.ty.simd_type()?;
        }
        Ident(ref name) => {
            if let Scalar(_) = e.ty {
                //  The identifier is a scalar defined outside the loop body, so we need to broadcast
                //  it into a vector.
                if broadcast_idens.contains(&name) {
                    // Don't continue if we replace this expression.
                    new_expr = Some(Expr::new_broadcast(e.clone())?);
                    cont = false;
                } else {
                    e.ty = e.ty.simd_type()?;
                }
            } else if let Struct(_) = e.ty {
                e.ty = e.ty.simd_type()?;
            }
        }
        GetField { .. } => {
            e.ty = e.ty.simd_type()?;
        }
        UnaryOp { .. } => {
            e.ty = e.ty.simd_type()?;
        }
        BinOp { .. } => {
            e.ty = e.ty.simd_type()?;
        }
        Select { .. } => {
            e.ty = e.ty.simd_type()?;
        }
        MakeStruct { .. } => {
            e.ty = e.ty.simd_type()?;
        }
        _ => {}
    }

    if let Some(val) = new_expr {
        *e = val;
    }
    Ok(cont)
}

/// Checks if the expression is a vectorizable newbuilder.
/// Returns Some(true) if it is an appender, merger or a struct of builders with at least one appender or merger.
/// Returns Some(false) if it is a builder, struct of builders, but no appender or mergers.
/// Returns None if it is not a builder.
fn vectorizable_builder(expr: &Expr) -> Option<bool> {
    use crate::ast::BuilderKind::*;
    match expr.kind {
        Ident(_) | NewBuilder(_) => {
            if let Builder(ref bk, _) = expr.ty {
                match *bk {
                    Appender(ref elem) | Merger(ref elem, _) => Some(elem.is_scalar()),
                    _ => Some(false),
                }
            } else {
                None
            }
        }
        MakeStruct { ref elems } => {
            let mut vectorizable = true;
            for elem in elems.iter() {
                match vectorizable_builder(elem) {
                    Some(val) => vectorizable &= val,
                    None => return None,
                }
            }
            Some(vectorizable)
        }
        _ => None,
    }
}

/// Checks basic vectorizability for a loop - this is a strong check which ensure that the only
/// expressions which appear in a function body are vectorizable expressions (see
/// `docs/internals/vectorization.md` for details)
fn vectorizable(for_loop: &Expr) -> Option<HashSet<Symbol>> {
    if let For {
        ref iters,
        builder: ref init_builder,
        ref func,
    } = for_loop.kind
    {
        // Check if the iterators are consumed.
        if vectorizable_iters(&iters) {
            // Check if at least one of the builders can be vectorized.
            if let Some(true) = vectorizable_builder(init_builder) {
                // Check the loop function.
                if let Lambda {
                    ref params,
                    ref body,
                } = func.kind
                {
                    let mut passed = true;

                    // Identifiers defined within the loop.
                    let mut defined_in_loop = HashSet::new();
                    for param in params.iter() {
                        defined_in_loop.insert(param.name.clone());
                    }

                    // Check if subexpressions in the body are all vectorizable.
                    body.traverse(&mut |f| {
                        if passed {
                            match f.kind {
                                Literal(_) => {}

                                Ident(ref name) => {
                                    if f.ty == params[1].ty && *name == params[1].name {
                                        // Used an index expression in the loop body.
                                        passed = false;
                                    }
                                }

                                UnaryOp { .. } => {}
                                BinOp { .. } => {}

                                Let { ref name, .. } => {
                                    defined_in_loop.insert(name.clone());
                                }

                                // TODO: do we want to allow all GetFields and MakeStructs, or look inside them?
                                GetField { .. } => {}

                                MakeStruct { .. } => {}

                                Merge { .. } => {}

                                Select { .. } => {}

                                _ => {
                                    passed = false;
                                }
                            }
                        }
                    });

                    if !passed {
                        trace!("Vectorization failed due to unsupported expression in loop body");
                        return None;
                    }

                    // If the data in the vector is not a Scalar, we can't vectorize it.
                    let mut check_arg_ty = false;
                    if let Scalar(_) = params[2].ty {
                        check_arg_ty = true;
                    } else if let Struct(ref field_tys) = params[2].ty {
                        if field_tys.iter().all(|t| match *t {
                            Scalar(_) => true,
                            _ => false,
                        }) {
                            check_arg_ty = true;
                        }
                    }

                    if !check_arg_ty {
                        trace!("Vectorization failed due to unsupported type");
                        return None;
                    }

                    let mut idens = HashSet::new();

                    // Check if there are identifiers defined outside the loop. If so, we need to
                    // broadcast them to vectorize them.
                    let mut passed = true;
                    body.traverse(&mut |e| match e.kind {
                        Ident(ref name) if !defined_in_loop.contains(name) => {
                            if let Scalar(_) = e.ty {
                                idens.insert(name.clone());
                            } else {
                                passed = false;
                            }
                        }
                        _ => {}
                    });

                    if !passed {
                        trace!("Unsupported pattern: non-scalar identifier that must be broadcast");
                        return None;
                    }
                    return Some(idens);
                }
            }
        }
    }
    trace!("Vectorization failed due to unsupported pattern");
    None
}

fn get_id_element(ty: &Type, op: BinOpKind) -> WeldResult<Option<Expr>> {
    let sk = &match *ty {
        Scalar(sk) => sk,
        _ => {
            return Ok(None);
        }
    };

    /* Dummy element to merge when predicate fails. */
    let identity = match op {
        BinOpKind::Add => match *sk {
            ScalarKind::I8 => Expr::new_literal(LiteralKind::I8Literal(0))?,
            ScalarKind::I32 => Expr::new_literal(LiteralKind::I32Literal(0))?,
            ScalarKind::I64 => Expr::new_literal(LiteralKind::I64Literal(0))?,
            ScalarKind::F32 => Expr::new_literal(LiteralKind::F32Literal(0f32.to_bits()))?,
            ScalarKind::F64 => Expr::new_literal(LiteralKind::F64Literal(0f64.to_bits()))?,
            _ => {
                return Ok(None);
            }
        },
        BinOpKind::Multiply => match *sk {
            ScalarKind::I8 => Expr::new_literal(LiteralKind::I8Literal(1))?,
            ScalarKind::I32 => Expr::new_literal(LiteralKind::I32Literal(1))?,
            ScalarKind::I64 => Expr::new_literal(LiteralKind::I64Literal(1))?,
            ScalarKind::F32 => Expr::new_literal(LiteralKind::F32Literal(1f32.to_bits()))?,
            ScalarKind::F64 => Expr::new_literal(LiteralKind::F64Literal(1f64.to_bits()))?,
            _ => {
                return Ok(None);
            }
        },
        _ => {
            return Ok(None);
        }
    };
    Ok(Some(identity))
}

fn make_select_for_kv(cond: Expr, kv: Expr, ident: Expr) -> WeldResult<Option<Expr>> {
    let mut sym_gen = SymbolGenerator::from_expression(&kv);
    let name = sym_gen.new_symbol("k");

    let kv_struct = Expr::new_ident(name.clone(), kv.ty.clone())?;
    let kv_ident = Expr::new_make_struct(vec![Expr::new_get_field(kv_struct.clone(), 0)?, ident])?; // use the original key and the identity as the value

    let sel = Expr::new_select(cond, kv_struct, kv_ident)?;
    let le = Expr::new_let(name, kv, sel)?; /* avoid copying key */
    Ok(Some(le))
}

/// Check whether a function has a vectorized Merge call. We'll use this to check whether function
/// bodies got vectorized.
#[cfg(test)]
fn has_vectorized_merge(expr: &Expr) -> bool {
    let mut found = false;
    expr.traverse(&mut |ref e| {
        if let Merge { ref value, .. } = e.kind {
            found |= value.ty.is_simd();
        }
    });
    found
}

#[test]
fn simple_merger() {
    let mut e =
        typed_expression("|v:vec[i32]| result(for(v, merger[i32,+], |b,i,e| merge(b,e+1)))");
    vectorize(&mut e);
    assert!(has_vectorized_merge(&e));
}

#[test]
fn predicated_merger() {
    let mut e = typed_expression("|v:vec[i32]| result(for(v, merger[i32,+], |b,i,e| @(predicate:true)if(e>0, merge(b,e), b)))");
    predicate_merge_expr(&mut e);
    vectorize(&mut e);
    assert!(has_vectorized_merge(&e));
}

#[test]
fn unpredicated_merger() {
    // This one shouldn't be vectorized since we didn't predicate it.
    let mut e = typed_expression(
        "|v:vec[i32]| result(for(v, merger[i32,+], |b,i,e| if(e>0, merge(b,e), b)))",
    );
    vectorize(&mut e);
    assert!(!has_vectorized_merge(&e));
}

#[test]
fn simple_appender() {
    let mut e =
        typed_expression("|v:vec[i32]| result(for(v, appender[i32], |b,i,e| merge(b,e+1)))");
    vectorize(&mut e);
    assert!(has_vectorized_merge(&e));
}

#[test]
fn predicated_appender() {
    // This code should NOT be vectorized because we can't predicate merges into vecbuilder.
    let mut e = typed_expression("|v:vec[i32]| result(for(v, appender[i32], |b,i,e| @(predicate:true)if(e>0, merge(b,e), b)))");
    predicate_merge_expr(&mut e);
    vectorize(&mut e);
    assert!(!has_vectorized_merge(&e));
}

#[test]
fn non_vectorizable_type() {
    // This code should NOT be vectorized because we can't vectorize merges of vectors.
    let mut e =
        typed_expression("|v:vec[i32]| result(for(v, appender[vec[i32]], |b,i,e| merge(b,v)))");
    vectorize(&mut e);
    assert!(!has_vectorized_merge(&e));
}

#[test]
fn non_vectorizable_expr() {
    // This code should NOT be vectorized because we can't vectorize lookup().
    let mut e = typed_expression(
        "|v:vec[i32]| result(for(v, appender[i32], |b,i,e| merge(b,lookup(v,i))))",
    );
    vectorize(&mut e);
    assert!(!has_vectorized_merge(&e));
}

#[test]
fn zipped_input() {
    let mut e = typed_expression(
        "|v:vec[i32]| result(for(zip(v,v), appender[i32], |b,i,e| merge(b,e.$0+e.$1)))",
    );
    vectorize(&mut e);
    assert!(has_vectorized_merge(&e));
}

// Pointless test as dictmerger cannot be vectorized anyway.
// #[test]
// fn zips_in_body() {
//     let mut e = typed_expression("|v:vec[i32]| result(for(v, dictmerger[{i32,i32},i32,+], |b,i,e| merge(b,{{e,e},e})))");
//     vectorize(&mut e);
//     assert!(has_vectorized_merge(&e));
// }
