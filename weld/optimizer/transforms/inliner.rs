//! Various inlining transforms.
//!
//! These transforms take a set of nested expressions and fuse them into a single one.

extern crate fnv;

use ast::*;
use ast::ExprKind::*;

use fnv::FnvHashMap;

use std::mem;

#[cfg(test)]
use tests::*;

/// Inlines GetField(MakeStruct(*)) expressions, which can occur during loop fusion when some
/// of the loops are zipping together multiple column vectors.
pub fn inline_get_field(expr: &mut Expr) {
    expr.transform(&mut |ref mut expr| {
        if let GetField { ref expr, index } = expr.kind {
            if let MakeStruct { ref elems } = expr.kind {
                return Some(elems[index as usize].clone());
            }
        }
        None
    });
}

/// Inlines Zip expressions as collections of iters. Using Zips outside of a For loop is currently
/// unsupported behavior. This transform handles the simple case of converting Zips in macros
/// such as map and filter into Iters in For loops.
///
/// TODO(shoumik): Perhaps Zip should just be a macro? Then macros need to be ordered.
pub fn inline_zips(expr: &mut Expr) {
    expr.transform(&mut |ref mut e| {
        if let For {
                   ref mut iters,
                   ref builder,
                   ref func,
               } = e.kind {
            if iters.len() == 1 {
                let ref first_iter = iters[0];
                if let Zip { ref vectors } = first_iter.data.kind {
                    let new_iters = vectors
                        .iter()
                        .map(|v| {
                                 Iter {
                                     data: Box::new(v.clone()),
                                     start: None,
                                     end: None,
                                     stride: None,
                                     kind: first_iter.kind.clone(),
                                     shape: None,
                                     strides: None,
                                 }
                             })
                        .collect::<Vec<_>>();
                    return Some(Expr {
                                    ty: e.ty.clone(),
                                    kind: For {
                                        iters: new_iters,
                                        builder: builder.clone(),
                                        func: func.clone(),
                                    },
                                    annotations: Annotations::new(),
                                });
                }
            }
        }
        None
    });
}

/// Inlines Apply nodes whose argument is a Lambda expression. These often arise during macro
/// expansion but it's simpler to inline them before doing type inference.
/// Unlike many of the other transformations, we make this one independent of types so that
/// we can apply it before type inference.
///
/// Caveats:
/// - Functions that reuse a parameter twice have its expansion appear twice, instead of assigning
///   it to a temporary as would happen with function application.
/// - Does not complete inlining if some of the functions take functions as arguments (in that
///   case, the expressions after inlining may lead to more inlining).
pub fn inline_apply(expr: &mut Expr) {
    expr.transform(&mut |ref mut expr| {
        if let Apply {
                   ref func,
                   params: ref args,
               } = expr.kind {
            if let Lambda {
                       ref params,
                       ref body,
                   } = func.kind {
                let mut new = *body.clone();
                for (param, arg) in params.iter().zip(args) {
                    new.substitute(&param.name, &arg);
                }
                return Some(new);
            }
        }
        None
    });
}

pub fn inline_let(expr: &mut Expr) {
    expr.uniquify().unwrap();
    let ref mut usages = FnvHashMap::default();
    count_symbols(expr, usages);
    trace!("Symbol count: {:?}", usages);
    inline_let_helper(expr, usages)
}

#[derive(Debug)]
struct SymbolTracker {
    count: i32,
    loop_nest: i32,
    value: Option<Box<Expr>>,
}

impl Default for SymbolTracker {
    fn default() -> SymbolTracker {
        SymbolTracker {
            count: 0,
            loop_nest: 0,
            value: None
        }
    }
}

/// Count the occurances of each symbol defined by a `Let` statement.
fn count_symbols(expr: &Expr, usage: &mut FnvHashMap<Symbol, SymbolTracker>) {
    match expr.kind {
        For { ref func, .. } | Iterate { update_func: ref func, .. } | Sort { cmpfunc: ref func, .. } => {
            // Mark all symbols seen so far as "in a loop"
            for value in usage.values_mut() {
                value.loop_nest += 1;
            }

            count_symbols(func, usage);

            for value in usage.values_mut() {
                value.loop_nest -= 1;
            }
        }
        Let { ref name, .. } => {
            debug_assert!(!usage.contains_key(name));
            let _ = usage.insert(name.clone(), SymbolTracker::default());
        }
        Ident(ref symbol) => {
            if let Some(ref mut tracker) = usage.get_mut(symbol) {
                if tracker.loop_nest == 0 {
                    tracker.count += 1;
                } else {
                    // Used in a loop!
                    tracker.count += 3;
                }
            }
        }
        _ => ()
    };

    // Recurse into children - skip functions that expressions may call repeatedly. We handled
    // those already.
    for child in expr.children() {
        match child.kind {
            Lambda { .. } => (),
            _ => count_symbols(child, usage),
        }
    }
}

/// Inlines Let calls if the symbol defined by the Let statement is used
/// never or only one time.
fn inline_let_helper(expr: &mut Expr, usages: &mut FnvHashMap<Symbol, SymbolTracker>) {
    let mut taken_body = None;
    match expr.kind {
        Let { ref mut name, ref mut value, ref mut body } => {
            // Check whether the symbol is used one or fewer times.
            if let Some(tracker) = usages.get_mut(name) {
                if tracker.count <= 1 {
                    taken_body = Some(body.take());
                    tracker.value = Some(value.take());
                }
            }
        }
        Ident(ref name) => {
            // Check if the identifier maps to one that should be inlined.
            if let Some(tracker) = usages.get_mut(name) {
                if tracker.count <= 1 {
                    // Value should have been set by a preceding Let.
                    debug_assert!(tracker.value.is_some());
                    // Value should only be swapped once.
                    debug_assert!(!tracker.value.as_ref().unwrap().is_placeholder());
                    mem::swap(&mut taken_body, &mut tracker.value);
                }
            }
        }
        _ => ()
    }

    // Set the body to this expression.
    if taken_body.is_some() {
        mem::swap(expr, taken_body.unwrap().as_mut());
        inline_let_helper(expr, usages);
    } else {
        for child in expr.children_mut() {
            inline_let_helper(child, usages);
        }
    }
}

/// Changes negations of literal values to be literal negated values.
pub fn inline_negate(expr: &mut Expr) {
    use ast::LiteralKind::*;
    use ast::constructors::literal_expr;
    expr.transform(&mut |ref mut expr| {
        if let Negate(ref child_expr) = expr.kind {
            if let Literal(ref literal_kind) = child_expr.kind {
                let res = match  *literal_kind {
                    I8Literal(a) => Some(literal_expr(I8Literal(-a)).unwrap()),
                    I16Literal(a) => Some(literal_expr(I16Literal(-a)).unwrap()),
                    I32Literal(a) => Some(literal_expr(I32Literal(-a)).unwrap()),
                    I64Literal(a) => Some(literal_expr(I64Literal(-a)).unwrap()),
                    F32Literal(a) => Some(literal_expr(F32Literal((-f32::from_bits(a)).to_bits())).unwrap()),
                    F64Literal(a) => Some(literal_expr(F64Literal((-f64::from_bits(a)).to_bits())).unwrap()),
                    _ => None,
                };
                return res;
            }
        }
        None
    });
}

/// Inline casts.
///
/// This changes casts of literal values to be literal values of the casted type. It additionally
/// removes "self casts" (e.g., `i64(x: i64)` becomes `x`).
pub fn inline_cast(expr: &mut Expr) {
    use ast::Type::Scalar;
    use ast::ScalarKind::*;
    use ast::LiteralKind::*;
    use ast::constructors::literal_expr;
    expr.transform(&mut |ref mut expr| {
        if let Cast { kind: ref scalar_kind, ref child_expr } = expr.kind {
            if let Literal(ref literal_kind) = child_expr.kind {
                return match (scalar_kind, literal_kind) {
                    (&F64, &I32Literal(a)) => Some(literal_expr(F64Literal((a as f64).to_bits())).unwrap()),
                    (&I64, &I32Literal(a)) => Some(literal_expr(I64Literal(a as i64)).unwrap()),
                    (&F64, &I64Literal(a)) => Some(literal_expr(F64Literal((a as f64).to_bits())).unwrap()),
                    _ => None,
                }
            }
            if let Scalar(ref kind) = child_expr.ty {
                if kind == scalar_kind {
                    // XXX Tombstone and mem::swap here!!
                    return Some(*child_expr.clone());
                }
            }
        }
        None
    });
}

/// Checks if `expr` is a `GetField` on an identifier with name `sym`. If so,
/// returns the field index being accessed.
fn getfield_on_symbol(expr: &Expr, sym: &Symbol) -> Option<u32> {
    if let GetField { ref expr, ref index } = expr.kind {
        if let Ident(ref ident_name) = expr.kind {
            if sym == ident_name {
                return Some(*index);
            }
        }
    }
    None
}

/// Simplifies branches with `<expr> == False` to just be over <expr>`
///
/// This switches the true condition and the false condition.
pub fn simplify_branch_conditions(expr: &mut Expr) {
    use ast::BinOpKind;
    use ast::LiteralKind::BoolLiteral;
    expr.uniquify().unwrap();
    expr.transform_up(&mut |ref mut expr| {
        if let If { ref mut cond, ref mut on_true, ref mut on_false } = expr.kind {
            let mut taken = None;
            if let &mut BinOp { ref mut kind, ref mut left, ref mut right } = &mut cond.kind {
                if *kind == BinOpKind::Equal {
                    if let Literal(BoolLiteral(false)) = left.kind {
                        taken = Some(right.take());
                    } else if let Literal(BoolLiteral(false)) = right.kind {
                        taken = Some(left.take());
                    }
                }
            };

            if let Some(ref mut expr) = taken {
                mem::swap(cond, expr);
                mem::swap(on_true, on_false);
            }
        }
        // We just updated the expression in place instead of replacing it.
        None
    });
}

/// Changes struct definitions assigned to a name and only used in `GetField` operations
/// to `Let` definitions over the struct elements themselves.
///
/// This transformation is similar to a simple SROA (scalar replacement of aggregates) transform in
/// other compilers.
///
/// ## Example
///
/// let a = {1, 2, 3, 4};
/// a.$0 + a.$1 + a.$2
///
/// Becomes
///
/// let us = 1;
/// let us#1 = 2;
/// let us#1 = 3;
/// let us#1 = 4;
/// us + us#1 + us#2
///
pub fn unroll_structs(expr: &mut Expr) {
    use ast::constructors::*;
    use util::SymbolGenerator;

    expr.uniquify().unwrap();
    let mut sym_gen = SymbolGenerator::from_expression(expr);
    expr.transform_up(&mut |ref mut expr| {
        match expr.kind {
            Let { ref name, ref value, ref body } => {
                if let MakeStruct { ref elems } = value.kind {

                    // First, ensure that the name is not used anywhere but a `GetField`.
                    let mut total_count: i32 = 0;
                    let mut getstruct_count: i32 = 0;
                    body.traverse(&mut |ref e| {
                        if getfield_on_symbol(e, name).is_some() {
                                    getstruct_count += 1;
                        }
                        if let Ident(ref ident_name) = e.kind {
                            if ident_name == name {
                                total_count += 1;
                            }
                        }
                    });

                    // We used the struct somewhere else, so we can't safely get rid of it.
                    if total_count != getstruct_count {
                        return None;
                    }

                    let mut new_body = body.as_ref().clone();
                    let symbols: Vec<_> = elems.iter().map(|_| sym_gen.new_symbol("us")).collect();
                    // Replace the new_body with the symbol we assigned the struct element to.
                    new_body.transform(&mut |ref mut expr2| {
                        if let Some(index) = getfield_on_symbol(expr2, name) {
                            let sym = symbols.get(index as usize).unwrap().clone();
                            return Some(ident_expr(sym, expr2.ty.clone()).unwrap())
                        }
                        None
                    });

                    // Unroll the struct elements by assigning each one to a name.
                    let mut prev = new_body;
                    for (i, sym) in symbols.into_iter().enumerate().rev() {
                        prev = let_expr(sym, elems[i].clone(), prev).unwrap();
                    }
                    return Some(prev);
                }
            },
            _ => ()
        }
        None
    });
}

#[test]
fn inline_lets() {
    let mut e1 = typed_expression("let a = 1; a + 2");
    inline_let(&mut e1);
    let e2 = typed_expression("1 + 2");
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    let mut e1 = typed_expression("let a = 1; a + a + 2");
    // The transform should fail since the identifier is used more than once.
    inline_let(&mut e1);
    let e2 = typed_expression("let a = 1; a + a + 2");
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    let mut e1 = typed_expression("let a = 1L; for([1L,2L,3L], appender, |b,i,e| merge(b, e + a \
                                   + 2L))");
    inline_let(&mut e1);
    // The transform should fail since the identifier is used in a loop.
    let e2 = typed_expression("let a = 1L; for([1L,2L,3L], appender, |b,i,e| merge(b, e + a + \
                               2L))");
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    let mut e1 = typed_expression("let a = 1; let b = 2; let c = 3; a + b + c");
    inline_let(&mut e1);
    let e2 = typed_expression("1 + 2 + 3");
    println!("{}, {}", e1.pretty_print(), e2.pretty_print());
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    let mut e1 = typed_expression("|input: vec[i32]|
        let b = 1;
        result(for(input, merger[i32,+], |b,i,e| let a = 1; merge(b, e + a))) + b");
    inline_let(&mut e1);

    let e2 = typed_expression("|input: vec[i32]|
        result(for(input, merger[i32,+], |b,i,e| merge(b, e + 1))) + 1");
    println!("{}, {}", e1.pretty_print(), e2.pretty_print());
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    let mut e1 = typed_expression("|input: vec[i32]|
        let b = 1;
        result(for(input, merger[i32,+], |b,i,e| let a = 1; merge(b, e + a + a))) + b");
    inline_let(&mut e1);

    let e2 = typed_expression("|input: vec[i32]|
        result(for(input, merger[i32,+], |b,i,e| let a = 1; merge(b, e + a + a))) + 1");
    println!("{}, {}", e1.pretty_print(), e2.pretty_print());
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());
}
