//! Various inlining transforms. These transforms take a set of nested expressions
//! and fuse them into a single one.

use ast::*;
use ast::ExprKind::*;

use annotations::*;

use super::uniquify::uniquify;

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

/// Inlines Let calls if the symbol defined by the Let statement is used
/// never or only one time.
pub fn inline_let(expr: &mut Expr) {
    if let Ok(_) = uniquify(expr) {
        expr.transform(&mut |ref mut expr| {
            if let Let {
                ref mut name,
                ref mut value,
                ref mut body,
            } = expr.kind {
                if symbol_usage_count(name, body) <= 1 {
                    body.transform(&mut |ref mut expr| {
                        if let Ident(ref symbol) = expr.kind {
                            if symbol == name {
                                return Some(*value.clone());
                            }
                        }
                        return None;
                    });
                    return Some(*body.clone());
                }
            }
            return None;
        });
    }
}

/// Changes negations of literal values to be literal negated values.
pub fn inline_negate(expr: &mut Expr) {
    use ast::LiteralKind::*;
    use exprs::literal_expr;
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

/// Changes casts of literal values to be literal values of the casted type.
pub fn inline_cast(expr: &mut Expr) {
    use ast::ScalarKind::*;
    use ast::LiteralKind::*;
    use exprs::literal_expr;
    expr.transform(&mut |ref mut expr| {
        if let Cast { kind: ref scalar_kind, ref child_expr } = expr.kind {
            if let Literal(ref literal_kind) = child_expr.kind {
                return match (scalar_kind, literal_kind) {
                    (&F64, &I32Literal(a)) => Some(literal_expr(F64Literal((a as f64).to_bits())).unwrap()),
                    (&I64, &I32Literal(a)) => Some(literal_expr(I64Literal(a as i64)).unwrap()),
                    (&F64, &I64Literal(a)) => Some(literal_expr(F64Literal((a as f64).to_bits())).unwrap()),
                    (&I64, &I64Literal(a)) => Some(literal_expr(I64Literal(a as i64)).unwrap()),
                    _ => None,
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

/// Changes struct definitions assigned to a name and only used in `GetField` operations
/// to `Let` definitions over the struct elements themselves.
///
/// For example:
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
    use exprs::*;
    use util::SymbolGenerator;

    uniquify(expr).unwrap();
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


/// Count the occurances of a `Symbol` in an expression.
fn symbol_usage_count(sym: &Symbol, expr: &Expr) -> u32 {
    let mut usage_count = 0;
    expr.traverse(&mut |ref e| {
        if let For { ref func, .. } = e.kind {
            // The other child expressions of the For will be counted by traverse.
            if symbol_usage_count(sym, func) >= 1 {
                usage_count += 3;
            }
        } else if let Ident(ref symbol) = e.kind {
            if sym == symbol {
                usage_count += 1;
            }
        }
    });

    usage_count
}
