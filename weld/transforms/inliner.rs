//! Various inlining transforms. These transforms take a set of nested expressions
//! and fuse them into a single one.

use ast::*;
use ast::ExprKind::*;

use annotations::*;

use super::uniquify::uniquify;

/// Inlines GetField(MakeStruct(*)) expressions, which can occur during loop fusion when some
/// of the loops are zipping together multiple column vectors.
pub fn inline_get_field<T: TypeBounds>(expr: &mut Expr<T>) {
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
pub fn inline_zips(expr: &mut Expr<Type>) {
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
pub fn inline_apply<T: TypeBounds>(expr: &mut Expr<T>) {
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
/// less than one time.
pub fn inline_let(expr: &mut Expr<Type>) {
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

/// Count the occurances of a `Symbol` in an expression.
fn symbol_usage_count(sym: &Symbol, expr: &Expr<Type>) -> u32 {
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
