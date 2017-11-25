//! Implements size inference for `For` loops.

use ast::*;
use ast::ExprKind::*;
use ast::Type::*;
use ast::BuilderKind::*;
use exprs::*;

struct NewAppender<'a> {
    elem_type: &'a Type
}

impl<'a> NewAppender<'a> {
    fn extract(expr: &'a TypedExpr) -> Option<NewAppender<'a>> {
        if let NewBuilder(_) = expr.kind {
            if let Builder(Appender(ref elem_type), _) = expr.ty {
                return Some(NewAppender{elem_type})
            }
        }
        return None
    }
}

fn newbuilder_with_size(builder: &TypedExpr, vector: &TypedExpr) -> Option<TypedExpr> {
    if let Some(na) = NewAppender::extract(builder) {
        let bk = Appender(Box::new(na.elem_type.clone()));
        let inferred_size = length_expr(vector.clone()).unwrap();
        return Some(newbuilder_expr(bk, Some(inferred_size)).unwrap());
    }
    if let MakeStruct { ref elems } = builder.kind {
        if elems.iter().all(|ref e| NewAppender::extract(e).is_some()) {
            let newbuilders = elems.iter().map(|ref e| newbuilder_with_size(e, vector).unwrap()).collect();
            return Some(makestruct_expr(newbuilders).unwrap());
        }
    }
    None
}

fn func_has_simple_merge(expr: &TypedExpr) -> bool {
    if let Lambda { ref params, ref body } = expr.kind {
        simple_merge(&params[0].name, body)
    } else {
        false
    }
}

/// Infers the size of an `Appender` in a `For` loop.
pub fn infer_size(expr: &mut Expr<Type>) {
    expr.transform_up(&mut |ref mut expr| {
        if let For { ref mut iters, ref mut builder, ref mut func } = expr.kind {
            if iters.len() > 0 && iters.iter().all(|ref iter| iter.is_simple()) && func_has_simple_merge(func) {
                if let Ident(_) = iters[0].data.kind {
                    if let Some(newbuilder) = newbuilder_with_size(builder, iters[0].data.as_ref()) {
                        *builder = Box::new(newbuilder);
                    }
                }
            }
        }
        None
    });
}

/// Checks that `expr` performs only one `Merge` per control path - this guarantees
/// that the loop containing `expr`'s size can be inferred.
fn simple_merge(sym: &Symbol, expr: &Expr<Type>) -> bool {
    match expr.kind {
        Merge { ref builder, ref value } => {
            if let Ident(ref s) = builder.kind {
                if s == sym {
                    return !value.contains_symbol(sym);
                }
            }
            return false;
        }
        If { ref cond, ref on_true, ref on_false } => {
            !cond.contains_symbol(sym) && simple_merge(sym, on_true) &&
                simple_merge(sym, on_false)
        }
        _ => false,
    }
}
