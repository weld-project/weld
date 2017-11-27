//! Analyzes the `loopsize` annotation to unroll loops into a series of `Lookup` nodes at compile
//! time.
//! 

use ast::*;
use ast::ExprKind::*;

use annotations::*;

use super::uniquify::uniquify;

struct MergeSingle<'a> {
    params: &'a Vec<TypedParameter>,
    value: &'a TypedExpr
}

impl<'a> MergeSingle<'a> {
    fn extract(expr: &'a TypedExpr) -> Option<MergeSingle<'a>> {
        if let Lambda{ref params, ref body} = expr.kind {
            if let Merge{ref builder, ref value} = body.kind {
                match builder.kind {
                    Ident(ref name) if *name == params[0].name =>
                        return Some(MergeSingle{params, value}),
                    _ => {}
                }
            }
        }
        return None
    }
}

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

struct ResForAppender<'a> {
    iters: &'a Vec<Iter<Type>>,
    func: &'a TypedExpr
}

impl<'a> ResForAppender<'a> {
    fn extract(expr: &'a TypedExpr) -> Option<ResForAppender<'a>> {
        if let Res{ref builder} = expr.kind {
            if let For{ref iters, ref builder, ref func} = builder.kind {
                if NewAppender::extract(builder).is_some() {
                    return Some(ResForAppender{iters, func})
                }
            }
        }
        return None;
    }
}

struct MapIter<'a> {
    iters: &'a Vec<Iter<Type>>,
    merge_params: &'a Vec<TypedParameter>,
    merge_value: &'a TypedExpr
}

impl<'a> MapIter<'a> {
    fn extract(iter: &'a TypedIter) -> Option<MapIter> {
        if iter.is_simple() {
            if let Some(rfa) = ResForAppender::extract(&iter.data) {
                if rfa.iters.iter().all(|ref i| i.is_simple()) {
                    if let Some(merge) = MergeSingle::extract(&rfa.func) {
                        return Some(MapIter {
                            iters: &rfa.iters,
                            merge_params: merge.params,
                            merge_value: merge.value
                        });
                    }
                }
            }
        }
        return None;
    }
}

// TODO When should the transformation be applied?

/// Takes a `MergeSingle` and returns a list of expressions which replace the element
/// in the merge with a Lookup.
fn unroll_values(merge_single: &'a MergeSingle, iters: &'a Vec<TypedIter>, loopsize: usize) -> WeldResult<Vec<TypedExpr>> {
    if merge_single.params.len() != 3 {
        return weld_err!("Expected three parameters to Merge function");
    }

    let ref elem_symbol = merge_single.params[2].name;
    // To match on the element.
    let ref elem_ident = ident_expr(elem_symbol.clone(), merge_single.params[2].ty.clone())?;
    let mut expressions = vec![];

    for i in 0..loopsize {
        let mut value = merge_single.value.clone();
        value.transform(&mut |ref mut e| {
            match e.kind {
                Ident(ref name) if name == elem_symbol && iters.len() == 1 {
                    // There is a single iterator, which means the type of the element is the type
                    // of the iterator's data. Replace it with a lookup into the vector.
                    return Ok(lookup_expr(iters[0].data.as_ref().clone(), literal_expr(LiteralKind::I64Literal(i as i64))?)?);
                }
                GetField { ref expr, ref index } if expr == elem_ident && iters.len() > 1 => {
                    // There are multiple iterators zipped into a struct, and this expression is
                    // pulling one of the elements out of that struct. Replace it with a lookup into the vector.
                    let data_expr = iters[index].data.as_ref().clone();
                    return Ok(lookup_expr(data_expr, literal_expr(LiteralKind::I64Literal(i as i64))?)?);
                }
            }
            None
        });
        expressions.push(value);
    }
    return Ok(expressions);
}

/// Combines the expressions in `values` into a single value based on the kind of builder the
/// values would have been merged into.
/// 
/// As an example, if `values` is [ Literal(1), Literal(2), Literal(3)] and the builder was a
/// merger[i32,+], this function will produce the expression Literal(1) + Literal(2) + Literal(3).
fn combine_unrolled_values(bk: BuilderKind, values: Vec<TypedExpr>) -> WeldResult<TypedExpr> {
    if values.len() == 0 {
        return weld_err!("Need at least one value to combine in unroller");
    }
    match bk {
        Merger(ref ty, ref binop) => {
            if values.iter().any(|ref expr| expr.ty != ty) {
                return weld_err!("Mismatched types in Merger and unrolled values.");
            }
            // Use the specified binary op to produce the final expression.
            let mut prev = None;
            for value in values.into_iter() {
                if prev.is_none() {
                    prev = Some(value);
                } else {
                    prev = Ok(binop_expr(*binop, prev.unwrap(), value)?)
                }
            }
            return prev.unwrap();
        }
        Appender(ref ty) => {
            if values.iter().any(|ref expr| expr.ty != ty) {
                return weld_err!("Mismatched types in Appender and unrolled values.");
            }
            return makevector_expr(values)?;
        }
        ref bk => {
            weld_err!("Unroller transform does not support loops with builder of kind {:?}", bk);
        }
    }
}

