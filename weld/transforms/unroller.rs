//! Analyzes the `loopsize` annotation to unroll loops into a series of `Lookup` nodes at compile
//! time.
//!

use std::error::Error;

use ast::*;

use ast::BuilderKind::*;
use ast::ExprKind::*;
use ast::Type::*;

use error::*;
use exprs::*;

use super::uniquify::uniquify;

/// Maximum number of iterations this transformation will unroll.
pub const UNROLL_LIMIT: i64 = 8;

/// A simple map pattern, which is a Result(For(.. with a single merge expression as the
/// For loop's function body.
struct UnrollPattern<'a> {
    loop_size: i64,
    iters: &'a Vec<TypedIter>,
    builder_kind: &'a BuilderKind,
    merge_params: &'a Vec<TypedParameter>,
    merge_value: &'a TypedExpr,
}

impl<'a> UnrollPattern<'a> {
    /// Extracts a `UnrollPattern` from the expression, or returns `None`.
    // TODO check annotation.
    // TODO check annotation cutoff.
    fn extract(expr: &'a TypedExpr) -> Option<UnrollPattern> {
        if let Res { ref builder } = expr.kind {
            if let Some(loopsize) = builder.annotations.loopsize() {
                if loopsize <= UNROLL_LIMIT {
                    if let For { ref iters, ref builder, ref func } = builder.kind {
                        if let Builder(ref bk, _) = builder.ty {
                            if let Lambda {ref params, ref body} = func.kind {
                                if let Merge { builder: ref builder2, ref value} = body.kind {
                                    match builder2.kind {
                                        Ident(ref name) if *name == params[0].name => {
                                            return Some(UnrollPattern {
                                                loop_size: loopsize,
                                                iters: iters,
                                                builder_kind: bk,
                                                merge_params: params,
                                                merge_value: value,
                                            });
                                        }
                                        _ => {
                                            return None;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }
}

pub fn unroll_static_loop(expr: &mut TypedExpr) {
    if let Err(_) = uniquify(expr) {
        return;
    }
    expr.transform(&mut |ref mut expr| {
        if let Some(pat) = UnrollPattern::extract(expr) {
            let vals = unroll_values(pat.merge_params, pat.merge_value, pat.iters, pat.loop_size);
            if vals.is_err() {
                trace!("Unroller error: {}", vals.unwrap_err().description());
                return None;
            }
            let vals = vals.unwrap();

            let combined_expr = combine_unrolled_values(pat.builder_kind.clone(), vals);
            if combined_expr.is_err() {
                trace!("Unroller error: {}", combined_expr.unwrap_err().description());
                return None;
            }
            let combined_expr = combined_expr.unwrap();
            Some(combined_expr)
        } else {
            None
        }
    });
}

fn is_same_ident(expr: &TypedExpr, other: &TypedExpr) -> bool {
    if let Ident(ref name) = other.kind {
        if let Ident(ref name2) = expr.kind {
            return name == name2 && expr.ty == other.ty;
        }
    }
    false
}

/// Takes a `MergeSingle` and returns a list of expressions which replace the element
/// in the merge with a Lookup.
fn unroll_values(parameters: &Vec<TypedParameter>, value: &TypedExpr, iters: &Vec<TypedIter>, loopsize: i64) -> WeldResult<Vec<TypedExpr>> {
    if parameters.len() != 3 {
        return weld_err!("Expected three parameters to Merge function");
    }

    let ref index_symbol = parameters[1].name;
    let ref elem_symbol = parameters[2].name;
    let ref elem_ident = ident_expr(elem_symbol.clone(), parameters[2].ty.clone())?;

    let mut expressions = vec![];
    for i in 0..loopsize {
        let mut unrolled_value = value.clone();
        unrolled_value.transform(&mut |ref mut e| {
            match e.kind {
                // TODO identifiers using the index value could also be handled here.
                Ident(ref name) if name == index_symbol => {
                    Some(literal_expr(LiteralKind::I64Literal(i as i64)).unwrap())
                }
                Ident(ref name) if name == elem_symbol && iters.len() == 1 => {
                    // There is a single iterator, which means the type of the element is the type
                    // of the iterator's data. Replace it with a lookup into the vector.
                    Some(lookup_expr(iters[0].data.as_ref().clone(), literal_expr(LiteralKind::I64Literal(i as i64)).unwrap()).unwrap())
                }
                GetField { ref expr, ref index } if is_same_ident(expr, elem_ident) && iters.len() > 1 => {
                    // There are multiple iterators zipped into a struct, and this expression is
                    // pulling one of the elements out of that struct. Replace it with a lookup into the vector.
                    let data_expr = iters[*index as usize].data.as_ref().clone();
                    Some(lookup_expr(data_expr, literal_expr(LiteralKind::I64Literal(i as i64)).unwrap()).unwrap())
                }
                _ => None
            }
        });
        expressions.push(unrolled_value);
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
            if values.iter().any(|ref expr| expr.ty != *ty.as_ref()) {
                return weld_err!("Mismatched types in Merger and unrolled values.");
            }
            // Use the specified binary op to produce the final expression.
            let mut prev = None;
            for value in values.into_iter() {
                if prev.is_none() {
                    prev = Some(value);
                } else {
                    prev = Some(binop_expr(*binop, prev.unwrap(), value)?);
                }
            }
            return Ok(prev.unwrap());
        }
        Appender(ref ty) => {
            if values.iter().any(|ref expr| expr.ty != *ty.as_ref()) {
                return weld_err!("Mismatched types in Appender and unrolled values.");
            }
            return makevector_expr(values);
        }
        ref bk => {
            return weld_err!("Unroller transform does not support loops with builder of kind {:?}", bk);
        }
    }
}
