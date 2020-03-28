//! Analyzes the `loopsize` annotation to unroll loops into a series of `Lookup` nodes at compile
//! time.
//!

use std::error::Error;

use crate::ast::*;

use crate::ast::BuilderKind::*;
use crate::ast::ExprKind::*;
use crate::ast::Type::*;

use crate::error::*;

/// Maximum number of iterations this transformation will unroll.
pub const UNROLL_LIMIT: u64 = 8;

/// A simple map pattern, which is a Result(For(.. with a single merge expression as the
/// For loop's function body.
struct UnrollPattern<'a> {
    loop_size: u64,
    iters: &'a Vec<Iter>,
    builder_kind: &'a BuilderKind,
    merge_params: &'a Vec<Parameter>,
    merge_value: &'a Expr,
}

/// Returns the loop size from the annotation.
///
/// Returns None if the annotation does not exist or cannot be parsed as a 64-bit unsigned integer.
trait LoopSizeAnnotation {
    fn loopsize(&self) -> Option<u64>;
}

impl LoopSizeAnnotation for Expr {
    fn loopsize(&self) -> Option<u64> {
        self.annotations
            .get("loopsize")
            .and_then(|value| value.parse::<u64>().ok())
    }
}

impl<'a> UnrollPattern<'a> {
    /// Extracts a `UnrollPattern` from the expression, or returns `None`.
    fn extract(expr: &'a Expr) -> Option<UnrollPattern<'_>> {
        if let Res { ref builder } = expr.kind {
            if let Some(loopsize) = builder.loopsize() {
                if loopsize <= UNROLL_LIMIT {
                    if let For {
                        ref iters,
                        ref builder,
                        ref func,
                    } = builder.kind
                    {
                        if let Builder(ref bk, _) = builder.ty {
                            if let Lambda {
                                ref params,
                                ref body,
                            } = func.kind
                            {
                                if let Merge {
                                    builder: ref builder2,
                                    ref value,
                                } = body.kind
                                {
                                    match builder2.kind {
                                        Ident(ref name) if *name == params[0].name => {
                                            return Some(UnrollPattern {
                                                loop_size: loopsize,
                                                iters,
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

pub fn unroll_static_loop(expr: &mut Expr) {
    use crate::util::SymbolGenerator;

    if expr.uniquify().is_err() {
        return;
    }

    let mut sym_gen = SymbolGenerator::from_expression(expr);
    expr.transform_up(&mut |ref mut expr| {
        if let Some(pat) = UnrollPattern::extract(expr) {
            // Create a vector of identifiers which will bind to the iterator data.
            let symbols: Vec<_> = (0..pat.iters.len())
                .map(|_| sym_gen.new_symbol("tmp"))
                .collect();
            let idents: Vec<_> = symbols
                .iter()
                .zip(pat.iters.iter())
                .map(|ref t| Expr::new_ident(t.0.clone(), t.1.data.ty.clone()).unwrap())
                .collect();

            let vals = unroll_values(pat.merge_params, pat.merge_value, &idents, pat.loop_size);
            if let Err(err) = vals {
                trace!("Unroller error: {}", err.description());
                return None;
            }
            let vals = vals.unwrap();

            let combined_expr = combine_unrolled_values(pat.builder_kind.clone(), vals);
            if let Err(err) = combined_expr {
                trace!("Unroller error: {}", err.description());
                return None;
            }

            // Prepend the identifier definitions to the combined expression.
            let mut prev = combined_expr.unwrap();
            for (ref sym, ref iter) in symbols.into_iter().rev().zip(pat.iters.iter().rev()) {
                // Construct this explicitly instead of using the `expr` based constructor
                // to allow a move of the boxed value, avoiding a copy.
                prev = Expr::new_let(sym.clone(), iter.data.as_ref().clone(), prev).unwrap();
            }
            Some(prev)
        } else {
            None
        }
    });
}

fn is_same_ident(expr: &Expr, other: &Expr) -> bool {
    if let Ident(ref name) = other.kind {
        if let Ident(ref name2) = expr.kind {
            return name == name2 && expr.ty == other.ty;
        }
    }
    false
}

/// Takes a `MergeSingle` and returns a list of expressions which replace the element
/// in the merge with a Lookup.
fn unroll_values(
    parameters: &[Parameter],
    value: &Expr,
    vectors: &[Expr],
    loopsize: u64,
) -> WeldResult<Vec<Expr>> {
    if parameters.len() != 3 {
        return compile_err!("Expected three parameters to Merge function");
    }

    let index_symbol = &parameters[1].name;
    let elem_symbol = &parameters[2].name;
    let elem_ident = &Expr::new_ident(elem_symbol.clone(), parameters[2].ty.clone())?;

    let mut expressions = vec![];
    for i in 0..loopsize {
        let mut unrolled_value = value.clone();
        unrolled_value.transform(&mut |ref mut e| {
            match e.kind {
                Ident(ref name) if name == index_symbol => {
                    // Index identifiers can be handled by just substituting a static index.
                    Some(Expr::new_literal(LiteralKind::I64Literal(i as i64)).unwrap())
                }
                Ident(ref name) if name == elem_symbol && vectors.len() == 1 => {
                    // There is a single iterator, which means the type of the element is the type
                    // of the iterator's data. Replace it with a lookup into the vector.
                    Some(
                        Expr::new_lookup(
                            vectors[0].clone(),
                            Expr::new_literal(LiteralKind::I64Literal(i as i64)).unwrap(),
                        )
                        .unwrap(),
                    )
                }
                GetField {
                    ref expr,
                    ref index,
                } if is_same_ident(expr, elem_ident) && vectors.len() > 1 => {
                    // There are multiple iterators zipped into a struct, and this expression is
                    // pulling one of the elements out of that struct. Replace it with a lookup into the vector.
                    let data_expr = vectors[*index as usize].clone();
                    Some(
                        Expr::new_lookup(
                            data_expr,
                            Expr::new_literal(LiteralKind::I64Literal(i as i64)).unwrap(),
                        )
                        .unwrap(),
                    )
                }
                _ => None,
            }
        });
        expressions.push(unrolled_value);
    }
    Ok(expressions)
}

/// Combines the expressions in `values` into a single value based on the kind of builder the
/// values would have been merged into.
///
/// As an example, if `values` is [ Literal(1), Literal(2), Literal(3)] and the builder was a
/// merger[i32,+], this function will produce the expression Literal(1) + Literal(2) + Literal(3).
fn combine_unrolled_values(bk: BuilderKind, values: Vec<Expr>) -> WeldResult<Expr> {
    if values.is_empty() {
        return compile_err!("Need at least one value to combine in unroller");
    }
    match bk {
        Merger(ref ty, ref binop) => {
            if values.iter().any(|ref expr| expr.ty != *ty.as_ref()) {
                return compile_err!("Mismatched types in Merger and unrolled values.");
            }
            // Use the specified binary op to produce the final expression.
            let mut prev = None;
            for value in values.into_iter() {
                if prev.is_none() {
                    prev = Some(value);
                } else {
                    prev = Some(Expr::new_bin_op(*binop, prev.unwrap(), value)?);
                }
            }
            Ok(prev.unwrap())
        }
        Appender(ref ty) => {
            if values.iter().any(|ref expr| expr.ty != *ty.as_ref()) {
                return compile_err!("Mismatched types in Appender and unrolled values.");
            }
            Expr::new_make_vector(values)
        }
        ref bk => compile_err!(
            "Unroller transform does not support loops with builder of kind {:?}",
            bk
        ),
    }
}

#[cfg(test)]
use crate::tests::*;

#[test]
fn simple_merger_loop() {
    let mut e = typed_expression(
        "|v:vec[i32]| result(
            @(loopsize:2L)
            for(v, merger[i32,+],
            |b,i,e| merge(b, e)))",
    );

    unroll_static_loop(&mut e);
    let expect = &typed_expression("|v:vec[i32]| let t0 = v; lookup(t0, 0L) + lookup(t0, 1L)");
    assert!(e.compare_ignoring_symbols(expect).unwrap());
}

#[test]
fn zipped_merger_loop() {
    let mut e = typed_expression(
        "|v:vec[i32], w: vec[i32]| result(
            @(loopsize:2L)
            for(zip(v, w), merger[i32,+],
            |b,i,e| merge(b, e.$0 * e.$1)))",
    );

    unroll_static_loop(&mut e);
    let expect = &typed_expression(
        "|v:vec[i32], w:vec[i32]| let t0 = v; let t1 = w;
                                      lookup(t0, 0L) * lookup(t1, 0L) +
                                      lookup(t0, 1L) * lookup(t1, 1L)",
    );
    assert!(e.compare_ignoring_symbols(expect).unwrap());
}

#[test]
fn simple_appender_loop() {
    let mut e = typed_expression(
        "|v:vec[i32]| result(
            @(loopsize:2L)
            for(v, appender,
            |b,i,e| merge(b, e)))",
    );

    unroll_static_loop(&mut e);
    let expect = &typed_expression("|v:vec[i32]| let t0 = v; [lookup(t0, 0L), lookup(t0, 1L)]");
    assert!(e.compare_ignoring_symbols(expect).unwrap());
}

#[test]
fn zipped_appender_loop() {
    let mut e = typed_expression(
        "|v:vec[i32], w: vec[i32]| result(
            @(loopsize:2L)
            for(zip(v, w), appender,
            |b,i,e| merge(b, e.$0 * e.$1)))",
    );

    unroll_static_loop(&mut e);
    let expect = &typed_expression(
        "|v:vec[i32], w:vec[i32]| let t0 = v; let t1 = w;
                                      [lookup(t0, 0L) * lookup(t1, 0L),
                                      lookup(t0, 1L) * lookup(t1, 1L)]",
    );
    assert!(e.compare_ignoring_symbols(expect).unwrap());
}

#[test]
fn large_merger_loop() {
    let mut e = typed_expression(
        format!(
            "|v:vec[i32]| result(
            @(loopsize:{}L)
            for(v, merger[i32,+],
            |b,i,e| merge(b, e)))",
            UNROLL_LIMIT + 1
        )
        .as_ref(),
    );
    // The annotation is more than the unroll limit, so don't unroll.
    let expect = &e.clone();
    unroll_static_loop(&mut e);
    assert!(e.compare_ignoring_symbols(expect).unwrap());
}
