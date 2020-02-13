//! Transforms which fuse loops to reduce memory movement and prevent unncessary
//! traversals of data.

use crate::ast::BuilderKind::*;
use crate::ast::ExprKind::*;
use crate::ast::LiteralKind::*;
use crate::ast::Type::*;
use crate::ast::*;
use crate::error::*;

use super::inliner::inline_apply;

use crate::util::SymbolGenerator;

#[cfg(test)]
use crate::tests::*;

/// Fuses for loops over the same vector in a zip into a single for loop which produces a vector of
/// structs directly.
///
/// Some examples:
///
/// for(zip(
///     result(for(a, appender, ...))
///     result(for(a, appender, ...))
/// ), ...)
///
/// will become for(result(for(a, ...))) where the nested for will produce a vector of structs with
/// two elements.
///
/// Caveats:
///     - Like all Zip-based transforms, this function currently assumes that the output of each
///     expression in the Zip is the same length.
///
pub fn fuse_loops_horizontal(expr: &mut Expr) {
    expr.transform(&mut |ref mut expr| {
        let mut sym_gen = SymbolGenerator::from_expression(expr);
        if let For {
            iters: ref all_iters,
            builder: ref outer_bldr,
            func: ref outer_func,
        } = expr.kind
        {
            if all_iters.len() > 1 {
                // Vector of tuples containing the params and expressions of functions in nested lambdas.
                let mut lambdas = vec![];
                let mut common_data = None;
                // Used to check if the same rows of each output are touched by the outer for.
                let first_iter = (&all_iters[0].start, &all_iters[0].end, &all_iters[0].stride);
                // First, check if all the lambdas are over the same vector and have a pattern we can merge.
                // Below, for each iterator in the for loop, we checked if each nested for loop is
                // over the same vector and has the same Iter parameters (i.e., same start, end, stride).
                let iters_same = all_iters.iter().all(|ref iter| {
                    if (&iter.start, &iter.end, &iter.stride) == first_iter {
                        // Make sure each nested for loop follows the ``result(for(a, appender, ...)) pattern.
                        if let Res {
                            builder: ref res_bldr,
                        } = iter.data.kind
                        {
                            if let For {
                                iters: ref iters2,
                                builder: ref bldr2,
                                func: ref lambda,
                            } = res_bldr.kind
                            {
                                if common_data.is_none() {
                                    common_data = Some(iters2.clone());
                                }
                                if iters2 == common_data.as_ref().unwrap() {
                                    if let NewBuilder(_) = bldr2.kind {
                                        if let Builder(ref kind, _) = bldr2.ty {
                                            if let Appender(_) = *kind {
                                                if let Lambda {
                                                    params: ref args,
                                                    ref body,
                                                } = lambda.kind
                                                {
                                                    if let Merge {
                                                        ref builder,
                                                        ref value,
                                                    } = body.kind
                                                    {
                                                        if let Ident(ref n) = builder.kind {
                                                            if *n == args[0].name {
                                                                // Save the arguments and expressions for the function so
                                                                // they can be used for fusion later.
                                                                lambdas.push((
                                                                    args.clone(),
                                                                    value.clone(),
                                                                ));
                                                                return true;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // The pattern doesn't match for some Iter -- abort the transform.
                    false
                });

                if iters_same {
                    // All Iters are over the same range and same vector, with a pattern we can
                    // transform. Produce the new expression by zipping the functions of each
                    // nested for into a single merge into a struct.

                    // Zip the expressions to create an appender whose merge (value) type is a struct.
                    let merge_type = Struct(
                        lambdas
                            .iter()
                            .map(|ref e| e.1.ty.clone())
                            .collect::<Vec<_>>(),
                    );
                    // TODO(Deepak): Fix this to something meaningful.
                    let builder_type =
                        Builder(Appender(Box::new(merge_type.clone())), Annotations::new());
                    // The element type remains unchanged.
                    let func_elem_type = lambdas[0].0[2].ty.clone();

                    // Parameters for the new fused function. Symbols names are generated using symbol
                    // names for the builder and element from an existing function.
                    let new_params = vec![
                        Parameter {
                            ty: builder_type.clone(),
                            name: sym_gen.new_symbol(&lambdas[0].0[0].name.name()),
                        },
                        Parameter {
                            ty: Scalar(ScalarKind::I64),
                            name: sym_gen.new_symbol(&lambdas[0].0[1].name.name()),
                        },
                        Parameter {
                            ty: func_elem_type.clone(),
                            name: sym_gen.new_symbol(&lambdas[0].0[2].name.name()),
                        },
                    ];

                    // Generate Ident expressions for the new symbols and substitute them in the
                    // functions' merge expressions.
                    let new_bldr_expr = Expr {
                        ty: builder_type.clone(),
                        kind: Ident(new_params[0].name.clone()),
                        annotations: Annotations::new(),
                    };
                    let new_index_expr = Expr {
                        ty: Scalar(ScalarKind::I64),
                        kind: Ident(new_params[1].name.clone()),
                        annotations: Annotations::new(),
                    };
                    let new_elem_expr = Expr {
                        ty: func_elem_type,
                        kind: Ident(new_params[2].name.clone()),
                        annotations: Annotations::new(),
                    };
                    for &mut (ref mut args, ref mut expr) in lambdas.iter_mut() {
                        expr.substitute(&args[0].name, &new_bldr_expr);
                        expr.substitute(&args[1].name, &new_index_expr);
                        expr.substitute(&args[2].name, &new_elem_expr);
                    }

                    // Build up the new expression. The new expression merges structs into an
                    // appender, where each struct field is an expression which was merged into an
                    // appender in one of the original functions. For example, if there were two
                    // zipped fors in the original expression with lambdas |b1,e1| merge(b1,
                    // e1+1) and |b2,e2| merge(b2, e2+2), the new expression would be merge(b,
                    // {e+1,e+2}) into a new builder b of type appender[{i32,i32}]. e1, e2, and e
                    // refer to the same element in the expressions above since we check to ensure
                    // each zipped for is over the same input data.
                    let new_merge_expr = Expr {
                        ty: builder_type.clone(),
                        kind: Merge {
                            builder: Box::new(new_bldr_expr),
                            value: Box::new(Expr {
                                ty: merge_type.clone(),
                                kind: MakeStruct {
                                    elems: lambdas
                                        .iter()
                                        .map(|ref lambda| *lambda.1.clone())
                                        .collect::<Vec<_>>(),
                                },
                                annotations: Annotations::new(),
                            }),
                        },
                        annotations: Annotations::new(),
                    };
                    let new_func = Expr {
                        ty: Function(
                            new_params
                                .iter()
                                .map(|ref p| p.ty.clone())
                                .collect::<Vec<_>>(),
                            Box::new(builder_type.clone()),
                        ),
                        kind: Lambda {
                            params: new_params,
                            body: Box::new(new_merge_expr),
                        },
                        annotations: Annotations::new(),
                    };
                    let new_iter_expr = Expr {
                        ty: Vector(Box::new(merge_type)),
                        kind: Res {
                            builder: Box::new(Expr {
                                ty: builder_type.clone(),
                                kind: For {
                                    iters: common_data.unwrap(),
                                    builder: Box::new(Expr {
                                        ty: builder_type,
                                        kind: NewBuilder(None),
                                        annotations: Annotations::new(),
                                    }),
                                    func: Box::new(new_func),
                                },
                                annotations: Annotations::new(),
                            }),
                        },
                        annotations: Annotations::new(),
                    };

                    // TODO(shoumik): Any way to avoid the clones here?
                    return Some(Expr {
                        ty: expr.ty.clone(),
                        kind: For {
                            iters: vec![Iter {
                                data: Box::new(new_iter_expr),
                                start: all_iters[0].start.clone(),
                                end: all_iters[0].end.clone(),
                                stride: all_iters[0].stride.clone(),
                                kind: all_iters[0].kind.clone(),
                                shape: all_iters[0].shape.clone(),
                                strides: all_iters[0].strides.clone(),
                            }],
                            builder: outer_bldr.clone(),
                            func: outer_func.clone(),
                        },
                        annotations: Annotations::new(),
                    });
                }
            }
        }
        None
    });
}

/// Fuses loops where one for loop takes another as it's input, which prevents intermediate results
/// from being materialized.
pub fn fuse_loops_vertical(expr: &mut Expr) {
    expr.transform_and_continue_res(&mut |ref mut expr| {
        let mut sym_gen = SymbolGenerator::from_expression(expr);
        if let For {
            iters: ref all_iters,
            builder: ref bldr1,
            func: ref nested,
        } = expr.kind
        {
            if all_iters.len() == 1 {
                let iter1 = &all_iters[0];
                if let Res {
                    builder: ref res_bldr,
                } = iter1.data.kind
                {
                    if let For {
                        iters: ref iters2,
                        builder: ref bldr2,
                        func: ref lambda,
                    } = res_bldr.kind
                    {
                        if iters2.iter().all(|ref i| consumes_all(&i)) {
                            if let NewBuilder(_) = bldr2.kind {
                                if let Builder(ref kind, _) = bldr2.ty {
                                    if let Appender(_) = *kind {
                                        let mut e = Expr::new_for(
                                            iters2.clone(),
                                            *bldr1.clone(),
                                            replace_builder(lambda, nested, &mut sym_gen)?,
                                        )?;
                                        e.annotations = expr.annotations.clone();
                                        return Ok((Some(e), true));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok((None, true))
    });
}

/// Given an iterator, returns whether the iterator consumes every element of its data vector.
fn consumes_all(iter: &Iter) -> bool {
    if let Iter {
        start: None,
        end: None,
        stride: None,
        ..
    } = *iter
    {
        return true;
    } else if let Iter {
        ref data,
        start: Some(ref start),
        end: Some(ref end),
        stride: Some(ref stride),
        ..
    } = *iter
    {
        // Checks if the stride is 1 and an entire vector represented by a symbol is consumed.
        if let (
            &Literal(I64Literal(1)),
            &Literal(I64Literal(0)),
            &Ident(ref name),
            &Length { data: ref v },
        ) = (&stride.kind, &start.kind, &data.kind, &end.kind)
        {
            if let Ident(ref vsym) = v.kind {
                return vsym == name;
            }
        }
        // Checks if an entire vector literal is consumed.
        if let (&Literal(I64Literal(1)), &Literal(I64Literal(0)), &MakeVector { ref elems }) =
            (&stride.kind, &start.kind, &data.kind)
        {
            let num_elems = elems.len() as i64;
            if let Literal(I64Literal(x)) = end.kind {
                return num_elems == x;
            }
        }
    }
    false
}

/// Given a lambda which takes a builder and an argument, returns a new function which takes a new
/// builder type and calls nested on the values it would've merged into its old builder. This
/// allows us to "compose" merge functions and avoid creating intermediate results.
fn replace_builder(
    lambda: &Expr,
    nested: &Expr,
    sym_gen: &mut SymbolGenerator,
) -> WeldResult<Expr> {
    // Tests whether an identifier and symbol refer to the same value by
    // comparing the symbols.
    fn same_iden(a: &ExprKind, b: &Symbol) -> bool {
        if let Ident(ref symbol) = *a {
            symbol == b
        } else {
            false
        }
    }

    if let Lambda {
        params: ref args,
        ref body,
    } = lambda.kind
    {
        if let Lambda {
            params: ref nested_args,
            ..
        } = nested.kind
        {
            let mut new_body = *body.clone();
            let old_bldr = &args[0];
            let old_index = &args[1];
            let old_arg = &args[2];
            let new_bldr_sym = sym_gen.new_symbol(&old_bldr.name.name());
            let new_index_sym = sym_gen.new_symbol(&old_index.name.name());
            let new_bldr = Expr::new_ident(new_bldr_sym.clone(), nested_args[0].ty.clone())?;
            let new_index = Expr::new_ident(new_index_sym.clone(), nested_args[1].ty.clone())?;

            // Fix expressions to use the new builder.
            new_body.transform_and_continue_res(&mut |ref mut e| match e.kind {
                Merge {
                    ref builder,
                    ref value,
                } if same_iden(&(*builder).kind, &old_bldr.name) => {
                    let params: Vec<Expr> =
                        vec![new_bldr.clone(), new_index.clone(), *value.clone()];
                    let mut expr = Expr::new_apply(nested.clone(), params)?;
                    inline_apply(&mut expr);
                    Ok((Some(expr), true))
                }
                For {
                    iters: ref data,
                    builder: ref bldr,
                    ref func,
                } if same_iden(&(*bldr).kind, &old_bldr.name) => {
                    let expr = Expr::new_for(
                        data.clone(),
                        new_bldr.clone(),
                        replace_builder(func, nested, sym_gen)?,
                    )?;
                    Ok((Some(expr), false))
                }
                Ident(ref mut symbol) if *symbol == old_bldr.name => {
                    Ok((Some(new_bldr.clone()), false))
                }
                Ident(ref mut symbol) if *symbol == old_index.name => {
                    Ok((Some(new_index.clone()), false))
                }
                _ => Ok((None, true)),
            });

            // Fix types to make sure the return type propagates through all subexpressions.
            match_types(&new_bldr.ty, &mut new_body);

            let new_params = vec![
                Parameter {
                    ty: new_bldr.ty,
                    name: new_bldr_sym,
                },
                Parameter {
                    ty: Scalar(ScalarKind::I64),
                    name: new_index_sym,
                },
                Parameter {
                    ty: old_arg.ty.clone(),
                    name: old_arg.name.clone(),
                },
            ];
            return Expr::new_lambda(new_params, new_body);
        }
    }
    return compile_err!("Inconsistency in replace_builder");
}

/// Given a root type, forces each expression to return that type. TODO For now, only supporting
/// expressions which can be builders. We might want to factor this out to be somewhere else.
fn match_types(root_ty: &Type, expr: &mut Expr) {
    expr.ty = root_ty.clone();
    match expr.kind {
        If {
            ref mut on_true,
            ref mut on_false,
            ..
        } => {
            match_types(root_ty, on_true);
            match_types(root_ty, on_false);
        }
        Select {
            ref mut on_true,
            ref mut on_false,
            ..
        } => {
            match_types(root_ty, on_true);
            match_types(root_ty, on_false);
        }
        Let { ref mut body, .. } => {
            match_types(root_ty, body);
        }
        _ => {}
    };
}

#[test]
fn simple_horizontal_loop_fusion() {
    // Two loops.
    let mut e1 = typed_expression(
        "for(zip(
            result(for([1,2,3], appender, |b,i,e| merge(b, e+1))),
            result(for([1,2,3], appender,|b2,i2,e2| merge(b2,e2+1)))
        ), appender, |b,i,e| merge(b, e.$0+1))",
    );
    fuse_loops_horizontal(&mut e1);
    let e2 = typed_expression(
        "for(result(for([1,2,3], appender, |b,i,e| merge(b, {e+1,e+1}))), \
         appender, |b,i,e| merge(b, e.$0+1))",
    );
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Three loops.
    let mut e1 = typed_expression(
        "for(zip(
            result(for([1,2,3], appender, |b,i,e| merge(b, e+1))),
            result(for([1,2,3], appender,|b2,i2,e2| merge(b2,e2+2))),
            result(for([1,2,3], appender,|b3,i3,e3| merge(b3,e3+3)))
        ), appender, |b,i,e| merge(b, e.$0+1))",
    );
    fuse_loops_horizontal(&mut e1);
    let e2 = typed_expression(
        "for(result(for([1,2,3], appender, |b,i,e| merge(b, \
         {e+1,e+2,e+3}))), appender, |b,i,e| merge(b, e.$0+1))",
    );
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Iters in inner loop
    let mut e1 = typed_expression(
        "for(zip(
            result(for(iter([1,2,3], 0L, 2L, 1L), appender, |b,i,e| merge(b, e+1))),
            result(for(iter([1,2,3], 0L, 2L, 1L), appender, |b,i,e| merge(b, e+2)))
        ), appender, |b,i,e| merge(b, e.$0+1))",
    );
    fuse_loops_horizontal(&mut e1);
    let e2 = typed_expression(
        "for(result(for(iter([1,2,3], 0L, 2L, 1L), appender, |b,i,e| \
         merge(b, {e+1,e+2}))), appender, |b,i,e| merge(b, e.$0+1))",
    );
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Iters in outer loop.
    let mut e1 = typed_expression(
        "for(zip(
            iter(result(for([1,2,3], appender, |b,i,e| merge(b, e+1))), 0L, 2L, 1L),
            iter(result(for([1,2,3], appender, |b,i,e| merge(b, e+2))), 0L, 2L, 1L)
        ), appender, |b,i,e| merge(b, e.$0+1))",
    );
    fuse_loops_horizontal(&mut e1);
    let e2 = typed_expression(
        "for(iter(result(for([1,2,3], appender, |b,i,e| merge(b, \
         {e+1,e+2}))), 0L, 2L, 1L), appender, |b,i,e| merge(b, e.$0+1))",
    );
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Two loops with different vectors; should fail.
    let mut e1 = typed_expression(
        "for(zip(
            result(for([1,2,3], appender, |b,i,e| merge(b, e+1))),
            result(for([1,2,4], appender,|b2,i2,e2| merge(b2,e2+1)))
        ), appender, |b,i,e| merge(b, e.$0+1))",
    );
    let e2 = e1.clone();
    fuse_loops_horizontal(&mut e1);
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());
}

#[test]
fn simple_vertical_loop_fusion() {
    // Two loops.
    let mut e1 = typed_expression(
        "for(result(for([1,2,3], appender, |b,i,e| merge(b,e+2))), \
         appender, |b,h,f| merge(b, f+1))",
    );
    fuse_loops_vertical(&mut e1);
    let e2 = typed_expression("for([1,2,3], appender, |b,i,e| merge(b, (e+2)+1))");
    println!("{}", print_expr_without_indent(&e1));
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Three loops.
    let mut e1 = typed_expression(
        "for(result(for(result(for([1,2,3], appender, |b,i,e| \
         merge(b,e+3))), appender, |b,i,e| merge(b,e+2))), appender, \
         |b,h,f| merge(b, f+1))",
    );
    fuse_loops_vertical(&mut e1);
    let e2 = typed_expression("for([1,2,3], appender, |b,i,e| merge(b, (((e+3)+2)+1)))");
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Merges in other positions, replace builder identifiers.
    let mut e1 = typed_expression(
        "for(result(for([1,2,3], appender, |b,i,e| if(e>5, \
         merge(b,e+2), b))), appender, |b,h,f| merge(b, f+1))",
    );
    fuse_loops_vertical(&mut e1);
    let e2 = typed_expression("for([1,2,3], appender, |b,i,e| if(e>5, merge(b, (e+2)+1), b))");
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Make sure correct builder is chosen.
    let mut e1 = typed_expression(
        "for(result(for([1,2,3], appender[i32], |b,i,e| \
         merge(b,e+2))), appender[f64], |b,h,f| merge(b, 1.0))",
    );
    fuse_loops_vertical(&mut e1);
    let e2 = typed_expression("for([1,2,3], appender[f64], |b,i,e| merge(b, 1.0))");
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Multiple inner loops.
    let mut e1 = typed_expression(
        "for(result(for(zip([1,2,3],[4,5,6]), appender, |b,i,e| \
         merge(b,e.$0+2))), appender, |b,h,f| merge(b, f+1))",
    );
    fuse_loops_vertical(&mut e1);
    let e2 = typed_expression("for(zip([1,2,3],[4,5,6]), appender, |b,i,e| merge(b, (e.$0+2)+1))");
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Iter where inner data consumed fully.
    let mut e1 = typed_expression(
        "let a = [1,2,3]; for(result(for(iter(a, 0L, len(a), 1L), \
         appender, |b,i,e| merge(b,e+2))), appender, |b,h,f| merge(b, \
         f+1))",
    );
    fuse_loops_vertical(&mut e1);
    let e2 = typed_expression(
        "let a = [1,2,3]; for(iter(a,0L,len(a),1L), appender, |b,i,e| \
         merge(b, (e+2)+1))",
    );
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Inner data not consumed fully.
    let mut e1 = typed_expression(
        "for(result(for(iter([1,2,3], 0L, 1L, 1L), appender, |b,i,e| \
         merge(b,e+2))), appender, |b,h,f| merge(b, f+1))",
    );
    fuse_loops_vertical(&mut e1);
    // Loop fusion should fail.
    let e2 = typed_expression(
        "for(result(for(iter([1,2,3], 0L, 1L, 1L), appender, |b,i,e| \
         merge(b,e+2))), appender, |b,h,f| merge(b, f+1))",
    );
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());
}
