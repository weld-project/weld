//! Common transformations on expressions.

use super::ast::*;
use super::ast::ExprKind::*;
use super::ast::Type::*;
use super::ast::BuilderKind::*;
use super::error::*;

use super::util::SymbolGenerator;

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
pub fn inline_apply<T:Clone+Eq>(expr: &mut Expr<T>) -> WeldResult<()> {
    expr.transform(&mut |ref mut expr| {
        if let Apply(ref func, ref args) = expr.kind {
            if let Lambda(ref params, ref body) = func.kind {
                let mut new = *body.clone();
                for (param, arg) in params.iter().zip(args) {
                    new.substitute(&param.name, &arg);
                }
                return Some(new);
            }
        }
        None
    });
    Ok(())
}

/// Fuses for loops over the same vector in a zip into a single for loop which produces multiple
/// builders.
pub fn fuse_loops_horizontal(expr: &mut Expr<Type>) -> WeldResult<()> {
    expr.transform(&mut |ref mut expr| {
        let mut sym_gen = SymbolGenerator::from_expression(expr);
        if let For(ref iters1, ref outer_bldr, ref outer_func) = expr.kind {
            // Collapses Zips with Fors over the same vector into a single For which produces multiple results.
            if iters1.len() > 1 {
                // Vector of tuples containing the params and expressions of functions in nested lambdas.
                let mut lambdas = vec![];
                let mut common_data = None;
                // Used to check if the same rows of each output are touched by the outer for.
                let iter_data = (&iters1[0].start, &iters1[0].end, &iters1[0].stride);
                // First, check if all the lambdas are over the same vector and have a pattern we can merge.
                if iters1.iter().all(|ref iter| {
                    if (&iter.start, &iter.end, &iter.stride) == iter_data {
                        if let Res(ref res_bldr) = iter.data.kind {
                            if let For(ref iters2, ref bldr2, ref lambda) = res_bldr.kind {
                                if common_data.is_none() {
                                    common_data = Some(iters2.clone());
                                }
                                if iters2 == common_data.as_ref().unwrap() { 
                                    if let NewBuilder = bldr2.kind {
                                        if let Builder(ref kind) = bldr2.ty {
                                            if let Appender(_) = *kind {
                                                if let Lambda(ref args, ref body) = lambda.kind {
                                                    if let Merge(ref bldr,  ref expr) = body.kind {
                                                        if let Ident(ref n) = bldr.kind {
                                                            if *n == args[0].name {
                                                                lambdas.push((args.clone(), expr.clone())); 
                                                                return true
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
                    return false
                }) {
                    // Zip the expressions to create an appender whose merge type is a struct.
                    let merge_type = Struct(lambdas.iter().map(|ref e| e.1.ty.clone()).collect::<Vec<_>>());
                    let builder_type = Builder(Appender(Box::new(merge_type.clone())));
                    let func_elem_type = lambdas[0].0[1].ty.clone();

                    // Parameters for the new fused function. Symbols names are generated using symbol
                    // names for the builder and element from an existing function.
                    let new_params = vec![
                        Parameter{ty: builder_type.clone(), name: sym_gen.new_symbol(&lambdas[0].0[0].name.name)},
                        Parameter{ty: func_elem_type.clone(), name: sym_gen.new_symbol(&lambdas[0].0[1].name.name)},
                    ];

                    // Generate Ident expressions for the new symbols and substitute them in the
                    // functions' merge expressions.
                    let new_bldr_expr = Expr {
                        ty: builder_type.clone(),
                        kind: Ident(new_params[0].name.clone())
                    };
                    let new_elem_expr = Expr {
                        ty: func_elem_type.clone(),
                        kind: Ident(new_params[1].name.clone())
                    };
                    for &mut (ref mut args, ref mut expr) in lambdas.iter_mut() {
                        expr.substitute(&args[0].name, &new_bldr_expr);
                        expr.substitute(&args[1].name, &new_elem_expr);
                    }

                    // Build up the new expression.
                    let new_merge_expr = Expr{
                        ty: builder_type.clone(),
                        kind: Merge(
                            Box::new(new_bldr_expr),
                            Box::new(Expr{
                                ty: merge_type.clone(),
                                kind: MakeStruct(lambdas.iter().map(|ref lambda| *lambda.1.clone()).collect::<Vec<_>>())
                            })
                            )
                    };
                    let new_func = Expr{
                        ty: Function(new_params.iter().map(|ref p| p.ty.clone()).collect::<Vec<_>>(), Box::new(builder_type.clone())),
                        kind: Lambda(new_params, Box::new(new_merge_expr))
                    };
                    let new_iter_expr = Expr{
                        ty: Vector(Box::new(merge_type.clone())),
                        kind: Res(Box::new(Expr{
                            ty: builder_type.clone(),
                            kind: For(common_data.unwrap(), Box::new(Expr{ty: builder_type.clone(), kind: NewBuilder}), Box::new(new_func))
                        })),
                    };

                    // TODO(shoumik): Any way to avoid the clones here?
                    return Some(Expr{
                        ty: expr.ty.clone(),
                        kind: For(vec![Iter{
                            data: Box::new(new_iter_expr),
                            start: iters1[0].start.clone(),
                            end: iters1[0].end.clone(),
                            stride: iters1[0].stride.clone()
                        }], outer_bldr.clone(), outer_func.clone())
                    });
                }
            }
        }
        None
    });
    Ok(())
}

/// Fuses loops where one for loop takes another as it's input, which prevents intermediate results
/// from being materialized.
pub fn fuse_loops(expr: &mut Expr<Type>) -> WeldResult<()> {
    expr.transform(&mut |ref mut expr| {
        let mut sym_gen = SymbolGenerator::from_expression(expr);
        if let For(ref iters1, ref bldr1, ref nested) = expr.kind {
            if iters1.len() == 1 {
                let ref iter1 = iters1[0];
                if let Res(ref res_bldr) = iter1.data.kind {
                    if let For(ref iters2, ref bldr2, ref lambda) = res_bldr.kind {
                        if iters2.iter().all(|ref i| consumes_all(&i)) {
                            if let NewBuilder = bldr2.kind {
                                if let Builder(ref kind) = bldr2.ty {
                                    if let Appender(_) = *kind {
                                        let e = Expr{
                                            ty: expr.ty.clone(),
                                            kind: For(iters2.clone(), bldr1.clone(), Box::new(replace_builder(lambda, nested, &mut sym_gen)))
                                        };
                                        return Some(e)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    });
    Ok(())
}

/// Given an iterator, returns whether the iterator consumes every element of its data vector.
fn consumes_all(iter: &Iter<Type>) -> bool {
    if let &Iter{start: None, end: None, stride: None, ..} = iter {
        return true
    } else if let &Iter{ref data, start: Some(ref start), end: Some(ref end), stride: Some(ref stride), ..} = iter {
        // Looks like Iter(A, 0, Len(A), 1)
        if let I64Literal(1) = stride.kind {
            if let I64Literal(0) = start.kind {
                if let Ident(ref name) = data.kind {
                    if let Length(ref v) = end.kind {
                        if let Ident(ref vsym) = v.kind {
                            return vsym == name
                        }
                    }
                } else if let MakeVector(ref exprs) = data.kind {
                    let num_elems = exprs.len() as i64;
                    if let I64Literal(x) = end.kind {
                        return num_elems == x
                    }
                }
            }
        }
    }
    false
}

/// Given a lambda which takes a builder and an argument, returns a new function which takes a new
/// builder type and calls nested on the old values it would've merged into its old builder. This
/// allows us to "compose" merge functions and avoid creating intermediate results.
fn replace_builder(lambda: &Expr<Type>,
                            nested: &Expr<Type>,
                            sym_gen: &mut SymbolGenerator) -> Expr<Type> {

    // Tests whether an identifier and symbol refer to the same value by
    // comparing the symbols.
    fn same_iden(a: &ExprKind<Type>, b: &Symbol) -> bool {
        if let Ident(ref symbol)  = *a { 
            symbol == b 
        } else {
            false
        }
    }

    let mut new_func = None;
    if let Lambda(ref args, ref body) = lambda.kind {
        if let Lambda(ref nested_args, _) = nested.kind {
            let mut new_body = *body.clone();
            let ref old_bldr = args[0];
            let ref old_arg = args[1];
            let new_sym = sym_gen.new_symbol(&old_bldr.name.name);
            let new_bldr = Expr{ty: nested_args[0].ty.clone(), kind: Ident(new_sym.clone())};
            new_body.transform(&mut |ref mut e| {
                match e.kind {
                    Merge(ref bldr, ref elem) if same_iden(&(*bldr).kind, &old_bldr.name) => {
                        let params: Vec<Expr<Type>> = vec![new_bldr.clone(), *elem.clone()];
                        let mut expr = Expr{ty: e.ty.clone(), kind: Apply(Box::new(nested.clone()), params)};
                        match inline_apply(&mut expr) {
                            Ok(_) => Some(expr),
                            Err(_) => None
                        }
                    },
                    For(ref data, ref bldr, ref func) if same_iden(&(*bldr).kind, &old_bldr.name) => {
                        Some(Expr{
                            ty: e.ty.clone(),
                            kind: For(data.clone(), Box::new(new_bldr.clone()), Box::new(replace_builder(func, nested, sym_gen)))
                        })
                    },
                    Ident(ref mut symbol) if *symbol == old_bldr.name => {
                        Some(new_bldr.clone())
                    },
                    _ => None
                }
            });
            let new_params = vec![
                Parameter{ty: new_bldr.ty.clone(), name: new_sym.clone()},
                Parameter{ty: old_arg.ty.clone(), name: old_arg.name.clone()}
            ];

            if let Function(_, ref ret_ty) = nested.ty {
                let new_func_type = Function(vec![new_bldr.ty.clone(), old_arg.ty.clone()], ret_ty.clone());
                new_func = Some(Expr{ty: new_func_type, kind: Lambda(new_params, Box::new(new_body))})
            }
        }
    }

    if let Some(new) = new_func {
        new
    } else {
        nested.clone()
    }
}
