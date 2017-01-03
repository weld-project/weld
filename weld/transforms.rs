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
    for child in expr.children_mut() {
        try!(inline_apply(child));
    }
    let mut new_expr = None;
    if let Apply(ref func, ref args) = expr.kind {
        if let Lambda(ref params, ref body) = func.kind {
            let mut new = *body.clone();
            for (param, arg) in params.iter().zip(args) {
                new.substitute(&param.name, &arg);
            }
            new_expr = Some(new);
        }
    }
    if let Some(new) = new_expr {
        *expr = new;
    }
    Ok(())
}

/// Fuses for loops over the same vector in a zip into a single for loop which produces multiple
/// builders.
pub fn fuse_loops_horizontal(expr: &mut Expr<Type>) -> WeldResult<()> {
    for child in expr.children_mut() {
        try!(fuse_loops(child));
    }
    let mut sym_gen = SymbolGenerator::from_expression(expr);
    let mut new_expr = None;
    if let For(ref iters1, ref outer_bldr, ref outer_func) = expr.kind {
        // Collapses Zips with Fors over the same vector into a single For which produces multiple
        // results.
        if iters1.len() > 1 {
            // Vector of tuples containing information about functions in nested fors.
            let mut fors = vec![];
            let mut old_bldr = None;
            let mut old_elem = None;
            let mut elem_type = None;
            // TODO(shoumik): Compare the entire Vec<Iter> (just implement PartialEq for all this..)
            let mut inner_for_data: Option<Expr<Type>> = None;
            // First, check if all the fors are over the same vector and have a pattern we can merge.
            if iters1.iter().all(|ref iter| {
                let mut common_data = None;
                if let Res(ref res_bldr) = iter.data.kind {
                    if let For(ref iters2, ref bldr2, ref lambda) = res_bldr.kind {
                        if iters2.iter().all(|ref i| consumes_all(&i)) {
                            // TODO(shoumik): Extend this to work with all kinds of iters.
                            // Basically just not doing this because it's tedious right now, it
                            // will be much easier once PartialEq is implemented for Iter/Expr.
                            if iters2.len() == 1 && iters2[0].start.is_none() && iters2[0].end.is_none() &&
                                iters2[0].stride.is_none() && common_data.unwrap_or(&iters2[0].data).compare(&iters2[0].data) {
                                if let NewBuilder = bldr2.kind {
                                    if let Builder(ref kind) = bldr2.ty {
                                        if let Appender(_) = *kind {
                                            if let Lambda(ref args, ref body) = lambda.kind {
                                                if let Merge(ref bldr,  ref expr) = body.kind {
                                                    if let Ident(ref n) = bldr.kind {
                                                        if *n == args[0].name {
                                                            fors.push((args.clone(), expr.clone())); 
                                                            old_bldr = Some(&args[0].name);
                                                            old_elem = Some(&args[1].name);
                                                            elem_type = Some(&args[1].ty);
                                                            // See comment above.
                                                            if common_data.is_none() {
                                                                common_data = Some(&iters2[0].data);
                                                            }
                                                            inner_for_data = Some(*common_data.unwrap().clone());
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
                // Zip the expressions to create a ``struct of builders'' type.
                let builder_type = Struct(fors.iter().map(|ref t| Builder(Appender(Box::new(t.1.ty.clone())))).collect::<Vec<_>>());
                // Generate symbols and identifier expressions for the new builder and element parameters.
                // TODO(shoumik): The old_bldr and old_elem aren't really needed; better way to gen
                // new symbol?
                let new_elem_sym = sym_gen.new_symbol(&old_elem.as_ref().unwrap().name);
                let new_bldr_sym = sym_gen.new_symbol(&old_bldr.as_ref().unwrap().name);
                let new_elem_expr = Expr {
                    ty: elem_type.unwrap().clone(),
                    kind: Ident(new_elem_sym.clone())
                };
                let new_bldr_expr = Expr {
                    ty: builder_type.clone(),
                    kind: Ident(new_bldr_sym.clone())
                };
                // Substitute old identifiers in the old merge expressions.
                for &mut (ref mut args, ref mut expr) in fors.iter_mut() {
                    let ref bldr_param = args[0].name;
                    let ref elem_param = args[1].name;
                    expr.substitute(bldr_param, &new_bldr_expr);
                    expr.substitute(elem_param, &new_elem_expr);
                }
                // Build up the new expression.
                let new_expr_kind = Merge(Box::new(new_bldr_expr), Box::new(Expr {
                    ty: Struct(fors.iter().map(|ref e| e.1.ty.clone()).collect::<Vec<_>>()),
                    kind: MakeStruct(fors.iter().map(|ref e| *e.1.clone()).collect::<Vec<_>>())
                }));
                let new_merge_expr = Expr {
                    ty: builder_type.clone(),
                    kind: new_expr_kind
                };
                let new_params = vec![
                    Parameter{ty: builder_type.clone(), name: new_bldr_sym},
                    Parameter{ty: elem_type.unwrap().clone(), name: new_elem_sym},
                ];
                let new_func = Expr {
                    ty: Function(new_params.iter().map(|ref p| p.ty.clone()).collect::<Vec<_>>(), Box::new(builder_type.clone())),
                    kind: Lambda(new_params, Box::new(new_merge_expr))
                };

                let new_iter_expr = Expr {
                    ty: Vector(Box::new(elem_type.unwrap().clone())),
                    kind: Res(Box::new(Expr {
                        ty: builder_type.clone(),
                        kind: For(vec![Iter{data: Box::new(inner_for_data.unwrap().clone()), start: None, end: None, stride: None}],
                            Box::new(Expr{ty: builder_type.clone(), kind: NewBuilder}), Box::new(new_func))
                    })),
                };
                new_expr = Some(Expr {
                    ty: expr.ty.clone(),
                    kind: For(vec![Iter{data: Box::new(new_iter_expr), start: None, end: None, stride: None}], outer_bldr.clone(), outer_func.clone())
                });
            }
        }
    }
    if let Some(new) = new_expr {
        *expr = new;
    }
    Ok(())
}

/// Fuses loops where one for loop takes another as it's input, which prevents intermediate results
/// from being materialized.
pub fn fuse_loops(expr: &mut Expr<Type>) -> WeldResult<()> {
    for child in expr.children_mut() {
        try!(fuse_loops(child));
    }
    let mut sym_gen = SymbolGenerator::from_expression(expr);
    let mut new_expr = None;
    if let For(ref iters1, ref bldr1, ref nested) = expr.kind {
        if iters1.len() == 1 {
            let ref iter1 = iters1[0];
            if let Res(ref res_bldr) = iter1.data.kind {
                if let For(ref iters2, ref bldr2, ref lambda) = res_bldr.kind {
                    if iters2.iter().all(|ref i| consumes_all(&i)) {
                        if let NewBuilder = bldr2.kind {
                            if let Builder(ref kind) = bldr2.ty {
                                if let Appender(_) = *kind {
                                    new_expr = Some(Expr{
                                        ty: expr.ty.clone(),
                                        kind: For(iters2.clone(), bldr1.clone(), Box::new(replace_builder(lambda, nested, &mut sym_gen)))
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
        if let Some(new) = new_expr {
        *expr = new;
    }
    Ok(())
}

/// Given an iterator, returns whether the iterator consumes every element of its data vector.
fn consumes_all(iter: &Iter<Type>) -> bool {
    if let &Iter{start: None, end: None, stride: None, ..} = iter {
        return true
    } else if let &Iter{ref data, start: Some(ref start), end: Some(ref end), stride: Some(ref stride), ..} = iter {
        // Looks like Iter(A, 0, Len(A), 1)
        if let I64Literal(1) = stride.kind {
            if let Ident(ref name) = data.kind {
                if let I64Literal(0) = start.kind {
                    if let Length(ref v) = end.kind {
                        if let Ident(ref vsym) = v.kind {
                            if vsym == name {
                                return true
                            }
                        }
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
