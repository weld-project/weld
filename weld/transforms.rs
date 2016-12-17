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
    if let Some(new) = new_expr {
        *expr = new;
    }
    Ok(())
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
            new_func = Some(Expr{ty: nested.ty.clone(), kind: Lambda(new_params, Box::new(new_body))})
        }
    }

    if let Some(new) = new_func {
        new
    } else {
        nested.clone()
    }
}
