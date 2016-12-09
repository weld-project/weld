//! Common transformations on expressions.

use super::ast::*;
use super::ast::ExprKind::*;
use super::ast::Type::*;
use super::ast::BuilderKind::*;
use super::error::*;

use super::pretty_print::*;

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
pub fn inline_apply<T:Clone>(expr: &mut Expr<T>) -> WeldResult<()> {
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

pub fn fuse_loops(expr: &mut Expr<Type>) -> WeldResult<()> {
    for child in expr.children_mut() {
        try!(fuse_loops(child));
    }
    let mut sym_gen = SymbolGenerator::from_expression(expr);
    let mut new_expr = None;
    if let For(ref iter1, ref bldr1, ref nested) = expr.kind {
        if let Res(ref res_bldr) = iter1.data.kind {
            if let For(ref iter2, ref bldr2, ref lambda) = res_bldr.kind {
                if let NewBuilder = bldr2.kind {
                    if let Builder(ref kind) = bldr2.ty {
                        if let Appender(_) = *kind {
                            // Everything matches up...fuse the loops.
                            println!("{}\n --------", print_expr(expr));
                            let mut new = Expr{
                                ty: expr.ty.clone(),
                                kind: For(iter2.clone(), bldr1.clone(), Box::new(replace_builder(lambda, nested, &mut sym_gen)))
                            };
                            try!(inline_apply(&mut new));
                            new_expr = Some(new);
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

/// Returns new function with nested builder applied to outer func.
fn replace_builder(lambda: &Expr<Type>,
                            nested: &Expr<Type>,
                            sym_gen: &mut SymbolGenerator) -> Expr<Type> {

    fn same_builder(a: &ExprKind<Type>, b: &Symbol) -> bool {
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
            // The old builder Parameter.
            let ref old_bldr = args[0];
            // The  old element Parameter.
            let ref old_arg = args[1];
            // A new symbol and identifier for the builder.
            let new_sym = sym_gen.new_symbol(&old_bldr.name.name);
            let new_bldr = Expr{ty: nested_args[0].ty.clone(), kind: Ident(new_sym.clone())};

            let replace = &mut |e: &mut Expr<Type>| {
                let mut new_expr = None;
                match e.kind {
                    Merge(ref bldr, ref elem) if same_builder(&(*bldr).kind, &old_bldr.name) => {
                        let params: Vec<Expr<Type>> = vec![new_bldr.clone(), *elem.clone()];
                        new_expr = Some(Expr{ty: e.ty.clone(), kind: Apply(Box::new(nested.clone()), params)});
                    }
                    For(ref data, ref bldr, ref func) if same_builder(&(*bldr).kind, &old_bldr.name) => {
                        new_expr = Some(Expr{
                            ty: e.ty.clone(),
                            kind: For(data.clone(), Box::new(new_bldr.clone()), Box::new(replace_builder(func, nested, sym_gen)))
                        });
                    },
                    Ident(ref mut symbol) if *symbol == new_sym => {
                        new_expr = Some(new_bldr.clone());
                        println!("replacing Merge!");
                    }
                    _ => {}
                };
                if let Some(new) = new_expr {
                    *e = new;
                }
            };

            // Mutate the new body to replace the nested builder.
            replace(&mut new_body);
            for child in new_body.children_mut() {
                replace(child);
            }

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
