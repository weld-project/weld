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

    // TODO(shoumik): Any way to nest these? box syntax is unstable...
    if let For(ref mut iter1, ref mut bldr1, ref mut nested) = (*expr).kind {
        let data = &mut iter1.data;
        if let Res(ref mut res_bldr) = (*data).kind {
            if let For(ref mut iter2, ref mut bldr2, ref lambda) = (*res_bldr).kind.clone() {
                if let NewBuilder = (*bldr2).kind {
                    let bldr_type = &bldr2.ty;
                    if let Builder(ref kind) = *bldr_type {
                        if let Appender(_) = *kind {
                            // Everything matches up...fuse the loops.
                            new_expr = Some(Expr{
                                ty: expr.ty.clone(),
                                kind: For(iter2.clone(), bldr1.clone(), Box::new(replace_builder(lambda, nested, &mut sym_gen)))
                            })
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

/// Takes a lambda expression and a set of expressions as parameters 
/// and applies the parameters to the lambda.
fn apply_lambda(lambda: &mut Expr<Type>, params: Vec<&Expr<Type>>) {
    let l_args: Vec<_>;
    if let Lambda(ref args, _) = (*lambda).kind {
        l_args = (*args).clone();
    } else {
        l_args = vec![];
    }
    assert!(l_args.len() == params.len());
    for (arg, param) in l_args.iter().zip(params) {
        lambda.substitute(&arg.name, param);
    }
}

fn replace_builder(lambda: &Expr<Type>,
                            nested: &mut Expr<Type>,
                            sym_gen: &mut SymbolGenerator) -> Expr<Type> {

    fn same_builder(a: &ExprKind<Type>, b: &Symbol) -> bool {
        if let Ident(ref symbol)  = *a { 
            symbol == b 
        } else {
            false
        }
    }

    let mut new_body = None;

    let new_params = if let Lambda(ref args, ref mut body) = lambda.clone().kind {
        new_body = Some(*body.clone());
        let ref old_bldr = args[0];
        // TODO(shoumik): index (change 1 -> 2).
        let ref old_arg = args[1];
        let new_sym = sym_gen.new_symbol(&old_bldr.name.name);
        let new_bldr = Expr{ty: nested.ty.clone(), kind: Ident(new_sym.clone())};
        for child in body.children_mut() {
            match child.kind {
                Merge(ref bldr, ref elem) if same_builder(&(*bldr).kind, &old_bldr.name) => {
                    apply_lambda(nested, vec![&new_bldr, elem]);
                }
                For(_, ref mut bldr, ref mut func) if same_builder(&(*bldr).kind, &old_bldr.name) => {
                    bldr.kind = new_bldr.kind.clone();
                    func.kind = replace_builder(func, nested, sym_gen).kind.clone();
                },
                Ident(ref mut symbol) if *symbol == new_sym => {
                    *symbol = new_sym.clone();
                }
                _ => {}
            };
        }

        vec![
            Parameter{ty: new_bldr.ty.clone(), name: new_sym.clone()},
            Parameter{ty: old_arg.ty.clone(), name: old_arg.name.clone()}
        ]
    } else {
        // TODO(shoumik): ...
        vec![]
    };

    if let Some(new) = new_body {
        Expr{ty: nested.ty.clone(), kind: Lambda(new_params, Box::new(new))}
    } else {
        nested.clone()
    }
        
}
