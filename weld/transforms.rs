//! Common transformations on expressions.

use super::ast::Expr;
use super::ast::ExprKind::*;
use super::error::*;

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

pub fn fuse_loops<T:Clone>(expr: &mut Expr<T>) -> WeldResult<()> {
    for child in expr.children_mut() {
        try!(fuse_loops(child));
    }

    let sym_gen = SymbolGenerator::from_expression(expr);

    // TODO(shoumik): Any way to nest these? box syntax is unstable...
    if let For(mut ref iter1, mut ref bldr1, mut ref func1) = expr.kind {
        let data = &iter1.data;
        if let Res(ref res_bldr) = data {
            if let For(ref iter2, ref bldr2, ref func2) = res_bldr.kind {
                if let NewBuilder = bldr2 {
                    let bldr_type = &bldr2.ty;
                    if let Builder(ref kind) = bldr_type {
                        if let Appender(_) = kind {
                            // Everything matches up...fuse the loops.
                            iter1 = iter2;
                            bldr1 = bldr2;
                            func1 = Box::new(replace_builder(func1, func2, sym_gen));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn apply_lambda<T:Clone>(lambda: &mut Expr<T>, params: Vec<Parameter<T>>) {
    if let Lambda(ref args, ref body) = lambda {
        // TODO(shoumik): Better error handling here.
        assert!(args.len() == params.len());
        let replace = args.iter().zip(params).collect::<HashMap<_,_>>();
        for (arg, param) in args.iter().zip(params) {
            lambda.substitute(arg, param);
        }
    }
}

fn replace_builder<T:Clone>(lambda: &mut Expr<T>,
                            nested: &Expr<T>i,
                            sym_gen: &mut SymbolGenerator) -> Expr<T> {

    if let Lambda(ref args, mut ref body) = lambda {
        let old_bldr = args[0];
        let old_arg = func.args[2];
        let new_bldr = sym_gen.new_symbol(old_bldr.symbol);
        for child in body.children_mut() {
            match child {
                Merge(ref bldr, ref elem) if bldr == old_bldr => {
                    // TODO(shoumik): Index.
                    apply_lambda(lambda, vec![new_bldr, elem])
                },
                For(ref iter, ref mut bldr, ref mut func) i fbldr == old_bldr => {
                    bldr = new_bldr;
                    func = replace_builder(func, nested, sym_gen)
                },
                Iden(ref mut symbol, _) if symbol == old_bldr.symbol {
                    symbol = new_bldr.symbol
                }
            };
        }
    }

    // TODO(shoumik): Index.
    Lambda{params: vec![new_bldr, old_arg], body: body}
}
