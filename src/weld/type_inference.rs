use std::collections::HashMap;

use super::ast::ScalarKind::*;
use super::error::*;
use super::parser::*;
use super::parser::Type::*;
use super::parser::ExprKind::*;

#[cfg(test)] use super::ast::BinOpKind::*;

type TypeMap = HashMap<String, Option<Type>>;

pub fn infer_types(expr: &mut Expr) -> WeldResult<()> {
    if has_some_types(expr) {
        return weld_err("Some types are already set")
    }
    loop {
        let mut env = TypeMap::new();
        let res = try!(infer_up(expr, &mut env));
        if res == false {
            if !has_all_types(expr) {
                return weld_err("Could not infer some types")
            }
            return Ok(())
        }
    }
}

fn has_some_types(expr: &Expr) -> bool {
    if expr.ty.is_some() {
        return true;
    }
    expr.children().any(|c| has_some_types(c))
}

fn has_all_types(expr: &Expr) -> bool {
    if expr.ty.is_none() {
        return false;
    }
    expr.children().all(|c| has_all_types(c))
}

/// Infer the types of expressions upward from the leaves of a tree, using infer_locally.
/// Return true if any new expression's type was inferred, or an error if types are inconsistent.
fn infer_up(expr: &mut Expr, env: &mut TypeMap) -> WeldResult<bool> {
    // Remember whether we inferred any new type
    let mut changed = false;

    // Special case: for Let expressions, add their type in env if known
    let mut old_binding = None;
    if let Let(ref symbol, ref value, _) = expr.kind {
        old_binding = env.insert(symbol.clone(), value.ty.clone());
    }

    // Infer types of children first (with new environment) 
    for c in expr.children_mut() {
        changed |= try!(infer_up(c, env));
    }

    // Infer our type
    changed |= try!(infer_locally(expr, env));

    // Undo the environment changes from Let
    if let Let(ref symbol, _, _) = expr.kind {
        match old_binding {
            Some(old) => env.insert(symbol.clone(), old),
            None => env.remove(symbol)
        };
    }

    Ok(changed)
}

/// Infer the type of expr or its children locally based on what is known about some of them.
/// Return true if any new expression's type was inferred, or an error if types are inconsistent.  
fn infer_locally(expr: &mut Expr, env: &mut TypeMap) -> WeldResult<bool> {
    match expr.kind {
        I32Literal(_) => {
            match expr.ty {
                None => {
                    expr.ty = Some(Scalar(I32));
                    Ok(true)
                }
                Some(Scalar(I32)) => Ok(false),
                Some(_) => weld_err("Wrong type ascribed to I32Literal")
            }
        }

        BoolLiteral(_) => {
            match expr.ty {
                None => {
                    expr.ty = Some(Scalar(Bool));
                    Ok(true)
                }
                Some(Scalar(Bool)) => Ok(false),
                Some(_) => weld_err("Wrong type ascribed to BoolLiteral")
            }
        }

        BinOp(_, ref mut left, ref mut right) => {
            let mut types_seen = Vec::<Type>::new();
            for &ty in [&expr.ty, &left.ty, &right.ty].iter() {
                types_seen.extend(ty.clone());
            }
            if !types_seen.is_empty() {
                let first_type = types_seen.pop().unwrap();
                for ty in &types_seen {
                    if *ty != first_type {
                        return weld_err("Mismatched types for BinOp")
                    }
                }
                let mut changed = false;
                for ty in [&mut expr.ty, &mut left.ty, &mut right.ty].iter_mut() {
                    if ty.is_none() {
                        **ty = Some(first_type.clone()); 
                        changed = true;
                    }
                }
                return Ok(changed)
            }
            Ok(false)
        }

        Ident(ref symbol) => {
            match env.get(symbol) {
                None => weld_err("Undefined identifier"),
                Some(ty_opt) => {
                    if ty_opt.is_some() && expr.ty.is_some() && *ty_opt != expr.ty {
                        weld_err("Mismatched types for Ident")
                    } else if ty_opt.is_some() && expr.ty.is_none() {
                        expr.ty = ty_opt.clone();
                        Ok(true)
                    } else {
                        // Don't try to propagate the type up from this Ident to its Let node.
                        Ok(false)
                    }
                } 
            }
        }

        Let (_, _, ref mut body) => {
            if expr.ty.is_some() && body.ty.is_some() && expr.ty != body.ty {
                weld_err("Mismatched types for Let body")
            } else if body.ty.is_some() && expr.ty.is_none() {
                expr.ty = body.ty.clone();
                Ok(true)
            } else if body.ty.is_none() && expr.ty.is_some() {
                body.ty = expr.ty.clone();
                Ok(true)
            } else {
                Ok(false)
            }
        }

        _ => Ok(false)
    }
}

#[test]
fn has_some_types_test() {
    let no_type = Box::new(Expr { ty: None, kind: I32Literal(1) });
    assert_eq!(has_some_types(&no_type), false);
    
    let with_type = Box::new(Expr { ty: Some(Scalar(I32)), kind: I32Literal(1) });
    assert_eq!(has_some_types(&with_type), true);

    let e = Expr { ty: None, kind: BinOp(Add, no_type.clone(), no_type.clone()) };
    assert_eq!(has_some_types(&e), false);

    let e = Expr { ty: None, kind: BinOp(Add, with_type.clone(), with_type.clone()) };
    assert_eq!(has_some_types(&e), true);
}

#[test]
fn infer_types_simple() {
    let int_lit = expr_box(I32Literal(1));
    let bool_lit = expr_box(BoolLiteral(false));
    let sum = expr_box(BinOp(Add, int_lit.clone(), int_lit.clone()));
    let prod = expr_box(BinOp(Multiply, sum.clone(), sum.clone()));

    let mut e = *int_lit.clone();
    assert!(infer_types(&mut e).is_ok());
    assert_eq!(e.ty, Some(Scalar(I32)));
    
    let mut e = *bool_lit.clone();
    assert!(infer_types(&mut e).is_ok());
    assert_eq!(e.ty, Some(Scalar(Bool)));
    
    let mut e = *sum.clone();
    assert!(infer_types(&mut e).is_ok());
    assert_eq!(e.ty, Some(Scalar(I32)));

    let mut e = *prod.clone();
    assert!(infer_types(&mut e).is_ok());
    assert_eq!(e.ty, Some(Scalar(I32)));
}

#[test]
fn infer_types_let() {
    let lit = expr_box(I32Literal(1));
    let lit_sum = expr_box(BinOp(Add, lit.clone(), lit.clone()));
    let ident = expr_box(Ident("a".to_string()));
    let ident_sum = expr_box(BinOp(Add, ident.clone(), ident.clone()));
    let let1 = expr_box(Let("a".to_string(), lit_sum.clone(), ident_sum.clone()));

    let mut e = *let1.clone();
    assert!(infer_types(&mut e).is_ok());
    assert_eq!(e.ty, Some(Scalar(I32)));

    let mut e = *ident_sum.clone();
    assert!(infer_types(&mut e).is_err());
}