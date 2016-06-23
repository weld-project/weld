use std::collections::HashMap;

use super::ast::ScalarKind::*;
use super::error::*;
use super::parser::*;
use super::parser::Type::*;
use super::parser::ExprKind::*;

#[cfg(test)] use super::grammar::parse_expr;
#[cfg(test)] use super::ast::BinOpKind::*;

type TypeMap = HashMap<String, Option<Type>>;

/// Partially inferred types about a function, which are passed down from parent nodes in ASTs.
struct FunctionTypes {
    params: Vec<Option<Type>>,
    result: Option<Type>
}

/// Infer the missing types of all expressions a tree, modifying it in place to set them.
pub fn infer_types(expr: &mut Expr) -> WeldResult<()> {
    // Note: we should also make sure that the types already set in expr are consistent; this will
    // be done by the first call to infer_up.
    loop {
        let mut env = TypeMap::new();
        let res = try!(infer_up(expr, &mut env, None));
        if res == false {
            if !has_all_types(expr) {
                return weld_err("Could not infer some types")
            }
            return Ok(())
        }
    }
}

/// Do expr or all of its descendants have types set?
fn has_all_types(expr: &Expr) -> bool {
    if expr.ty.is_none() {
        return false;
    }
    expr.children().all(|c| has_all_types(c))
}

/// Infer the types of expressions upward from the leaves of a tree, using infer_locally.
/// Return true if any new expression's type was inferred, or an error if types are inconsistent.
fn infer_up(expr: &mut Expr, env: &mut TypeMap, ft: Option<FunctionTypes>) -> WeldResult<bool> {
    // Remember whether we inferred any new type
    let mut changed = false;

    // Special case: for Let expressions, add their type in env if known
    let mut old_binding = None;
    if let Let(ref symbol, ref value, _) = expr.kind {
        old_binding = env.insert(symbol.clone(), value.ty.clone());
    }

    // Infer types of children first (with new environment) 
    for c in expr.children_mut() {
        changed |= try!(infer_up(c, env, None));
    }

    // Infer our type
    changed |= try!(infer_locally(expr, env, None));

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
fn infer_locally(expr: &mut Expr, env: &mut TypeMap, ft: Option<FunctionTypes>) -> WeldResult<bool> {
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

        Lambda(ref mut params, ref mut body) => {
            let (expected_params, expected_result) = match ft {
                Some(FunctionTypes { params, result }) => (params, result),
                None => (vec![None; params.len()], None)
            };

            let mut changed = false;

            // Harmonize parameter types
            for (i, p) in params.iter_mut().enumerate() {
                let expected = expected_params.get(i).unwrap();
                if p.ty.is_some() && expected.is_some() && p.ty != *expected {
                    return weld_err("Mismatched parameter types for Lambda");
                } else if p.ty.is_none() && expected.is_some() {
                    p.ty = expected.clone();
                    changed = true;
                }
            }

            // Harmonize body type
            if body.ty.is_some() && expected_result.is_some() && body.ty != expected_result {
                return weld_err("Mismatched return type for Lambda");
            } else if body.ty.is_none() && expected_result.is_some() {
                body.ty = expected_result.clone();
                changed = true;
            }

            // Set our overall type if we can
            let have_params = params.iter().all(|p| p.ty.is_some());
            if have_params && body.ty.is_some() {
                let param_types = params.iter().map(|p| p.ty.clone().unwrap()).collect();
                let func_type = Some(Function(param_types, Box::new(body.ty.clone().unwrap())));
                if expr.ty.is_some() && expr.ty != func_type {
                    return weld_err("Mismatched overall function type for Lambda");
                } else if expr.ty.is_none() {
                    expr.ty = func_type;
                    changed = true;
                }
            }

            Ok(changed)
        }

        _ => Ok(false)
    }
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
    let mut e = *parse_expr("a := 1; a + a").unwrap();
    assert!(infer_types(&mut e).is_ok());
    assert_eq!(e.ty, Some(Scalar(I32)));

    let mut e = *parse_expr("a + a").unwrap();
    assert!(infer_types(&mut e).is_err());

    let mut e = *parse_expr("a:i32 := 1; a").unwrap();
    assert!(infer_types(&mut e).is_ok());
    assert_eq!(e.ty, Some(Scalar(I32)));

    let mut e = *parse_expr("a:bool := 1; a").unwrap();
    assert!(infer_types(&mut e).is_err());

    let mut e = *parse_expr("a := 1; a:bool").unwrap();
    assert!(infer_types(&mut e).is_err());
}
