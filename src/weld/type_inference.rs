use std::collections::HashMap;

use super::ast::*;
use super::ast::ScalarKind::*;
use super::ast::Type::*;
use super::ast::ExprKind::*;
use super::error::*;

#[cfg(test)] use super::parser::parse_expr;
#[cfg(test)] use super::ast::BinOpKind::*;

type TypeMap = HashMap<Symbol, Option<Type>>;

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
                return weld_err!("Could not infer some types")
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
fn infer_up(expr: &mut Expr, env: &mut TypeMap, fts: Option<FunctionTypes>) -> WeldResult<bool> {
    // Remember whether we inferred any new type
    let mut changed = false;

    // Special case: for Lets and Lambdas, add the types of identifiers they (re-)define to env.
    let mut old_bindings: Vec<(Symbol, Option<Option<Type>>)> = Vec::new();
    match expr.kind {
        Let(ref symbol, ref value, _) => {
            old_bindings.push((symbol.clone(), env.insert(symbol.clone(), value.ty.clone())));
        }

        Lambda(ref params, _) => {
            for p in params {
                old_bindings.push((p.name.clone(), env.insert(p.name.clone(), p.ty.clone())));
            }
        }

        _ => ()
    }

    // Infer types of children first (with new environment)
    match expr.kind {
        Map(ref mut data, ref mut func) => {
            changed |= try!(infer_up(data, env, None));
            let data_type = match data.ty {
                Some(Vector(ref elem_type)) => Some(*elem_type.clone()),
                Some(_) => return weld_err!("Mismatched types for Map"),
                _ => None
            };
            let result_type = match expr.ty {
                Some(Vector(ref elem_type)) => Some(*elem_type.clone()),
                Some(_) => return weld_err!("Mismatched types for Map"),
                _ => None
            };
            let expected_fts = FunctionTypes { params: vec![data_type], result: result_type };
            changed |= try!(infer_up(func, env, Some(expected_fts)));
        }

        _ => {
            for c in expr.children_mut() {
                changed |= try!(infer_up(c, env, None));
            }
        }
    }

    // Infer our type
    changed |= try!(infer_locally(expr, env, fts));

    // Undo the environment changes from Let
    for (symbol, opt) in old_bindings {
        match opt {
            Some(old) => env.insert(symbol, old),
            None => env.remove(&symbol)
        };
    }

    Ok(changed)
}

/// Infer the type of expr or its children locally based on what is known about some of them.
/// Return true if any new expression's type was inferred, or an error if types are inconsistent.  
fn infer_locally(expr: &mut Expr, env: &mut TypeMap, fts: Option<FunctionTypes>) -> WeldResult<bool> {
    match expr.kind {
        I32Literal(_) =>
            push_concrete_type(&mut expr.ty, Scalar(I32), "Wrong type ascribed to I32Literal"),

        BoolLiteral(_) =>
            push_concrete_type(&mut expr.ty, Scalar(Bool), "Wrong type ascribed to BoolLiteral"),

        BinOp(_, ref mut lefts, ref mut right) => {
            let mut types_seen = Vec::<Type>::new();
            for &ty in [&expr.ty, &lefts.ty, &right.ty].iter() {
                types_seen.extend(ty.clone());
            }
            if !types_seen.is_empty() {
                let first_type = Some(types_seen.pop().unwrap());
                let mut changed = false;
                for ty in [&mut expr.ty, &mut lefts.ty, &mut right.ty].iter_mut() {
                    changed |= try!(push_type(ty, &first_type, "Mismatched types for BinOp"));
                }
                return Ok(changed)
            }
            Ok(false)
        }

        Ident(ref symbol) => {
            match env.get(symbol) {
                None => weld_err!("Undefined identifier"),
                Some(t) => push_type(&mut expr.ty, t, "Mismatched types for Ident")
            }
        }

        Let (_, _, ref mut body) => {
            let mut changed = false;
            changed |= try!(push_type(&mut expr.ty, &body.ty, "Mismatched types for Let body"));
            changed |= try!(push_type(&mut body.ty, &expr.ty, "Mismatched types for Let body"));
            Ok(changed)
        }

        MakeVector(ref exprs) if exprs.len() > 0 => {
            // TODO: check types of all other vector elements!
            let first_expr = exprs.first().unwrap();
            match first_expr.ty {
                Some(ref elem_type) => {
                    for other_expr in exprs {
                        if other_expr.ty != first_expr.ty {
                            return weld_err!("Mismatched types for MakeVector");
                        }
                    }
                    let vec_type = Some(Vector(Box::new(elem_type.clone())));
                    push_type(&mut expr.ty, &vec_type, "Mismatched types for MakeVector")
                }
                None => Ok(false)
            }
        }

        Lambda(ref mut params, ref mut body) => {
            let (expected_params, expected_result) = match fts {
                Some(FunctionTypes { params, result }) => (params, result),
                None => (vec![None; params.len()], None)
            };

            let mut changed = false;

            // Harmonize parameter types
            for (i, p) in params.iter_mut().enumerate() {
                let expected = expected_params.get(i).unwrap();
                changed |= try!(push_type(&mut p.ty, &expected, "Mismatched Lambda param types"));
            }

            // Harmonize body type
            changed |= try!(push_type(
                &mut body.ty, &expected_result, "Mismatched Lambda return type"));

            // Set our overall type if we can
            let have_params = params.iter().all(|p| p.ty.is_some());
            if have_params && body.ty.is_some() {
                let param_types = params.iter().map(|p| p.ty.clone().unwrap()).collect();
                let func_type = Some(Function(param_types, Box::new(body.ty.clone().unwrap())));
                changed |= try!(push_type(&mut expr.ty, &func_type, "Mismatched type for Lambda"))
            }

            Ok(changed)
        }

        Map(_, ref func) => {
            let expected_type = match func.ty {
                Some(Function(_, ref res)) => Some(Vector(res.clone())),
                _ => None
            };
            push_type(&mut expr.ty, &expected_type, "Mismatched types for Map")
        }

        _ => Ok(false)
    }
}

/// Force the given type to be assigned to an Option<Type>, or report an error if it has the wrong
/// type. Return a result indicating whether the option has changed (i.e. a new type as added).
fn push_concrete_type(opt: &mut Option<Type>, expected: Type, error: &str) -> WeldResult<bool> {
    match *opt {
        None => {
            *opt = Some(expected);
            Ok(true) 
        }
        Some(ref t) if *t == expected => Ok(false),
        _ => weld_err!("{}", error)
    }
}

/// Force the type of `dest` to be `src` if `src` is set, or report an error if it has the wrong
/// type. Return a result indicating whether the type of `dest` has changed.
fn push_type(dest: &mut Option<Type>, src: &Option<Type>, error: &str) -> WeldResult<bool> {
    if dest.is_some() && src.is_some() && *dest != *src {
        weld_err!("{}", error)
    } else if src.is_some() && dest.is_none() {
        *dest = src.clone();
        Ok(true)
    } else {
        Ok(false)
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
    let mut e = *parse_expr("let a = 1; a + a").unwrap();
    assert!(infer_types(&mut e).is_ok());
    assert_eq!(e.ty, Some(Scalar(I32)));

    let mut e = *parse_expr("a + a").unwrap();
    assert!(infer_types(&mut e).is_err());

    let mut e = *parse_expr("let a:i32 = 1; a").unwrap();
    assert!(infer_types(&mut e).is_ok());
    assert_eq!(e.ty, Some(Scalar(I32)));

    let mut e = *parse_expr("let a:bool = 1; a").unwrap();
    assert!(infer_types(&mut e).is_err());

    let mut e = *parse_expr("let a = 1; a:bool").unwrap();
    assert!(infer_types(&mut e).is_err());
}
