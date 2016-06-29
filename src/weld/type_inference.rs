use std::collections::HashMap;

use super::ast::ScalarKind::*;
use super::ast::Symbol;
use super::ast::ExprKind::*;
use super::error::*;
use super::partial_types::PartialExpr;
use super::partial_types::PartialType;
use super::partial_types::PartialType::*;

#[cfg(test)] use super::grammar::*;
#[cfg(test)] use super::partial_types::expr_box;
#[cfg(test)] use super::ast::BinOpKind::*;

type TypeMap = HashMap<Symbol, PartialType>;

/// Partially inferred types about a function, which are passed down from parent nodes in ASTs.

/// Infer the missing types of all expressions a tree, modifying it in place to set them.
pub fn infer_types(expr: &mut PartialExpr) -> WeldResult<()> {
    // Note: we should also make sure that the types already set in expr are consistent; this will
    // be done by the first call to infer_up.
    loop {
        let mut env = TypeMap::new();
        let res = try!(infer_up(expr, &mut env));
        if res == false {
            if !has_all_types(expr) {
                return weld_err!("Could not infer some types")
            }
            return Ok(())
        }
    }
}

/// Do expr or all of its descendants have types set?
fn has_all_types(expr: &PartialExpr) -> bool {
    if !expr.ty.is_complete() {
        return false;
    }
    expr.children().all(|c| has_all_types(c))
}

/// Infer the types of expressions upward from the leaves of a tree, using infer_locally.
/// Return true if any new expression's type was inferred, or an error if types are inconsistent.
fn infer_up(expr: &mut PartialExpr, env: &mut TypeMap) -> WeldResult<bool> {
    // Remember whether we inferred any new type
    let mut changed = false;

    // For Lets and Lambdas, add the identifiers they (re-)define to env
    let mut old_bindings: Vec<(Symbol, Option<PartialType>)> = Vec::new();
    match expr.kind {
        Let(ref symbol, ref value, _) => {
            old_bindings.push((symbol.clone(), env.insert(symbol.clone(), value.ty.clone())));
        }

        Lambda(ref params, _) => {
            for p in params {
                old_bindings.push(
                    (p.name.clone(), env.insert(p.name.clone(), p.ty.clone())));
            }
        }

        _ => ()
    }

    // Infer types of children first (with new environment)
    for c in expr.children_mut() {
        changed |= try!(infer_up(c, env));
    }

    // Undo the changes to env from Let and Lambda
    for (symbol, opt) in old_bindings {
        match opt {
            Some(old) => env.insert(symbol, old),
            None => env.remove(&symbol)
        };
    }

    // Infer our type
    changed |= try!(infer_locally(expr, env));

    Ok(changed)
}

/// Infer the type of expr or its children locally based on what is known about some of them.
/// Return true if any new expression's type was inferred, or an error if types are inconsistent.  
fn infer_locally(expr: &mut PartialExpr, env: &mut TypeMap) -> WeldResult<bool> {
    match expr.kind {
        I32Literal(_) =>
            push_complete_type(&mut expr.ty, Scalar(I32), "Wrong type ascribed to I32Literal"),

        BoolLiteral(_) =>
            push_complete_type(&mut expr.ty, Scalar(Bool), "Wrong type ascribed to BoolLiteral"),

        BinOp(_, ref mut lefts, ref mut right) => {
            let mut best_type = Unknown;
            for &ty in [&expr.ty, &lefts.ty, &right.ty].iter() {
                try!(push_type(&mut best_type, ty, "Mismatched types for BinOp"));
            }
            let mut changed = false;
            for ty in [&mut expr.ty, &mut lefts.ty, &mut right.ty].iter_mut() {
                changed |= try!(push_type(ty, &best_type, "Mismatched types for BinOp"));
            }
            Ok(changed)
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

        MakeVector(ref mut exprs) if exprs.len() > 0 => {
            let mut changed = false;
            let mut elem_type = Unknown;
            for ref e in exprs.iter() {
                try!(push_type(&mut elem_type, &e.ty, "Mismatched types for MakeVector"));
            }
            for ref mut e in exprs.iter_mut() {
                changed |= try!(push_type(
                    &mut e.ty, &elem_type, "Mismatched types for MakeVector"));
            }
            let vec_type = Vector(Box::new(elem_type));
            changed |= try!(push_type(&mut expr.ty, &vec_type, "Mismatched types for MakeVector"));
            Ok(changed)
        }

        Lambda(ref mut params, ref mut body) => {
            let mut changed = false;

            let base_type = Function(vec![Unknown; params.len()], Box::new(Unknown));
            changed |= try!(push_type(&mut expr.ty, &base_type, "Mismatched types for Lambda"));

            if let Function(ref mut param_types, ref mut res_type) = expr.ty {
                changed |= try!(push_type(res_type, &body.ty,
                    "Mismatched return types for Lambda"));
                changed |= try!(push_type(&mut body.ty, res_type,
                    "Mismatched return types for Lambda"));
                for (param_ty, param_expr) in param_types.iter_mut().zip(params.iter_mut()) {               
                    changed |= try!(push_type(param_ty, &param_expr.ty,
                        "Mismatched parameter types for Lambda"));
                    changed |= try!(push_type(&mut param_expr.ty, param_ty,
                        "Mismatched parameter types for Lambda"));
                }
            } else {
                return weld_err!("Internal error: type of Lambda was not Function");
            }

            Ok(changed)
        }

        Map(ref mut data, ref mut func) => {
            let mut changed = false;

            // Push data's type into func
            let data_type = match data.ty {
                Vector(ref elem_type) => *elem_type.clone(),
                Unknown => Unknown,
                _ => return weld_err!("Mismatched types for Map")
            };
            let result_type = match expr.ty {
                Vector(ref elem_type) => *elem_type.clone(),
                Unknown => Unknown,
                _ => return weld_err!("Mismatched types for Map")
            };
            let func_type = Function(vec![data_type], Box::new(result_type));
            changed |= try!(push_type(&mut func.ty, &func_type, "Mismatched types for Map"));

            // Pull up our type from function
            let our_type = match func.ty {
                Function(_, ref res) => Vector(res.clone()),
                _ => Vector(Box::new(Unknown))
            };
            changed |= try!(push_type(&mut expr.ty, &our_type, "Mismatched types for Map"));

            Ok(changed)
        }

        _ => Ok(false)
    }
}

/// Force the given type to be assigned to a PartialType, or report an error if it has the wrong
/// type. Return a Result indicating whether the option has changed (i.e. a new type as added).
fn push_complete_type(dest: &mut PartialType, src: PartialType, error: &str) -> WeldResult<bool> {
    match src {
        Scalar(_) => {
            if *dest == Unknown {
                *dest = src;
                Ok(true)
            } else if *dest == src {
                Ok(false)
            } else {
                weld_err!("{}", error)
            }
        }

        _ => weld_err!("Internal error: no push_complete_type for {:?}", src)
    }
}

/// Force the type of `dest` to be at least as specific as `src`, or report an error if it has an
/// incompatible type. Return a Result indicating whether the type of `dest` has changed.
fn push_type(dest: &mut PartialType, src: &PartialType, error: &str) -> WeldResult<bool> {
    if *src == Unknown {
        return Ok(false);
    }
    match *dest {
        Unknown => {
            *dest = src.clone();
            Ok(true)
        },

        Scalar(ref d) => match *src {
            Scalar(ref s) if d == s => Ok(false),
            _ => weld_err!("{}", error)
        },

        Vector(ref mut dest_elem) => match *src {
            Vector(ref src_elem) => push_type(dest_elem, src_elem, error),
            _ => weld_err!("{}", error)
        },

        Function(ref mut dest_params, ref mut dest_res) => match *src {
            Function(ref src_params, ref src_res) => {
                let mut changed = false;
                if dest_params.len() != src_params.len() {
                    return weld_err!("{}", error);
                }
                for (dest_param, src_param) in dest_params.iter_mut().zip(src_params) {
                    changed |= try!(push_type(dest_param, src_param, error));
                }
                changed |= try!(push_type(dest_res, src_res, error));
                Ok(changed)
            },
            _ => weld_err!("{}", error)
        },

        _ => weld_err!("push_type not implemented for {:?}", dest)
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
    assert_eq!(e.ty, Scalar(I32));
    
    let mut e = *bool_lit.clone();
    assert!(infer_types(&mut e).is_ok());
    assert_eq!(e.ty, Scalar(Bool));
    
    let mut e = *sum.clone();
    assert!(infer_types(&mut e).is_ok());
    assert_eq!(e.ty, Scalar(I32));

    let mut e = *prod.clone();
    assert!(infer_types(&mut e).is_ok());
    assert_eq!(e.ty, Scalar(I32));
}

#[test]
fn infer_types_let() {
    let mut e = *parse_expr("let a = 1; a + a").unwrap();
    assert!(infer_types(&mut e).is_ok());
    assert_eq!(e.ty, Scalar(I32));

    let mut e = *parse_expr("a + a").unwrap();
    assert!(infer_types(&mut e).is_err());

    let mut e = *parse_expr("let a:i32 = 1; a").unwrap();
    assert!(infer_types(&mut e).is_ok());
    assert_eq!(e.ty, Scalar(I32));

    let mut e = *parse_expr("let a:bool = 1; a").unwrap();
    assert!(infer_types(&mut e).is_err());

    let mut e = *parse_expr("let a = 1; a:bool").unwrap();
    assert!(infer_types(&mut e).is_err());
}
