use std::collections::HashMap;

use weld_ast::ast::ScalarKind::*;
use weld_ast::ast::Symbol;
use weld_ast::ast::ExprKind::*;
use weld_ast::partial_types::PartialExpr;
use weld_ast::partial_types::PartialType;
use weld_ast::partial_types::PartialType::*;
use weld_ast::partial_types::PartialBuilderKind::*;
use weld_error::*;

#[cfg(test)] use weld_ast::partial_types::expr_box;
#[cfg(test)] use weld_ast::ast::BinOpKind::*;
#[cfg(test)] use weld_parser::*;

type TypeMap = HashMap<Symbol, PartialType>;

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
            push_complete_type(&mut expr.ty, Scalar(I32), "I32Literal"),

        BoolLiteral(_) =>
            push_complete_type(&mut expr.ty, Scalar(Bool), "BoolLiteral"),

        BinOp(_, ref mut lefts, ref mut right) => {
            let mut best_type = Unknown;
            for &ty in [&expr.ty, &lefts.ty, &right.ty].iter() {
                try!(push_type(&mut best_type, ty, "BinOp"));
            }
            let mut changed = false;
            for ty in [&mut expr.ty, &mut lefts.ty, &mut right.ty].iter_mut() {
                changed |= try!(push_type(ty, &best_type, "BinOp"));
            }
            Ok(changed)
        }

        Ident(ref symbol) => {
            match env.get(symbol) {
                None => weld_err!("Undefined identifier: {}", symbol),
                Some(t) => push_type(&mut expr.ty, t, "Ident")
            }
        }

        Let (_, _, ref mut body) => {
            sync_types(&mut expr.ty, &mut body.ty, "Let body")
        }

        MakeVector(ref mut exprs) => {
            let mut changed = false;
            let mut elem_type = Unknown;
            for ref e in exprs.iter() {
                try!(push_type(&mut elem_type, &e.ty, "MakeVector"));
            }
            for ref mut e in exprs.iter_mut() {
                changed |= try!(push_type(
                    &mut e.ty, &elem_type, "MakeVector"));
            }
            let vec_type = Vector(Box::new(elem_type));
            changed |= try!(push_type(&mut expr.ty, &vec_type, "MakeVector"));
            Ok(changed)
        }

        MakeStruct(ref mut elems) => {
            let mut changed = false;

            let base_type = Struct(vec![Unknown; elems.len()]);
            changed |= try!(push_type(&mut expr.ty, &base_type, "MakeStruct"));

            if let Struct(ref mut elem_types) = expr.ty {
                for (elem_ty, elem_expr) in elem_types.iter_mut().zip(elems.iter_mut()) {
                    changed |= try!(sync_types(elem_ty, &mut elem_expr.ty, "MakeStruct"));
                }
            } else {
                return weld_err!("Internal error: type of MakeStruct was not Struct");
            }

            Ok(changed)
        }

        GetField(ref mut param, index) => {
            if let Struct(ref mut elem_types) = param.ty {
                let index = index as usize;
                if index >= elem_types.len() {
                    return weld_err!("Invalid index for GetField")
                }
                sync_types(&mut elem_types[index], &mut expr.ty, "MakeStruct")
            } else if param.ty == Unknown {
                Ok(false)
            } else {
                weld_err!("Internal error: GetField called on {:?}", param.ty)
            }
        }

        Lambda(ref mut params, ref mut body) => {
            let mut changed = false;

            let base_type = Function(vec![Unknown; params.len()], Box::new(Unknown));
            changed |= try!(push_type(&mut expr.ty, &base_type, "Lambda"));

            if let Function(ref mut param_types, ref mut res_type) = expr.ty {
                changed |= try!(sync_types(&mut body.ty, res_type, "Lambda return"));
                for (param_ty, param_expr) in param_types.iter_mut().zip(params.iter_mut()) {
                    changed |= try!(sync_types(param_ty, &mut param_expr.ty, "Lambda parameter"));
                }
            } else {
                return weld_err!("Internal error: type of Lambda was not Function");
            }

            Ok(changed)
        }

        Merge(ref mut builder, ref mut value) => {
            let mut changed = false;
            match builder.ty {
                Builder(ref mut b) => {
                    let mty = b.merge_type_mut();
                    changed |= try!(sync_types(mty, &mut value.ty, "Merge"));
                }
                Unknown => (),
                _ => return weld_err!("Internal error: Merge called on non-builder")
            }
            changed |= try!(sync_types(
                &mut expr.ty, &mut builder.ty, "Merge"));
            Ok(changed)
        }

        Res(ref mut builder) => {
            let mut changed = false;
            match builder.ty {
                Builder(ref mut b) => {
                    let rty = b.result_type();
                    changed |= try!(push_type(&mut expr.ty, &rty, "Res"));
                }
                Unknown => (),
                _ => return weld_err!("Internal error: Result called on non-builder")
            }
            Ok(changed)
        }

        For(ref mut data, ref mut builder, ref mut func) => {
            let mut changed = false;

            // Push data and builder type into func
            let elem_type = match data.ty {
                Vector(ref elem) => *elem.clone(),
                Unknown => Unknown,
                _ => return weld_err!("For")
            };
            let bldr_type = builder.ty.clone();
            let func_type = Function(vec![bldr_type.clone(), elem_type], Box::new(bldr_type));
            changed |= try!(push_type(&mut func.ty, &func_type, "For"));

            // Push func's argument type and return type into builder
            match func.ty {
                Function(ref params, ref result) if params.len() == 2 => {
                    changed |= try!(push_type(
                        &mut builder.ty, &params[0], "For"));
                    changed |= try!(push_type(
                        &mut builder.ty, result.as_ref(), "For"));
                }
                _ => return weld_err!("For")
            };

            // Push builder's type to our expression
            changed |= try!(push_type(&mut expr.ty, &builder.ty, "For"));

            Ok(changed)
        }

        NewBuilder => {
            match expr.ty {
                Unknown | Builder(_) => Ok(false),
                _ => weld_err!("Wrong type ascribed to NewBuilder")
            }
        }

        If(ref mut cond, ref mut on_true, ref mut on_false) => {
            let mut changed = false;
            changed |= try!(push_complete_type(
                &mut cond.ty, Scalar(Bool), "If"));
            changed |= try!(sync_types(&mut expr.ty, &mut on_true.ty, "If"));
            changed |= try!(sync_types(&mut expr.ty, &mut on_false.ty, "If"));
            Ok(changed)
        }

        Apply(ref mut func, ref mut params) => {
            let mut changed = false;

            let func_type = Function(vec![Unknown; params.len()], Box::new(Unknown));
            changed |= try!(push_type(&mut func.ty, &func_type, "Apply"));

            let fty = &mut func.ty;
            match *fty {
                Function(ref mut param_types, ref mut res_type) => {
                    changed |= try!(sync_types(&mut expr.ty, res_type, "Apply result"));
                    for (param_ty, param_expr) in param_types.iter_mut().zip(params.iter_mut()) {
                        changed |= try!(sync_types(
                            param_ty, &mut param_expr.ty, "Apply parameter"));
                    }
                }

                _ => return weld_err!("Internal error: Apply was not called on a function")
            }

            Ok(changed)
        }
    }
}

/// Force the given type to be assigned to a PartialType, or report an error if it has the wrong
/// type. Return a Result indicating whether the option has changed (i.e. a new type as added).
fn push_complete_type(dest: &mut PartialType, src: PartialType, context: &str)
        -> WeldResult<bool> {
    match src {
        Scalar(_) => {
            if *dest == Unknown {
                *dest = src;
                Ok(true)
            } else if *dest == src {
                Ok(false)
            } else {
                weld_err!("Mismatched types in {}", context)
            }
        }

        _ => weld_err!("Internal error: push_complete_type not implemented for {:?}", src)
    }
}

/// Force the type of `dest` to be at least as specific as `src`, or report an error if it has an
/// incompatible type. Return a Result indicating whether the type of `dest` has changed.
fn push_type(dest: &mut PartialType, src: &PartialType, context: &str) -> WeldResult<bool> {
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
            _ => weld_err!("Mismatched types in {}", context)
        },

        Vector(ref mut dest_elem) => match *src {
            Vector(ref src_elem) => push_type(dest_elem, src_elem, context),
            _ => weld_err!("Mismatched types in {}", context)
        },

        Struct(ref mut dest_elems) => match *src {
            Struct(ref src_elems) => {
                let mut changed = false;
                if dest_elems.len() != src_elems.len() {
                    return weld_err!("Mismatched types in {}", context);
                }
                for (dest_elem, src_elem) in dest_elems.iter_mut().zip(src_elems) {
                    changed |= try!(push_type(dest_elem, src_elem, context));
                }
                Ok(changed)
            },
            _ => weld_err!("Mismatched types in {}", context)
        },

        Function(ref mut dest_params, ref mut dest_res) => match *src {
            Function(ref src_params, ref src_res) => {
                let mut changed = false;
                if dest_params.len() != src_params.len() {
                    return weld_err!("Mismatched types in {}", context);
                }
                for (dest_param, src_param) in dest_params.iter_mut().zip(src_params) {
                    changed |= try!(push_type(dest_param, src_param, context));
                }
                changed |= try!(push_type(dest_res, src_res, context));
                Ok(changed)
            },
            _ => weld_err!("Mismatched types in {}", context)
        },

        Builder(Appender(ref mut dest_elem)) => match *src {
            Builder(Appender(ref src_elem)) =>
                push_type(dest_elem.as_mut(), src_elem.as_ref(), context),
            _ => weld_err!("Mismatched types in {}", context)
        },

        _ => weld_err!("Internal error: push_type not implemented for {:?}", dest)
    }
}

/// Force two types to be equal, calling `push_type` in each direction. Return true if any type
/// has changed in this process or an error if the types cannot be made to match.
fn sync_types(t1: &mut PartialType, t2: &mut PartialType, error: &str) -> WeldResult<bool> {
    Ok(try!(push_type(t1, t2, error)) | try!(push_type(t2, t1, error)))
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
