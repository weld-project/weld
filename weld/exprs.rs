//! Constructors for creating typed expressions.

use super::ast::*;
use super::ast::ExprKind::*;
use super::ast::Type::*;
use super::ast::BuilderKind::*;
use super::ast::LiteralKind::*;
use super::error::*;

fn new_expr(kind: ExprKind<Type>, ty: Type) -> WeldResult<Expr<Type>> {
    Ok(Expr {
           kind: kind,
           ty: ty,
           annotations: Annotations::new(),
       })
}

pub fn literal_expr(kind: LiteralKind) -> WeldResult<Expr<Type>> {
    new_expr(Literal(kind),
             Scalar(match kind {
                        BoolLiteral(_) => ScalarKind::Bool,
                        I8Literal(_) => ScalarKind::I8,
                        I32Literal(_) => ScalarKind::I32,
                        I64Literal(_) => ScalarKind::I64,
                        F32Literal(_) => ScalarKind::F32,
                        F64Literal(_) => ScalarKind::F64,
                    }))
}

pub fn ident_expr(symbol: Symbol, ty: Type) -> WeldResult<Expr<Type>> {
    new_expr(Ident(symbol.clone()), ty.clone())
}

pub fn binop_expr(kind: BinOpKind, left: Expr<Type>, right: Expr<Type>) -> WeldResult<Expr<Type>> {
    if left.ty != right.ty {
        weld_err!("Internal error: Mismatched types in binop_expr")
    } else {
        let ty = left.ty.clone();
        new_expr(BinOp {
                     kind: kind,
                     left: Box::new(left),
                     right: Box::new(right),
                 },
                 ty)
    }
}

pub fn cast_expr(kind: ScalarKind, expr: Expr<Type>) -> WeldResult<Expr<Type>> {
    let ty = if let Scalar(_) = expr.ty {
        Scalar(kind)
    } else {
        return weld_err!("Internal error: Mismatched types in cast_expr");
    };
    new_expr(Cast {
                 kind: kind,
                 child_expr: Box::new(expr),
             },
             ty)
}

pub fn negate_expr(expr: Expr<Type>) -> WeldResult<Expr<Type>> {
    let ty = if let Scalar(ref k) = expr.ty {
        Scalar(k.clone())
    } else {
        return weld_err!("Internal error: Mismatched types in negate_expr");
    };
    new_expr(Negate(Box::new(expr)), ty)
}

pub fn tovec_expr(expr: Expr<Type>) -> WeldResult<Expr<Type>> {
    let ty = if let Dict(ref kt, ref vt) = expr.ty {
        Struct(vec![*kt.clone(), *vt.clone()])
    } else {
        return weld_err!("Internal error: Mismatched types in tovec_expr");
    };
    new_expr(ToVec { child_expr: Box::new(expr.clone()) }, ty)
}

pub fn makestruct_expr(exprs: Vec<Expr<Type>>) -> WeldResult<Expr<Type>> {
    let ty = Struct(exprs.iter().map(|e| e.ty.clone()).collect());
    new_expr(MakeStruct { elems: exprs }, ty)
}

pub fn makevector_expr(exprs: Vec<Expr<Type>>) -> WeldResult<Expr<Type>> {
    let ty = exprs[0].ty.clone();
    if exprs.iter().all(|e| e.ty == ty) {
        new_expr(MakeVector { elems: exprs }, Vector(Box::new(ty)))
    } else {
        weld_err!("Internal error: Mismatched types in makevector_expr")
    }
}

pub fn getfield_expr(expr: Expr<Type>, index: u32) -> WeldResult<Expr<Type>> {
    let ty = if let Struct(ref tys) = expr.ty {
        tys[index as usize].clone()
    } else {
        return weld_err!("Internal error: Mismatched types in getfield_expr");
    };
    new_expr(GetField {
                 expr: Box::new(expr),
                 index: index,
             },
             ty)
}

pub fn length_expr(expr: Expr<Type>) -> WeldResult<Expr<Type>> {
    if let Vector(_) = expr.ty {
        new_expr(Length { data: Box::new(expr) }, Scalar(ScalarKind::I64))
    } else {
        weld_err!("Internal error: Mismatched types in length_expr")
    }
}

pub fn lookup_expr(data: Expr<Type>, index: Expr<Type>) -> WeldResult<Expr<Type>> {
    let err = weld_err!("Internal error: Mismatched types in lookup_expr");
    let ty = if let Vector(ref ty) = data.ty {
        *ty.clone()
    } else {
        return err;
    };

    if let Scalar(ScalarKind::I64) = index.ty {
        new_expr(Lookup {
                     data: Box::new(data),
                     index: Box::new(index),
                 },
                 ty)
    } else {
        err
    }
}

pub fn keyexists_expr(data: Expr<Type>, key: Expr<Type>) -> WeldResult<Expr<Type>> {
    let err = weld_err!("Internal error: Mismatched types in keyexists_expr");
    let kt = if let Dict(ref kt, _) = data.ty {
        *kt.clone()
    } else {
        return err;
    };

    if key.ty != kt {
        return err;
    }
    new_expr(KeyExists {
                 data: Box::new(data),
                 key: Box::new(key),
             },
             Scalar(ScalarKind::Bool))
}

pub fn slice_expr(data: Expr<Type>, index: Expr<Type>, size: Expr<Type>) -> WeldResult<Expr<Type>> {
    let mut type_checked = 0;

    if let Vector(_) = data.ty {
        type_checked += 1;
    }

    if let Scalar(ScalarKind::I64) = index.ty {
        type_checked += 1;
    }

    if let Scalar(ScalarKind::I64) = size.ty {
        type_checked += 1;
    }

    if type_checked != 3 {
        return weld_err!("Internal error: Mismatched types in slice_expr");
    }

    let ty = data.ty.clone();
    new_expr(Slice {
                 data: Box::new(data),
                 index: Box::new(index),
                 size: Box::new(size),
             },
             ty)
}

pub fn exp_expr(expr: Expr<Type>) -> WeldResult<Expr<Type>> {
    if expr.ty != Scalar(ScalarKind::F32) && expr.ty != Scalar(ScalarKind::F64) {
        weld_err!("Internal error: Mismatched types in exp_expr")
    } else {
        let ty = expr.ty.clone();
        new_expr(Exp { value: Box::new(expr) }, ty)
    }
}

pub fn let_expr(name: Symbol, value: Expr<Type>, body: Expr<Type>) -> WeldResult<Expr<Type>> {
    let ty = body.ty.clone();
    new_expr(Let {
                 name: name,
                 value: Box::new(value),
                 body: Box::new(body),
             },
             ty)
}

pub fn if_expr(cond: Expr<Type>,
               on_true: Expr<Type>,
               on_false: Expr<Type>)
               -> WeldResult<Expr<Type>> {
    let err = weld_err!("Internal error: Mismatched types in if_expr");
    if cond.ty != Scalar(ScalarKind::Bool) {
        return err;
    }

    if on_true.ty != on_false.ty {
        return err;
    }

    let ty = on_true.ty.clone();
    new_expr(If {
                 cond: Box::new(cond),
                 on_true: Box::new(on_true),
                 on_false: Box::new(on_false),
             },
             ty)
}

pub fn lambda_expr(params: Vec<Parameter<Type>>, body: Expr<Type>) -> WeldResult<Expr<Type>> {
    let ty = Function(params.iter().map(|p| p.ty.clone()).collect(),
                      Box::new(body.ty.clone()));
    new_expr(Lambda {
                 params: params,
                 body: Box::new(body),
             },
             ty)
}

pub fn apply_expr(func: Expr<Type>, params: Vec<Expr<Type>>) -> WeldResult<Expr<Type>> {
    let err = weld_err!("Internal error: Mismatched types in apply_expr");
    let mut passed = false;
    let mut ty = None;
    if let Function(ref param_tys, ref ret_ty) = func.ty {
        if params.iter().zip(param_tys).all(|e| e.0.ty == *e.1) {
            passed = true;
            ty = Some(ret_ty.clone());
        }
    }

    if !passed {
        return err;
    }

    new_expr(Apply {
                 func: Box::new(func),
                 params: params,
             },
             *ty.unwrap())
}

pub fn cudf_expr(sym_name: String,
                 args: Vec<Expr<Type>>,
                 return_ty: Type)
                 -> WeldResult<Expr<Type>> {
    new_expr(CUDF {
                 sym_name: sym_name,
                 args: args,
                 return_ty: Box::new(return_ty.clone()),
             },
             return_ty)
}

pub fn newbuilder_expr(kind: BuilderKind, expr: Option<Expr<Type>>) -> WeldResult<Expr<Type>> {
    let passed = match kind {
        Merger(ref ty, _) => {
            let mut passed = false;
            if let Some(ref e) = expr {
                if &e.ty == ty.as_ref() {
                    passed = true;
                }
            } else {
                // Argument is optional.
                passed = true;
            }
            passed
        }
        VecMerger(ref ty, _) => {
            let mut passed = false;
            if let Some(ref e) = expr {
                if let Vector(ref elem_ty) = e.ty {
                    if elem_ty == ty {
                        passed = true;
                    }
                }
            }
            passed
        }
        _ => expr.is_none(),
    };

    if !passed {
        return weld_err!("Internal error: Mismatched types in newbuilder_expr");
    }

    new_expr(NewBuilder(expr.map(|e| Box::new(e))),
             Builder(kind, Annotations::new()))
}

pub fn for_expr(iters: Vec<Iter<Type>>,
                builder: Expr<Type>,
                func: Expr<Type>)
                -> WeldResult<Expr<Type>> {

    let vec_tys = iters.iter().map(|i| i.data.ty.clone()).collect::<Vec<_>>();
    let mut vec_elem_tys = vec![];
    for ty in vec_tys.iter() {
        if let Vector(ref elem_ty) = *ty {
            vec_elem_tys.push(*elem_ty.clone())
        }
    }

    if vec_tys.len() != vec_elem_tys.len() {
        return weld_err!("Internal error: Mismatched types in for_expr - non vector type in iter");
    }

    let builder_ty = builder.ty.clone();

    if let Function(ref params, ref ret_ty) = func.ty {
        // Make sure the function parameters match.
        let ref param_0_ty = params[0];
        let ref param_1_ty = params[1];
        let ref param_2_ty = params[2];

        // Check builder.
        if param_0_ty != &builder_ty {
            return weld_err!("Internal error: Mismatched types in for_expr - function builder type",);
        }

        // Check the index.
        if *param_1_ty != Scalar(ScalarKind::I64) {
            return weld_err!("Internal error: Mismatched types in for_expr - function index type");
        }

        if iters.len() != vec_elem_tys.len() {
            return weld_err!("Internal error: Mismatched types in for_expr - iters and vec_tys length",);
        }

        // Check the element type.
        if iters.len() == 1 {
            if *param_2_ty != vec_elem_tys[0] {
                return weld_err!("Internal error: Mismatched types in for_expr - function elem type",);
            }
        } else {
            let composite_ty = Struct(vec_elem_tys.clone());
            if *param_2_ty != composite_ty {
                return weld_err!("Internal error: Mismatched types in for_expr - function zipped elem type",);
            }
        }

        // Function return type should match builder type.
        if ret_ty.as_ref() != &builder_ty {
            return weld_err!("Internal error: Mismatched types in for_expr - function return type");
        }
    }

    new_expr(For {
                 iters: iters,
                 builder: Box::new(builder),
                 func: Box::new(func),
             },
             builder_ty)
}

pub fn merge_expr(builder: Expr<Type>, value: Expr<Type>) -> WeldResult<Expr<Type>> {
    let err = weld_err!("Internal error: Mismatched types in merge_expr");
    if let Builder(ref bk, _) = builder.ty {
        match *bk {
            Appender(ref elem_ty) => {
                if elem_ty.as_ref() != &value.ty {
                    return err;
                }
            }
            Merger(ref elem_ty, _) => {
                if elem_ty.as_ref() != &value.ty {
                    return err;
                }
            }
            // TODO other builders...
            _ => {
                return err;
            }
        }
    }

    let ty = builder.ty.clone();
    new_expr(Merge {
                 builder: Box::new(builder),
                 value: Box::new(value),
             },
             ty)
}

pub fn result_expr(builder: Expr<Type>) -> WeldResult<Expr<Type>> {
    let err = weld_err!("Internal error: Mismatched types in result_expr");
    let ty = if let Builder(ref bk, _) = builder.ty {
        match *bk {
            Appender(ref elem_ty) => Vector(elem_ty.clone()),
            Merger(ref elem_ty, _) => *elem_ty.clone(),
            DictMerger(ref kt, ref vt, _) => Dict(kt.clone(), vt.clone()),
            GroupMerger(ref kt, ref vt) => Dict(kt.clone(), Box::new(Vector(vt.clone()))),
            VecMerger(ref elem_ty, _) => Vector(elem_ty.clone()),
        }
    } else {
        return err;
    };
    new_expr(Res { builder: Box::new(builder) }, ty)
}

#[cfg(test)]
use super::pretty_print::*;

#[test]
fn literal_test() {
    let expr = literal_expr(LiteralKind::I32Literal(1)).unwrap();
    assert_eq!(print_expr_without_indent(&expr), "1");
    let expr = literal_expr(LiteralKind::F32Literal(1.0)).unwrap();
    assert_eq!(print_expr_without_indent(&expr), "1.0F");

}

#[test]
fn binop_test() {
    let right = literal_expr(LiteralKind::I32Literal(1)).unwrap();
    let left = literal_expr(LiteralKind::I32Literal(1)).unwrap();
    let expr = binop_expr(BinOpKind::Add, left, right).unwrap();

    assert_eq!(print_expr_without_indent(&expr), "(1+1)");
    assert_eq!(expr.ty, Scalar(ScalarKind::I32));
}

#[test]
fn builder_exprs_test() {
    let bk = BuilderKind::Merger(Box::new(Scalar(ScalarKind::I32)), BinOpKind::Add);
    let builder_type = Builder(bk.clone(), Annotations::new());

    let builder = newbuilder_expr(bk.clone(), None).unwrap();
    assert_eq!(builder.ty, builder_type);

    let i32_literal = literal_expr(LiteralKind::I32Literal(5)).unwrap();
    let f32_literal = literal_expr(LiteralKind::F32Literal(5.0)).unwrap();

    // Construct a Merge expression.
    let merge = merge_expr(builder.clone(), i32_literal.clone()).unwrap();
    assert_eq!(merge.ty, builder_type);

    // Wrong type merged - should fail.
    assert!(merge_expr(builder.clone(), f32_literal.clone()).is_err());

    // Now, create a for loop.
    let vector = makevector_expr(vec![i32_literal.clone()]).unwrap();

    let iter = Iter {
        data: Box::new(vector),
        start: None,
        end: None,
        stride: None,
    };

    // Create a list of params for the for loop's function.
    let params = vec![Parameter {
                          name: Symbol::new("b", 0),
                          ty: builder_type.clone(),
                      },
                      Parameter {
                          name: Symbol::new("i", 0),
                          ty: Scalar(ScalarKind::I64),
                      },
                      Parameter {
                          name: Symbol::new("e", 0),
                          ty: i32_literal.ty.clone(),
                      }];

    let builder_iden = ident_expr(params[0].name.clone(), params[0].ty.clone()).unwrap();
    let elem_iden = ident_expr(params[2].name.clone(), params[2].ty.clone()).unwrap();
    let for_body = merge_expr(builder_iden.clone(), elem_iden.clone()).unwrap();

    let for_func = lambda_expr(params, for_body.clone()).unwrap();
    let for_loop = for_expr(vec![iter], builder.clone(), for_func).unwrap();

    assert_eq!(print_expr_without_indent(&for_loop),
               "for([5],merger[i32,+],|b,i,e|merge(b,e))");

    let result = result_expr(for_loop).unwrap();
    assert_eq!(result.ty, i32_literal.ty);
}
