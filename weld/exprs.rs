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

pub fn cast_expr(kind: ScalarKind, expr: &Expr<Type>) -> WeldResult<Expr<Type>> {
    if let Scalar(_) = expr.ty {
        new_expr(Cast {
                     kind: kind,
                     child_expr: Box::new(expr.clone()),
                 },
                 Scalar(kind))
    } else {
        weld_err!("Internal error: Mismatched types in cast_expr")
    }
}

pub fn tovec_expr(expr: &Expr<Type>) -> WeldResult<Expr<Type>> {
    if let Dict(ref kt, ref vt) = expr.ty {
        let ty = Struct(vec![*kt.clone(), *vt.clone()]);
        new_expr(ToVec { child_expr: Box::new(expr.clone()) }, ty)
    } else {
        weld_err!("Internal error: Mistmatched types in tovec_expr")
    }
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
        weld_err!("Internal error: Mistmatched types in makevector_expr")
    }
}

pub fn getfield_expr(expr: Expr<Type>, index: u32) -> WeldResult<Expr<Type>> {
    let ty = if let Struct(ref tys) = expr.ty {
        tys[index as usize].clone()
    } else {
        return weld_err!("Internal error: Mistmatched types in makevector_expr");
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
        weld_err!("Internal error: Mistmatched types in length_expr")
    }
}

pub fn lookup_expr(data: Expr<Type>, index: Expr<Type>) -> WeldResult<Expr<Type>> {
    let err = weld_err!("Internal error: Mistmatched types in lookup_expr");
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
    let builder_ty = builder.ty.clone();

    if let Function(ref params, ref ret_ty) = func.ty {
        // Make sure the function parameters match.
        let ref param_0_ty = params[0];
        let ref param_1_ty = params[1];
        let ref param_2_ty = params[2];

        if iters.len() != vec_tys.len() {
            return weld_err!("Internal error: Mismatched types in for_expr - iters and vec_tys length",);
        }

        // Check the first function parameter type.
        if iters.len() == 1 {
            if *param_0_ty != vec_tys[0] {
                return weld_err!("Internal error: Mismatched types in for_expr - function elem type",);
            }
        } else {
            let composite_ty = Struct(vec_tys.clone());
            if *param_0_ty != composite_ty {
                return weld_err!("Internal error: Mismatched types in for_expr - function zipped elem type",);
            }
        }

        // Check the index.
        if *param_1_ty != Scalar(ScalarKind::I64) {
            return weld_err!("Internal error: Mismatched types in for_expr - function index type");
        }

        // Check builder.
        if param_2_ty != &builder_ty {
            return weld_err!("Internal error: Mismatched types in for_expr - function builder type",);
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
            _ => {
                return err;
            }
        }
    } else {
        return err;
    };
    new_expr(Res { builder: Box::new(builder) }, ty)
}
