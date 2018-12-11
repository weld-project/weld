//! Builder API for Weld expressions.

use ast::*;
use ast::ExprKind::*;
use ast::Type::*;
use ast::BuilderKind::*;
use ast::LiteralKind::*;
use error::*;

pub trait NewExpr {
    fn new(kind: ExprKind, ty: Type) -> WeldResult<Expr>;
    fn literal(kind: LiteralKind) -> WeldResult<Expr>;
    fn ident(symbol: Symbol, ty: Type) -> WeldResult<Expr>;
    fn binop(kind: BinOpKind, left: Expr, right: Expr) -> WeldResult<Expr>;
    fn unaryop(kind: UnaryOpKind, value: Expr) -> WeldResult<Expr>;
    fn cast(kind: ScalarKind, expr: Expr) -> WeldResult<Expr>;
    fn negate(expr: Expr) -> WeldResult<Expr>;
    fn not(expr: Expr) -> WeldResult<Expr>;
    fn broadcast(expr: Expr) -> WeldResult<Expr>;
    fn tovec(expr: Expr) -> WeldResult<Expr>;
    fn makestruct(exprs: Vec<Expr>) -> WeldResult<Expr>;
    fn makevector(exprs: Vec<Expr>) -> WeldResult<Expr>;
    fn makevector_typed(exprs: Vec<Expr>, ty: Type) -> WeldResult<Expr>;
    fn getfield(expr: Expr, index: u32) -> WeldResult<Expr>;
    fn length(expr: Expr) -> WeldResult<Expr>;
    fn lookup(data: Expr, index: Expr) -> WeldResult<Expr>;
    fn keyexists(data: Expr, key: Expr) -> WeldResult<Expr>;
    fn slice(data: Expr, index: Expr, size: Expr) -> WeldResult<Expr>;
    fn sort(data: Expr, cmpfunc: Expr) -> WeldResult<Expr>;
    fn let(name: Symbol, value: Expr, body: Expr) -> WeldResult<Expr>;
    fn if(cond: Expr, on_true: Expr, on_false: Expr) -> WeldResult<Expr>;
    fn select(cond: Expr, on_true: Expr, on_false: Expr) -> WeldResult<Expr>;
    fn lambda(params: Vec<Parameter>, body: Expr) -> WeldResult<Expr>;
    fn apply(func: Expr, params: Vec<Expr>) -> WeldResult<Expr>;
    fn cudf(sym_name: String, args: Vec<Expr>, return_ty: Type) -> WeldResult<Expr>;
    fn newbuilder(kind: BuilderKind, expr: Option<Expr>) -> WeldResult<Expr>;
    fn for(iters: Vec<Iter>, builder: Expr, func: Expr, vectorized: bool) -> WeldResult<Expr>;
    fn merge(builder: Expr, value: Expr) -> WeldResult<Expr>;
    fn result(builder: Expr) -> WeldResult<Expr>;
}

impl NewExpr for Expr {

    pub fn new(kind: ExprKind, ty: Type) -> WeldResult<Expr> {
        Ok(Expr {
            kind: kind,
            ty: ty,
            annotations: Annotations::new(),
        })
    }

    pub fn literal(kind: LiteralKind) -> WeldResult<Expr> {
        let ty = match kind {
            BoolLiteral(_) => Scalar(ScalarKind::Bool),
            I8Literal(_) => Scalar(ScalarKind::I8),
            I16Literal(_) => Scalar(ScalarKind::I16),
            I32Literal(_) => Scalar(ScalarKind::I32),
            I64Literal(_) => Scalar(ScalarKind::I64),
            U8Literal(_) => Scalar(ScalarKind::U8),
            U16Literal(_) => Scalar(ScalarKind::U16),
            U32Literal(_) => Scalar(ScalarKind::U32),
            U64Literal(_) => Scalar(ScalarKind::U64),
            F32Literal(_) => Scalar(ScalarKind::F32),
            F64Literal(_) => Scalar(ScalarKind::F64),
            StringLiteral(_) => Vector(Box::new(Scalar(ScalarKind::I8))),
        };

        Self::new(Literal(kind.clone()), ty)
    }

    pub fn ident(symbol: Symbol, ty: Type) -> WeldResult<Expr> {
        Self::new(Ident(symbol), ty)
    }

    pub fn binop(kind: BinOpKind, left: Expr, right: Expr) -> WeldResult<Expr> {
        if left.ty != right.ty {
            return compile_err!("Internal error: Mismatched types in binary operator");
        }

        let ty = if kind.is_comparison() {
            Scalar(ScalarKind::Bool)
        } else {
            left.ty.clone()
        };

        Self::new(BinOp { kind: kind, left: Box::new(left), right: Box::new(right) }, ty)
    }

    pub fn unaryop(kind: UnaryOpKind, value: Expr) -> WeldResult<Expr> {
        let ty = value.ty.clone();
        Self::new(UnaryOp { kind: kind, value: Box::new(value) }, value.ty.clone())
    }

    pub fn cast(kind: ScalarKind, expr: Expr) -> WeldResult<Expr> {
        let ty = if let Scalar(_) = expr.ty {
            Scalar(kind)
        } else {
            return compile_err!("Internal error: Mismatched types inSelf::cast");
        };
        Self::new(Cast {
            kind: kind,
            Self::child: Box::new(expr),
        },
        ty)
    }

    pub fn negate(expr: Expr) -> WeldResult<Expr> {
        let ty = if let Scalar(ref k) = expr.ty {
            Scalar(k.clone())
        } else {
            return compile_err!("Internal error: Mismatched types inSelf::negate");
        };
        Self::new(Negate(Box::new(expr)), ty)
    }

    pub fn not(expr: Expr) -> WeldResult<Expr> {
        let ty = if let Scalar(ref k) = expr.ty {
            Scalar(k.clone())
        } else {
            return compile_err!("Internal error: Mismatched types inSelf::not");
        };
        Self::new(Not(Box::new(expr)), ty)
    }

    pub fn broadcast(expr: Expr) -> WeldResult<Expr> {
        let ty = if let Scalar(ref k) = expr.ty {
            Simd(k.clone())
        } else {
            return compile_err!("Internal error: Mismatched types inSelf::broadcast");
        };
        Self::new(Broadcast(Box::new(expr)), ty)
    }

    pub fn tovec(expr: Expr) -> WeldResult<Expr> {
        let ty = if let Dict(ref kt, ref vt) = expr.ty {
            Struct(vec![*kt.clone(), *vt.clone()])
        } else {
            return compile_err!("Internal error: Mismatched types inSelf::tovec");
        };
        Self::new(ToVec {Self::child: Box::new(expr.clone()) }, ty)
    }

    pub fn makestruct(exprs: Vec<Expr>) -> WeldResult<Expr> {
        let ty = Struct(exprs.iter().map(|e| e.ty.clone()).collect());
        Self::new(MakeStruct { elems: exprs }, ty)
    }

    pub fn makevector(exprs: Vec<Expr>) -> WeldResult<Expr> {
        let ty = exprs[0].ty.clone();
        if exprs.iter().all(|e| e.ty == ty) {
            Self::new(MakeVector { elems: exprs }, Vector(Box::new(ty)))
        } else {
            compile_err!("Internal error: Mismatched types inSelf::makevector")
        }
    }

    /// Version ofSelf::makevector that is compatible with empty vectors.
    pub fn makevector_typed(exprs: Vec<Expr>, ty: Type) -> WeldResult<Expr> {
        if exprs.iter().all(|e| e.ty == ty) {
            Self::new(MakeVector { elems: exprs }, Vector(Box::new(ty.clone())))
        } else {
            compile_err!("Internal error: Mismatched types inSelf::makevector")
        }
    }

    pub fn getfield(expr: Expr, index: u32) -> WeldResult<Expr> {
        let ty = if let Struct(ref tys) = expr.ty {
            tys[index as usize].clone()
        } else {
            return compile_err!("Internal error: Mismatched types inSelf::getfield");
        };
        Self::new(GetField {
            expr: Box::new(expr),
            index: index,
        },
        ty)
    }

    pub fn length(expr: Expr) -> WeldResult<Expr> {
        if let Vector(_) = expr.ty {
            Self::new(Length { data: Box::new(expr) }, Scalar(ScalarKind::I64))
        } else {
            compile_err!("Internal error: Mismatched types inSelf::length")
        }
    }

    pub fn lookup(data: Expr, index: Expr) -> WeldResult<Expr> {
        let err = compile_err!("Internal error: Mismatched types inSelf::lookup");
        let ty = if let Vector(ref ty) = data.ty {
            *ty.clone()
        } else {
            return err;
        };

        if let Scalar(ScalarKind::I64) = index.ty {
            Self::new(Lookup {
                data: Box::new(data),
                index: Box::new(index),
            },
            ty)
        } else {
            err
        }
    }

    pub fn keyexists(data: Expr, key: Expr) -> WeldResult<Expr> {
        let err = compile_err!("Internal error: Mismatched types inSelf::keyexists");
        let kt = if let Dict(ref kt, _) = data.ty {
            *kt.clone()
        } else {
            return err;
        };

        if key.ty != kt {
            return err;
        }
        Self::new(KeyExists {
            data: Box::new(data),
            key: Box::new(key),
        },
        Scalar(ScalarKind::Bool))
    }

    pub fn slice(data: Expr, index: Expr, size: Expr) -> WeldResult<Expr> {
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
            return compile_err!("Internal error: Mismatched types inSelf::slice");
        }

        let ty = data.ty.clone();
        Self::new(Slice {
            data: Box::new(data),
            index: Box::new(index),
            size: Box::new(size),
        },
        ty)
    }

    pub fn sort(data: Expr, cmpfunc: Expr) -> WeldResult<Expr> {
        let mut type_checked = false;

        if let Vector(ref vec_ty) = data.ty {
            if let Function(ref params, ref body) = cmpfunc.ty {
                if params.len() == 2 && params[0] == **vec_ty {
                    // Return type must be i32.
                    if let Scalar(ScalarKind::I32) = **body {
                        // Both parameters must be of the same type.
                        if params[0] == params[1] {
                            type_checked = true;
                        }
                    }
                }
            }
        }

        if !type_checked {
            return compile_err!("Internal error: Mismatched types inSelf::sort")
        }

        let ty = data.ty.clone();
        Self::new(Sort {
            data: Box::new(data),
            cmpfunc: Box::new(cmpfunc),
        },
        ty)
    }

    pub fn let(name: Symbol, value: Expr, body: Expr) -> WeldResult<Expr> {
        let ty = body.ty.clone();
        Self::new(Let {
            name: name,
            value: Box::new(value),
            body: Box::new(body),
        },
        ty)
    }

    pub fn if(cond: Expr, on_true: Expr, on_false: Expr) -> WeldResult<Expr> {
        let err = compile_err!("Internal error: Mismatched types inSelf::if");
        if cond.ty != Scalar(ScalarKind::Bool) {
            return err;
        }

        if on_true.ty != on_false.ty {
            return err;
        }

        let ty = on_true.ty.clone();
        Self::new(If {
            cond: Box::new(cond),
            on_true: Box::new(on_true),
            on_false: Box::new(on_false),
        },
        ty)
    }

    pub fn select(cond: Expr, on_true: Expr, on_false: Expr) -> WeldResult<Expr> {
        let err = compile_err!("Internal error: Mismatched types inSelf::select");
        if cond.ty != Scalar(ScalarKind::Bool) && cond.ty != Simd(ScalarKind::Bool) {
            return err;
        }

        if on_true.ty != on_false.ty {
            return err;
        }

        let ty = on_true.ty.clone();
        Self::new(Select {
            cond: Box::new(cond),
            on_true: Box::new(on_true),
            on_false: Box::new(on_false),
        },
        ty)
    }

    pub fn lambda(params: Vec<Parameter>, body: Expr) -> WeldResult<Expr> {
        let ty = Function(params.iter().map(|p| p.ty.clone()).collect(),
        Box::new(body.ty.clone()));
        Self::new(Lambda {
            params: params,
            body: Box::new(body),
        },
        ty)
    }

    pub fn apply(func: Expr, params: Vec<Expr>) -> WeldResult<Expr> {
        let err = compile_err!("Internal error: Mismatched types inSelf::apply");
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

        Self::new(Apply {
            func: Box::new(func),
            params: params,
        },
        *ty.unwrap())
    }

    pub fn cudf(sym_name: String, args: Vec<Expr>, return_ty: Type) -> WeldResult<Expr> {
        Self::new(CUDF {
            sym_name: sym_name,
            args: args,
            return_ty: Box::new(return_ty.clone()),
        },
        return_ty)
    }

    pub fn newbuilder(kind: BuilderKind, expr: Option<Expr>) -> WeldResult<Expr> {
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
            Appender(_) => {
                let mut passed = false;
                if let Some(ref e) = expr {
                    if let Scalar(ScalarKind::I64) = e.ty {
                        passed = true;
                    }
                }
                passed
            }
            _ => expr.is_none(),
        };

        if !passed {
            return compile_err!("Internal error: Mismatched types inSelf::newbuilder");
        }

        Self::new(NewBuilder(expr.map(|e| Box::new(e))),
        Builder(kind, Annotations::new()))
    }

    // TODO - the vectorized flag is temporary!
    pub fn for(iters: Vec<Iter>, builder: Expr, func: Expr, vectorized: bool) -> WeldResult<Expr> {

        let vec_tys = iters.iter().map(|i| i.data.ty.clone()).collect::<Vec<_>>();
        let mut vec_elem_tys = vec![];
        for ty in vec_tys.iter() {
            if let Vector(ref elem_ty) = *ty {
                vec_elem_tys.push(*elem_ty.clone())
            }
        }

        if vec_tys.len() != vec_elem_tys.len() {
            return compile_err!("Internal error: Mismatched types inSelf::for - non vector type in iter");
        }

        let builder_ty = builder.ty.clone();

        if let Function(ref params, ref ret_ty) = func.ty {
            // Make sure the function parameters match.
            let ref param_0_ty = params[0];
            let ref param_1_ty = params[1];
            let ref param_2_ty = params[2];

            // Check builder.
            if param_0_ty != &builder_ty {
                return compile_err!("Internal error: Mismatched types inSelf::for - function builder type",);
            }

            // Check the index.
            if *param_1_ty != Scalar(ScalarKind::I64) {
                return compile_err!("Internal error: Mismatched types inSelf::for - function index type");
            }

            if iters.len() != vec_elem_tys.len() {
                return compile_err!("Internal error: Mismatched types inSelf::for - iters and vec_tys length",);
            }

            // Check the element type.
            if iters.len() == 1 {
                let elem_ty = if vectorized {
                    if let Scalar(ref sk) = vec_elem_tys[0] {
                        Simd(sk.clone())
                    } else {
                        return compile_err!("Internal error: Mismatched types inSelf::for - bad vector",);
                    }
                } else {
                    vec_elem_tys[0].clone()
                };
                if *param_2_ty != elem_ty {
                    return compile_err!("Internal error: Mismatched types inSelf::for - function elem type {} != {}",
                                        param_2_ty, &elem_ty);
                }
            } else {
                let composite_ty = if vectorized {
                    let mut vec_elem_tys_simd = vec![];
                    for ty in vec_elem_tys {
                        if let Scalar(ref sk) = ty {
                            vec_elem_tys_simd.push(Simd(sk.clone()))
                        } else {
                            return compile_err!("Internal error: Mismatched types inSelf::for - bad vector",);
                        }
                    }
                    Struct(vec_elem_tys_simd)
                } else {
                    Struct(vec_elem_tys.clone())
                };

                if *param_2_ty != composite_ty {
                    return compile_err!("Internal error: Mismatched types inSelf::for - function zipped elem type",);
                }
            }

            // Function return type should match builder type.
            if ret_ty.as_ref() != &builder_ty {
                return compile_err!("Internal error: Mismatched types inSelf::for - function return type");
            }
        }

        Self::new(For {
            iters: iters,
            builder: Box::new(builder),
            func: Box::new(func),
        },
        builder_ty)
    }

    pub fn merge(builder: Expr, value: Expr) -> WeldResult<Expr> {
        let err = compile_err!("Internal error: Mismatched types inSelf::merge");
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
                DictMerger(ref elem_ty1, ref elem_ty2, _) => {
                    if let Struct(ref v_ty) = value.ty {
                        if v_ty.len() < 2 { return err; }

                        if elem_ty1.as_ref() != &v_ty[0] {
                            return err;
                        }
                        if elem_ty2.as_ref() != &v_ty[1] {
                            return err;
                        }
                    } else {
                        return err;
                    }
                }
                GroupMerger(ref elem_ty1, ref elem_ty2) => {
                    if let Struct(ref v_ty) = value.ty {
                        if v_ty.len() < 2 { return err; }

                        if elem_ty1.as_ref() != &v_ty[0] {
                            return err;
                        }
                        if elem_ty2.as_ref() != &v_ty[1] {
                            return err;
                        }
                    } else {
                        return err;
                    }
                }
                VecMerger(ref elem_ty, _) => {
                    if let Struct(ref tys) = value.ty {
                        if tys.len() != 2 {
                            return err;
                        }
                        if tys[0] != Scalar(ScalarKind::I64) {
                            return err;
                        }
                        if &tys[1] != elem_ty.as_ref() {
                            return err;
                        }
                    } else {
                        return err;
                    }
                }
            }
        }

        let ty = builder.ty.clone();
        Self::new(Merge {
            builder: Box::new(builder),
            value: Box::new(value),
        },
        ty)
    }

    pub fn result(builder: Expr) -> WeldResult<Expr> {
        let err = compile_err!("Internal error: Mismatched types inSelf::result");
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
        Self::new(Res { builder: Box::new(builder) }, ty)
    }
}
