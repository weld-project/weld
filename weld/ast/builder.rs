//! Builder API for Weld expressions.

use ast::*;
use ast::ExprKind::*;
use ast::Type::*;
use error::*;

pub trait NewExpr {
    fn new(kind: ExprKind) -> WeldResult<Expr>;
    fn new_with_type(kind: ExprKind, ty: Type) -> WeldResult<Expr>;
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
    fn r#let(name: Symbol, value: Expr, body: Expr) -> WeldResult<Expr>;
    fn r#if(cond: Expr, on_true: Expr, on_false: Expr) -> WeldResult<Expr>;
    fn select(cond: Expr, on_true: Expr, on_false: Expr) -> WeldResult<Expr>;
    fn lambda(params: Vec<Parameter>, body: Expr) -> WeldResult<Expr>;
    fn apply(func: Expr, params: Vec<Expr>) -> WeldResult<Expr>;
    fn cudf(sym_name: String, args: Vec<Expr>, return_ty: Type) -> WeldResult<Expr>;
    fn newbuilder(kind: BuilderKind, expr: Option<Expr>) -> WeldResult<Expr>;
    fn r#for(iters: Vec<Iter>, builder: Expr, func: Expr) -> WeldResult<Expr>;
    fn merge(builder: Expr, value: Expr) -> WeldResult<Expr>;
    fn result(builder: Expr) -> WeldResult<Expr>;
}

impl NewExpr for Expr {

    fn new(kind: ExprKind) -> WeldResult<Expr> {
        Self::new_with_type(kind, Unknown)
    }

    fn new_with_type(kind: ExprKind, ty: Type) -> WeldResult<Expr> {
        let mut expr = Expr {
            kind: kind,
            ty: ty,
            annotations: Annotations::new(),
        };

        expr.infer_local()?;
        Ok(expr)
    }

    fn literal(kind: LiteralKind) -> WeldResult<Expr> {
        use ast::LiteralKind::*;
        use ast::Type::Scalar;

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

        Self::new_with_type(Literal(kind.clone()), ty)
    }

    fn ident(symbol: Symbol, ty: Type) -> WeldResult<Expr> {
        Self::new_with_type(Ident(symbol), ty)
    }

    fn binop(kind: BinOpKind, left: Expr, right: Expr) -> WeldResult<Expr> {
        Self::new(BinOp {
            kind: kind,
            left: Box::new(left),
            right: Box::new(right) 
        })
    }

    fn unaryop(kind: UnaryOpKind, value: Expr) -> WeldResult<Expr> {
        Self::new(UnaryOp {
            kind: kind,
            value: Box::new(value)
        })
    }

    fn cast(kind: ScalarKind, expr: Expr) -> WeldResult<Expr> {
        Self::new(Cast {
            kind: kind,
            child_expr: Box::new(expr),
        })
    }

    fn negate(expr: Expr) -> WeldResult<Expr> {
        Self::new(Negate(Box::new(expr)))
    }

    fn not(expr: Expr) -> WeldResult<Expr> {
        Self::new(Not(Box::new(expr)))
    }

    fn broadcast(expr: Expr) -> WeldResult<Expr> {
        Self::new(Broadcast(Box::new(expr)))
    }

    fn tovec(expr: Expr) -> WeldResult<Expr> {
        Self::new(ToVec {
            child_expr: Box::new(expr.clone())
        })
    }

    fn makestruct(exprs: Vec<Expr>) -> WeldResult<Expr> {
        Self::new(MakeStruct {
            elems: exprs,
        })
    }

    fn makevector(exprs: Vec<Expr>) -> WeldResult<Expr> {
        Self::new(MakeVector {
            elems: exprs,
        })
    }

    fn makevector_typed(exprs: Vec<Expr>, ty: Type) -> WeldResult<Expr> {
        Self::new_with_type(MakeVector { elems: exprs }, Vector(Box::new(ty)))
    }

    fn getfield(expr: Expr, index: u32) -> WeldResult<Expr> {
        Self::new(GetField {
            expr: Box::new(expr),
            index: index,
        })
    }

    fn length(expr: Expr) -> WeldResult<Expr> {
        Self::new(Length {
            data: Box::new(expr)
        })
    }

    fn lookup(data: Expr, index: Expr) -> WeldResult<Expr> {
        Self::new(Lookup {
            data: Box::new(data),
            index: Box::new(index),
        })
    }

    fn keyexists(data: Expr, key: Expr) -> WeldResult<Expr> {
        Self::new(KeyExists {
            data: Box::new(data),
            key: Box::new(key),
        })
    }

    fn slice(data: Expr, index: Expr, size: Expr) -> WeldResult<Expr> {
        Self::new(Slice {
            data: Box::new(data),
            index: Box::new(index),
            size: Box::new(size),
        })
    }

    fn sort(data: Expr, cmpfunc: Expr) -> WeldResult<Expr> {
        Self::new(Sort {
            data: Box::new(data),
            cmpfunc: Box::new(cmpfunc),
        })
    }

    fn r#let(name: Symbol, value: Expr, body: Expr) -> WeldResult<Expr> {
        Self::new(Let {
            name: name,
            value: Box::new(value),
            body: Box::new(body),
        })
    }

    fn r#if(cond: Expr, on_true: Expr, on_false: Expr) -> WeldResult<Expr> {
        Self::new(If {
            cond: Box::new(cond),
            on_true: Box::new(on_true),
            on_false: Box::new(on_false),
        })
    }

    fn select(cond: Expr, on_true: Expr, on_false: Expr) -> WeldResult<Expr> {
        Self::new(Select {
            cond: Box::new(cond),
            on_true: Box::new(on_true),
            on_false: Box::new(on_false),
        })
    }

    fn lambda(params: Vec<Parameter>, body: Expr) -> WeldResult<Expr> {
        Self::new(Lambda {
            params: params,
            body: Box::new(body),
        })
    }

    fn apply(func: Expr, params: Vec<Expr>) -> WeldResult<Expr> {
        Self::new(Apply {
            func: Box::new(func),
            params: params,
        })
    }

    fn cudf(sym_name: String, args: Vec<Expr>, return_ty: Type) -> WeldResult<Expr> {
        Self::new(CUDF {
            sym_name: sym_name,
            args: args,
            return_ty: Box::new(return_ty.clone()),
        })
    }

    fn newbuilder(kind: BuilderKind, expr: Option<Expr>) -> WeldResult<Expr> {
        let expr = expr.map(|e| Box::new(e));
        Self::new_with_type(NewBuilder(expr), Builder(kind, Annotations::new()))
    }

    fn r#for(iters: Vec<Iter>, builder: Expr, func: Expr) -> WeldResult<Expr> {
        Self::new(For {
            iters: iters,
            builder: Box::new(builder),
            func: Box::new(func),
        })
    }

    fn merge(builder: Expr, value: Expr) -> WeldResult<Expr> {
        Self::new(Merge {
            builder: Box::new(builder),
            value: Box::new(value),
        })
    }

    fn result(builder: Expr) -> WeldResult<Expr> {
        Self::new(Res {
            builder: Box::new(builder)
        })
    }
}
