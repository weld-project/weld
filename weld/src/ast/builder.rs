//! Constructors for Weld expressions.
//!
//! This module provides constructors with type checking for Weld expressions.

use crate::ast::ExprKind::*;
use crate::ast::Type::*;
use crate::ast::*;
use crate::error::*;

/// A trait for initializing expressions with type inference.
///
/// This trait provides constructors with type checking to expression variants. The type of each
/// expression is inferred locally based on its immediate children. Each function returns an error
/// if a type error is detected.
///
/// The `new` and `new_with_type` functions take `ExprKind` values directly and perform type
/// checking and inference on the built expression.
///
/// This trait is re-exported as part of the `ast` module and should not need to be imported
/// directly.
pub trait NewExpr {
    /// Creates a new expression with the given kind.
    ///
    /// Returns an error if types in the `kind` mismatch.
    fn new(kind: ExprKind) -> WeldResult<Expr>;
    /// Creates new expression with the given kind and type.
    ///
    /// Returns an error if types in the `kind` mismatch.
    fn new_with_type(kind: ExprKind, ty: Type) -> WeldResult<Expr>;
    /// Creates a new literal expression.
    fn new_literal(kind: LiteralKind) -> WeldResult<Expr>;
    /// Creates a new typed identifier expression.
    fn new_ident(symbol: Symbol, ty: Type) -> WeldResult<Expr>;
    /// Creates a new binary operator expression.
    fn new_bin_op(kind: BinOpKind, left: Expr, right: Expr) -> WeldResult<Expr>;
    /// Creates a new unary operator expression.
    fn new_unary_op(kind: UnaryOpKind, value: Expr) -> WeldResult<Expr>;
    /// Creates a new cast operator expression.
    fn new_cast(kind: ScalarKind, expr: Expr) -> WeldResult<Expr>;
    /// Creates a new negation operator expression.
    fn new_negate(expr: Expr) -> WeldResult<Expr>;
    /// Creates a new not operator expression.
    fn new_not(expr: Expr) -> WeldResult<Expr>;
    /// Creates a new scalar to SIMD broadcast expression.
    fn new_broadcast(expr: Expr) -> WeldResult<Expr>;
    /// Creates a new dictionary to vector expression.
    fn new_to_vec(expr: Expr) -> WeldResult<Expr>;
    /// Creates a new struct literal expression.
    fn new_make_struct(exprs: Vec<Expr>) -> WeldResult<Expr>;
    /// Creates a new vector literal expression.
    fn new_make_vector(exprs: Vec<Expr>) -> WeldResult<Expr>;
    /// Creates a new typed vector literal expression.
    ///
    /// This version can be used if `exprs` is empty and the type cannot be inferred with
    /// `new_make_vector`.
    fn new_make_vector_typed(exprs: Vec<Expr>, ty: Type) -> WeldResult<Expr>;
    /// Creates a new field access expression on struct.
    fn new_get_field(expr: Expr, index: u32) -> WeldResult<Expr>;
    /// Creates a new vector length expression.
    fn new_length(expr: Expr) -> WeldResult<Expr>;
    /// Creates a new vector or dictionary lookup expression.
    fn new_lookup(data: Expr, index: Expr) -> WeldResult<Expr>;
    /// Creates a new dictionary optional lookup expression.
    fn new_opt_lookup(data: Expr, index: Expr) -> WeldResult<Expr>;
    /// Creates a new dictionary key exists expression.
    fn new_key_exists(data: Expr, key: Expr) -> WeldResult<Expr>;
    /// Creates a new vector slicing expression.
    fn new_slice(data: Expr, index: Expr, size: Expr) -> WeldResult<Expr>;
    /// Creates a new vector sort expression.
    fn new_sort(data: Expr, cmpfunc: Expr) -> WeldResult<Expr>;
    /// Creates a new let expression.
    fn new_let(name: Symbol, value: Expr, body: Expr) -> WeldResult<Expr>;
    /// Creates a new if expression.
    fn new_if(cond: Expr, on_true: Expr, on_false: Expr) -> WeldResult<Expr>;
    /// Creates a new select expression.
    fn new_select(cond: Expr, on_true: Expr, on_false: Expr) -> WeldResult<Expr>;
    /// Creates a new lambda expression.
    fn new_lambda(params: Vec<Parameter>, body: Expr) -> WeldResult<Expr>;
    /// Creates a new apply expression.
    fn new_apply(func: Expr, params: Vec<Expr>) -> WeldResult<Expr>;
    /// Creates a new CUDF expression.
    fn new_cudf(sym_name: String, args: Vec<Expr>, return_ty: Type) -> WeldResult<Expr>;
    /// Creates a builder initialization expression.
    fn new_new_builder(kind: BuilderKind, expr: Option<Expr>) -> WeldResult<Expr>;
    /// Creates a new parallel for loop expression.
    fn new_for(iters: Vec<Iter>, builder: Expr, func: Expr) -> WeldResult<Expr>;
    /// Creates a new merge expression.
    fn new_merge(builder: Expr, value: Expr) -> WeldResult<Expr>;
    /// Creates a new result expression.
    fn new_result(builder: Expr) -> WeldResult<Expr>;
    /// Creates a new serialize expression.
    fn new_serialize(value: Expr) -> WeldResult<Expr>;
    /// Creates a new deserialize expression.
    fn new_deserialize(value: Expr, ty: Type) -> WeldResult<Expr>;
}

impl NewExpr for Expr {
    fn new(kind: ExprKind) -> WeldResult<Expr> {
        Self::new_with_type(kind, Unknown)
    }

    fn new_with_type(kind: ExprKind, ty: Type) -> WeldResult<Expr> {
        let mut expr = Expr {
            kind,
            ty,
            annotations: Annotations::new(),
        };

        // Check the type/infer unknown types locally.
        expr.infer_local()?;
        Ok(expr)
    }

    fn new_literal(kind: LiteralKind) -> WeldResult<Expr> {
        use crate::ast::LiteralKind::*;
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

        Self::new_with_type(Literal(kind), ty)
    }

    fn new_ident(symbol: Symbol, ty: Type) -> WeldResult<Expr> {
        Self::new_with_type(Ident(symbol), ty)
    }

    fn new_bin_op(kind: BinOpKind, left: Expr, right: Expr) -> WeldResult<Expr> {
        Self::new(BinOp {
            kind,
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    fn new_unary_op(kind: UnaryOpKind, value: Expr) -> WeldResult<Expr> {
        Self::new(UnaryOp {
            kind,
            value: Box::new(value),
        })
    }

    fn new_cast(kind: ScalarKind, expr: Expr) -> WeldResult<Expr> {
        Self::new(Cast {
            kind,
            child_expr: Box::new(expr),
        })
    }

    fn new_negate(expr: Expr) -> WeldResult<Expr> {
        Self::new(Negate(Box::new(expr)))
    }

    fn new_not(expr: Expr) -> WeldResult<Expr> {
        Self::new(Not(Box::new(expr)))
    }

    fn new_broadcast(expr: Expr) -> WeldResult<Expr> {
        Self::new(Broadcast(Box::new(expr)))
    }

    fn new_to_vec(expr: Expr) -> WeldResult<Expr> {
        Self::new(ToVec {
            child_expr: Box::new(expr),
        })
    }

    fn new_make_struct(exprs: Vec<Expr>) -> WeldResult<Expr> {
        Self::new(MakeStruct { elems: exprs })
    }

    fn new_make_vector(exprs: Vec<Expr>) -> WeldResult<Expr> {
        Self::new(MakeVector { elems: exprs })
    }

    fn new_make_vector_typed(exprs: Vec<Expr>, ty: Type) -> WeldResult<Expr> {
        Self::new_with_type(MakeVector { elems: exprs }, Vector(Box::new(ty)))
    }

    fn new_get_field(expr: Expr, index: u32) -> WeldResult<Expr> {
        Self::new(GetField {
            expr: Box::new(expr),
            index,
        })
    }

    fn new_length(expr: Expr) -> WeldResult<Expr> {
        Self::new(Length {
            data: Box::new(expr),
        })
    }

    fn new_lookup(data: Expr, index: Expr) -> WeldResult<Expr> {
        Self::new(Lookup {
            data: Box::new(data),
            index: Box::new(index),
        })
    }

    fn new_opt_lookup(data: Expr, index: Expr) -> WeldResult<Expr> {
        Self::new(OptLookup {
            data: Box::new(data),
            index: Box::new(index),
        })
    }

    fn new_key_exists(data: Expr, key: Expr) -> WeldResult<Expr> {
        Self::new(KeyExists {
            data: Box::new(data),
            key: Box::new(key),
        })
    }

    fn new_slice(data: Expr, index: Expr, size: Expr) -> WeldResult<Expr> {
        Self::new(Slice {
            data: Box::new(data),
            index: Box::new(index),
            size: Box::new(size),
        })
    }

    fn new_sort(data: Expr, cmpfunc: Expr) -> WeldResult<Expr> {
        Self::new(Sort {
            data: Box::new(data),
            cmpfunc: Box::new(cmpfunc),
        })
    }

    fn new_let(name: Symbol, value: Expr, body: Expr) -> WeldResult<Expr> {
        Self::new(Let {
            name,
            value: Box::new(value),
            body: Box::new(body),
        })
    }

    fn new_if(cond: Expr, on_true: Expr, on_false: Expr) -> WeldResult<Expr> {
        Self::new(If {
            cond: Box::new(cond),
            on_true: Box::new(on_true),
            on_false: Box::new(on_false),
        })
    }

    fn new_select(cond: Expr, on_true: Expr, on_false: Expr) -> WeldResult<Expr> {
        Self::new(Select {
            cond: Box::new(cond),
            on_true: Box::new(on_true),
            on_false: Box::new(on_false),
        })
    }

    fn new_lambda(params: Vec<Parameter>, body: Expr) -> WeldResult<Expr> {
        Self::new(Lambda {
            params,
            body: Box::new(body),
        })
    }

    fn new_apply(func: Expr, params: Vec<Expr>) -> WeldResult<Expr> {
        Self::new(Apply {
            func: Box::new(func),
            params,
        })
    }

    fn new_cudf(sym_name: String, args: Vec<Expr>, return_ty: Type) -> WeldResult<Expr> {
        Self::new(CUDF {
            sym_name,
            args,
            return_ty: Box::new(return_ty),
        })
    }

    fn new_new_builder(kind: BuilderKind, expr: Option<Expr>) -> WeldResult<Expr> {
        let expr = expr.map(Box::new);
        Self::new_with_type(NewBuilder(expr), Builder(kind, Annotations::new()))
    }

    fn new_for(iters: Vec<Iter>, builder: Expr, func: Expr) -> WeldResult<Expr> {
        Self::new(For {
            iters,
            builder: Box::new(builder),
            func: Box::new(func),
        })
    }

    fn new_merge(builder: Expr, value: Expr) -> WeldResult<Expr> {
        Self::new(Merge {
            builder: Box::new(builder),
            value: Box::new(value),
        })
    }

    fn new_result(builder: Expr) -> WeldResult<Expr> {
        Self::new(Res {
            builder: Box::new(builder),
        })
    }

    fn new_serialize(value: Expr) -> WeldResult<Expr> {
        Self::new(Serialize(Box::new(value)))
    }

    fn new_deserialize(value: Expr, ty: Type) -> WeldResult<Expr> {
        Self::new(Deserialize {
            value: Box::new(value),
            value_ty: Box::new(ty),
        })
    }
}
