//! Constructors for Weld expressions.
//!
//! This module provides constructors with type checking for Weld expressions.

use ast::*;
use ast::ExprKind::*;
use ast::Type::*;
use error::*;

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
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let expr = Expr::new_literal(I32Literal(1)).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_literal(kind: LiteralKind) -> WeldResult<Expr>;
    /// Creates a new typed identifier expression.
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let name = Symbol::new("a", 0);
    /// let ty = Scalar(I32);
    /// let expr = Expr::new_ident(name, ty.clone()).unwrap();
    /// assert_eq!(expr.ty, ty);
    /// ```
    fn new_ident(symbol: Symbol, ty: Type) -> WeldResult<Expr>;
    /// Creates a new binary operator expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_bin_op(Add, one.clone(), one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_bin_op(kind: BinOpKind, left: Expr, right: Expr) -> WeldResult<Expr>;
    /// Creates a new unary operator expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(F32Literal((1.0_f32).to_bits())).unwrap();
    /// let expr = Expr::new_unary_op(Cos, one).unwrap();
    /// assert_eq!(expr.ty, Scalar(F32));
    /// ```
    fn new_unary_op(kind: UnaryOpKind, value: Expr) -> WeldResult<Expr>;
    /// Creates a new cast operator expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_cast(F32, one).unwrap();
    /// assert_eq!(expr.ty, Scalar(F32));
    /// ```
    fn new_cast(kind: ScalarKind, expr: Expr) -> WeldResult<Expr>;
    /// Creates a new negation operator expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_negate(one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_negate(expr: Expr) -> WeldResult<Expr>;
    /// Creates a new not operator expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let value = Expr::new_literal(BoolLiteral(false)).unwrap();
    /// let expr = Expr::new_not(value).unwrap();
    /// assert_eq!(expr.ty, Scalar(Bool));
    /// ```
    fn new_not(expr: Expr) -> WeldResult<Expr>;
    /// Creates a new scalar to SIMD broadcast expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_broadcast(one).unwrap();
    /// assert_eq!(expr.ty, Simd(I32));
    /// ```
    fn new_broadcast(expr: Expr) -> WeldResult<Expr>;
    /// Creates a new dictionary to vector expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let dict_ty = Dict(Box::new(Scalar(I32)), Box::new(Scalar(I32)));
    /// let dict = Expr::new_ident(Symbol::new("d", 0), dict_ty).unwrap();
    /// let expr = Expr::new_to_vec(dict).unwrap();
    /// assert_eq!(expr.ty, Vector(Box::new(Struct(vec![Scalar(I32), Scalar(I32)]))));
    /// ```
    fn new_to_vec(expr: Expr) -> WeldResult<Expr>;
    /// Creates a new struct literal expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_make_struct(vec![one.clone(), one.clone(), one]).unwrap();
    /// assert_eq!(expr.ty, Struct(vec![Scalar(I32), Scalar(I32), Scalar(I32)]));
    /// ```
    fn new_make_struct(exprs: Vec<Expr>) -> WeldResult<Expr>;
    /// Creates a new vector literal expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_make_vector(vec![one.clone(), one.clone(), one]).unwrap();
    /// assert_eq!(expr.ty, Vector(Box::new(Scalar(I32))));
    /// ```
    fn new_make_vector(exprs: Vec<Expr>) -> WeldResult<Expr>;
    /// Creates a new typed vector literal expression.
    ///
    /// This version can be used if `exprs` is empty and the type cannot be inferred with
    /// `new_make_vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_bin_op(Add, one.clone(), one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_make_vector_typed(exprs: Vec<Expr>, ty: Type) -> WeldResult<Expr>;                   ///////////////////////////// IM HERE!
    /// Creates a new field access expression on struct.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_bin_op(Add, one.clone(), one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_get_field(expr: Expr, index: u32) -> WeldResult<Expr>;
    /// Creates a new vector length expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_bin_op(Add, one.clone(), one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_length(expr: Expr) -> WeldResult<Expr>;
    /// Creates a new vector or dictionary lookup expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_bin_op(Add, one.clone(), one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_lookup(data: Expr, index: Expr) -> WeldResult<Expr>;
    /// Creates a new dictionary optional lookup expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_bin_op(Add, one.clone(), one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_opt_lookup(data: Expr, index: Expr) -> WeldResult<Expr>;
    /// Creates a new dictionary key exists expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_bin_op(Add, one.clone(), one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_key_exists(data: Expr, key: Expr) -> WeldResult<Expr>;
    /// Creates a new vector slicing expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_bin_op(Add, one.clone(), one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_slice(data: Expr, index: Expr, size: Expr) -> WeldResult<Expr>;
    /// Creates a new vector sort expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_bin_op(Add, one.clone(), one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_sort(data: Expr, cmpfunc: Expr) -> WeldResult<Expr>;
    /// Creates a new let expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_bin_op(Add, one.clone(), one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_let(name: Symbol, value: Expr, body: Expr) -> WeldResult<Expr>;
    /// Creates a new if expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_bin_op(Add, one.clone(), one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_if(cond: Expr, on_true: Expr, on_false: Expr) -> WeldResult<Expr>;
    /// Creates a new select expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_bin_op(Add, one.clone(), one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_select(cond: Expr, on_true: Expr, on_false: Expr) -> WeldResult<Expr>;
    /// Creates a new lambda expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_bin_op(Add, one.clone(), one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_lambda(params: Vec<Parameter>, body: Expr) -> WeldResult<Expr>;
    /// Creates a new apply expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_bin_op(Add, one.clone(), one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_apply(func: Expr, params: Vec<Expr>) -> WeldResult<Expr>;
    /// Creates a new CUDF expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_bin_op(Add, one.clone(), one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_cudf(sym_name: String, args: Vec<Expr>, return_ty: Type) -> WeldResult<Expr>;
    /// Creates a builder initialization expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_bin_op(Add, one.clone(), one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_new_builder(kind: BuilderKind, expr: Option<Expr>) -> WeldResult<Expr>;
    /// Creates a new parallel for loop expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_bin_op(Add, one.clone(), one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_for(iters: Vec<Iter>, builder: Expr, func: Expr) -> WeldResult<Expr>;
    /// Creates a new merge expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_bin_op(Add, one.clone(), one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_merge(builder: Expr, value: Expr) -> WeldResult<Expr>;
    /// Creates a new result expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_bin_op(Add, one.clone(), one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_result(builder: Expr) -> WeldResult<Expr>;
    /// Creates a new serialize expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_bin_op(Add, one.clone(), one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_serialize(value: Expr) -> WeldResult<Expr>;
    /// Creates a new deserialize expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use weld::ast::prelude::*;
    ///
    /// let one = Expr::new_literal(I32Literal(1)).unwrap();
    /// let expr = Expr::new_bin_op(Add, one.clone(), one).unwrap();
    /// assert_eq!(expr.ty, Scalar(I32));
    /// ```
    fn new_deserialize(value: Expr, ty: Type) -> WeldResult<Expr>;
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

    fn new_literal(kind: LiteralKind) -> WeldResult<Expr> {
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

    fn new_ident(symbol: Symbol, ty: Type) -> WeldResult<Expr> {
        Self::new_with_type(Ident(symbol), ty)
    }

    fn new_bin_op(kind: BinOpKind, left: Expr, right: Expr) -> WeldResult<Expr> {
        Self::new(BinOp {
            kind: kind,
            left: Box::new(left),
            right: Box::new(right) 
        })
    }

    fn new_unary_op(kind: UnaryOpKind, value: Expr) -> WeldResult<Expr> {
        Self::new(UnaryOp {
            kind: kind,
            value: Box::new(value)
        })
    }

    fn new_cast(kind: ScalarKind, expr: Expr) -> WeldResult<Expr> {
        Self::new(Cast {
            kind: kind,
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
            child_expr: Box::new(expr.clone())
        })
    }

    fn new_make_struct(exprs: Vec<Expr>) -> WeldResult<Expr> {
        Self::new(MakeStruct {
            elems: exprs,
        })
    }

    fn new_make_vector(exprs: Vec<Expr>) -> WeldResult<Expr> {
        Self::new(MakeVector {
            elems: exprs,
        })
    }

    fn new_make_vector_typed(exprs: Vec<Expr>, ty: Type) -> WeldResult<Expr> {
        Self::new_with_type(MakeVector { elems: exprs }, Vector(Box::new(ty)))
    }

    fn new_get_field(expr: Expr, index: u32) -> WeldResult<Expr> {
        Self::new(GetField {
            expr: Box::new(expr),
            index: index,
        })
    }

    fn new_length(expr: Expr) -> WeldResult<Expr> {
        Self::new(Length {
            data: Box::new(expr)
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
            name: name,
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
            params: params,
            body: Box::new(body),
        })
    }

    fn new_apply(func: Expr, params: Vec<Expr>) -> WeldResult<Expr> {
        Self::new(Apply {
            func: Box::new(func),
            params: params,
        })
    }

    fn new_cudf(sym_name: String, args: Vec<Expr>, return_ty: Type) -> WeldResult<Expr> {
        Self::new(CUDF {
            sym_name: sym_name,
            args: args,
            return_ty: Box::new(return_ty.clone()),
        })
    }

    fn new_new_builder(kind: BuilderKind, expr: Option<Expr>) -> WeldResult<Expr> {
        let expr = expr.map(|e| Box::new(e));
        Self::new_with_type(NewBuilder(expr), Builder(kind, Annotations::new()))
    }

    fn new_for(iters: Vec<Iter>, builder: Expr, func: Expr) -> WeldResult<Expr> {
        Self::new(For {
            iters: iters,
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
            builder: Box::new(builder)
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
