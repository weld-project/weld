//! Abstract syntax tree for Weld.

use std::vec;

/// A symbol (identifier name); for now these are strings, but we may add some kind of scope ID.
pub type Symbol = String;

/// A data type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Scalar(ScalarKind),
    Vector(Box<Type>),
    Builder(BuilderKind),
    Function(Vec<Type>, Box<Type>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarKind {
    Bool,
    I32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BuilderKind {
    Appender(Box<Type>),
    Merger(Box<Type>, BinOpKind)
}

/// An expression tree, having type annotations of type T. We make this parametrized because
/// expressions have different "kinds" of types attached to them at different points in the
/// compilation process -- namely PartialType when parsed and then Type after type inference.
#[derive(Clone, Debug, PartialEq)]
pub struct Expr<T> {
    pub ty: T,
    pub kind: ExprKind<T>
}

impl<T> Expr<T> {
    /// Get an iterator for the children of this expression.
    pub fn children(&self) -> vec::IntoIter<&Expr<T>> {
        use self::ExprKind::*;
        match self.kind {
            BinOp(_, ref left, ref right) => vec![left.as_ref(), right.as_ref()],
            Let(_, ref value, ref body) => vec![value.as_ref(), body.as_ref()],
            Lambda(_, ref body) => vec![body.as_ref()],
            Map(ref data, ref body) => vec![data.as_ref(), body.as_ref()],
            MakeVector(ref exprs) => exprs.iter().collect(),
            _ => vec![]
        }.into_iter()
    }

    /// Get an iterator of mutable references to the children of this expression.
    pub fn children_mut(&mut self) -> vec::IntoIter<&mut Expr<T>> {
        use self::ExprKind::*;
        match self.kind {
            BinOp(_, ref mut left, ref mut right) => vec![left.as_mut(), right.as_mut()],
            Let(_, ref mut value, ref mut body) => vec![value.as_mut(), body.as_mut()],
            Lambda(_, ref mut body) => vec![body.as_mut()],
            Map(ref mut data, ref mut body) => vec![data.as_mut(), body.as_mut()],
            MakeVector(ref mut exprs) => exprs.iter_mut().collect(),
            _ => vec![]
        }.into_iter()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind<T> {
    BoolLiteral(bool),
    I32Literal(i32),
    BinOp(BinOpKind, Box<Expr<T>>, Box<Expr<T>>),
    Ident(Symbol),
    MakeVector(Vec<Expr<T>>),
    /// name, value, body
    Let(Symbol, Box<Expr<T>>, Box<Expr<T>>),
    /// condition, on_true, on_false 
    If(Box<Expr<T>>, Box<Expr<T>>, Box<Expr<T>>),
    /// variables, body
    Lambda(Vec<Parameter<T>>, Box<Expr<T>>),
    /// data, function
    Map(Box<Expr<T>>, Box<Expr<T>>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOpKind {
    Add,
    Subtract,
    Multiply,
    Divide,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Parameter<T> {
    pub name: Symbol,
    pub ty: T
}

/// A typed expression struct.
pub type TypedExpr = Expr<Type>;

/// A typed parameter.
pub type TypedParameter = Parameter<Type>;