//! Abstract syntax tree for Weld.

use std::vec;

pub type Symbol = String;

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

#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub ty: Option<Type>,
    pub kind: ExprKind
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    BoolLiteral(bool),
    I32Literal(i32),
    BinOp(BinOpKind, Box<Expr>, Box<Expr>),
    Ident(Symbol),
    MakeVector(Vec<Expr>),
    /// name, value, body
    Let(Symbol, Box<Expr>, Box<Expr>),
    /// condition, on_true, on_false 
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    /// variables, body
    Lambda(Vec<Parameter>, Box<Expr>),
    /// data, function
    Map(Box<Expr>, Box<Expr>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOpKind {
    Add,
    Subtract,
    Multiply,
    Divide,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Parameter {
    pub name: Symbol,
    pub ty: Option<Type>
}

/// Create a box containing an untyped expression of the given kind.
pub fn expr_box(kind: ExprKind) -> Box<Expr> {
    Box::new(Expr { ty: None, kind: kind })
}

impl Expr {
    /// Get an iterator for the children of this expression.
    pub fn children(&self) -> vec::IntoIter<&Expr> {
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
    pub fn children_mut(&mut self) -> vec::IntoIter<&mut Expr> {
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