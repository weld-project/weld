use std::vec;

use super::ast::BinOpKind;
use super::ast::ScalarKind;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Scalar(ScalarKind),
    Vector(Box<Type>),
    Builder(BuilderKind),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BuilderKind {
    Appender(Box<Type>),
    Merger(BinOpKind)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Symbol {
    pub ty: Option<Type>,
    pub name: String
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
    Ident(String),
    /// name, value, body
    Let(String, Box<Expr>, Box<Expr>),
    /// condition, on_true, on_false 
    If(Box<Expr>, Box<Expr>, Box<Expr>),
}

pub fn expr_box(kind: ExprKind) -> Box<Expr> {
    Box::new(Expr { ty: None, kind: kind })
}

impl Expr {
    pub fn children(&self) -> vec::IntoIter<&Expr> {
        use self::ExprKind::*;
        match self.kind {
            BinOp(_, ref left, ref right) => vec![left.as_ref(), right.as_ref()],
            Let(_, ref value, ref body) => vec![value.as_ref(), body.as_ref()],
            _ => vec![]
        }.into_iter()
    }

    pub fn children_mut(&mut self) -> vec::IntoIter<&mut Expr> {
        use self::ExprKind::*;
        match self.kind {
            BinOp(_, ref mut left, ref mut right) => vec![left.as_mut(), right.as_mut()],
            Let(_, ref mut value, ref mut body) => vec![value.as_mut(), body.as_mut()],
            _ => vec![]
        }.into_iter()
    }
}