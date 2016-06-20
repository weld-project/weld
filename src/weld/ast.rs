//! Abstract syntax tree for Weld.

/// Data types.
#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Scalar(ScalarKind),
    Vector(Box<Type>),
    Builder(BuilderKind),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarKind {
    Bool,
    I32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BuilderKind {
    Appender(Box<Type>),
    Merger(Box<Type>, BinOpKind)
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Symbol(pub String);

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    BoolLiteral(bool),
    I32Literal(i32),
    BinOp(Type, BinOpKind, Box<Expr>, Box<Expr>),
    Ident(Type, Symbol),
    Let {
        out_type: Type,
        symbol: Symbol,
        value: Box<Expr>,
        body: Box<Expr>,
    },
    If {
        out_type: Type,
        condition: Box<Expr>,
        on_true: Box<Expr>,
        on_false: Box<Expr>,
    },
}

pub fn get_type(expr: &Expr) -> Type {
    use self::Expr::*;
    use self::Type::*;
    use self::ScalarKind::*;
    match *expr {
        BoolLiteral(_) => Scalar(Bool),
        I32Literal(_) => Scalar(I32),
        BinOp(ref ty, _, _, _) => ty.clone(),
        Ident(ref ty, _) => ty.clone(),
        Let { ref out_type, .. } => out_type.clone(),
        If { ref out_type, .. } => out_type.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOpKind {
    Add,
    Subtract,
    Multiply,
    Divide,
}