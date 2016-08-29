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
    Struct(Vec<Type>),
    Function(Vec<Type>, Box<Type>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarKind {
    Bool,
    I32,
    I64,
    F32,
    F64,
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
pub struct Expr<T:Clone> {
    pub ty: T,
    pub kind: ExprKind<T>
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind<T:Clone> {
    // TODO: maybe all of these should take named parameters
    BoolLiteral(bool),
    I32Literal(i32),
    I64Literal(i64),
    F32Literal(f32),
    F64Literal(f64),
    BinOp(BinOpKind, Box<Expr<T>>, Box<Expr<T>>),
    Ident(Symbol),
    NewBuilder,  // TODO: this may need to take a parameter
    MakeStruct(Vec<Expr<T>>),
    MakeVector(Vec<Expr<T>>),
    GetField(Box<Expr<T>>, u32),
    /// name, value, body
    Let(Symbol, Box<Expr<T>>, Box<Expr<T>>),
    /// condition, on_true, on_false
    If(Box<Expr<T>>, Box<Expr<T>>, Box<Expr<T>>),
    /// variables, body
    Lambda(Vec<Parameter<T>>, Box<Expr<T>>),
    /// function, params
    Apply(Box<Expr<T>>, Vec<Expr<T>>),
    /// data, builder, func (very basic version for now)
    For(Box<Expr<T>>, Box<Expr<T>>, Box<Expr<T>>),
    /// builder, elem
    Merge(Box<Expr<T>>, Box<Expr<T>>),
    /// builder
    Res(Box<Expr<T>>)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOpKind {
    Add,
    Subtract,
    Multiply,
    Divide,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Parameter<T:Clone> {
    pub name: Symbol,
    pub ty: T
}

/// A typed expression struct.
pub type TypedExpr = Expr<Type>;

/// A typed parameter.
pub type TypedParameter = Parameter<Type>;

impl<T:Clone> Expr<T> {
    /// Get an iterator for the children of this expression.
    pub fn children(&self) -> vec::IntoIter<&Expr<T>> {
        use self::ExprKind::*;
        match self.kind {
            BinOp(_, ref left, ref right) => vec![left.as_ref(), right.as_ref()],
            Let(_, ref value, ref body) => vec![value.as_ref(), body.as_ref()],
            Lambda(_, ref body) => vec![body.as_ref()],
            MakeStruct(ref exprs) => exprs.iter().collect(),
            MakeVector(ref exprs) => exprs.iter().collect(),
            GetField(ref expr, _) => vec![expr.as_ref()],
            Merge(ref bldr, ref value) => vec![bldr.as_ref(), value.as_ref()],
            Res(ref bldr) => vec![bldr.as_ref()],
            For(ref data, ref bldr, ref func) =>
                vec![data.as_ref(), bldr.as_ref(), func.as_ref()],
            If(ref cond, ref on_true, ref on_false) =>
                vec![cond.as_ref(), on_true.as_ref(), on_false.as_ref()],
            Apply(ref func, ref params) => {
                let mut res = vec![func.as_ref()];
                res.extend(params.iter());
                res
            }
            // Explicitly list types instead of doing _ => ... to remember to add new types.
            BoolLiteral(_) | I32Literal(_) | I64Literal(_) | F32Literal(_) | F64Literal(_) | Ident(_) | NewBuilder => vec![]
        }.into_iter()
    }

    /// Get an iterator of mutable references to the children of this expression.
    pub fn children_mut(&mut self) -> vec::IntoIter<&mut Expr<T>> {
        use self::ExprKind::*;
        match self.kind {
            BinOp(_, ref mut left, ref mut right) => vec![left.as_mut(), right.as_mut()],
            Let(_, ref mut value, ref mut body) => vec![value.as_mut(), body.as_mut()],
            Lambda(_, ref mut body) => vec![body.as_mut()],
            MakeStruct(ref mut exprs) => exprs.iter_mut().collect(),
            MakeVector(ref mut exprs) => exprs.iter_mut().collect(),
            GetField(ref mut expr, _) => vec![expr.as_mut()],
            Merge(ref mut bldr, ref mut value) => vec![bldr.as_mut(), value.as_mut()],
            Res(ref mut bldr) => vec![bldr.as_mut()],
            For(ref mut data, ref mut bldr, ref mut func) =>
                vec![data.as_mut(), bldr.as_mut(), func.as_mut()],
            If(ref mut cond, ref mut on_true, ref mut on_false) =>
                vec![cond.as_mut(), on_true.as_mut(), on_false.as_mut()],
            Apply(ref mut func, ref mut params) => {
                let mut res = vec![func.as_mut()];
                res.extend(params.iter_mut());
                res
            }
            // Explicitly list types instead of doing _ => ... to remember to add new types.
            BoolLiteral(_) | I32Literal(_) | I64Literal(_) | F32Literal(_) | F64Literal(_) | Ident(_) | NewBuilder => vec![]
        }.into_iter()
    }

    /// Substitute Ident nodes with the given symbol for another expression, stopping when an
    /// expression in the tree redefines the symbol (e.g. Let or Lambda parameters).
    pub fn substitute(&mut self, symbol: &Symbol, replacement: &Expr<T>) {
        // Replace ourselves if we are exactly the symbol.
        use self::ExprKind::*;
        let mut self_matches = false;
        match self.kind {
            Ident(ref sym) if *sym == *symbol => self_matches = true,
            _ => ()
        }
        if self_matches {
            *self = (*replacement).clone();
            return;
        }

        // Otherwise, replace any relevant children, unless we redefine the symbol.
        match self.kind {
            Let(ref name, ref mut value, ref mut body) => {
                value.substitute(symbol, replacement);
                if name != symbol {
                    body.substitute(symbol, replacement);
                }
            }

            Lambda(ref params, ref mut body) => {
                if params.iter().all(|p| *p.name != *symbol) {
                    body.substitute(symbol, replacement);
                }
            }

            _ => {
                for c in self.children_mut() {
                    c.substitute(symbol, replacement);
                }
            }
        }
    }
}
