//! Abstract syntax tree for Weld.

use std::vec;
use std::fmt;

/// A symbol (identifier name); for now these are strings, but we may add some kind of scope ID.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Symbol {
    pub name: String,
    pub id: i32
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.id == 0 {
            write!(f, "{}", self.name)
        } else {
            write!(f, "{}#{}", self.name, self.id)
        }
    }
}

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
pub struct Expr<T:Clone + Eq> {
    pub ty: T,
    pub kind: ExprKind<T>
}

/// An iterator, which specifies a vector to iterate over and optionally a start index,
/// end index, and stride.
#[derive(Clone, Debug, PartialEq)]
pub struct Iter<T:Clone + Eq> {
    pub data: Box<Expr<T>>,
    pub start: Option<Box<Expr<T>>>,
    pub end: Option<Box<Expr<T>>>,
    pub stride: Option<Box<Expr<T>>>
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind<T:Clone+Eq> {
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
    // TODO(shoumik): BoxIter<T>> -> Vec<Iter<T>>
    /// iterator, builder, func
    For(Vec<Iter<T>>, Box<Expr<T>>, Box<Expr<T>>),
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
    Modulo,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    LogicalAnd,
    LogicalOr,
    BitwiseAnd,
    BitwiseOr,
    Xor,
}

impl BinOpKind {
    pub fn is_comparison(&self) -> bool {
        use ast::BinOpKind::*;
        match *self {
            Equal | NotEqual | LessThan | GreaterThan => true,
            _ => false
        }
    }
}

impl fmt::Display for BinOpKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ast::BinOpKind::*;
        let text = match *self {
            Add => "+",
            Subtract => "-",
            Multiply => "*",
            Divide => "/",
            Modulo => "%",
            Equal => "==",
            NotEqual => "!=",
            LessThan => "<",
            LessThanOrEqual => "<=",
            GreaterThan => ">",
            GreaterThanOrEqual => ">=",
            LogicalAnd => "&&",
            LogicalOr => "||",
            BitwiseAnd => "&",
            BitwiseOr => "|",
            Xor => "^",
        };
        f.write_str(text)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Parameter<T:Clone+Eq> {
    pub name: Symbol,
    pub ty: T
}

/// A typed expression struct.
pub type TypedExpr = Expr<Type>;

/// A typed parameter.
pub type TypedParameter = Parameter<Type>;

impl<T:Clone+Eq> Expr<T> {
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
            For(ref iters, ref bldr, ref func) => {
                let mut res: Vec<&Expr<T>> = vec![];
                for iter in iters {
                    res.push(iter.data.as_ref());
                    if let Some(ref s) = iter.start {
                        res.push(s);
                    }
                    if let Some(ref e) = iter.end {
                        res.push(e);
                    }
                    if let Some(ref s) = iter.stride {
                        res.push(s);
                    }
                }
                res.push(bldr.as_ref());
                res.push(func.as_ref());
                res
            }
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
            For(ref mut iters, ref mut bldr, ref mut func) => {
                let mut res: Vec<&mut Expr<T>> = vec![];
                for iter in iters {
                    res.push(iter.data.as_mut());
                    if let Some(ref mut s) = iter.start {
                        res.push(s);
                    }
                    if let Some(ref mut e) = iter.end {
                        res.push(e);
                    }
                    if let Some(ref mut s) = iter.stride {
                        res.push(s);
                    }
                }
                res.push(bldr.as_mut());
                res.push(func.as_mut());
                res
            }
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

    /// Compares two expression trees, returning true if they are the same.
    pub fn compare(&self, other: &Expr<T>) -> bool {
        use self::ExprKind::*;
        if self.ty != other.ty {
            return false
        }
        let same_kind = match (&self.kind, &other.kind) {
            (&BinOp(kind1, _, _), &BinOp(kind2, _, _)) if kind1 == kind2 => true,
            (&Let(ref sym1, _, _), &Let(ref sym2, _, _)) if sym1 == sym2 => true,
            (&NewBuilder, &NewBuilder) => true,
            (&Lambda(ref params1, _), &Lambda(ref params2, _)) => {
                params1.len() == params2.len() && params1.iter().zip(params2).all(|t| t.0 == t.1)
            }
            (&MakeStruct(..), &MakeStruct(..)) => true,
            (&MakeVector(..), &MakeVector(..)) => true,
            (&GetField(_, idx1), &GetField(_, idx2)) if idx1 == idx2 => true,
            (&Merge(..), &Merge(..)) => true,
            (&Res(..), &Res(..)) => true,
            (&For(..), &For(..)) => true,
            (&If(..), &If(..)) => true,
            (&Apply(..), &Apply(..)) => true,
            (&BoolLiteral(ref l), &BoolLiteral(ref r)) if l == r => true,
            (&I32Literal(ref l), &I32Literal(ref r)) if l == r => true,
            (&I64Literal(ref l), &I64Literal(ref r)) if l == r => true,
            (&F32Literal(ref l), &F32Literal(ref r)) if l == r => true,
            (&F64Literal(ref l), &F64Literal(ref r)) if l == r => true,
            (&Ident(ref l), &Ident(ref r)) if l == r => true,
            _ => false // all else fail.
        };
        if !same_kind {
            return false
        }
        let children: Vec<_> = self.children().collect();
        let other_children: Vec<_> = other.children().collect();
        if children.len() == other_children.len() {
            children.iter().zip(other_children).all(|e| e.0.compare(&e.1))
        } else {
            false
        }
    }

    /// Compares two expression trees, returning true if they are the same modulo symbol names.
    /// Symbols in the two expressions must have a one to one correspondance for the trees to be
    /// considered equal.
    ///
    /// Caveats:
    ///     - If an undeclared symbol is encountered, this method return false.
    pub fn compare_ignoring_symbols(&self, other: &Expr<T>) -> bool {
        use self::ExprKind::*;
        use std::collections::HashMap;
        let mut sym_map: HashMap<&Symbol, &Symbol> = HashMap::new();

        fn _compare_ignoring_symbols<'b, 'a, U:Clone+Eq>(e1: &'a Expr<U>, e2: &'b Expr<U>, sym_map: &mut HashMap<&'a Symbol, &'b Symbol>) -> bool {
            if e1.ty != e2.ty {
                return false
            }
            let same_kind = match (&e1.kind, &e2.kind) {
                (&BinOp(ref kind1, _, _), &BinOp(ref kind2, _, _)) if kind1 == kind2 => true,
                (&Let(ref sym1, _, _), &Let(ref sym2, _, _)) => {
                    sym_map.insert(sym1, sym2);
                    true
                },
                (&Lambda(ref params1, _), &Lambda(ref params2, _)) => {
                    // Just compare types, and assume the symbol names "match up".
                    if params1.len() == params2.len() && params1.iter().zip(params2).all(|t| t.0.ty == t.1.ty) {
                        for (p1, p2) in params1.iter().zip(params2) {
                            sym_map.insert(&p1.name,  &p2.name);
                        }
                        true
                    } else {
                        false
                    }
                },
                (&NewBuilder, &NewBuilder) => true,
                (&MakeStruct(..), &MakeStruct(..)) => true,
                (&MakeVector(..), &MakeVector(..)) => true,
                (&GetField(_, idx1), &GetField(_, idx2)) if idx1 == idx2 => true,
                (&Merge(..), &Merge(..)) => true,
                (&Res(..), &Res(..)) => true,
                (&For(..), &For(..)) => true,
                (&If(..), &If(..)) => true,
                (&Apply(..), &Apply(..)) => true,
                (&BoolLiteral(ref l), &BoolLiteral(ref r)) if l == r => true,
                (&I32Literal(ref l), &I32Literal(ref r)) if l == r => true,
                (&I64Literal(ref l), &I64Literal(ref r)) if l == r => true,
                (&F32Literal(ref l), &F32Literal(ref r)) if l == r => true,
                (&F64Literal(ref l), &F64Literal(ref r)) if l == r => true,
                (&Ident(ref l), &Ident(ref r)) => {
                    if let Some(lv) = sym_map.get(l) {
                        **lv == *r
                    } else {
                        false
                    }
                }
                _ => false // all else fail.
            };
            if !same_kind {
                return false
            }
            let children: Vec<_> = e1.children().collect();
            let e2_children: Vec<_> = e2.children().collect();
            if children.len() == e2_children.len() {
                children.iter().zip(e2_children).all(|e| _compare_ignoring_symbols(&e.0, &e.1, sym_map))
            } else {
                false
            }
        }
        _compare_ignoring_symbols(self, other, &mut sym_map)
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
                if params.iter().all(|p| p.name != *symbol) {
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

    /// Run a closure on this expression and every child, in pre-order.
    pub fn traverse<F>(&self, func: &mut F) where F: FnMut(&Expr<T>) -> () {
        func(self);
        for c in self.children() {
            c.traverse(func);
        }
    }

    /// Recursively transforms an expression in place by running a function on it and optionally replacing it with another expression.
    pub fn transform<F>(&mut self, func: &mut F) where F: FnMut(&mut Expr<T>) -> Option<Expr<T>> {
        if let Some(e) = func(self) {
            *self = e;
        }
        for c in self.children_mut() {
            c.transform(func);
        }
    }
}
