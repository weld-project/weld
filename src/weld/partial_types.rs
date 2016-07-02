//! Partial types and expressions tagged with them, used during parsing and type inference.

use std::vec::Vec;

use super::ast::*;
use super::error::*;

/// A partial data type, where some parameters may not be known.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PartialType {
    Unknown,
    Scalar(ScalarKind),
    Vector(Box<PartialType>),
    Builder(PartialBuilderKind),
    Function(Vec<PartialType>, Box<PartialType>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PartialBuilderKind {
    Appender(Box<PartialType>),
    Merger(Box<PartialType>, BinOpKind)
}

/// A partially typed expression.
pub type PartialExpr = Expr<PartialType>;

/// A partially typed parameter.
pub type PartialParameter = Parameter<PartialType>;

/// Create a box containing an untyped expression of the given kind.
pub fn expr_box(kind: ExprKind<PartialType>) -> Box<PartialExpr> {
    Box::new(PartialExpr { ty: PartialType::Unknown, kind: kind })
}

impl PartialType {
    /// Convert a PartialType to a Type if neither it nor any of its parameters are unknown.
    pub fn to_type(&self) -> WeldResult<Type> {
        use self::PartialType::*;
        use self::PartialBuilderKind::*;
        match *self {
            Unknown =>
                weld_err!("Incomplete partial type"),
            Scalar(kind) =>
                Ok(Type::Scalar(kind)),
            Vector(ref elem) =>
                Ok(Type::Vector(Box::new(try!(elem.to_type())))),
            Builder(Appender(ref elem)) =>
                Ok(Type::Builder(BuilderKind::Appender(Box::new(try!(elem.to_type()))))),
            Builder(Merger(ref elem, op)) =>
                Ok(Type::Builder(BuilderKind::Merger(Box::new(try!(elem.to_type())), op))),
            Function(ref params, ref res) => {
                let mut new_params = Vec::with_capacity(params.len());
                for p in params {
                    new_params.push(try!(p.to_type()));
                }
                let new_res = try!(res.to_type());
                Ok(Type::Function(new_params, Box::new(new_res)))
            }
        }
    }

    /// Is a PartialType complete (i.e. does it have no unkowns)?
    pub fn is_complete(&self) -> bool {
        use self::PartialType::*;
        use self::PartialBuilderKind::*;
        match *self {
            Unknown => false,
            Scalar(_) => true,
            Vector(ref elem) => elem.is_complete(),
            Builder(Appender(ref elem)) => elem.is_complete(),
            Builder(Merger(ref elem, _)) => elem.is_complete(),
            Function(ref params, ref res) =>
                params.iter().all(|p| p.is_complete()) && res.is_complete()
        }
    }
}

impl PartialBuilderKind {
    pub fn merge_type(&mut self) -> &PartialType {
        use self::PartialBuilderKind::*;
        match *self {
            Appender(ref elem) => elem.as_ref(),
            Merger(ref elem, _) => elem.as_ref(),
        }
    }

    pub fn merge_type_mut(&mut self) -> &mut PartialType {
        use self::PartialBuilderKind::*;
        match *self {
            Appender(ref mut elem) => elem.as_mut(),
            Merger(ref mut elem, _) => elem.as_mut(),
        }
    }

    pub fn result_type(&self) -> PartialType {
        use self::PartialType::*;
        use self::PartialBuilderKind::*;
        match *self {
            Appender(ref elem) => Vector((*elem).clone()),
            Merger(ref elem, _) => *elem.clone(),
        }
    }
}

impl PartialParameter {
    pub fn to_typed(&self) -> WeldResult<TypedParameter> {
        let t = try!(self.ty.to_type());
        Ok(TypedParameter { name: self.name.clone(), ty: t })
    }
}

impl PartialExpr {
    pub fn to_typed(&self) -> WeldResult<TypedExpr> {
        use ast::ExprKind::*;

        fn typed_box(expr: &Box<PartialExpr>) -> WeldResult<Box<TypedExpr>> {
            let e = try!(expr.to_typed());
            Ok(Box::new(e))
        }

        let new_kind: ExprKind<Type> = match self.kind {
            BoolLiteral(b) => BoolLiteral(b),
            I32Literal(i) => I32Literal(i),
            Ident(ref name) => Ident(name.clone()),
            NewBuilder => NewBuilder,

            BinOp(op, ref left, ref right) =>
                BinOp(op, try!(typed_box(left)), try!(typed_box(right))),
            
            Let(ref name, ref value, ref body) => 
                Let(name.clone(), try!(typed_box(value)), try!(typed_box(body))),

            Lambda(ref params, ref body) => {
                let params: WeldResult<Vec<_>> = params.iter().map(|p| p.to_typed()).collect();
                Lambda(try!(params), try!(typed_box(body)))
            }

            MakeVector(ref exprs) => {
                let exprs: WeldResult<Vec<_>> = exprs.iter().map(|e| e.to_typed()).collect();
                MakeVector(try!(exprs))
            }

            Merge(ref bldr, ref value) =>
                Merge(try!(typed_box(bldr)), try!(typed_box(value))),

            Res(ref bldr) => Res(try!(typed_box(bldr))),

            For(ref data, ref bldr, ref func) =>
                For(try!(typed_box(data)), try!(typed_box(bldr)), try!(typed_box(func))),

            If(ref cond, ref on_true, ref on_false) =>
                If(try!(typed_box(cond)), try!(typed_box(on_true)), try!(typed_box(on_false))),

            Apply(ref func, ref params) => {
                let params: WeldResult<Vec<_>> = params.iter().map(|p| p.to_typed()).collect();
                Apply(try!(typed_box(func)), try!(params))
            }
        };

        Ok(TypedExpr { ty: try!(self.ty.to_type()), kind: new_kind })
    }
}