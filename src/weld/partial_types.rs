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
}
