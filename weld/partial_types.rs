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
    Dict(Box<PartialType>, Box<PartialType>),
    Builder(PartialBuilderKind, Annotations),
    Struct(Vec<PartialType>),
    Function(Vec<PartialType>, Box<PartialType>),
}

impl TypeBounds for PartialType {}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PartialBuilderKind {
    Appender(Box<PartialType>),
    // Key type, value type, merge type (struct of <key,value> pairs)
    DictMerger(Box<PartialType>, Box<PartialType>, Box<PartialType>, BinOpKind),
    // Key type, value type, merge type (struct of <key,value> pairs)
    GroupMerger(Box<PartialType>, Box<PartialType>, Box<PartialType>),
    // elem type, merge type (struct of <index, value> pairs
    VecMerger(Box<PartialType>, Box<PartialType>, BinOpKind),
    Merger(Box<PartialType>, BinOpKind),
}

/// A partially typed expression.
pub type PartialExpr = Expr<PartialType>;

/// A partially typed parameter.
pub type PartialParameter = Parameter<PartialType>;

/// Create a box containing an untyped expression of the given kind.
pub fn expr_box(kind: ExprKind<PartialType>) -> Box<PartialExpr> {
    Box::new(PartialExpr {
                 ty: PartialType::Unknown,
                 kind: kind,
                 annotations: Annotations::new(),
             })
}

impl PartialType {
    /// Convert a PartialType to a Type if neither it nor any of its parameters are unknown.
    pub fn to_type(&self) -> WeldResult<Type> {
        use self::PartialType::*;
        use self::PartialBuilderKind::*;
        match *self {
            Unknown => weld_err!("Incomplete partial type"),
            Scalar(kind) => Ok(Type::Scalar(kind)),
            Vector(ref elem) => Ok(Type::Vector(Box::new(try!(elem.to_type())))),
            Dict(ref kt, ref vt) => {
                Ok(Type::Dict(Box::new(try!(kt.to_type())), Box::new(try!(vt.to_type()))))
            }
            Builder(Appender(ref elem), ref annotations) => {
                Ok(Type::Builder(BuilderKind::Appender(Box::new(try!(elem.to_type()))),
                                 annotations.clone()))
            }
            Builder(DictMerger(ref kt, ref vt, _, op), ref annotations) => {
                Ok(Type::Builder(BuilderKind::DictMerger(Box::new(try!(kt.to_type())),
                                                         Box::new(try!(vt.to_type())),
                                                         op),
                                 annotations.clone()))
            }
            Builder(GroupMerger(ref kt, ref vt, _), ref annotations) => {
                Ok(Type::Builder(BuilderKind::GroupMerger(Box::new(try!(kt.to_type())),
                                                          Box::new(try!(vt.to_type()))),
                                 annotations.clone()))
            }
            Builder(VecMerger(ref elem, _, op), ref annotations) => {
                Ok(Type::Builder(BuilderKind::VecMerger(Box::new(try!(elem.to_type())), op),
                                 annotations.clone()))
            }
            Builder(Merger(ref elem, op), ref annotations) => {
                Ok(Type::Builder(BuilderKind::Merger(Box::new(try!(elem.to_type())), op),
                                 annotations.clone()))
            }
            Struct(ref elems) => {
                let mut new_elems = Vec::with_capacity(elems.len());
                for e in elems {
                    new_elems.push(try!(e.to_type()));
                }
                Ok(Type::Struct(new_elems))
            }
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
            Dict(ref kt, ref vt) => kt.is_complete() && vt.is_complete(),
            Builder(Appender(ref elem), _) => elem.is_complete(),
            Builder(DictMerger(ref kt, ref vt, _, _), _) => kt.is_complete() && vt.is_complete(),
            Builder(GroupMerger(ref kt, ref vt, _), _) => kt.is_complete() && vt.is_complete(),
            Builder(VecMerger(ref elem, _, _), _) => elem.is_complete(),
            Builder(Merger(ref elem, _), _) => elem.is_complete(),
            Struct(ref elems) => elems.iter().all(|e| e.is_complete()),
            Function(ref params, ref res) => {
                params.iter().all(|p| p.is_complete()) && res.is_complete()
            }
        }
    }
}

impl PartialBuilderKind {
    pub fn merge_type(&mut self) -> PartialType {
        use self::PartialBuilderKind::*;
        match *self {
            Appender(ref elem) => *elem.clone(),
            DictMerger(_, _, ref mt, _) => *mt.clone(),
            GroupMerger(_, _, ref mt) => *mt.clone(),
            VecMerger(_, ref mt, _) => *mt.clone(),
            Merger(ref elem, _) => *elem.clone(),
        }
    }

    pub fn merge_type_mut(&mut self) -> &mut PartialType {
        use self::PartialBuilderKind::*;
        match *self {
            Appender(ref mut elem) => elem.as_mut(),
            DictMerger(_, _, ref mut mt, _) => mt.as_mut(),
            GroupMerger(_, _, ref mut mt) => mt.as_mut(),
            VecMerger(_, ref mut mt, _) => mt.as_mut(),
            Merger(ref mut elem, _) => elem.as_mut(),
        }
    }

    pub fn result_type(&self) -> PartialType {
        use self::PartialType::*;
        use self::PartialBuilderKind::*;
        match *self {
            Appender(ref elem) => Vector((*elem).clone()),
            DictMerger(ref kt, ref vt, _, _) => Dict((*kt).clone(), (*vt).clone()),
            GroupMerger(ref kt, ref vt, _) => Dict((*kt).clone(), Box::new(Vector((*vt).clone()))),
            VecMerger(ref elem, _, _) => Vector((*elem).clone()),
            Merger(ref elem, _) => *elem.clone(),
        }
    }
}

impl PartialParameter {
    pub fn to_typed(&self) -> WeldResult<TypedParameter> {
        let t = try!(self.ty.to_type());
        Ok(TypedParameter {
               name: self.name.clone(),
               ty: t,
           })
    }
}

impl PartialExpr {
    pub fn to_typed(&self) -> WeldResult<TypedExpr> {
        use ast::ExprKind::*;
        use ast::LiteralKind::*;

        fn typed_box(expr: &Box<PartialExpr>) -> WeldResult<Box<TypedExpr>> {
            let e = try!(expr.to_typed());
            Ok(Box::new(e))
        }

        let new_kind: ExprKind<Type> = match self.kind {
            Literal(BoolLiteral(v)) => Literal(BoolLiteral(v)),
            Literal(I8Literal(v)) => Literal(I8Literal(v)),
            Literal(I32Literal(v)) => Literal(I32Literal(v)),
            Literal(I64Literal(v)) => Literal(I64Literal(v)),
            Literal(F32Literal(v)) => Literal(F32Literal(v)),
            Literal(F64Literal(v)) => Literal(F64Literal(v)),
            Ident(ref name) => Ident(name.clone()),

            CUDF {
                ref sym_name,
                ref args,
                ref return_ty,
            } => {
                let sym_name: String = sym_name.clone();
                let args: WeldResult<Vec<_>> = args.iter().map(|e| e.to_typed()).collect();
                let return_ty: Box<Type> = Box::new(try!(return_ty.to_type()));
                CUDF {
                    sym_name: sym_name,
                    args: try!(args),
                    return_ty: return_ty,
                }
            }

            BinOp {
                kind,
                ref left,
                ref right,
            } => {
                BinOp {
                    kind: kind,
                    left: try!(typed_box(left)),
                    right: try!(typed_box(right)),
                }
            }

            Cast {
                kind,
                ref child_expr,
            } => {
                Cast {
                    kind: kind,
                    child_expr: try!(typed_box(child_expr)),
                }
            }

            ToVec { ref child_expr } => ToVec { child_expr: try!(typed_box(child_expr)) },

            Let {
                ref name,
                ref value,
                ref body,
            } => {
                Let {
                    name: name.clone(),
                    value: try!(typed_box(value)),
                    body: try!(typed_box(body)),
                }
            }

            Lambda {
                ref params,
                ref body,
            } => {
                let params: WeldResult<Vec<_>> = params.iter().map(|p| p.to_typed()).collect();
                Lambda {
                    params: try!(params),
                    body: try!(typed_box(body)),
                }
            }

            MakeVector { ref elems } => {
                let elems: WeldResult<Vec<_>> = elems.iter().map(|e| e.to_typed()).collect();
                MakeVector { elems: try!(elems) }
            }

            Zip { ref vectors } => {
                let vectors: WeldResult<Vec<_>> = vectors.iter().map(|e| e.to_typed()).collect();
                Zip { vectors: try!(vectors) }
            }

            MakeStruct { ref elems } => {
                let elems: WeldResult<Vec<_>> = elems.iter().map(|e| e.to_typed()).collect();
                MakeStruct { elems: try!(elems) }
            }

            GetField { ref expr, index } => {
                GetField {
                    expr: try!(typed_box(expr)),
                    index: index,
                }
            }

            Length { ref data } => Length { data: try!(typed_box(data)) },
            Lookup {
                ref data,
                ref index,
            } => {
                Lookup {
                    data: try!(typed_box(data)),
                    index: try!(typed_box(index)),
                }
            }
            KeyExists { ref data, ref key } => {
                KeyExists {
                    data: try!(typed_box(data)),
                    key: try!(typed_box(key)),
                }
            }

            Slice {
                ref data,
                ref index,
                ref size,
            } => {
                Slice {
                    data: try!(typed_box(data)),
                    index: try!(typed_box(index)),
                    size: try!(typed_box(size)),
                }
            }

            Exp { ref value } => Exp { value: try!(typed_box(value)) },

            Merge {
                ref builder,
                ref value,
            } => {
                Merge {
                    builder: try!(typed_box(builder)),
                    value: try!(typed_box(value)),
                }
            }

            Res { ref builder } => Res { builder: try!(typed_box(builder)) },

            For {
                ref iters,
                ref builder,
                ref func,
            } => {
                let mut typed_iters = vec![];
                for iter in iters {
                    let start = match iter.start {
                        Some(ref s) => Some(try!(typed_box(s))),
                        None => None,
                    };
                    let end = match iter.end {
                        Some(ref e) => Some(try!(typed_box(e))),
                        None => None,
                    };
                    let stride = match iter.stride {
                        Some(ref s) => Some(try!(typed_box(s))),
                        None => None,
                    };
                    let typed_iter = Iter {
                        data: try!(typed_box(&iter.data)),
                        start: start,
                        end: end,
                        stride: stride,
                    };
                    typed_iters.push(typed_iter);
                }
                For {
                    iters: typed_iters,
                    builder: try!(typed_box(builder)),
                    func: try!(typed_box(func)),
                }
            }

            If {
                ref cond,
                ref on_true,
                ref on_false,
            } => {
                If {
                    cond: try!(typed_box(cond)),
                    on_true: try!(typed_box(on_true)),
                    on_false: try!(typed_box(on_false)),
                }
            }

            Apply {
                ref func,
                ref params,
            } => {
                let params: WeldResult<Vec<_>> = params.iter().map(|p| p.to_typed()).collect();
                Apply {
                    func: try!(typed_box(func)),
                    params: try!(params),
                }
            }
            NewBuilder(ref arg) => {
                let typed_arg = match *arg {
                    Some(ref e) => Some(try!(typed_box(e))),
                    None => None,
                };
                NewBuilder(typed_arg)
            }
            Negate(ref expr) => Negate(try!(typed_box(expr))),
        };

        Ok(TypedExpr {
               ty: try!(self.ty.to_type()),
               kind: new_kind,
               annotations: Annotations::new(),
           })
    }
}
