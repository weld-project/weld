//! Defines the Weld abstract syntax tree.

use annotation::Annotations;
use error::*;
use util;

use self::ExprKind::*;
use self::ScalarKind::*;
use self::BinOpKind::*;

use std::fmt;
use std::vec;

#[cfg(test)]
use tests::*;

/// A type on a Weld expression.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    /// A scalar.
    Scalar(ScalarKind),
    /// A SIMD vector.
    Simd(ScalarKind),
    /// A variable-length vector.
    Vector(Box<Type>),
    /// A dictionary mapping keys to values.
    Dict(Box<Type>, Box<Type>),
    /// A mutable builder to construct results.
    Builder(BuilderKind, Annotations),
    /// An ordered struct or tuple.
    Struct(Vec<Type>),
    /// A function with a list of arguments and return type.
    Function(Vec<Type>, Box<Type>),
    /// An unknown type, used only before type inference.
    Unknown,
}

impl Type {
    /// Returns the child types of this `Type`.
    pub fn children(&self) -> vec::IntoIter<&Type> {
        use self::Type::*;
        use self::BuilderKind::*;
        match *self {
            Unknown | Scalar(_) | Simd(_) => vec![],
            Vector(ref elem) => {
                vec![elem.as_ref()]
            }
            Dict(ref key, ref value) => {
                vec![key.as_ref(), value.as_ref()]
            }
            Builder(ref kind, _) => match kind {
                Appender(ref elem) => {
                    vec![elem.as_ref()]
                }
                Merger(ref elem, _) => {
                    vec![elem.as_ref()]
                }
                DictMerger(ref key, ref value, _) => {
                    vec![key.as_ref(), value.as_ref()]
                }
                GroupMerger(ref key, ref value) =>  {
                    vec![key.as_ref(), value.as_ref()]
                }
                VecMerger(ref elem, _) => {
                    vec![elem.as_ref()]
                }
            },
            Struct(ref elems) => elems.iter().collect(),
            Function(ref params, ref res) => {
                let mut children = vec![];
                for param in params.iter() {
                    children.push(param);
                }
                children.push(res.as_ref());
                children
            }
        }.into_iter()
    }

    /// Returns whether this `Type` is a SIMD value.
    ///
    /// A value is a SIMD value if its a `Simd` type or it is a `Struct` where each member is a
    /// `Simd` type. We additionally consider each of the builders to be SIMD values, since they
    /// can operate over SIMD values as inputs.
    pub fn is_simd(&self) -> bool {
        use self::Type::*;
        match *self {
            Simd(_) | Builder(_, _) => true,
            Struct(ref fields) => fields.iter().all(|f| f.is_simd()),
            _ => false
        }
    }

    pub fn is_scalar(&self) -> bool {
        use self::Type::Scalar;
        match *self {
            Scalar(_) => true,
            _ => false
        }
    }

    /// Return the vectorized version of a type.
    ///
    /// This method returns an error if this `Type` is not vectorizable.
    pub fn simd_type(&self) -> WeldResult<Type> {
        use self::Type::*;
        match *self {
            Scalar(kind) => Ok(Simd(kind)),
            Builder(_, _) => Ok(self.clone()),
            Struct(ref fields) => {
                let result: WeldResult<_> = fields.iter().map(|f| f.simd_type()).collect();
                Ok(Struct(result?))
            }
            _ => compile_err!("simd_type called on non-scalar type {}", self)
        }
    }

    /// Return the scalar version of a type.
    ///
    /// This method returns an error if this `Type` is not scalarizable.
    pub fn scalar_type(&self) -> WeldResult<Type> {
        use self::Type::*;
        match *self {
            Simd(kind) => Ok(Scalar(kind)),
            Builder(_, _) => Ok(self.clone()),
            Struct(ref fields) => {
                let result: WeldResult<_> = fields.iter().map(|f| f.scalar_type()).collect();
                Ok(Struct(result?))
            }
            _ => compile_err!("scalar_type called on non-SIMD type {}", self)
        }
    }

    /// Returns the type merged into a builder.
    ///
    /// Returns an error if this `Type` is not a builder type.
    pub fn merge_type(&self) -> WeldResult<Type> {
        use self::Type::Builder;
        if let Builder(ref kind, _) = *self {
            Ok(kind.merge_type())

        } else {
            compile_err!("merge_type called on non-builder type {}", self)
        }
    }

    /// Returns whether this `Type` is partial.
    ///
    /// A type is partial if it or any of its subtypes is `Unknown`.
    pub fn partial_type(&self) -> bool {
        use self::Type::Unknown;
        match *self {
            Unknown => true,
            _ => self.children().any(|t| t.partial_type()),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::Type::*;
        let ref text = match *self {
            Scalar(ref kind) => {
                format!("{}", kind)
            }
            Simd(ref kind) => {
                format!("simd[{}]", kind)
            }
            Vector(ref elem) => {
                format!("vec[{}]", elem)
            }
            Dict(ref key, ref value) => {
                format!("dict[{},{}]", key, value)
            }
            Struct(ref elems) => {
                util::join("{", ",", "}", elems.iter().map(|e| e.to_string()))
            }
            Function(ref params, ref return_type) => {
                let mut res = util::join("|", ",", "|(", params.iter().map(|e| e.to_string()));
                res.push_str(&return_type.to_string());
                res.push_str(")");
                res
            }
            Builder(ref kind, ref annotations) => {
                format!("{}{}", annotations, kind)
            }
            Unknown => String::from("?")
        };
        f.write_str(text)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Scalar types in the Weld IR.
pub enum ScalarKind {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
}

impl ScalarKind {
    /// Returns whether this scalar is a floating-point type.
    ///
    /// The current floating point kinds are `F32` and `F64`.
    pub fn is_float(&self) -> bool {
        match *self {
            F32 | F64 => true,
            _ => false
        }
    }

    /// Returns whether this scalar is a boolean.
    pub fn is_bool(&self) -> bool {
        match *self {
            Bool => true,
            _ => false
        }
    }

    /// Returns whether this scalar is a signed integer.
    ///
    /// Booleans are not considered to be signed integers.
    pub fn is_signed_integer(&self) -> bool {
        match *self {
            I8 | I16 | I32 | I64 => true,
            _ => false
        }
    }

    /// Returns whether this scalar is an unsigned integer.
    ///
    /// Booleans are not considered to be unsigned integers.
    pub fn is_unsigned_integer(&self) -> bool {
        match *self {
            U8 | U16 | U32 | U64 => true,
            _ => false
        }
    }

    /// Returns whether this scalar is an integer.
    ///
    /// Booleans are not considered to be integers.
    pub fn is_integer(&self) -> bool {
        return self.is_signed_integer() || self.is_unsigned_integer();
    }

    /// Return the length of this scalar type in bits.
    pub fn bits(&self) -> u32 {
        match *self {
            Bool => 1,
            I8 | U8 => 8,
            I16 | U16 => 16,
            I32 | U32 | F32 => 32,
            I64 | U64 | F64 => 64
        }
    }
}

impl fmt::Display for ScalarKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let text = match *self {
            Bool => "bool",
            I8 => "i8",
            I16 => "i16",
            I32 => "i32",
            I64 => "i64",
            U8 => "u8",
            U16 => "u16",
            U32 => "u32",
            U64 => "u64",
            F32 => "f32",
            F64 => "f64",
        };
        f.write_str(text)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Defines the builder types in Weld.
pub enum BuilderKind {
    /// A builder that appends items to a list.
    Appender(Box<Type>),
    /// A builder that constructs a scalar or struct.
    ///
    /// Results are merged using an associative binary operator.
    Merger(Box<Type>, BinOpKind),
    /// A builder that creates a dictionary.
    ///
    /// Results are grouped by key, and values are combined using an associative binary operator.
    DictMerger(Box<Type>, Box<Type>, BinOpKind),
    /// A builder that creates groups.
    ///
    /// Results are grouped by key, where values are appended to a list.
    GroupMerger(Box<Type>, Box<Type>),
    /// A builder that constructs a vector by updating elements.
    ///
    /// Elements are updated by index using an associative binary operator.
    VecMerger(Box<Type>, BinOpKind),
}

impl BuilderKind {
    /// Returns the type merged into this `BuilderKind`.
    pub fn merge_type(&self) -> Type {
        use self::BuilderKind::*;
        use self::Type::*;
        match *self {
            Appender(ref elem) => *elem.clone(),
            Merger(ref elem, _) => *elem.clone(),
            DictMerger(ref key, ref value, _) => Struct(vec![*key.clone(), *value.clone()]),
            GroupMerger(ref key, ref value) => Struct(vec![*key.clone(), *value.clone()]),
            VecMerger(ref elem, _) => Struct(vec![Scalar(I64), *elem.clone()]),
        }
    }

    /// Returns the type produced by this `BuilderKind`.
    pub fn result_type(&self) -> Type {
        use self::Type::*;
        use self::BuilderKind::*;
        match *self {
            Appender(ref elem) => Vector(elem.clone()),
            Merger(ref elem, _) => *elem.clone(),
            DictMerger(ref key, ref value, _) => Dict(key.clone(), value.clone()),
            GroupMerger(ref key, ref value) => Dict(key.clone(), Box::new(Vector(value.clone()))),
            VecMerger(ref elem, _) => Vector(elem.clone()),
        }
    }
}

impl fmt::Display for BuilderKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::BuilderKind::*;
        let ref text = match *self {
            Appender(ref t) => {
                format!("appender[{}]", t)
            }
            DictMerger(ref key, ref value, op) => {
                format!("dictmerger[{},{},{}]", key, value, op)
            }
            GroupMerger(ref key, ref value) => {
                format!("groupmerger[{},{}]", key, value)
            }
            VecMerger(ref elem, op) => {
                format!("vecmerger[{},{}]", elem, op)
            }
            Merger(ref elem, op) => {
                format!("merger[{},{}]", elem, op)
            }
        };
        f.write_str(text)
    }
}

// -------------------------------

/// A named symbol in the Weld AST.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol {
    pub name: String,
    pub id: i32,
}

impl Symbol {
    pub fn new(name: &str, id: i32) -> Symbol {
        Symbol {
            name: name.into(),
            id: id,
        }
    }

    pub fn name(name: &str) -> Symbol {
        Symbol {
            name: name.into(),
            id: 0,
        }
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.id == 0 {
            write!(f, "{}", self.name)
        } else {
            write!(f, "{}__{}", self.name, self.id)
        }
    }
}

/// An expression tree, having type annotations of type T. We make this parametrized because
/// expressions have different "kinds" of types attached to them at different points in the
/// compilation process -- namely PartialType when parsed and then Type after type inference.
#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub ty: Type,
    pub kind: ExprKind,
    pub annotations: Annotations,
}

/// An iterator kind, which specifies how data should be loaded and passed to a `For` loop.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum IterKind {
    ScalarIter, // A standard scalar iterator.
    SimdIter, // A vector iterator.
    FringeIter, // A fringe iterator, handling the fringe of a vector iter.
    NdIter,     // multi-dimensional nd-iter
    RangeIter,
}

impl fmt::Display for IterKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::IterKind::*;
        let ref text = match *self {
            ScalarIter => "",
            SimdIter => "simd",
            FringeIter => "fringe",
            NdIter => "nditer",
            RangeIter => "range",
        };
        f.write_str(text)?;
        f.write_str("iter")
    }
}

/// An iterator, which specifies a vector to iterate over and optionally a start index,
/// end index, and stride.
#[derive(Clone, Debug, PartialEq)]
pub struct Iter {
    pub data: Box<Expr>,
    pub start: Option<Box<Expr>>,
    pub end: Option<Box<Expr>>,
    pub stride: Option<Box<Expr>>,
    pub kind: IterKind,
    pub strides: Option<Box<Expr>>,
    pub shape: Option<Box<Expr>>,
}

impl Iter {
    /// Returns true if this is a simple iterator with no start/stride/end specified
    /// (i.e., it iterates over all the input data) and kind `ScalarIter`.
    pub fn is_simple(&self) -> bool {
        return self.start.is_none() && self.end.is_none() && self.stride.is_none() &&
            self.kind == IterKind::ScalarIter;
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    Literal(LiteralKind),
    Ident(Symbol),
    Negate(Box<Expr>),
    // Broadcasts a scalar into a vector, e.g., 1 -> <1, 1, 1, 1>
    Broadcast(Box<Expr>),
    BinOp {
        kind: BinOpKind,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    UnaryOp {
        kind: UnaryOpKind,
        value: Box<Expr>,
    },
    Cast {
        kind: ScalarKind,
        child_expr: Box<Expr>,
    },
    ToVec { child_expr: Box<Expr> },
    MakeStruct { elems: Vec<Expr> },
    MakeVector { elems: Vec<Expr> },
    Zip { vectors: Vec<Expr> },
    GetField { expr: Box<Expr>, index: u32 },
    Length { data: Box<Expr> },
    Lookup {
        data: Box<Expr>,
        index: Box<Expr>,
    },
    KeyExists {
        data: Box<Expr>,
        key: Box<Expr>,
    },
    Slice {
        data: Box<Expr>,
        index: Box<Expr>,
        size: Box<Expr>,
    },
    Sort {
        data: Box<Expr>,
        keyfunc: Box<Expr>,
    },
    Let {
        name: Symbol,
        value: Box<Expr>,
        body: Box<Expr>,
    },
    If {
        cond: Box<Expr>,
        on_true: Box<Expr>,
        on_false: Box<Expr>,
    },
    Iterate {
        initial: Box<Expr>,
        update_func: Box<Expr>,
    },
    Select {
        cond: Box<Expr>,
        on_true: Box<Expr>,
        on_false: Box<Expr>,
    },
    Lambda {
        params: Vec<Parameter>,
        body: Box<Expr>,
    },
    Apply {
        func: Box<Expr>,
        params: Vec<Expr>,
    },
    CUDF {
        sym_name: String,
        args: Vec<Expr>,
        return_ty: Box<Type>,
    },
    Serialize(Box<Expr>),
    Deserialize {
        value: Box<Expr>,
        value_ty: Box<Type>,
    },
    NewBuilder(Option<Box<Expr>>),
    For {
        iters: Vec<Iter>,
        builder: Box<Expr>,
        func: Box<Expr>,
    },
    Merge {
        builder: Box<Expr>,
        value: Box<Expr>,
    },
    Res { builder: Box<Expr> },
}

impl ExprKind {
    /// Return a readable name for the kind which is independent of any subexpressions.
    pub fn name(&self) -> &str {
        match *self {
            Literal(_) => "Literal",
            Ident(_) => "Ident",
            Negate(_) => "Negate",
            Broadcast(_) => "Broadcast",
            BinOp{ .. } => "BinOp",
            UnaryOp { .. } => "UnaryOp",
            Cast { .. } => "Cast",
            ToVec { .. } => "ToVec",
            MakeStruct { .. } => "MakeStruct",
            MakeVector { .. } => "MakeVector",
            Zip { .. } => "Zip",
            GetField { .. } => "GetField",
            Length { .. } => "Length",
            Lookup { .. } => "Lookup",
            KeyExists { .. } => "KeyExists",
            Slice { .. } => "Slice",
            Sort { .. } => "Sort",
            Let { .. } => "Let",
            If { .. } => "If",
            Iterate { .. } => "Iterate",
            Select { .. } => "Select",
            Lambda  { .. } => "Lambda",
            Apply { .. } => "Apply",
            CUDF { .. } => "CUDF",
            Serialize(_) => "Serialize",
            Deserialize { .. } => "Deserialize",
            NewBuilder(_) => "NewBuilder",
            For { .. } => "For",
            Merge { .. } => "Merge",
            Res { .. } => "Res",
        }
    }

    pub fn is_builder_expr(&self) -> bool {
        match *self {
            Merge { .. } | Res { .. } | For { .. } | NewBuilder(_) => true,
            _ => false,
        }
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum LiteralKind {
    BoolLiteral(bool),
    I8Literal(i8),
    I16Literal(i16),
    I32Literal(i32),
    I64Literal(i64),
    U8Literal(u8),
    U16Literal(u16),
    U32Literal(u32),
    U64Literal(u64),
    F32Literal(u32),
    F64Literal(u64),
    StringLiteral(String),
}

impl fmt::Display for LiteralKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::LiteralKind::*;
        let ref text = match *self {
            BoolLiteral(v) => format!("{}", v),
            I8Literal(v) => format!("{}c", v),
            I16Literal(v) => format!("{}si", v),
            I32Literal(v) => format!("{}", v),
            I64Literal(v) => format!("{}L", v),
            U8Literal(v) => format!("{}", v),
            U16Literal(v) => format!("{}", v),
            U32Literal(v) => format!("{}", v),
            U64Literal(v) => format!("{}", v),
            F32Literal(v) => {
                let mut res = format!("{}", f32::from_bits(v));
                // Hack to disambiguate from integers.
                if !res.contains(".") {
                    res.push_str(".0");
                }
                res.push_str("F");
                res
            }
            F64Literal(v) => {
                let mut res = format!("{}", f64::from_bits(v));
                // Hack to disambiguate from integers.
                if !res.contains(".") {
                    res.push_str(".0");
                }
                res
            }
            StringLiteral(ref v) => format!("\"{}\"", v),
        };
        f.write_str(text)
    }
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
    Max,
    Min,
    Pow,
}

impl BinOpKind {
    pub fn is_comparison(&self) -> bool {
        match *self {
            Equal | NotEqual | LessThan | GreaterThan | LessThanOrEqual | GreaterThanOrEqual => {
                true
            }
            _ => false,
        }
    }
}

impl fmt::Display for BinOpKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
            Max => "max",
            Min => "min",
            Pow => "pow",
        };
        f.write_str(text)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOpKind {
    Exp,
    Log,
    Sqrt,
    Sin,
    Cos,
    Tan,
    ASin,
    ACos,
    ATan,
    Sinh,
    Cosh,
    Tanh,
    Erf,
}

impl fmt::Display for UnaryOpKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let text = format!("{:?}", self);
        f.write_str(text.to_lowercase().as_ref())
    }
}



#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Parameter {
    pub name: Symbol,
    pub ty: Type,
}

impl fmt::Display for Parameter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{}", self.name, self.ty)
    }
}

impl Expr {
    /// Get an iterator for the children of this expression.
    pub fn children(&self) -> vec::IntoIter<&Expr> {
        use self::ExprKind::*;
        match self.kind {
            BinOp {
                ref left,
                ref right,
                ..
            } => vec![left.as_ref(), right.as_ref()],
            UnaryOp {ref value, ..} => vec![value.as_ref()],
            Cast { ref child_expr, .. } => vec![child_expr.as_ref()],
            ToVec { ref child_expr } => vec![child_expr.as_ref()],
            Let {
                ref value,
                ref body,
                ..
            } => vec![value.as_ref(), body.as_ref()],
            Lambda { ref body, .. } => vec![body.as_ref()],
            MakeStruct { ref elems } => elems.iter().collect(),
            MakeVector { ref elems } => elems.iter().collect(),
            Zip { ref vectors } => vectors.iter().collect(),
            GetField { ref expr, .. } => vec![expr.as_ref()],
            Length { ref data } => vec![data.as_ref()],
            Lookup {
                ref data,
                ref index,
            } => vec![data.as_ref(), index.as_ref()],
            KeyExists { ref data, ref key } => vec![data.as_ref(), key.as_ref()],
            Slice {
                ref data,
                ref index,
                ref size,
            } => vec![data.as_ref(), index.as_ref(), size.as_ref()],
            Sort {
                ref data,
                ref keyfunc,
            } => vec![data.as_ref(), keyfunc.as_ref()],
            Merge {
                ref builder,
                ref value,
            } => vec![builder.as_ref(), value.as_ref()],
            Res { ref builder } => vec![builder.as_ref()],
            For {
                ref iters,
                ref builder,
                ref func,
            } => {
                let mut res: Vec<&Expr> = vec![];
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
                res.push(builder.as_ref());
                res.push(func.as_ref());
                res
            }
            If {
                ref cond,
                ref on_true,
                ref on_false,
            } => vec![cond.as_ref(), on_true.as_ref(), on_false.as_ref()],
            Iterate {
                ref initial,
                ref update_func,
            } => vec![initial.as_ref(), update_func.as_ref()],
            Select {
                ref cond,
                ref on_true,
                ref on_false,
            } => vec![cond.as_ref(), on_true.as_ref(), on_false.as_ref()],
            Apply {
                ref func,
                ref params,
            } => {
                let mut res = vec![func.as_ref()];
                res.extend(params.iter());
                res
            }
            NewBuilder(ref opt) => {
                if let Some(ref e) = *opt {
                    vec![e.as_ref()]
                } else {
                    vec![]
                }
            }
            Serialize(ref e) => vec![e.as_ref()],
            Deserialize { ref value, .. } => vec![value.as_ref()],
            CUDF { ref args, .. } => args.iter().collect(),
            Negate(ref t) => vec![t.as_ref()],
            Broadcast(ref t) => vec![t.as_ref()],
            // Explicitly list types instead of doing _ => ... to remember to add new types.
            Literal(_) | Ident(_) => vec![],
        }.into_iter()
    }

    /// Get an iterator of mutable references to the children of this expression.
    pub fn children_mut(&mut self) -> vec::IntoIter<&mut Expr> {
        use self::ExprKind::*;
        match self.kind {
            BinOp {
                ref mut left,
                ref mut right,
                ..
            } => vec![left.as_mut(), right.as_mut()],
            UnaryOp { ref mut value, .. } => vec![value.as_mut()],
            Cast { ref mut child_expr, .. } => vec![child_expr.as_mut()],
            ToVec { ref mut child_expr } => vec![child_expr.as_mut()],
            Let {
                ref mut value,
                ref mut body,
                ..
            } => vec![value.as_mut(), body.as_mut()],
            Lambda { ref mut body, .. } => vec![body.as_mut()],
            MakeStruct { ref mut elems } => elems.iter_mut().collect(),
            MakeVector { ref mut elems } => elems.iter_mut().collect(),
            Zip { ref mut vectors } => vectors.iter_mut().collect(),
            GetField { ref mut expr, .. } => vec![expr.as_mut()],
            Length { ref mut data } => vec![data.as_mut()],
            Lookup {
                ref mut data,
                ref mut index,
            } => vec![data.as_mut(), index.as_mut()],
            KeyExists {
                ref mut data,
                ref mut key,
            } => vec![data.as_mut(), key.as_mut()],
            Slice {
                ref mut data,
                ref mut index,
                ref mut size,
            } => vec![data.as_mut(), index.as_mut(), size.as_mut()],
            Sort {
                ref mut data,
                ref mut keyfunc,
            } => vec![data.as_mut(), keyfunc.as_mut()],
            Merge {
                ref mut builder,
                ref mut value,
            } => vec![builder.as_mut(), value.as_mut()],
            Res { ref mut builder } => vec![builder.as_mut()],
            For {
                ref mut iters,
                ref mut builder,
                ref mut func,
            } => {
                let mut res: Vec<&mut Expr> = vec![];
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
                res.push(builder.as_mut());
                res.push(func.as_mut());
                res
            }
            If {
                ref mut cond,
                ref mut on_true,
                ref mut on_false,
            } => vec![cond.as_mut(), on_true.as_mut(), on_false.as_mut()],
            Iterate {
                ref mut initial,
                ref mut update_func,
            } => vec![initial.as_mut(), update_func.as_mut()],
            Select {
                ref mut cond,
                ref mut on_true,
                ref mut on_false,
            } => vec![cond.as_mut(), on_true.as_mut(), on_false.as_mut()],

            Apply {
                ref mut func,
                ref mut params,
            } => {
                let mut res = vec![func.as_mut()];
                res.extend(params.iter_mut());
                res
            }
            NewBuilder(ref mut opt) => {
                if let Some(ref mut e) = *opt {
                    vec![e.as_mut()]
                } else {
                    vec![]
                }
            }
            Serialize(ref mut e) => vec![e.as_mut()],
            Deserialize { ref mut value, .. } => vec![value.as_mut()],
            CUDF { ref mut args, .. } => args.iter_mut().collect(),
            Negate(ref mut t) => vec![t.as_mut()],
            Broadcast(ref mut t) => vec![t.as_mut()],
            // Explicitly list types instead of doing _ => ... to remember to add new types.
            Literal(_) | Ident(_) => vec![],
        }.into_iter()
    }


    /// Returns whether this `Expr` is partially typed.
    ///
    /// A type is partial if it or any of its subtypes is `Unknown`.
    pub fn partially_typed(&self) -> bool {
        use self::Type::Unknown;
        match self.ty {
            Unknown => true,
            _ => self.children().any(|e| e.ty.partial_type()),
        }
    }

    /// Compares two expression trees, returning true if they are the same modulo symbol names.
    /// Symbols in the two expressions must have a one to one correspondance for the trees to be
    /// considered equal. If an undefined symbol is encountered during the comparison, it must
    /// be the same in both expressions (e.g. for a symbol captured from an outer scope).
    pub fn compare_ignoring_symbols(&self, other: &Expr) -> WeldResult<bool> {
        use self::ExprKind::*;
        use std::collections::HashMap;
        let mut sym_map: HashMap<&Symbol, &Symbol> = HashMap::new();
        let mut reverse_sym_map: HashMap<&Symbol, &Symbol> = HashMap::new();

        fn _compare_ignoring_symbols<'b, 'a>(e1: &'a Expr,
                                             e2: &'b Expr,
                                             sym_map: &mut HashMap<&'a Symbol, &'b Symbol>,
                                             reverse_sym_map: &mut HashMap<&'b Symbol, &'a Symbol>)
            -> WeldResult<bool> {
            // First, check the type.
            if e1.ty != e2.ty {
                return Ok(false);
            }
            // Check the kind of each expression. same_kind is true if each *non-expression* field
            // is equal and the kind of the expression matches. Also records corresponding symbol names.
            let same_kind = match (&e1.kind, &e2.kind) {
                (&BinOp { kind: ref kind1, .. }, &BinOp { kind: ref kind2, .. }) if kind1 ==
                                                                                    kind2 => {
                    Ok(true)
                }
                (&UnaryOp { .. }, &UnaryOp { .. }) => Ok(true),
                (&Cast { kind: ref kind1, .. }, &Cast { kind: ref kind2, .. }) if kind1 ==
                                                                                  kind2 => Ok(true),
                (&ToVec { .. }, &ToVec { .. }) => Ok(true),
                (&Let { name: ref sym1, .. }, &Let { name: ref sym2, .. }) => {
                    sym_map.insert(sym1, sym2);
                    reverse_sym_map.insert(sym2, sym1);
                    Ok(true)
                }
                (&Lambda { params: ref params1, .. }, &Lambda { params: ref params2, .. }) => {
                    // Just compare types, and assume the symbol names "match up".
                    if params1.len() == params2.len() &&
                       params1.iter().zip(params2).all(|t| t.0.ty == t.1.ty) {
                        for (p1, p2) in params1.iter().zip(params2) {
                            sym_map.insert(&p1.name, &p2.name);
                            reverse_sym_map.insert(&p2.name, &p1.name);
                        }
                        Ok(true)
                    } else {
                        Ok(false)
                    }
                }
                (&NewBuilder(_), &NewBuilder(_)) => Ok(true),
                (&Negate(_), &Negate(_)) => Ok(true),
                (&Broadcast(_), &Broadcast(_)) => Ok(true),
                (&MakeStruct { .. }, &MakeStruct { .. }) => Ok(true),
                (&MakeVector { .. }, &MakeVector { .. }) => Ok(true),
                (&Zip { .. }, &Zip { .. }) => Ok(true),
                (&GetField { index: idx1, .. }, &GetField { index: idx2, .. }) if idx1 == idx2 => {
                    Ok(true)
                }
                (&Length { .. }, &Length { .. }) => Ok(true),
                (&Lookup { .. }, &Lookup { .. }) => Ok(true),
                (&KeyExists { .. }, &KeyExists { .. }) => Ok(true),
                (&Slice { .. }, &Slice { .. }) => Ok(true),
                (&Sort { .. }, &Sort { .. }) => Ok(true),
                (&Merge { .. }, &Merge { .. }) => Ok(true),
                (&Res { .. }, &Res { .. }) => Ok(true),
                (&For { iters: ref liters, .. }, &For { iters: ref riters, .. })  => {
                    // If the iter kinds match for each iterator, the non-expression fields match.
                    Ok(liters.iter().zip(riters.iter()).all(|(ref l, ref r)| l.kind == r.kind))
                },
                (&If { .. }, &If { .. }) => Ok(true),
                (&Iterate { .. }, &Iterate { .. }) => Ok(true),
                (&Select { .. }, &Select { .. }) => Ok(true),
                (&Apply { .. }, &Apply { .. }) => Ok(true),
                (&CUDF {
                     sym_name: ref sym_name1,
                     return_ty: ref return_ty1,
                     ..
                 },
                 &CUDF {
                     sym_name: ref sym_name2,
                     return_ty: ref return_ty2,
                     ..
                 }) => {
                    let mut matches = sym_name1 == sym_name2;
                    matches = matches && return_ty1 == return_ty2;
                    Ok(matches)
                }
                (&Serialize(_), &Serialize(_)) => Ok(true),
                (&Deserialize { ref value_ty, .. }, &Deserialize { value_ty: ref value_ty2, .. }) if value_ty == value_ty2 => Ok(true),
                (&Literal(ref l), &Literal(ref r)) if l == r => Ok(true),
                (&Ident(ref l), &Ident(ref r)) => {
                    if let Some(lv) = sym_map.get(l) {
                        Ok(**lv == *r)
                    } else if reverse_sym_map.contains_key(r) {
                        Ok(false) // r was defined in other expression but l wasn't defined in self.
                    } else {
                        Ok(*l == *r)
                    }
                }
                _ => Ok(false), // all else fail.
            };

            // Return if encountered and error or kind doesn't match.
            if same_kind.is_err() || !same_kind.as_ref().unwrap() {
                return same_kind;
            }

            // Recursively check the children.
            let e1_children: Vec<_> = e1.children().collect();
            let e2_children: Vec<_> = e2.children().collect();
            if e1_children.len() != e2_children.len() {
                return Ok(false);
            }
            for (c1, c2) in e1_children.iter().zip(e2_children) {
                let res = _compare_ignoring_symbols(&c1, &c2, sym_map, reverse_sym_map);
                if res.is_err() || !res.as_ref().unwrap() {
                    return res;
                }
            }
            return Ok(true);
        }
        _compare_ignoring_symbols(self, other, &mut sym_map, &mut reverse_sym_map)
    }

    /// Substitute Ident nodes with the given symbol for another expression, stopping when an
    /// expression in the tree redefines the symbol (e.g. Let or Lambda parameters).
    pub fn substitute(&mut self, symbol: &Symbol, replacement: &Expr) {
        // Replace ourselves if we are exactly the symbol.
        use self::ExprKind::*;
        let mut self_matches = false;
        match self.kind {
            Ident(ref sym) if *sym == *symbol => self_matches = true,
            _ => (),
        }
        if self_matches {
            *self = (*replacement).clone();
            return;
        }

        // Otherwise, replace any relevant children, unless we redefine the symbol.
        match self.kind {
            Let {
                ref name,
                ref mut value,
                ref mut body,
            } => {
                value.substitute(symbol, replacement);
                if name != symbol {
                    body.substitute(symbol, replacement);
                }
            }

            Lambda {
                ref params,
                ref mut body,
            } => {
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
    pub fn traverse<F>(&self, func: &mut F)
        where F: FnMut(&Expr) -> ()
    {
        func(self);
        for c in self.children() {
            c.traverse(func);
        }
    }

    /// Returns `true` if this expression contains the symbol `sym` in an `Ident`.
    pub fn contains_symbol(&self, sym: &Symbol) -> bool {
        let mut found = false;
        self.traverse(&mut |ref mut e| {
            if let ExprKind::Ident(ref s) = e.kind {
                if *sym == *s {
                    found = true;
                }
            }
        });
        found
    }

    /// Recursively transforms an expression in place by running a function on it and optionally replacing it with another expression.
    pub fn transform_and_continue<F>(&mut self, func: &mut F)
        where F: FnMut(&mut Expr) -> (Option<Expr>, bool)
    {
        match func(self) {
            (Some(e), true) => {
                *self = e;
                return self.transform_and_continue(func);
            }
            (Some(e), false) => {
                *self = e;
            }
            (None, true) => {
                for c in self.children_mut() {
                    c.transform_and_continue(func);
                }
            }
            (None, false) => {}
        }
    }

    /// Recursively transforms an expression in place by running a function on it and optionally replacing it with another expression.
    /// Supports returning an error, which is treated as returning (None, false)
    pub fn transform_and_continue_res<F>(&mut self, func: &mut F)
        where F: FnMut(&mut Expr) -> WeldResult<(Option<Expr>, bool)>
        {
            if let Ok(result) = func(self) {
                match result {
                    (Some(e), true) => {
                        *self = e;
                        return self.transform_and_continue_res(func);
                    }
                    (Some(e), false) => {
                        *self = e;
                    }
                    (None, true) => {
                        for c in self.children_mut() {
                            c.transform_and_continue_res(func);
                        }
                    }
                    (None, false) => {}
                }
            }
        }


    /// Recursively transforms an expression in place by running a function on it and optionally replacing it with another expression.
    pub fn transform<F>(&mut self, func: &mut F)
        where F: FnMut(&mut Expr) -> Option<Expr>
    {
        if let Some(e) = func(self) {
            *self = e;
            return self.transform(func);
        }
        for c in self.children_mut() {
            c.transform(func);
        }
    }

    /// Recursively transforms an expression in place by running a function first on its children, then on the root
    /// expression itself; this can be more efficient than `transform` for some cases
    pub fn transform_up<F>(&mut self, func: &mut F)
        where F: FnMut(&mut Expr) -> Option<Expr>
    {
        for c in self.children_mut() {
            c.transform(func);
        }
        if let Some(e) = func(self) {
            *self = e;
        }
    }

    /// Returns true if this expressions contains `other`.
    pub fn contains(&self, other: &Expr) -> bool {
        if *self == *other {
            return true;
        }
        for c in other.children() {
            if self.contains(c) {
                return true;
            }
        }
        return false;
    }
}

/// Create a box containing an untyped expression of the given kind.
pub fn expr_box(kind: ExprKind, annot: Annotations) -> Box<Expr> {
    Box::new(Expr {
                 ty: Type::Unknown,
                 kind: kind,
                 annotations: annot,
             })
}

#[test]
fn compare_expressions() {
    let e1 = parse_expr("for([1,2], appender, |e| e+1)").unwrap();
    let e2 = parse_expr("for([1,2], appender, |f| f+1)").unwrap();
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    let e1 = parse_expr("let a = 2; a").unwrap();
    let e2 = parse_expr("let b = 2; b").unwrap();
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    let e2 = parse_expr("let b = 2; c").unwrap();
    assert!(!e1.compare_ignoring_symbols(&e2).unwrap());

    // Undefined symbols should work as long as the symbol is the same.
    let e1 = parse_expr("[1, 2, 3, d]").unwrap();
    let e2 = parse_expr("[1, 2, 3, d]").unwrap();
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    let e2 = parse_expr("[1, 2, 3, e]").unwrap();
    assert!(!e1.compare_ignoring_symbols(&e2).unwrap());

    // Symbols can be substituted, so equal.
    let e1 = parse_expr("|a, b| a + b").unwrap();
    let e2 = parse_expr("|c, d| c + d").unwrap();
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Symbols don't match up.
    let e2 = parse_expr("|c, d| d + c").unwrap();
    assert!(!e1.compare_ignoring_symbols(&e2).unwrap());
}
