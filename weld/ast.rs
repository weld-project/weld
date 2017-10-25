//! Abstract syntax tree for Weld.

use std::vec;
use std::fmt;

use super::error::*;

/// A symbol (identifier name); for now these are strings, but we may add some kind of scope ID.
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
            write!(f, "{}#{}", self.name, self.id)
        }
    }
}

/// A data type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Scalar(ScalarKind),
    Simd(ScalarKind),
    Vector(Box<Type>),
    Dict(Box<Type>, Box<Type>),
    Builder(BuilderKind, Annotations),
    Struct(Vec<Type>),
    Function(Vec<Type>, Box<Type>),
}

impl Type {
    /// Return true if a given type is SIMD vectorized, which concretely means that it is
    /// either a simd[T], a builder, or a struct of SIMD types.
    ///
    /// TODO: we should investigate just using simd[struct[T]] for structs and turning it
    /// to the nested form at runtime.
    pub fn is_simd(&self) -> bool {
        use self::Type::*;
        match *self {
            Simd(_) => true,
            Builder(_, _) => true,
            Struct(ref fields) => fields.iter().all(|f| f.is_simd()),
            _ => false
        }
    }

    pub fn is_scalar(&self) -> bool {
        use self::Type::*;
        match *self {
            Scalar(_) => true,
            _ => false
        }
    }

    /// Get the vectorized version of a type, if it is vectorizable.
    pub fn simd_type(&self) -> WeldResult<Type> {
        use self::Type::*;
        match *self {
            Scalar(kind) => Ok(Simd(kind)),

            Builder(_, _) => Ok(self.clone()),

            Struct(ref fields) => {
                let mut new_fields = vec![];
                for f in fields {
                    new_fields.push(f.simd_type()?)
                }
                Ok(Struct(new_fields))
            }

            _ => weld_err!("simd_type called on non-SIMD {:?}", self)
        }
    }

    /// Get the scalar version of a type if it `is_simd`.
    pub fn scalar_type(&self) -> WeldResult<Type> {
        use self::Type::*;
        match *self {
            Simd(kind) => Ok(Scalar(kind)),

            Builder(_, _) => Ok(self.clone()),

            Struct(ref fields) => {
                let mut new_fields = vec![];
                for f in fields {
                    new_fields.push(f.scalar_type()?)
                }
                Ok(Struct(new_fields))
            }

            _ => weld_err!("scalar_type called on non-SIMD {:?}", self)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    /// Is this scalar a floating point type (F32 or F64)?
    pub fn is_float(&self) -> bool {
        use ast::ScalarKind::*;
        match *self {
            F32 | F64 => true,
            _ => false
        }
    }

    /// Is this scalar a Bool?
    pub fn is_bool(&self) -> bool {
        use ast::ScalarKind::*;
        match *self {
            Bool => true,
            _ => false
        }
    }

    /// Is this type a signed integer of any type (excludes Bool)?
    pub fn is_signed_integer(&self) -> bool {
        use ast::ScalarKind::*;
        match *self {
            I8 | I16 | I32 | I64 => true,
            _ => false
        }
    }

    /// Is this type a signed integer of any type (excludes Bool)?
    pub fn is_unsigned_integer(&self) -> bool {
        use ast::ScalarKind::*;
        match *self {
            U8 | U16 | U32 | U64 => true,
            _ => false
        }
    }

    /// Is this type a signed or unsigned integer of any type (excludes Bool)?
    pub fn is_integer(&self) -> bool {
        return self.is_signed_integer() || self.is_unsigned_integer();
    }

    /// Return the length of this scalar type in bits
    pub fn bits(&self) -> u32 {
        use ast::ScalarKind::*;
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
        use ast::ScalarKind::*;
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
pub enum BuilderImplementationKind {
    Local,
    Global,
}

impl fmt::Display for BuilderImplementationKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ast::BuilderImplementationKind::*;
        let text = match *self {
            Local => "local",
            Global => "global",
        };
        f.write_str(text)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Annotations {
    builder_implementation: Option<BuilderImplementationKind>,

    predicate: Option<bool>,
    vectorize: Option<bool>,
    tile_size: Option<i32>,
    grain_size: Option<i32>,
    always_use_runtime: Option<bool>,

    size: Option<i64>,
    branch_selectivity: Option<i32>, // Fractions of 10,000.
    num_keys: Option<i64>, /* If additional fields are added, remember to add a getter / setter
                            * as well as necessary logic in fmt::Display and is_empty below. */
}

impl Annotations {
    pub fn new() -> Annotations {
        return Annotations {
            builder_implementation: None,
            predicate: None,
            vectorize: None,
            tile_size: None,
            grain_size: None,
            always_use_runtime: None,
            size: None,
            branch_selectivity: None,
            num_keys: None,
        };
    }

    pub fn builder_implementation(&self) -> &Option<BuilderImplementationKind> {
        &self.builder_implementation
    }

    pub fn set_builder_implementation(&mut self,
                                      builder_implementation: BuilderImplementationKind) {
        self.builder_implementation = Some(builder_implementation);
    }

    pub fn predicate(&self) -> &Option<bool> {
        &self.predicate
    }

    pub fn set_predicate(&mut self, predicate: bool) {
        self.predicate = Some(predicate)
    }

    pub fn vectorize(&self) -> &Option<bool> {
        &self.vectorize
    }

    pub fn set_vectorize(&mut self, vectorize: bool) {
        self.vectorize = Some(vectorize)
    }

    pub fn always_use_runtime(&self) -> &Option<bool> {
        &self.always_use_runtime
    }

    pub fn set_always_use_runtime(&mut self, always_use_runtime: bool) {
        self.always_use_runtime = Some(always_use_runtime)
    }

    pub fn tile_size(&self) -> &Option<i32> {
        &self.tile_size
    }

    pub fn set_tile_size(&mut self, tile_size: i32) {
        self.tile_size = Some(tile_size)
    }

    pub fn grain_size(&self) -> &Option<i32> {
        &self.grain_size
    }

    pub fn set_grain_size(&mut self, grain_size: i32) {
        self.grain_size = Some(grain_size)
    }

    pub fn size(&self) -> &Option<i64> {
        &self.size
    }

    pub fn set_size(&mut self, size: i64) {
        self.size = Some(size)
    }

    pub fn selectivity(&self) -> &Option<i32> {
        &self.branch_selectivity
    }

    pub fn set_selectivity(&mut self, selectivity: i32) {
        self.branch_selectivity = Some(selectivity)
    }

    pub fn num_keys(&self) -> &Option<i64> {
        &self.num_keys
    }

    pub fn set_num_keys(&mut self, num_keys: i64) {
        self.num_keys = Some(num_keys)
    }

    pub fn is_empty(&self) -> bool {
        let mut is_empty = true;
        if let Some(_) = self.builder_implementation {
            is_empty = false;
        }
        if let Some(_) = self.predicate {
            is_empty = false;
        }
        if let Some(_) = self.vectorize {
            is_empty = false;
        }
        if let Some(_) = self.tile_size {
            is_empty = false;
        }
        if let Some(_) = self.grain_size {
            is_empty = false;
        }
        if let Some(_) = self.size {
            is_empty = false;
        }
        if let Some(_) = self.branch_selectivity {
            is_empty = false;
        }
        if let Some(_) = self.num_keys {
            is_empty = false;
        }
        is_empty
    }
}

impl fmt::Display for Annotations {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut annotations = Vec::new();
        if let Some(ref e) = self.builder_implementation {
            annotations.push(format!("impl:{}", e));
        }
        if let Some(ref e) = self.predicate {
            annotations.push(format!("predicate:{}", e));
        }
        if let Some(ref e) = self.vectorize {
            annotations.push(format!("vectorize:{}", e));
        }
        if let Some(ref e) = self.always_use_runtime {
            annotations.push(format!("always_use_runtime:{}", e));
        }
        if let Some(ref e) = self.tile_size {
            annotations.push(format!("tile_size:{}", e));
        }
        if let Some(ref e) = self.grain_size {
            annotations.push(format!("grain_size:{}", e));
        }
        if let Some(ref e) = self.size {
            annotations.push(format!("size:{}", e));
        }
        if let Some(ref e) = self.branch_selectivity {
            annotations.push(format!("selectivity:{}", e));
        }
        if let Some(ref e) = self.num_keys {
            annotations.push(format!("num_keys:{}", e));
        }

        if annotations.len() == 0 {
            return write!(f, "");
        }
        return write!(f, "@({})", annotations.join(","));
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BuilderKind {
    Appender(Box<Type>),
    Merger(Box<Type>, BinOpKind),
    /// key_type, value_type, binop
    DictMerger(Box<Type>, Box<Type>, BinOpKind),
    GroupMerger(Box<Type>, Box<Type>),
    /// elem_type, binop
    VecMerger(Box<Type>, BinOpKind),
}

pub trait TypeBounds: Clone + PartialEq {}

impl TypeBounds for Type {}

/// An expression tree, having type annotations of type T. We make this parametrized because
/// expressions have different "kinds" of types attached to them at different points in the
/// compilation process -- namely PartialType when parsed and then Type after type inference.
#[derive(Clone, Debug, PartialEq)]
pub struct Expr<T: TypeBounds> {
    pub ty: T,
    pub kind: ExprKind<T>,
    pub annotations: Annotations,
}

/// An iterator kind, which specifies how data should be loaded and passed to a `For` loop.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum IterKind {
    ScalarIter, // A standard scalar iterator.
    SimdIter, // A vector iterator.
    FringeIter, // A fringe iterator, handling the fringe of a vector iter.
}

impl fmt::Display for IterKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ast::IterKind::*;
        let text = match *self {
            ScalarIter => "scalar",
            SimdIter => "vectorized",
            FringeIter => "fringe",
        };
        f.write_str(text)
    }
}

/// An iterator, which specifies a vector to iterate over and optionally a start index,
/// end index, and stride.
#[derive(Clone, Debug, PartialEq)]
pub struct Iter<T: TypeBounds> {
    pub data: Box<Expr<T>>,
    pub start: Option<Box<Expr<T>>>,
    pub end: Option<Box<Expr<T>>>,
    pub stride: Option<Box<Expr<T>>>,
    pub kind: IterKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind<T: TypeBounds> {
    // TODO: maybe all of these should take named parameters
    Literal(LiteralKind),
    Ident(Symbol),
    Negate(Box<Expr<T>>),
    // Broadcasts a scalar into a vector, e.g., 1 -> <1, 1, 1, 1>
    Broadcast(Box<Expr<T>>),
    BinOp {
        kind: BinOpKind,
        left: Box<Expr<T>>,
        right: Box<Expr<T>>,
    },
    UnaryOp {
        kind: UnaryOpKind,
        value: Box<Expr<T>>,
    },
    Cast {
        kind: ScalarKind,
        child_expr: Box<Expr<T>>,
    },
    ToVec { child_expr: Box<Expr<T>> },
    MakeStruct { elems: Vec<Expr<T>> },
    MakeVector { elems: Vec<Expr<T>> },
    Zip { vectors: Vec<Expr<T>> },
    GetField { expr: Box<Expr<T>>, index: u32 },
    Length { data: Box<Expr<T>> },
    Lookup {
        data: Box<Expr<T>>,
        index: Box<Expr<T>>,
    },
    KeyExists {
        data: Box<Expr<T>>,
        key: Box<Expr<T>>,
    },
    Slice {
        data: Box<Expr<T>>,
        index: Box<Expr<T>>,
        size: Box<Expr<T>>,
    },
    Let {
        name: Symbol,
        value: Box<Expr<T>>,
        body: Box<Expr<T>>,
    },
    If {
        cond: Box<Expr<T>>,
        on_true: Box<Expr<T>>,
        on_false: Box<Expr<T>>,
    },
    /// Sequential loop
    Iterate {
        initial: Box<Expr<T>>,
        update_func: Box<Expr<T>>,
    },
    Select {
        cond: Box<Expr<T>>,
        on_true: Box<Expr<T>>,
        on_false: Box<Expr<T>>,
    },
    Lambda {
        params: Vec<Parameter<T>>,
        body: Box<Expr<T>>,
    },
    Apply {
        func: Box<Expr<T>>,
        params: Vec<Expr<T>>,
    },
    CUDF {
        sym_name: String,
        args: Vec<Expr<T>>,
        return_ty: Box<T>,
    },
    NewBuilder(Option<Box<Expr<T>>>),
    For {
        iters: Vec<Iter<T>>,
        builder: Box<Expr<T>>,
        func: Box<Expr<T>>,
    },
    Merge {
        builder: Box<Expr<T>>,
        value: Box<Expr<T>>,
    },
    Res { builder: Box<Expr<T>> },
}

#[derive(Clone, Copy, Debug, PartialEq)]
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
    F32Literal(f32),
    F64Literal(f64),
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOpKind {
    Exp,
    Log,
    Sqrt,
    Erf,
}

impl BinOpKind {
    pub fn is_comparison(&self) -> bool {
        use ast::BinOpKind::*;
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
            Max => "max",
            Min => "min",
        };
        f.write_str(text)
    }
}

impl fmt::Display for UnaryOpKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ast::UnaryOpKind::*;
        let text = match *self {
            Exp => "exp",
            Log => "log",
            Sqrt => "sqrt",
            Erf => "erf",
        };
        f.write_str(text)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Parameter<T: TypeBounds> {
    pub name: Symbol,
    pub ty: T,
}

/// A typed expression struct.
pub type TypedExpr = Expr<Type>;

/// A typed parameter.
pub type TypedParameter = Parameter<Type>;

impl<T: TypeBounds> Expr<T> {
    /// Get an iterator for the children of this expression.
    pub fn children(&self) -> vec::IntoIter<&Expr<T>> {
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
            CUDF { ref args, .. } => args.iter().collect(),
            Negate(ref t) => vec![t.as_ref()],
            Broadcast(ref t) => vec![t.as_ref()],
            // Explicitly list types instead of doing _ => ... to remember to add new types.
            Literal(_) | Ident(_) => vec![],
        }.into_iter()
    }

    /// Get an iterator of mutable references to the children of this expression.
    pub fn children_mut(&mut self) -> vec::IntoIter<&mut Expr<T>> {
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
            CUDF { ref mut args, .. } => args.iter_mut().collect(),
            Negate(ref mut t) => vec![t.as_mut()],
            Broadcast(ref mut t) => vec![t.as_mut()],
            // Explicitly list types instead of doing _ => ... to remember to add new types.
            Literal(_) | Ident(_) => vec![],
        }.into_iter()
    }

    /// Compares two expression trees, returning true if they are the same modulo symbol names.
    /// Symbols in the two expressions must have a one to one correspondance for the trees to be
    /// considered equal. If an undefined symbol is encountered in &self during the comparison,
    /// returns an error.
    pub fn compare_ignoring_symbols(&self, other: &Expr<T>) -> WeldResult<bool> {
        use self::ExprKind::*;
        use std::collections::HashMap;
        let mut sym_map: HashMap<&Symbol, &Symbol> = HashMap::new();

        fn _compare_ignoring_symbols<'b, 'a, U: TypeBounds>(e1: &'a Expr<U>,
                                                            e2: &'b Expr<U>,
                                                            sym_map: &mut HashMap<&'a Symbol,
                                                                                  &'b Symbol>)
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
                    Ok(true)
                }
                (&Lambda { params: ref params1, .. }, &Lambda { params: ref params2, .. }) => {
                    // Just compare types, and assume the symbol names "match up".
                    if params1.len() == params2.len() &&
                       params1.iter().zip(params2).all(|t| t.0.ty == t.1.ty) {
                        for (p1, p2) in params1.iter().zip(params2) {
                            sym_map.insert(&p1.name, &p2.name);
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
                (&Merge { .. }, &Merge { .. }) => Ok(true),
                (&Res { .. }, &Res { .. }) => Ok(true),
                (&For { .. }, &For { .. }) => Ok(true), // TODO need to check Iters?
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
                (&Literal(ref l), &Literal(ref r)) if l == r => Ok(true),
                (&Ident(ref l), &Ident(ref r)) => {
                    if let Some(lv) = sym_map.get(l) {
                        Ok(**lv == *r)
                    } else {
                        weld_err!("undefined symbol {} when comparing expressions", l)
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
                let res = _compare_ignoring_symbols(&c1, &c2, sym_map);
                if res.is_err() || !res.as_ref().unwrap() {
                    return res;
                }
            }
            return Ok(true);
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
        where F: FnMut(&Expr<T>) -> ()
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
        where F: FnMut(&mut Expr<T>) -> (Option<Expr<T>>, bool)
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
        where F: FnMut(&mut Expr<T>) -> WeldResult<(Option<Expr<T>>, bool)>
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
        where F: FnMut(&mut Expr<T>) -> Option<Expr<T>>
    {
        if let Some(e) = func(self) {
            *self = e;
            return self.transform(func);
        }
        for c in self.children_mut() {
            c.transform(func);
        }
    }

    /// Returns true if this expressions contains `other`.
    pub fn contains(&self, other: &Expr<T>) -> bool {
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
