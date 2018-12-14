//! Defines the Weld abstract syntax tree.

use error::*;
use util;

use self::ExprKind::*;
use self::ScalarKind::*;
use self::BinOpKind::*;

use std::rc::Rc;
use std::fmt;
use std::vec;
use std::collections::BTreeMap;

/// Name used for placeholder expressions.
const PLACEHOLDER_NAME: &'static str = "#placeholder";

/// An annotation over a type or expression.
///
/// Annotations are unstructured String key-value pairs. They can be added on expressions and
/// interpreted in different ways by the compiler.
///
/// ## Limitations
///
/// Currently, the parser is only capable of parsing annotation values that are either identifiers (i.e.,
/// single-token strings), boolean literals, floating point literals, and signed integer literals.
/// Keys must be identifiers (i.e., non-numeric, non-boolean strings that are not reserved words).
///
/// The annotation system should in theory support arbitrary string key/value pairs: the parser
/// will eventually be updated to support this.
#[derive(Clone,Debug,PartialEq,Eq,Hash)]
pub struct Annotations {
    /// Holds the annotations.
    ///
    /// This is wrapped in an option so we don't need to allocate memory in cases where the HashMap
    /// is empty (which will usually be the case).
    values: Option<BTreeMap<String, String>>,
}

impl Annotations {
    /// Create a new set of empty annotations.
    pub fn new() -> Annotations {
        Annotations {
            values: None,
        }
    }

    /// Set an annotation with key associated with value.
    ///
    /// The previous value for this key is returned if there was one. 
    pub fn set<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) -> Option<String> {
        if self.values.is_none() {
            self.values = Some(BTreeMap::new());
        }
        self.values
            .as_mut()
            .unwrap()
            .insert(key.into(), value.into())
    }

    /// Get the annotation value associated with a key.
    ///
    /// Returns `None` if the key was not found.
    pub fn get<K: AsRef<str>>(&self, key: K) -> Option<&str> {
        if self.values.is_none() {
            return None
        }
        self.values
            .as_ref()
            .unwrap()
            .get(key.as_ref())
            .map(|v| v.as_ref())
    }

    /// Return whether the annotations are empty.
    pub fn is_empty(&self) -> bool {
        self.values.as_ref().map(|f| f.len()).unwrap_or(0) == 0
    }

    /// Clears the annotations.
    pub fn clear(&mut self) {
        self.values = None;
    }
}

impl fmt::Display for Annotations {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.values.is_none() {
            return write!(f, "");
        }

        // Annotations are always sorted alphabetically (due to BTreeMap's iter).
        let annotations = self.values.as_ref().unwrap()
            .iter()
            .map(|(k, v)| format!("{}:{}", k, v))
            .collect::<Vec<String>>()
            .join(",");

        write!(f, "@({})", annotations)
    }
}

/// Types in the Weld IR.
#[derive(Clone,Debug,PartialEq,Eq,Hash)]
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

    /// Returns the type of a string.
    pub fn string_type() -> Type {
        Type::Vector(Box::new(Type::Scalar(ScalarKind::I8)))
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


    /// Returns whether this `Type` contains a builder.
    pub fn contains_builder(&self) -> bool {
        use self::Type::Builder;
        match *self {
            Builder(_, _) => true,
            _ => self.children().any(|t| t.contains_builder()),
        }
    }

    /// Returns whether this `Type` is a builder.
    pub fn is_builder(&self) -> bool {
        use self::Type::{Builder, Struct};
        match *self {
            Builder(_, _) => true,
            Struct(ref tys) => tys.iter().all(|t| t.is_builder()),
            _ => false
        }
    }

    /// Returns whether this `Type` is a hashable.
    pub fn is_hashable(&self) -> bool {
        use self::Type::*;
        match *self {
            Scalar(_) => true,
            // XXX Is this hashable...?
            Simd(_) => true,
            Struct(ref tys) => tys.iter().all(|t| t.is_hashable()),
            Vector(ref elem) => elem.is_hashable(),
            Builder(_, _) => false,
            Dict(_, _) => false,
            Function(_, _) | Unknown => false
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

    /// Returns whether this scalar is signed.
    pub fn is_signed(&self) -> bool {
        self.is_signed_integer() || self.is_float()
    }

    /// Returns whether this scalar is an integer.
    ///
    /// Booleans are not considered to be integers.
    pub fn is_integer(&self) -> bool {
        self.is_signed_integer() || self.is_unsigned_integer()
    }

    /// Returns whether the scalar is a numeric.
    pub fn is_numeric(&self) -> bool {
        self.is_integer() || self.is_float()
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

    /// Returns whether this type is smaller in bits than `target`.
    pub fn is_upcast(&self, target: &ScalarKind) -> bool {
        target.bits() >= self.bits()
    }

    /// Returns whether this type is strictly smaller in bits than `target`.
    pub fn is_strict_upcast(&self, target: &ScalarKind) -> bool {
        target.bits() > self.bits()
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
/// Builder types in the Weld IR.
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
    name: Rc<String>,
    id: i32,
}

impl Symbol {
    pub fn new<T: Into<String>>(name: T, id: i32) -> Symbol {
        Symbol {
            name: Rc::new(name.into()),
            id: id,
        }
    }

    pub fn placeholder() -> Symbol {
        Symbol::new(PLACEHOLDER_NAME, 0)
    }

    pub fn name(&self) -> String {
        self.name.to_string()
    }

    pub fn id(&self) -> i32 {
        self.id
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

/// A typed Weld expression tree.
#[derive(Clone,Debug,PartialEq,Eq,Hash)]
pub struct Expr {
    pub ty: Type,
    pub kind: ExprKind,
    pub annotations: Annotations,
}

/// Iterator kinds in the Weld IR.
///
/// An iterator defines how a for loop iterates over data.
#[derive(Clone,Debug,Eq,PartialEq,Hash)]
pub enum IterKind {
    /// A standard scalar iterator.
    ScalarIter,
    /// A SIMD iterator.
    ///
    /// This iterator fetches multiple elements packed into a SIMD value per loop iteration.
    SimdIter,
    /// A fringe iterator.
    ///
    /// This iterator handles the fringe of a SIMD iterator (i.e., if the number of elements in the
    /// source is not a multiple of the fixed SIMD width). The elements are processed using scalar
    /// code.
    FringeIter,
    /// An interator over a N-dimensional tensor.
    NdIter,
    /// An interator over a finite integer range.
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
#[derive(Clone,Debug,PartialEq,Eq,Hash)]
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
    /// Returns true if this is a simple iterator.
    ///
    /// An iterator is simple if it has no start/stride/end specified (i.e., it iterates over all
    /// the input data) and kind `ScalarIter`.
    pub fn is_simple(&self) -> bool {
        return self.start.is_none() && self.end.is_none() && self.stride.is_none() &&
            self.kind == IterKind::ScalarIter;
    }
}

/// Expressions in the Weld IR.
///
/// This enumeration defines the operators in the Weld IR. Each operator relies on zero or more
/// sub-expressions, forming an expression tree. We use the term "expression" to refer to a
/// particular `ExprKind`.
#[derive(Clone,Debug,PartialEq,Hash,Eq)]
pub enum ExprKind {
    /// A literal expression.
    ///
    /// Weld supports numerical scalar literals (e.g., 1.0 and -5) and ASCII string literals.
    Literal(LiteralKind),
    /// An identifier.
    Ident(Symbol),
    /// Invert a boolean expression.
    Not(Box<Expr>),
    /// Negates a numerical expression.
    Negate(Box<Expr>),
    /// Broadcasts a scalar into a vector.
    Broadcast(Box<Expr>),
    /// Applies a binary operator to the child expressions.
    BinOp {
        kind: BinOpKind,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Applies a unary operator to the child expressions.
    UnaryOp {
        kind: UnaryOpKind,
        value: Box<Expr>,
    },
    /// Cast a scalar or SIMD child expression to another type.
    Cast {
        kind: ScalarKind,
        child_expr: Box<Expr>,
    },
    /// Convert a dictionary into a vector of key/value pairs.
    ToVec { child_expr: Box<Expr> },
    /// Construct a struct from a list of child expressions.
    MakeStruct { elems: Vec<Expr> },
    /// Construct a vector from a list of child expressions.
    MakeVector { elems: Vec<Expr> },
    /// Zip vectors together for iteration.
    ///
    /// This operator is only available within a For loop.
    Zip { vectors: Vec<Expr> },
    /// Access a struct field at the given index.
    GetField { expr: Box<Expr>, index: u32 },
    /// Get the length of a vector as an `i64`.
    Length { data: Box<Expr> },
    /// Lookup a value in a collection.
    ///
    /// If `data` is a vector, `index` must be an `i64` specifying the vector index. If `data` is a
    /// dictionary, `index` is a key. If the key is not present in the dictionary, this expression
    /// raises a `KeyNotFoundError`.
    Lookup {
        data: Box<Expr>,
        index: Box<Expr>,
    },
    /// A variant of lookup on dictionaries that does not result in an error.
    ///
    /// Returns a `{bool, V}`, where the `bool` indicates whether the value was in the dictionary.
    OptLookup {
        data: Box<Expr>,
        index: Box<Expr>,
    },
    /// Check whether a key exists in a dictionary.
    KeyExists {
        data: Box<Expr>,
        key: Box<Expr>,
    },
    /// Slices a vector, creating a view into it.
    ///
    /// This does not allocate new data.
    Slice {
        data: Box<Expr>,
        index: Box<Expr>,
        size: Box<Expr>,
    },
    /// Sorts a vector.
    /// 
    /// The sort operator takes a vector comprised of any non-builder, non-SIMD, or non-dictionary type
    /// and returns a new sorted vector. The sort order is determined by `cmpfunc`.
    ///
    /// The comparator takes two arguments `x` and `y` whose type is the vector element type and
    /// returns a positive `i32` if `x > y`, zero if `x == y`, and a negative number of `x < y`.
    Sort {
        data: Box<Expr>,
        cmpfunc: Box<Expr>,
    },
    /// Assign a `value` to `name`, and then evaluate `body`.
    ///
    /// The environment is updated with `name` before `body` is run.
    Let {
        name: Symbol,
        value: Box<Expr>,
        body: Box<Expr>,
    },
    /// Evaluate `on_true` or `on_false` depending on `cond`.
    If {
        cond: Box<Expr>,
        on_true: Box<Expr>,
        on_false: Box<Expr>,
    },
    /// Iterate sequentially using an update function.
    ///
    /// The initial value has type `T`, and the update function has type `|T| -> {T, bool}`.
    /// Iteration continues until the update function returns `false`.
    Iterate {
        initial: Box<Expr>,
        update_func: Box<Expr>,
    },
    /// Select `on_true` or `on_false` depending on `cond`.
    ///
    /// Both `on_true`and `on_false` are evaluated unconditionally.
    Select {
        cond: Box<Expr>,
        on_true: Box<Expr>,
        on_false: Box<Expr>,
    },
    /// An expression representing a function.
    Lambda {
        params: Vec<Parameter>,
        body: Box<Expr>,
    },
    /// Apply a function using a list of parameters.
    Apply {
        func: Box<Expr>,
        params: Vec<Expr>,
    },
    /// A C UDF called by symbol name.
    CUDF {
        sym_name: String,
        args: Vec<Expr>,
        return_ty: Box<Type>,
    },
    /// Serialize an expression into a vector of bytes.
    Serialize(Box<Expr>),
    /// Deserialize an expression from a vector of bytes.
    ///
    /// The expression should be serialized using `Serialize`.
    Deserialize {
        value: Box<Expr>,
        value_ty: Box<Type>,
    },
    /// Create a new builder.
    NewBuilder(Option<Box<Expr>>),
    /// Update a builder in parallel by iterating over data.
    For {
        iters: Vec<Iter>,
        builder: Box<Expr>,
        func: Box<Expr>,
    },
    /// Update a builder value, returning a new builder.
    Merge {
        builder: Box<Expr>,
        value: Box<Expr>,
    },
    /// Consume a builder and return its result.
    Res { builder: Box<Expr> },
}

impl ExprKind {
    /// Return a readable name for the kind.
    ///
    /// The name is independent of any subexpressions.
    pub fn name(&self) -> &str {
        match *self {
            Literal(_) => "Literal",
            Ident(_) => "Ident",
            Not(_) => "Not",
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
            OptLookup { .. } => "OptLookup",
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

    /// Returns `true` if the expression is over builders.
    pub fn is_builder_expr(&self) -> bool {
        match *self {
            Merge { .. } | Res { .. } | For { .. } | NewBuilder(_) => true,
            _ => false,
        }
    }
}

/// Literal types in the Weld IR.
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

/// Binary operators over numerical values in the Weld IR.
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

/// Unary operators over numerical values in the Weld IR.
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

/// A Parameter in a Lambda.
///
/// A parameter is a typed `Symbol`.
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
            OptLookup {
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
                ref cmpfunc,
            } => vec![data.as_ref(), cmpfunc.as_ref()],
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
            Not(ref t) => vec![t.as_ref()],
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
            OptLookup {
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
                ref mut cmpfunc,
            } => vec![data.as_mut(), cmpfunc.as_mut()],
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
            Not(ref mut t) => vec![t.as_mut()],
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
            c.transform_up(func);
        }
        if let Some(e) = func(self) {
            *self = e;
        }
    }

    /// Transform an expression by replacing its kind with another `ExprKind`.
    ///
    /// The type of the expression is unmodified.
    pub fn transform_kind<F>(&mut self, func: &mut F)
        where F: FnMut(&mut Expr) -> Option<ExprKind>,
    {
        if let Some(k) = func(self) {
            self.kind = k;
            return self.transform_kind(func);
        }
        for c in self.children_mut() {
            c.transform_kind(func);
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

/// Creates a placeholder expression.
pub trait Placeholder {
    /// Returns whether this expression is a placeholder.
    fn is_placeholder(&self) -> bool;

    /// Create a new placeholder.
    fn new_placeholder() -> Self;
}

impl Placeholder for Expr {
    fn is_placeholder(&self) -> bool {
        if let Ident(ref name) = self.kind {
            return name.name.as_ref() == PLACEHOLDER_NAME
        }
        return false
    }

    fn new_placeholder() -> Expr {
        Expr {
            ty: Type::Unknown,
            kind: Ident(Symbol::placeholder()),
            annotations: Annotations::new()
        }
    }
}

impl Placeholder for Box<Expr> {
    fn is_placeholder(&self) -> bool {
        self.as_ref().is_placeholder()
    }

    fn new_placeholder() -> Box<Expr> {
        Box::new(Expr {
            ty: Type::Unknown,
            kind: Ident(Symbol::placeholder()),
            annotations: Annotations::new()
        })
    }
}

/// Takes an expression, replacing it with a placeholder.
pub trait Takeable: Placeholder {
    fn take(&mut self) -> Self;
}

impl Takeable for Expr {
    fn take(&mut self) -> Expr {
        use std::mem;
        let mut new = Self::new_placeholder();
        new.ty = self.ty.clone();
        mem::swap(self, &mut new);
        new
    }
}

impl Takeable for Box<Expr> {
    fn take(&mut self) -> Box<Expr> {
        use std::mem;
        let mut new = Self::new_placeholder();
        new.ty = self.ty.clone();
        mem::swap(self.as_mut(), new.as_mut());
        new
    }
}

