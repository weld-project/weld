//! Defines types in the Weld IR.

use super::annotations::Annotations;
use super::ast::BinOpKind;
use super::error::*;

use self::Type::*;
use self::ScalarKind::*;
use self::BuilderKind::*;

use std::fmt;
use std::vec;

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
    /// An unknown type, used only for type inference.
    Unknown,
}

impl Type {
    /// Returns the child types of this `Type`.
    pub fn children(&self) -> vec::IntoIter<&Type> {
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
        match *self {
            Simd(_) | Builder(_, _) => true,
            Struct(ref fields) => fields.iter().all(|f| f.is_simd()),
            _ => false
        }
    }

    pub fn is_scalar(&self) -> bool {
        match *self {
            Scalar(_) => true,
            _ => false
        }
    }

    /// Return the vectorized version of a type.
    ///
    /// This method returns an error if this `Type` is not vectorizable.
    pub fn simd_type(&self) -> WeldResult<Type> {
        match *self {
            Scalar(kind) => Ok(Simd(kind)),
            Builder(_, _) => Ok(self.clone()),
            Struct(ref fields) => {
                let result: WeldResult<_> = fields.iter().map(|f| f.simd_type()).collect();
                Ok(Struct(result?))
            }
            _ => compile_err!("simd_type called on non-SIMD {:?}", self)
        }
    }

    /// Return the scalar version of a type.
    ///
    /// This method returns an error if this `Type` is not scalarizable.
    pub fn scalar_type(&self) -> WeldResult<Type> {
        match *self {
            Simd(kind) => Ok(Scalar(kind)),
            Builder(_, _) => Ok(self.clone()),
            Struct(ref fields) => {
                let result: WeldResult<_> = fields.iter().map(|f| f.scalar_type()).collect();
                Ok(Struct(result?))
            }
            _ => compile_err!("scalar_type called on non-SIMD {:?}", self)
        }
    }

    /// Returns the type merged into a builder.
    ///
    /// Returns an error if this `Type` is not a builder type.
    pub fn merge_type(&self) -> WeldResult<Type> {
        if let Builder(ref kind, _) = *self {
            Ok(kind.merge_type())

        } else {
            compile_err!("merge_type called on non-builder type {:?}", self)
        }
    }

    /// Returns whether this `Type` is partial.
    ///
    /// A type is partial if it or any of its subtypes is `Unknown`.
    pub fn partial_type(&self) -> bool {
        match *self {
            Unknown => false,
            _ => self.children().any(|t| t.partial_type()),
        }
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
        match *self {
            Appender(ref elem) => Vector(elem.clone()),
            Merger(ref elem, _) => *elem.clone(),
            DictMerger(ref key, ref value, _) => Dict(key.clone(), value.clone()),
            GroupMerger(ref key, ref value) => Dict(key.clone(), Box::new(Vector(value.clone()))),
            VecMerger(ref elem, _) => Vector(elem.clone()),
        }
    }
}
