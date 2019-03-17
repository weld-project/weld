//! Structures that can be passed into Weld.
//!
//! This currently defines the type layout specified for the single threaded backend. In general,
//! type layouts vary from backend to backend especially for builders.
//!
//! # Primitives
//!
//! Primitives in Weld match their Rust counterparts, _except for booleans_. Booleans in Weld are
//! guaranteed to be one byte in size, but are defined as `_Bool` from `stdbool.h` in Rust when
//! defined in a struct with `repr(C)`.
//!
//! # Vectors
//!
//! Vectors will always have the same layout: a pointer followed by a 64-bit length.
//!
//! # Builders
//!
//! Builders are backend-specific and have layouts that may change at any time. Therefore, the
//! builder definitions here should be used as _opaque sized types_ rather than as structs whose
//! fields can be accessed.

use std::convert::AsRef;
use std::marker::PhantomData;

use std::fmt;

/// A boolean in Weld.
///
/// Weld booleans are always defined as a single-byte unsigned value. Weld will always return a
/// boolean with value 0 or 1, corresponding to `false` and `true` respectively. When passing
/// booleans as input, Weld will consider _any_ non-zero value to be `true`, and 0 to be false.
pub type WeldBool = u8;

/// A dynamically sized constant vector.
///
/// Vectors are always defined as a pointer and a length.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct WeldVec<T> {
    pub data: *const T,
    pub len: i64,
}

unsafe impl<T> Send for WeldVec<T> {}
unsafe impl<T> Sync for WeldVec<T> {}

impl<T> WeldVec<T> {
    /// Return a new WeldVec from a pointer and a length.
    ///
    /// Consider using `WeldVec::from` instead, which automatically derives the length.
    pub fn new(ptr: *const T, len: i64) -> WeldVec<T> {
        WeldVec { data: ptr, len }
    }
}

impl<'a, T, U> From<&'a U> for WeldVec<T>
where
    U: AsRef<[T]>,
{
    fn from(s: &'a U) -> WeldVec<T> {
        WeldVec::new(s.as_ref().as_ptr(), s.as_ref().len() as i64)
    }
}

impl<T> PartialEq for WeldVec<T>
where
    T: PartialEq + Clone,
{
    fn eq(&self, other: &WeldVec<T>) -> bool {
        if self.len != other.len {
            return false;
        }
        for i in 0..self.len {
            let v1 = unsafe { (*self.data.offset(i as isize)).clone() };
            let v2 = unsafe { (*other.data.offset(i as isize)).clone() };
            if v1 != v2 {
                return false;
            }
        }
        true
    }
}

impl<T> fmt::Display for WeldVec<T>
where
    T: fmt::Display + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[ ")?;
        for i in 0..self.len {
            let v = unsafe { (*self.data.offset(i as isize)).clone() };
            write!(f, "{} ", v)?;
        }
        write!(f, "] ")?;
        write!(f, "(length={})", self.len)
    }
}

/// The `appender` builder type.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct Appender<T> {
    pointer: *mut T,
    size: i64,
    capacity: i64,
}

/// The dictionary type.
///
/// Like builders, dictionaries currently have an opaque format. At some point, dictionaries will
/// have methods for accessing keys and values and iterating over them. For now, these operations
/// require compiling a Weld program.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct Dict<K, V> {
    // Dictionaries are just opaque pointers.
    pointer: *mut (),
    phantom_key: PhantomData<K>, // 0-sized
    phantom_val: PhantomData<V>, // 0-sized
}

/// The `dictmerger` builder type.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct DictMerger<K, V> {
    d: Dict<K, V>,
}

/// The `groupmerger` builder type.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct GroupMerger<K, V> {
    d: Dict<K, WeldVec<V>>,
}

// Ensures that the sizes of the types defined here match the sizes of the types in the backend.
#[test]
fn size_check() {
    use crate::ast::BinOpKind::Add;
    use crate::ast::ScalarKind::I32;
    use crate::ast::*;
    use crate::codegen::size_of;
    use std::mem;

    let i32_ty = Box::new(Type::Scalar(I32));

    let vector = &Type::Vector(i32_ty.clone());
    assert_eq!(size_of(vector), mem::size_of::<WeldVec<i32>>());

    let dict = &Type::Dict(i32_ty.clone(), i32_ty.clone());
    assert_eq!(size_of(dict), mem::size_of::<Dict<i32, i32>>());

    let appender = &Type::Builder(BuilderKind::Appender(i32_ty.clone()), Annotations::new());
    assert_eq!(size_of(appender), mem::size_of::<Appender<i32>>());

    let dictmerger = &Type::Builder(
        BuilderKind::DictMerger(i32_ty.clone(), i32_ty.clone(), Add),
        Annotations::new(),
    );
    assert_eq!(size_of(dictmerger), mem::size_of::<DictMerger<i32, i32>>());

    let groupmerger = &Type::Builder(
        BuilderKind::GroupMerger(i32_ty.clone(), i32_ty.clone()),
        Annotations::new(),
    );
    assert_eq!(
        size_of(groupmerger),
        mem::size_of::<GroupMerger<i32, i32>>()
    );
}
