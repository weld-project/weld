use std::fmt;

use std::collections::BTreeMap;

/// A kind of annotation that can be set on an expression.
#[derive(Clone, Debug, Ord, PartialOrd, PartialEq, Eq, Hash)]
pub enum AnnotationKind {
    BuilderImplementation,
    Predicate,
    Vectorize,
    TileSize,
    GrainSize,
    AlwaysUseRuntime,
    Size,
    BranchSelectivity,
    NumKeys,
}

impl fmt::Display for AnnotationKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}",
               match *self {
                   AnnotationKind::BuilderImplementation => "impl",
                   AnnotationKind::TileSize => "tile_size",
                   AnnotationKind::GrainSize => "grain_size",
                   AnnotationKind::Size => "size",
                   AnnotationKind::BranchSelectivity => "branch_selectivity",
                   AnnotationKind::NumKeys => "num_keys",
                   AnnotationKind::Predicate => "predicate",
                   AnnotationKind::Vectorize => "vectorize",
                   AnnotationKind::AlwaysUseRuntime => "always_use_runtime",
               })
    }
}

#[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
/// A annotation value for the way a builder is implemented.
pub enum BuilderImplementationKind {
    Local,
    Global,
}

impl fmt::Display for BuilderImplementationKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use annotations::BuilderImplementationKind::*;
        let text = match *self {
            Local => "local",
            Global => "global",
        };
        f.write_str(text)
    }
}

/// An internal representation of annotation values.
#[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
enum AnnotationValue {
    VBuilderImplementation(BuilderImplementationKind),
    VTileSize(i32),
    VGrainSize(i32),
    VSize(i64),
    VNumKeys(i64),
    VBranchSelectivity(i32), // Fractions of 10,000
    VPredicate,
    VVectorize,
    VAlwaysUseRuntime,
}

impl fmt::Display for AnnotationValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
       write!(f, "{}",
              match *self {
                   AnnotationValue::VBuilderImplementation(ref kind) => format!("{}", kind),
                   AnnotationValue::VTileSize(ref v) => format!("{}", v),
                   AnnotationValue::VGrainSize(ref v) => format!("{}", v),
                   AnnotationValue::VSize(ref v) => format!("{}", v),
                   AnnotationValue::VBranchSelectivity(ref v) => format!("{}", v),
                   AnnotationValue::VNumKeys(ref v) => format!("{}", v),
                   // These are flags, so their existence indicates that the value is `true`.
                   AnnotationValue::VPredicate => "true".to_string(),
                   AnnotationValue::VVectorize => "true".to_string(),
                   AnnotationValue::VAlwaysUseRuntime => "true".to_string(),
               })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Annotations {
    values: BTreeMap<AnnotationKind, AnnotationValue>,
}

impl Annotations {
    pub fn new() -> Annotations {
        return Annotations {
            values: BTreeMap::new(),
        };
    }

    pub fn builder_implementation(&self) -> Option<BuilderImplementationKind> {
        if let Some(s) = self.values.get(&AnnotationKind::BuilderImplementation) {
            if let AnnotationValue::VBuilderImplementation(ref s) = *s {
                return Some(s.clone());
            }
        }
        None
    }

    pub fn set_builder_implementation(&mut self, value: BuilderImplementationKind) {
        self.values.insert(AnnotationKind::BuilderImplementation, AnnotationValue::VBuilderImplementation(value));
    }

    pub fn tile_size(&self) -> Option<i32> {
        if let Some(s) = self.values.get(&AnnotationKind::TileSize) {
            if let AnnotationValue::VTileSize(ref s) = *s {
                return Some(s.clone());
            }
        }
        None
    }

    pub fn set_tile_size(&mut self, value: i32) {
        self.values.insert(AnnotationKind::TileSize, AnnotationValue::VTileSize(value));
    }

    pub fn grain_size(&self) -> Option<i32> {
        if let Some(s) = self.values.get(&AnnotationKind::GrainSize) {
            if let AnnotationValue::VGrainSize(ref s) = *s {
                return Some(s.clone());
            }
        }
        None
    }

    pub fn set_grain_size(&mut self, value: i32) {
        self.values.insert(AnnotationKind::GrainSize, AnnotationValue::VGrainSize(value));
    }

    pub fn size(&self) -> Option<i64> {
        if let Some(s) = self.values.get(&AnnotationKind::Size) {
            if let AnnotationValue::VSize(ref s) = *s {
                return Some(s.clone());
            }
        }
        None
    }

    pub fn set_size(&mut self, value: i64) {
        self.values.insert(AnnotationKind::Size, AnnotationValue::VSize(value));
    }

    pub fn num_keys(&self) -> Option<i64> {
        if let Some(s) = self.values.get(&AnnotationKind::NumKeys) {
            if let AnnotationValue::VNumKeys(ref s) = *s {
                return Some(s.clone());
            }
        }
        None
    }

    pub fn set_num_keys(&mut self, value: i64) {
        self.values.insert(AnnotationKind::NumKeys, AnnotationValue::VNumKeys(value));
    }

    pub fn branch_selectivity(&self) -> Option<i32> {
        if let Some(s) = self.values.get(&AnnotationKind::BranchSelectivity) {
            if let AnnotationValue::VBranchSelectivity(ref s) = *s {
                return Some(s.clone());
            }
        }
        None
    }

    pub fn set_branch_selectivity(&mut self, value: i32) {
        self.values.insert(AnnotationKind::BranchSelectivity, AnnotationValue::VBranchSelectivity(value));
    }

    pub fn predicate(&self) -> bool {
        return self.values.contains_key(&AnnotationKind::Predicate);
    }

    pub fn set_predicate(&mut self, val: bool) {
        if val {
            self.values.insert(AnnotationKind::Predicate, AnnotationValue::VPredicate);
        } else {
            self.values.remove(&AnnotationKind::Predicate);
        }
    }

    pub fn vectorize(&self) -> bool {
        return self.values.contains_key(&AnnotationKind::Vectorize);
    }

    pub fn set_vectorize(&mut self, val: bool) {
        if val {
            self.values.insert(AnnotationKind::Vectorize, AnnotationValue::VVectorize);
        } else {
            self.values.remove(&AnnotationKind::Vectorize);
        }
    }

    pub fn always_use_runtime(&self) -> bool {
        return self.values.contains_key(&AnnotationKind::AlwaysUseRuntime);
    }

    pub fn set_always_use_runtime(&mut self, val: bool) {
        if val {
            self.values.insert(AnnotationKind::AlwaysUseRuntime, AnnotationValue::VAlwaysUseRuntime);
        } else {
            self.values.remove(&AnnotationKind::AlwaysUseRuntime);
        }
    }

    pub fn is_empty(&self) -> bool {
        return self.values.is_empty();
    }
}

impl fmt::Display for Annotations {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut annotations = Vec::new();
        for (ref kind, ref value) in self.values.iter() {
            annotations.push(format!("{}:{}", kind, value));
        }

        // Sort the annotations alphabetically when displaying them so the result is deterministic.
        annotations.sort();

        if annotations.len() == 0 {
            write!(f, "")
        } else {
            write!(f, "@({})", annotations.join(","))
        }
    }
}
