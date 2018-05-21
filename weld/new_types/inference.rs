//! Type inference for Weld expressions.

extern crate fnv;

use super::ast::Symbol;
use super::error::*;

use fnv::FnvHashMap;

type TypeMap = FnvHashMap<Symbol, Type>;

pub trait InferTypes {
    /// Checks and infers types in place.
    ///
    /// Returns an error if types are inconsistent.
    fn infer_types(&mut self) -> WeldResult<()>;
}

impl InferTypes for Expr {
    fn infer_types(&mut self) -> WeldResult<()> {
        self.infer_types_internal()
    }
}

/// A module-internal implementation of type inference.
///
/// This trait contains additional helper methods that are not exposed outside this module.
trait InferTypesInternal {
    fn infer_types_internal(&mut self) -> WeldResult<()>
    fn infer_up(&mut self, &mut TypeMap) -> WeldResult<()>;
    fn assign_type(&mut self, &TypeMap) -> WeldResult<()>; 
}


impl InferTypesInternal for Expr {
    fn infer_types_internal(&mut self) -> WeldResult<()> {
        Ok(())
    }

    fn infer_up(&mut self, &mut TypeMap) -> WeldResult<()> {
        Ok(())
    }

    fn assign_type(&mut self, &TypeMap) -> WeldResult<()> {
        Ok(())
    }
}

