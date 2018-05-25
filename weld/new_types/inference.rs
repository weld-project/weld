//! Type inference for Weld expressions.

extern crate fnv;

use new_types::*;

use fnv::FnvHashMap;

type TypeMap = FnvHashMap<Symbol, Type>;
type Binding = (Symbol, Option<Type>);

pub trait InferTypes {
    /// Checks and infers types in place.
    ///
    /// Returns an error if types are inconsistent or all types could not be inferred. That is, if
    /// the resulting value has any partial types, this method will return an error.
    fn infer_types(&mut self) -> WeldResult<()>;
}

impl InferTypes for Expr {
    /// Checks and infers types in place.
    fn infer_types(&mut self) -> WeldResult<()> {
        self.infer_types_internal()
    }
}

/// A trait for updating a type based on types around it.
///
/// This trait is implemented by `Type`.
trait PushType {
    /// Push the type of `other` into this `Type` and return if this `Type` changed.
    ///
    /// Pushing the type forces this `Type` to be at least as specific as `other (i.e., the same
    /// type with fewer Unknowns in any subtypes). The function returns an error if an incompatible
    /// type was pushed, and otherwise returns a boolean indicating whether this `Type` changed.
    fn push(&mut self, other: &Self) -> WeldResult<bool>;

    /// Sync this `Type` with `other`, calling push in both directions.
    ///
    /// This function returns an error if an incompatible type was pushed in either direction, and
    /// otherwise returns a boolean indicating whether either `Type` changed.
    fn sync(&mut self, other: &mut Self) -> WeldResult<bool> {
        Ok(self.push(other)? || other.push(self)?)
    }

    /// Sets this `Type` to be `other`.
    ///
    /// This function returns an error if `other` is incompatible with this `Type`.
    fn set(&mut self, other: Self) -> WeldResult<bool>;
}

impl PushType for Type {

    fn set(&mut self, other: Type) -> WeldResult<bool> {
        match other {
            Scalar(_) | Simd(_) if *self == Unknown => {
                *self = other;
                Ok(true)
            }
            Scalar(_) | Simd(_) if *self == other => {
                Ok(false)
            }
            Vector(ref elem) if *self == Unknown => {
                *self = Vector(elem.clone());
                Ok(true)
            }
            Vector(ref elem) => {
                if let Vector(ref mut dest) = *self {
                    dest.set(elem.as_ref().clone())
                } else {
                    compile_err!("blah")
                }
            }
            _ => {
                compile_err!("blah")
            }
        }
    }

    /// Push the type of `other` into this `Type` and return if this `Type` changed.
    fn push(&mut self, other: &Self) -> WeldResult<bool> {
        // If the other type is Unknown, we cannot infer anything. If this type is Unknown,
        // copy the other type to this one.
        if *other == Unknown {
            return Ok(false);
        } else if *self == Unknown {
            *self = other.clone();
            return Ok(true);
        }

        // Match this `Type` with `other` to make sure the variant is the same. Then, recursively
        // call `push` on any subtypes.
        match (self, other) {
            (&mut Scalar(a), &Scalar(b)) if a == b => {
                Ok(false)
            }
            (&mut Simd(a), &Simd(b)) if a == b => {
                Ok(false)
            }
            (&mut Vector(ref mut elem), &Vector(ref other_elem)) => {
                elem.push(other_elem)
            }
            (&mut Dict(ref mut key, ref mut value), &Dict(ref other_key, ref other_value)) => {
                Ok(key.push(other_key)? || value.push(other_value)?) 
            }
            (&mut Struct(ref mut types), &Struct(ref other_types)) if types.len() == other_types.len() => {
                let mut changed = false;
                for (this, other) in types.iter_mut().zip(other_types) {
                    changed |= this.push(other)?;
                }
                Ok(changed)
            }
            (&mut Function(ref mut params, ref mut body), &Function(ref other_params, ref other_body)) if params.len() == other_params.len() => {
                let mut changed = false;
                for (this, other) in params.iter_mut().zip(other_params) {
                    changed |= this.push(other)?;
                }
                changed |= body.push(other_body)?;
                Ok(changed)
            }
            (&mut Builder(ref mut kind, ref mut annotations), &Builder(ref other_kind, ref other_annotations)) => {
                // Perform type checking on the BuilderKind, followed by the annotations.
                let mut changed = match (kind, other_kind) {
                    (&mut Appender(ref mut elem), &Appender(ref other_elem)) => {
                        Ok(elem.push(other_elem)?)
                    }
                    (&mut DictMerger(ref mut key, ref mut value, ref mut op), &DictMerger(ref other_key, ref other_value, ref other_op)) if *op == *other_op => {
                        // TODO other type checking for DictMergers? We could in theory do this in
                        // a separate pass.
                        Ok(key.push(other_key)? | value.push(other_value)?)
                    }
                    (&mut GroupMerger(ref mut key, ref mut value), &GroupMerger(ref other_key, ref other_value)) => {
                        Ok(key.push(other_key)? | value.push(other_value)?)
                    }
                    (&mut VecMerger(ref mut elem, ref mut op), &VecMerger(ref other_elem, ref other_op)) if *op == *other_op => {
                        Ok(elem.push(other_elem)?)
                    }
                    (&mut Merger(ref mut elem, ref mut op), &Merger(ref other_elem, ref other_op)) if *op == *other_op => {
                        Ok(elem.push(other_elem)?)
                    }
                    // Mismatches in the binary operator of DictMerger, VecMerger, and GroupMerger,
                    // or other type mismatches in the BuilderKind. We list them explicitly so the
                    // compiler will throw an error if we add new types.
                    (&mut Appender(_), _) | (&mut DictMerger(_,_,_), _) |
                        (&mut GroupMerger(_,_), _) | (&mut VecMerger(_,_), _) | (&mut Merger(_,_), _) => {
                            compile_err!("Mismatched types")
                        }
                }?;

                // Check the annotations.
                if *annotations != *other_annotations {
                    if !annotations.is_empty() {
                        *annotations = other_annotations.clone();
                        changed = true;
                    }
                }
                Ok(changed)
            }
            (ref this, ref other) => {
                // TODO: pretty print the types here.
                compile_err!("Mismatched types: could not push type {:?} to {:?}", other, this)
            }
        }
    }
}

/// A module-internal implementation of type inference.
///
/// This trait contains additional helper methods that are not exposed outside this module.
trait InferTypesInternal {
    fn infer_types_internal(&mut self) -> WeldResult<()>;
    fn infer_locally(&mut self, env: &TypeMap) -> WeldResult<bool>;
    fn infer_up(&mut self, &mut TypeMap) -> WeldResult<bool>;
}


impl InferTypesInternal for Expr {
    /// Internal implementation of type inference.
    fn infer_types_internal(&mut self) -> WeldResult<()> {
        loop {
            let ref mut env = TypeMap::default();
            if !self.infer_up(env)? {
                if self.ty.partial_type() {
                    return compile_err!("Could not infer some types")
                } else {
                    return Ok(())
                }
            }
        }
    }

    /// Infer types for an expression upward.
    ///
    /// Returns whether the type of this expression or any subexpressions changed, or an error if
    /// one occurred.
    fn infer_up(&mut self, env: &mut TypeMap) -> WeldResult<bool> {
        // Remember whether we inferred any new type.
        let mut changed = false;
        // Remember the old bindings so they can be restored.
        let mut old_bindings: Vec<Binding> = Vec::new();
        // Let and Lambda are the only expressions that change bindings.
        match self.kind {
            Let { ref mut name, ref mut value, ref mut body } => {
                // First perform type inference on the value. Then, take the old binding to `name`
                // and perform type inference on the body. Finally, restore the original
                // environment.
                changed |= value.infer_up(env)?;
                old_bindings.push((name.clone(), env.insert(name.clone(), value.ty.clone())));
                changed |= body.infer_up(env)?;
            }
            Lambda { ref mut params, ref mut body } =>  {
                for p in params {
                    old_bindings.push((p.name.clone(), env.insert(p.name.clone(), p.ty.clone())));
                }
                changed |= body.infer_up(env)?;
            }
            _ => ()
        };

        // Undo symbol bindings.
        for (symbol, opt) in old_bindings {
            match opt {
                Some(old) => env.insert(symbol, old),
                None => env.remove(&symbol),
            };
        }

        // Infer/check the type of this expression.
        changed |= self.infer_locally(env)?;
        Ok(changed)
    }

    fn infer_locally(&mut self, env: &TypeMap) -> WeldResult<bool> {
        Ok(true)
    }
}
