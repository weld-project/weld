//! Type inference for Weld expressions.

extern crate fnv;

use new_types::*;
use new_types::LiteralKind::*;
use new_types::BuilderKind::*;
use new_types::ScalarKind::*;

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
    fn push_complete(&mut self, other: Self) -> WeldResult<bool>;
}

impl PushType for Type {
    /// Sets this `Type` to be `other`.
    fn push_complete(&mut self, other: Type) -> WeldResult<bool> {
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
                    dest.push_complete(elem.as_ref().clone())
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

/// Force `expr`, which has kind `Lambda`, to have parameters that match `tys`.
fn sync_function(expr: &mut Expr, tys: Vec<&Type>) -> WeldResult<bool> {
    if let Function(ref mut params, _) = expr.ty {
        if params.len() != tys.len() {
            compile_err!("TODO")
        } else {
            let mut changed = false;
            for (param, ty) in params.iter_mut().zip(tys) {
                changed |= param.push(ty)?;
            }
            Ok(changed)
        }
    } else {
        compile_err!("TODO")
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
        match self.kind {
            // Literal values - force the type to match the literal.
            Literal(I8Literal(_)) => self.ty.push_complete(Scalar(I8)),
            Literal(I16Literal(_)) => self.ty.push_complete(Scalar(I16)),
            Literal(I32Literal(_)) => self.ty.push_complete(Scalar(I32)),
            Literal(I64Literal(_)) => self.ty.push_complete(Scalar(I64)),
            Literal(U8Literal(_)) => self.ty.push_complete(Scalar(U8)),
            Literal(U16Literal(_)) => self.ty.push_complete(Scalar(U16)),
            Literal(U32Literal(_)) => self.ty.push_complete(Scalar(U32)),
            Literal(U64Literal(_)) => self.ty.push_complete(Scalar(U64)),
            Literal(F32Literal(_)) => self.ty.push_complete(Scalar(F32)),
            Literal(F64Literal(_)) => self.ty.push_complete(Scalar(F64)),
            Literal(BoolLiteral(_)) => self.ty.push_complete(Scalar(Bool)),
            Literal(StringLiteral(_)) => self.ty.push_complete(Vector(Box::new(Scalar(I8)))),

            BinOp { kind: op, ref mut left, ref mut right } => {
                // First, sync the left and right types into the elem_type.
                let ref mut elem_type = Unknown;
                elem_type.push(&left.ty)?;
                elem_type.push(&right.ty)?;

                if !op.is_comparison() {
                    elem_type.push(&self.ty)?;
                }

                // Now, attempt to push the elem_type back into left and right to "sync" them.
                let mut changed = left.ty.push(elem_type)?;
                changed |= right.ty.push(elem_type)?;

                // For comparisons, force the type to be Bool.
                if op.is_comparison() {
                    if let Simd(_) = left.ty {
                        changed |= self.ty.push(&Simd(Bool))?;
                    } else {
                        changed |= self.ty.push(&Scalar(Bool))?;
                    }
                } else {
                    changed |= self.ty.push(elem_type)?;
                }
                Ok(changed)
            }

            UnaryOp { ref value, .. } => {
                match value.ty {
                    Scalar(ref kind) if kind.is_float() => {
                        self.ty.push(&value.ty)
                    }
                    _ => {
                        compile_err!("TODO")
                    }
                }
            }

            Cast { kind, .. } => {
                self.ty.push_complete(Scalar(kind))
            }

            ToVec { ref mut child_expr } => {
                // The base type is vec[{?,?}] - infer the key and value type.
                let ref base_type = Vector(Box::new(Struct(vec![Unknown, Unknown])));
                let mut changed = self.ty.push(base_type)?;

                let mut set_types = false;
                if let Dict(ref key, ref value) = child_expr.ty {
                    if let Vector(ref mut elem) = self.ty {
                        if let Struct(ref mut fields) = **elem {
                            let pair_type = vec![key.clone(), value.clone()];
                            for (ref mut field, ref child) in fields.iter_mut().zip(pair_type) {
                                changed |= field.push(child)?;
                            }
                            set_types = true;
                        }
                    }
                }

                if !set_types {
                    compile_err!("TODO")
                } else {
                    Ok(changed)
                }
            }

            Ident(ref symbol) => {
                if let Some(ref ty) = env.get(symbol) {
                    self.ty.push(ty)
                } else {
                    compile_err!("TODO")
                }
            }

            Negate(ref c) => {
                self.ty.push(&c.ty)
            }

            Broadcast(ref c) => {
                if let Scalar(ref kind) = c.ty {
                    self.ty.push(&Simd(kind.clone()))
                } else {
                    compile_err!("TODO")
                }
            }

            CUDF { ref return_ty, .. } => {
                self.ty.push(return_ty)
            }

            Serialize(_) => {
                let serialized_type = Vector(Box::new(Scalar(I8)));
                self.ty.push_complete(serialized_type)
            }

            Deserialize{ ref value_ty, .. } => {
                self.ty.push(value_ty)
            }

            Let { ref mut body, .. } => {
                self.ty.sync(&mut body.ty)
            }

            MakeVector { ref mut elems } => {
                // Sync the types of each element.
                let mut elem_type = Unknown;
                for e in elems.iter() {
                    elem_type.push(&e.ty)?;
                }

                let mut changed = false;
                for e in elems.iter_mut() {
                    changed |= e.ty.push(&elem_type)?;
                }

                // If no error eccord, push the element type to the expression.
                let ref vec_type = Vector(Box::new(elem_type));
                changed |= self.ty.push(vec_type)?;
                Ok(changed)
            }

            Zip { ref mut vectors } => {
                let mut changed = false;

                let ref base_type = Vector(Box::new(Struct(vec![Unknown; vectors.len()])));
                changed |= self.ty.push(base_type)?;

                // Push the vector type to each vector.
                let mut types = vec![];
                let mut set_types = false;
                if let Vector(ref mut elem_type) = self.ty {
                    if let Struct(ref mut vec_types) = **elem_type {
                        for (vec_ty, vec_expr) in vec_types.iter_mut().zip(vectors.iter_mut()) {
                            if let Vector(ref elem_type) = vec_expr.ty {
                                changed |= vec_ty.push(elem_type)?;
                            }
                            types.push(vec_ty.clone());
                        }
                        set_types = true;
                    }
                }

                if !set_types || types.len() != vectors.len() {
                    compile_err!("TODO")
                } else {
                    let ref base_type = Vector(Box::new(Struct(types)));
                    changed |= self.ty.push(base_type)?;
                    Ok(changed)
                }
            }

        MakeStruct { ref mut elems } => {
            let mut changed = false;
            let ref base_type = Struct(vec![Unknown; elems.len()]);
            changed |= self.ty.push(base_type)?;

            if let Struct(ref mut elem_types) = self.ty {
                // Sync the type of each element and the expression struct type.
                for (elem_ty, elem_expr) in elem_types.iter_mut().zip(elems.iter_mut()) {
                    changed |= elem_ty.sync(&mut elem_expr.ty)?;
                }
                Ok(changed)
            } else {
                compile_err!("TODO")
            }
        }

        GetField { expr: ref mut param, index } => {
            if let Struct(ref mut elem_types) = param.ty {
                let index = index as usize;
                if index >= elem_types.len() {
                    compile_err!("IndexError")
                } else {
                    self.ty.sync(&mut elem_types[index])
                }
            } else {
                compile_err!("TODO")
            }
        }

        Length { ref mut data } => {
            if let Vector(_) = data.ty {
                self.ty.push_complete(Scalar(I64))
            } else {
                compile_err!("TODO")
            }
        }

        Slice { ref mut data, ref mut index, ref mut size } => {
            if let Vector(_) = data.ty {
                let mut changed = false;
                changed |= index.ty.push_complete(Scalar(I64))?;
                changed |= size.ty.push_complete(Scalar(I64))?;
                changed |= self.ty.push(&data.ty)?;
                Ok(changed)
            } else {
                compile_err!("TODO")
            }
        }

        Sort { ref mut data, ref mut keyfunc } => {
            if let Vector(ref elem_type) = data.ty {
                let mut changed = sync_function(keyfunc, vec![&elem_type])?;
                changed |= self.ty.push(&data.ty)?;
                Ok(changed)
            } else {
                compile_err!("TODO")
            }
        }

        /*
        Lookup {
            ref mut data,
            ref mut index,
        } => {
            if let Vector(ref elem_type) = data.ty {
                let mut changed = false;
                changed |= try!(push_complete_type(&mut index.ty, Scalar(I64), "Lookup"));
                changed |= try!(push_type(&mut expr.ty, &elem_type, "Lookup"));
                Ok(changed)
            } else if let Dict(ref key_type, ref value_type) = data.ty {
                let mut changed = false;
                changed |= try!(push_type(&mut index.ty, &key_type, "Lookup"));
                changed |= try!(push_type(&mut expr.ty, &value_type, "Lookup"));
                Ok(changed)
            } else if data.ty == Unknown {
                Ok(false)
            } else {
                compile_err!("Internal error: Lookup called on {:?}", data.ty)
            }
        }

        KeyExists {
            ref mut data,
            ref mut key,
        } => {
            if let Dict(ref key_type, _) = data.ty {
                let mut changed = false;
                changed |= try!(push_type(&mut key.ty, &key_type, "KeyExists"));
                changed |= try!(push_complete_type(&mut expr.ty, Scalar(Bool), "KeyExists"));
                Ok(changed)
            } else if data.ty == Unknown {
                Ok(false)
            } else {
                compile_err!("Internal error: KeyExists called on {:?}", data.ty)
            }
        }

        Lambda {
            ref mut params,
            ref mut body,
        } => {
            let mut changed = false;

            let base_type = Function(vec![Unknown; params.len()], Box::new(Unknown));
            changed |= try!(push_type(&mut expr.ty, &base_type, "Lambda"));

            if let Function(ref mut param_types, ref mut res_type) = expr.ty {
                changed |= try!(sync_types(&mut body.ty, res_type, "Lambda return"));
                for (param_ty, param_expr) in param_types.iter_mut().zip(params.iter_mut()) {
                    changed |= try!(sync_types(param_ty, &mut param_expr.ty, "Lambda parameter"));
                }
            } else {
                return compile_err!("Internal error: type of Lambda was not Function");
            }

            Ok(changed)
        }
        */

            _ => unimplemented!()
        }
    }
}
