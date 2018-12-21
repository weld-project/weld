//! Type inference for Weld expressions.

extern crate fnv;

use ast::*;
use ast::ExprKind::*;
use ast::Type::*;
use ast::LiteralKind::*;
use ast::BuilderKind::*;
use ast::ScalarKind::*;

use error::*;

use fnv::FnvHashMap;


type TypeMap = FnvHashMap<Symbol, Type>;
type Binding = (Symbol, Option<Type>);

/// A trait for checking and infering types in-place. 
pub trait InferTypes {
    /// Checks and infers types in place.
    ///
    /// Returns an error if types are inconsistent or all types could not be inferred. That is, if
    /// the resulting value has any partial types, this method will return an error.
    fn infer_types(&mut self) -> WeldResult<()>;

    /// Locally check and infer types.
    ///
    /// This method will check types (and infer unknown ones) for a given expression without
    /// recursing into its children. Returns an error if the expression or its direct
    /// sub-expression have invalid types.
    ///
    /// Because this function only looks at local sub-expressions, identifiers with `Unknown` type
    /// are disallowed.
    fn infer_local(&mut self) -> WeldResult<()>;
}

impl InferTypes for Expr {
    /// Checks and infers types in place.
    fn infer_types(&mut self) -> WeldResult<()> {
        self.infer_types_internal()
    }

    fn infer_local(&mut self) -> WeldResult<()> {
        let ref mut env = TypeMap::default();
        if let Ident(ref sym) = self.kind {
            env.insert(sym.clone(), self.ty.clone());
        }

        self.infer_locally(env).map(|_| ())
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

fn key_hashable(key: &Type) -> WeldResult<()> {
    if *key != Unknown && !key.is_hashable() {
        compile_err!("Non-hashable type {} as dictionary key", key)
    } else {
        Ok(())
    }
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
                    compile_err!("Type mismatch: expected {} but got {}", &other, self)
                }
            }
            _ => {
                compile_err!("Type mismatch: expected {} but got {}", &other, self)
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
                let mut changed = key.push(other_key)? || value.push(other_value)?;
                key_hashable(key.as_ref())?;
                Ok(changed)
            }
            (&mut Struct(ref mut types), &Struct(ref other_types)) if types.len() == other_types.len() => {
                let mut changed = false;
                for (this, other) in types.iter_mut().zip(other_types) {
                    changed |= this.push(other)?;
                }
                Ok(changed)
            }
            (&mut Function(ref mut params, ref mut body),
             &Function(ref other_params, ref other_body))
                if params.len() == other_params.len() => {

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
                    (
                        &mut Appender(ref mut elem),
                        &Appender(ref other_elem)
                    ) => {
                        elem.push(other_elem)
                    }
                    (
                        &mut DictMerger(ref mut key, ref mut value, ref mut op),
                        &DictMerger(ref other_key, ref other_value, ref other_op)
                    ) if *op == *other_op => {
                        let mut changed = key.push(other_key)? || value.push(other_value)?;
                        key_hashable(key.as_ref())?;
                        Ok(changed)
                    }
                    (
                        &mut GroupMerger(ref mut key, ref mut value),
                        &GroupMerger(ref other_key, ref other_value)
                    ) => {
                        let mut changed = key.push(other_key)? || value.push(other_value)?;
                        key_hashable(key.as_ref())?;
                        Ok(changed)
                    }
                    (
                        &mut VecMerger(ref mut elem, ref mut op),
                        &VecMerger(ref other_elem, ref other_op)
                    ) if *op == *other_op => {
                        elem.push(other_elem)
                    }
                    (
                        &mut Merger(ref mut elem, ref mut op),
                        &Merger(ref other_elem, ref other_op)
                    ) if *op == *other_op => {
                        elem.push(other_elem)
                    }
                    // Mismatches in the binary operator of DictMerger, VecMerger, and GroupMerger,
                    // or other type mismatches in the BuilderKind. We list them explicitly so the
                    // compiler will throw an error if we add new types.
                    (&mut Appender(_), _) | (&mut DictMerger(_,_,_), _) |
                        (&mut GroupMerger(_,_), _) | (&mut VecMerger(_,_), _) |
                        (&mut Merger(_,_), _) => {
                            compile_err!("Type mismatch: expected builder type {}", other)
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
                compile_err!("Type mismatch: expected {} but got {}",
                             other, this)
            }
        }
    }
}

/// Force `expr`, which has kind `Lambda`, to have parameters that match `tys`.
fn sync_function(expr: &mut Expr, tys: Vec<&Type>) -> WeldResult<bool> {
    match expr.ty {
        Function(ref mut params, _) if params.len() == tys.len() => {
            let mut changed = false;
            for (param, ty) in params.iter_mut().zip(tys) {
                changed |= param.push(ty)?;
            }
            Ok(changed)
        }
        _ => {
            compile_err!("Expected function with {} arguments, but got {}", tys.len(), &expr.ty)
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
                if self.partially_typed() {
                    return compile_err!("Could not infer some types")
                } else {
                    return Ok(())
                }
            }
        }
    }

    /// Infer types for an expression upward.
    ///
    /// This method iterates over each expression in the AST in post-order, operating on the trees
    /// leaves and propogating types up. The method returns whether the type of this expression or
    /// any subexpressions changed, or an error if one occurred.
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
                let previous = env.insert(name.clone(), value.ty.clone());
                old_bindings.push((name.clone(), previous));
                changed |= body.infer_up(env)?;
            }
            Lambda { ref mut params, ref mut body } =>  {
                for p in params {
                    let previous = env.insert(p.name.clone(), p.ty.clone());
                    old_bindings.push((p.name.clone(), previous));
                }
                changed |= body.infer_up(env)?;
            }
            _ => ()
        };

        // For Let and Lambda, we already inferred the types of each child. For all other
        // expressions, we still need to do it after undoing the binding..
        match self.kind {
            Let { .. } | Lambda { .. } => (),
            _ => {
                // Infer/check the type of the subexpressions.
                for child in self.children_mut() {
                    changed |= child.infer_up(env)?;
                }
            }
        };

        // Undo symbol bindings.
        for (symbol, opt) in old_bindings {
            match opt {
                Some(old) => env.insert(symbol, old),
                None => env.remove(&symbol),
            };
        }

        // Infer local type.
        changed |= self.infer_locally(env)?;
        Ok(changed)
    }

    /// Infer the types of an expression based on direct subexpressions.
    fn infer_locally(&mut self, env: &TypeMap) -> WeldResult<bool> {
        match self.kind {
            Literal(I8Literal(_)) =>
                self.ty.push_complete(Scalar(I8)),
            Literal(I16Literal(_)) =>
                self.ty.push_complete(Scalar(I16)),
            Literal(I32Literal(_)) =>
                self.ty.push_complete(Scalar(I32)),
            Literal(I64Literal(_)) =>
                self.ty.push_complete(Scalar(I64)),
            Literal(U8Literal(_)) =>
                self.ty.push_complete(Scalar(U8)),
            Literal(U16Literal(_)) =>
                self.ty.push_complete(Scalar(U16)),
            Literal(U32Literal(_)) =>
                self.ty.push_complete(Scalar(U32)),
            Literal(U64Literal(_)) =>
                self.ty.push_complete(Scalar(U64)),
            Literal(F32Literal(_)) =>
                self.ty.push_complete(Scalar(F32)),
            Literal(F64Literal(_)) =>
                self.ty.push_complete(Scalar(F64)),
            Literal(BoolLiteral(_)) =>
                self.ty.push_complete(Scalar(Bool)),
            Literal(StringLiteral(_)) =>
                self.ty.push_complete(Type::string_type()),

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
                        changed |= self.ty.push_complete(Simd(Bool))?;
                    } else {
                        changed |= self.ty.push_complete(Scalar(Bool))?;
                    }
                } else {
                    changed |= self.ty.push(elem_type)?;
                }
                Ok(changed)
            }

            UnaryOp { ref value, ref kind } => {
                match value.ty {
                    Scalar(ref kind) | Simd(ref kind) if kind.is_float() => {
                        self.ty.push(&value.ty)
                    }
                    Unknown => Ok(false),
                    _ => {
                        compile_err!("Expected floating-point type for unary op '{}'", kind)
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
                } else if child_expr.ty == Unknown {
                    return Ok(false)
                }

                if !set_types {
                    compile_err!("Expected dictionary argument for tovec(...), got {}", &child_expr.ty)
                } else {
                    Ok(changed)
                }
            }

            Ident(ref symbol) => {
                if let Some(ref ty) = env.get(symbol) {
                    self.ty.push(ty)
                } else {
                    compile_err!("Symbol {} is not defined", symbol)
                }
            }

            Negate(ref c) => {
                self.ty.push(&c.ty)
            }

            Not(ref value) => {
                match value.ty {
                    Scalar(ref kind) | Simd(ref kind) if kind.is_bool() => {
                        self.ty.push(&value.ty)
                    }
                    Unknown => Ok(false),
                    _ => {
                        compile_err!("Expected boolean type for ! operator")
                    }
                }
            }

            Broadcast(ref c) => {
                if let Scalar(ref kind) = c.ty {
                    self.ty.push(&Simd(kind.clone()))
                } else if c.ty == Unknown {
                    Ok(false)
                } else {
                    compile_err!("Expected scalar argument for broadcast, got {}", &c.ty)
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
                    compile_err!("Expected vector types in zip, got types {}",
                                 types.iter()
                                    .map(|t| t.to_string())
                                    .collect::<Vec<_>>()
                                    .join(","))
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
                    compile_err!("Type mismatch in struct literal")
                }
            }

            GetField { expr: ref mut param, index } => {
                if let Struct(ref mut elem_types) = param.ty {
                    let index = index as usize;
                    if index >= elem_types.len() {
                        compile_err!("struct index error")
                    } else {
                        self.ty.sync(&mut elem_types[index])
                    }
                } else if param.ty == Unknown {
                    Ok(false)
                } else {
                    compile_err!("Expected struct type for struct field access, got {}", &param.ty)
                }
            }

            Length { ref mut data } => {
                if let Vector(_) = data.ty {
                    self.ty.push_complete(Scalar(I64))
                } else if data.ty == Unknown {
                    Ok(false)
                } else {
                    compile_err!("Expected vector type in len, got {}", &data.ty)
                }
            }

            Slice { ref mut data, ref mut index, ref mut size } => {
                if let Vector(_) = data.ty {
                    let mut changed = false;
                    changed |= index.ty.push_complete(Scalar(I64))?;
                    changed |= size.ty.push_complete(Scalar(I64))?;
                    changed |= self.ty.push(&data.ty)?;
                    Ok(changed)
                } else if data.ty == Unknown {
                    Ok(false)
                } else {
                    compile_err!("Expected vector type in slice, got {}", &data.ty)
                }
            }

            Sort { ref mut data, ref mut cmpfunc } => {
                if let Vector(ref elem_type) = data.ty {
                    let mut changed = sync_function(cmpfunc, vec![&elem_type, &elem_type])?;
                    changed |= self.ty.push(&data.ty)?;
                    Ok(changed)
                } else if data.ty == Unknown {
                    Ok(false)
                } else {
                    compile_err!("Expected vector type in sort, got {}", &data.ty)
                }
            }

            Lookup { ref mut data, ref mut index } => {
                let mut changed = false;
                match data.ty {
                    Vector(ref elem_type) => {
                        changed |= index.ty.push_complete(Scalar(I64))?;
                        changed |= self.ty.push(elem_type)?;
                        Ok(changed)
                    }
                    Dict(ref key_type, ref value_type) => {
                        changed |= index.ty.push(key_type)?;
                        changed |= self.ty.push(value_type)?;
                        Ok(changed)
                    }
                    Unknown => Ok(false),
                    _ => {
                        compile_err!("Expected vector or dict type in lookup, got {}", &data.ty)
                    }
                }
            }

            OptLookup { ref mut data, ref mut index } => {
                debug!("optlookup type inference: optlookup({}, {}): {}", data.ty, index.ty, self.ty);
                match data.ty {
                    Dict(ref key_type, ref value_type) => {
                        let mut my_type = vec![ Scalar(Bool), value_type.as_ref().clone() ];
                        let mut changed = false;
                        changed |= index.ty.push(key_type)?;

                        // Push the value type.
                        changed |= my_type.get_mut(1).unwrap().push(value_type)?;

                        let ref mut struct_ty = Struct(my_type);
                        changed |= self.ty.sync(struct_ty)?;

                        Ok(changed)
                    }
                    Unknown => Ok(false),
                    _ => {
                        compile_err!("Expected dict type in optlookup, got {}", &data.ty)
                    }
                }
            }

            KeyExists { ref mut data, ref mut key } => {
                if let Dict(ref key_type, _) = data.ty {
                    let mut changed = false;
                    changed |= key.ty.push(&key_type)?;
                    changed |= self.ty.push_complete(Scalar(Bool))?;
                    Ok(changed)
                } else if data.ty == Unknown {
                    Ok(false)
                } else {
                    compile_err!("Expected dict type in lookup, got {}", &data.ty)
                }
            }

            Lambda { ref mut params, ref mut body } => {
                let mut changed = false;
                let base_type = Function(vec![Unknown; params.len()], Box::new(Unknown));

                changed |= self.ty.push(&base_type)?;

                if let Function(ref mut param_types, ref mut res_type) = self.ty {
                    changed |= body.ty.sync(res_type)?;
                    for (param_ty, param_expr) in param_types.iter_mut().zip(params.iter_mut()) {
                        changed |= param_ty.sync(&mut param_expr.ty)?;
                    }
                    Ok(changed)
                } else {
                    compile_err!("Expected function type for lambda, got {}", &self.ty)
                }
            }

            If { ref mut cond, ref mut on_true, ref mut on_false } => {
                let mut changed = false;
                changed |= cond.ty.push_complete(Scalar(Bool))?;
                changed |= self.ty.sync(&mut on_true.ty)?;
                changed |= self.ty.sync(&mut on_false.ty)?;
                Ok(changed)
            }

            Iterate { ref mut initial, ref mut update_func } => {
                let mut changed = self.ty.sync(&mut initial.ty)?;
                match update_func.ty {
                    Function(ref mut params, _) if params.len() == 1 => {
                        changed |= params.get_mut(0).unwrap().sync(&mut initial.ty)?;
                        Ok(changed)
                    }
                    _ => compile_err!("Expected function with single type in iterate, got {}", &update_func.ty)
                }
            }

            Select { ref mut cond, ref mut on_true, ref mut on_false } => {
                let mut changed = false;
                let cond_type = if cond.ty.is_simd() {
                    Simd(Bool)
                } else {
                    Scalar(Bool)
                };
                changed |= cond.ty.push_complete(cond_type)?;
                changed |= self.ty.sync(&mut on_true.ty)?;
                changed |= self.ty.sync(&mut on_false.ty)?;
                Ok(changed)
            }

            Apply { ref mut func, ref mut params } => {
                let func_type = Function(vec![Unknown; params.len()], Box::new(Unknown));
                let mut changed = func.ty.push(&func_type)?;

                let ref mut fty = func.ty;
                if let Function(ref mut param_types, ref mut res_type) = *fty {
                    changed |= self.ty.sync(res_type)?;
                    for (param_ty, param_expr) in param_types.iter_mut().zip(params.iter_mut()) {
                        changed |= param_ty.sync(&mut param_expr.ty)?;
                    }
                    Ok(changed)
                } else {
                    compile_err!("Expected function type in apply, got {}", fty)
                }
            }

            NewBuilder(ref mut argument) => {
                // NewBuilder is a special case where the expression can currently only be created
                // via a type. If it doesn't have a type, throw a type inference error.
                if let Builder(ref kind, _) = self.ty {
                    // Handle the builders that may take arguments.
                    match *kind {
                        VecMerger(ref elem, _) => {
                            if let Some(ref mut argument) = argument {
                                argument.ty.push(&Vector(elem.clone()))
                            } else {
                                compile_err!("Expected single vector argument in vecmerger")
                            }
                        }
                        Merger(ref elem, _) => {
                            if let Some(ref mut argument) = argument {
                                argument.ty.push(elem)
                            } else {
                                Ok(false)
                            }
                        }
                        Appender(_) => {
                            if let Some(ref mut argument) = argument {
                                argument.ty.push(&Scalar(I64))
                            } else {
                                Ok(false)
                            }
                        }
                        _ => Ok(false)
                    }
                } else if self.ty == Unknown {
                    Ok(false)
                } else {
                    compile_err!("non-builder type when creating new builder")
                }
            }

            Merge { ref mut builder, ref mut value } => {
                use std::mem;
                let mut changed = false;

                // Get the merge type, which is the expected type of the value merged into the
                // builder, and the builder kind, whose precise type we will infer and then set
                // back into the builder.
                let (mut merge_type, mut kind) = if let Builder(ref builder_kind, _) = builder.ty {
                    let mut merge_type = builder_kind.merge_type();
                    // If we are merging a SIMD value, sync with the SIMD merge type.
                    // NOTE: This currently only works under the assumption that each
                    // value in a SIMD program is SIMD-valued.
                    if value.ty.is_simd() {
                        merge_type = merge_type.simd_type()?;
                    }
                    (merge_type, builder_kind.clone())
                } else if builder.ty == Unknown {
                    return Ok(false)
                } else {
                    return compile_err!("Expected builder type in merge, got {}", &builder.ty)
                };

                changed |= value.ty.sync(&mut merge_type)?;

                // To sync the builder type, remove any SIMD type: SIMD shouldn't appear
                // in the Builder kind.
                if merge_type.is_simd() {
                    merge_type = merge_type.scalar_type()?;
                }

                // Set the builder kind type.
                match kind {
                    Appender(ref mut elem) => {
                        **elem = merge_type;
                    }
                    Merger(ref mut elem, _) => {
                        **elem = merge_type;
                    }
                    DictMerger(ref mut key, ref mut value, _) => {
                        if let Struct(mut tys) = merge_type {
                            mem::swap(key.as_mut(), tys.get_mut(0).unwrap());
                            mem::swap(value.as_mut(), tys.get_mut(1).unwrap());
                        } else {
                            unreachable!()
                        }
                    }
                    GroupMerger(ref mut key, ref mut value) => {
                        if let Struct(mut tys) = merge_type {
                            mem::swap(key.as_mut(), tys.get_mut(0).unwrap());
                            mem::swap(value.as_mut(), tys.get_mut(1).unwrap());
                        } else {
                            unreachable!()
                        }
                    }
                    VecMerger(ref mut elem, _) => {
                        if let Struct(mut tys) = merge_type {
                            // tys[0] is the index.
                            mem::swap(elem.as_mut(), tys.get_mut(1).unwrap());
                        } else {
                            unreachable!()
                        }
                    }
                };

                if let Builder(ref mut builder_kind, _) = builder.ty {
                    *builder_kind = kind;
                } else {
                    unreachable!()
                }
                changed |= self.ty.sync(&mut builder.ty)?;
                Ok(changed)
            }

            Res { ref mut builder } => {
                if let Builder(ref mut kind, _) = builder.ty {
                    self.ty.push(&kind.result_type())
                } else if builder.ty == Unknown {
                    Ok(false)
                } else {
                    compile_err!("Expected builder type in result, got {}", &builder.ty)
                }
            }

            For { ref mut iters, ref mut builder, ref mut func } => {
                let mut changed = false;
                // First, for each Iter, if it has a start, end, stride, etc., make sure the types of
                // those expressions is Scalar(I64).
                for iter in iters.iter_mut() {
                    // For ScalarIter, SimdIter, and RangeIter, start, end and stride must all be
                    // None or Some.
                    if iter.start.is_some() {
                        changed |= iter.start.as_mut().unwrap().ty.push_complete(Scalar(I64))?;
                        changed |= iter.end.as_mut().unwrap().ty.push_complete(Scalar(I64))?;
                        changed |= iter.stride.as_mut().unwrap().ty.push_complete(Scalar(I64))?;
                    }

                    // For NDIter, the same rule applies for shape and stride.
                    if iter.strides.is_some() {
                        changed |= iter.strides.as_mut().unwrap().ty.push_complete(Vector(Box::new(Scalar(I64))))?;
                        changed |= iter.shape.as_mut().unwrap().ty.push_complete(Vector(Box::new(Scalar(I64))))?;
                    }
                }

                // Now get the vector data types.
                let mut elem_types: Vec<_> = iters.iter().map(|iter| {
                    // If the iterator is a RangeIter, special case it -- the data must be a "dummy"
                    // empty vector with type vec[i64].
                    if iter.kind == IterKind::RangeIter {
                        Ok(Scalar(I64))
                    } else {
                        // Make sure the Iter's data is a Vector, and pull out its element kind.
                        match iter.data.ty {
                            Vector(ref elem) => Ok(elem.as_ref().clone()),
                            Unknown => Ok(Unknown),
                            _ => compile_err!("Expected vector type in for loop iter, got {}", &iter.data.ty)
                        }
                    }
                }).collect::<WeldResult<_>>()?;

                // Convert the vector into a Type, which will either be a Struct or a single type.
                let mut elem_types = if elem_types.len() == 1 {
                    elem_types[0].clone()
                } else {
                    Struct(elem_types)
                };

                // Check the For loop's function, and change the element types to be SIMD types
                // if necessary.
                match func.kind {
                    Lambda { ref params, .. } if params.len() == 3 => {
                        // Convert the expected element types to SIMD if the parameter in the builder
                        // functinon is SIMD.
                        if params[2].ty.is_simd() {
                            if !iters.iter().all(|i| i.kind == IterKind::SimdIter) {
                                return compile_err!("for loop requires that either all or none of the iters are simditer")
                            }
                            elem_types = elem_types.simd_type()?;
                        } else if iters.iter().any(|i| i.kind == IterKind::SimdIter) {
                            return compile_err!("for loop requires that either all or none of the iters are simditer")
                        }
                    }
                    _ => {
                        // Invalid builder function.
                        return compile_err!("Expected builder function of type |builder, i64, elements|")
                    }
                }

                // Impose the correct types on the function.
                // Function should be (builder, i64, elems) -> builder.
                let builder_type = builder.ty.clone();
                let ref func_type = Function(vec![builder_type.clone(), Scalar(I64), elem_types], Box::new(builder_type));
                changed |= func.ty.push(func_type)?;

                // This will never be false, since we pushed the Function type above.
                // Make sure the Builder type and the function builder argument and return types match.
                if let Function(ref params, ref result) = func.ty {
                    changed |= builder.ty.push(&params[0])?;
                    changed |= builder.ty.push(result.as_ref())?;
                } else {
                    unreachable!()
                }

                // Push builder's type to our expression
                changed |= self.ty.push(&builder.ty)?;
                Ok(changed)
            }
        }
    }
}

#[test]
fn infer_types_test() {
    use tests::*;
    let e = parse_expr("a").unwrap();
    assert_eq!(print_typed_expr_without_indent(&e).as_str(), "a:?");

    let e = Expr {
        kind: ExprKind::Ident(Symbol::new("a", 1)),
        ty: Type::Unknown,
        annotations: Annotations::new(),
    };
    assert_eq!(print_typed_expr_without_indent(&e).as_str(), "a__1:?");

    let e = parse_expr("a:i32").unwrap();
    assert_eq!(print_typed_expr_without_indent(&e).as_str(), "a:i32");

    let mut e = parse_expr("let a = 2; a").unwrap();
    assert_eq!(print_typed_expr_without_indent(&e).as_str(),
               "(let a:?=(2);a:?)");
    e.infer_types().unwrap();
    assert_eq!(print_typed_expr_without_indent(&e).as_str(),
               "(let a:i32=(2);a:i32)");

    let mut e = parse_expr("let a = 2; let a = false; a").unwrap();
    e.infer_types().unwrap();
    assert_eq!(print_typed_expr_without_indent(&e).as_str(),
               "(let a:i32=(2);(let a:bool=(false);a:bool))");

    // Types should propagate from function parameters to body
    let mut e = parse_expr("|a:i32, b:i32| a + b").unwrap();
    e.infer_types().unwrap();
    assert_eq!(print_typed_expr_without_indent(&e).as_str(),
               "|a:i32,b:i32|(a:i32+b:i32)");

    let mut e = parse_expr("|a:f32, b:f32| a + b").unwrap();
    e.infer_types().unwrap();
    assert_eq!(print_typed_expr_without_indent(&e).as_str(),
               "|a:f32,b:f32|(a:f32+b:f32)");

    let mut e = parse_expr("let a = [1, 2, 3]; 1").unwrap();
    e.infer_types().unwrap();
    assert_eq!(print_typed_expr_without_indent(&e).as_str(),
               "(let a:vec[i32]=([1,2,3]);1)");

    // Mismatched types in MakeVector
    let mut e = parse_expr("[1, true]").unwrap();
    assert!(e.infer_types().is_err());

    let mut e = parse_expr("for([1],appender[?],|b,i,x|merge(b,x))").unwrap();
    e.infer_types().unwrap();
    assert_eq!(print_typed_expr_without_indent(&e).as_str(),
               "for([1],appender[i32],|b:appender[i32],i:i64,x:i32|merge(b:appender[i32],x:i32))");
}
