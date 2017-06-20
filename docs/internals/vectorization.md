## Vectorization in Weld

Weld represents vector (SIMD) operations via the `Vectorized` type. Any scalar type can be
vectorized; vectorization for other types is not supported.

Vectorization occurs in the context of a `For` loop. If a loop is vectorized, each iteration of the
loop operates over multiple elements in each iteration, using SIMD instructions. For example, the
following loop is easily vectorizable:

```
for(v, merger[i32,+], |b,i,e| merge(b, e))
```

Vectorization allows multiple elements to be merged into the builder `b` in a single iteration, in
parallel.

### Vectorization Support for Builders

The compiler only vectorizes builders which are known to support vectorization. At the moment, the
following builders have vectorization support:

* `merger`

When a loop is vectorized, its function may merge elements of type `Vectorized` into the builder.
The builder contract for supporting vectorization is as follows:

If the builder supports vectorization for a type `Scalar(T)`:

* It must support merges of type `Vectorized(T)`
* It must support merges of type `Scalar(T)` (in the same builder)
* The `Result` operation must assume that any preceding operations on the builder could have
  involved both vectors and scalars.

Builders which support vectorization need not support vectorization for types other than
`Scalar(T)`. For example, if `merger` supports vectorization, the type `merger[{i32,i32},+]` does
not need to support vectorization.

Builders *do not* support vector types as their explicit merge types. For example, a builder of type
`merger[<? x i32>,+]` is invalid. It is therefore currently impossible to build a vector type in Weld.

### Vectorization Internals 

The vectorizer performs the following steps:

1. Check if the all the constraints for performing vectorization are met. See [this section](#current-limitations-and-to-dos).
2. Copy the loop being vectorized an call it `vectorized_body`.
3. For each expression in `vectorized_body`: Change the type of `Scalar` expressions to `Vector`
   expressions.
4. Change the loop's iterator type to `VectorIter`. This directs the code generation to stop
   iterating at a multiple of the vector size.
5. Use the original (scalar) loop body to create a new "fringe" loop. Change the iterator type to
   `FringeIter`, which directs the code generation to only handle elements from where the
   `VectorIter` would have stopped to the end of the iterator.
6. Code generation. Most of the added complexity comes from computing index bounds for the fringe
   iterator and vector iterator.

### Current Limitations and To Dos

* Only the `merger` builder is supported
* Vectorization fails if more than one iterator is present
* Scatters and gathers are not supported (iterators must look at all elements), and index
  computations are disallowed in the for loop body)
* Broadcast of variables defined outside the loop body not currently supported. For example, the
  following example will fail to vectorize because `a` must be broadcast into a vector:

  ```
  |v: vec[i32], a: i32| for(v, merger[i32,+], |b,i,e| merge(b, e+a))
  ```

* Predication not yet implemented.
* Nested loops not allowed. Indeed, only loops whose bodies contain the following expression kinds
  are allowed: `Literal`, `Ident`, `BinOp`, `Let`, `Merge`.
