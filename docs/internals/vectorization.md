## Vectorization in Weld

Weld represents vector (SIMD) operations via the `Simd` type. Any scalar type can be
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

When a loop is vectorized, its function may merge elements of type `Simd` into the builder.
The builder contract for supporting vectorization is as follows:

If the builder supports vectorization for a type `Scalar(T)`:

* It must support merges of type `Simd(T)`
* It must support merges of type `Scalar(T)` (in the same builder)
* The `Result` operation must assume that any preceding operations on the builder could have
  involved both vectors and scalars.

Builders which support vectorization need not support vectorization for types other than
`Scalar(T)`. For example, if `merger` supports vectorization, the type `merger[{i32,i32},+]` does
not need to support vectorization.

Builders *do not* support vector types as their explicit merge types. For example, a builder of type
`merger[simd[i32],+]` is invalid. It is therefore currently impossible to build a vector type in Weld.

### Vectorization Internals 

The vectorizer performs the following steps:

1. Check if the all the constraints for performing vectorization are met. See [this section](#current-limitations-and-to-dos).
2. Copy the loop being vectorized an call it `vectorized_body`.
3. For each expression in `vectorized_body`: Change the type of `Scalar` expressions to `Simd`
   expressions.
4. Change the loop's iterator type to `SimdIter`. This directs the code generation to stop
   iterating at a multiple of the vector size.
5. Use the original (scalar) loop body to create a new "fringe" loop. Change the iterator type to
   `FringeIter`, which directs the code generation to only handle elements from where the
   `SimdIter` would have stopped to the end of the iterator.
6. Code generation. Most of the added complexity comes from computing index bounds for the fringe
   iterator and vector iterator.

#### New Expressions

* `broadcast(s)` broadcasts the scalar value in `s` to a vector.
* `select(cond, on_true, on_false)` evaluates `cond`, `on_true`, and `on_false` and returns a value
  based on whether `cond` is `true` or `false`.
* The `simd[T]` type is used to specify SIMD vectors.

#### Iterators

Fringing is handled via iterators, which specify which elements in a `for` loop to load. There are
now three kinds of iterators:

* `ScalarIter`: The grammar for this is `iter(x, ...)`. This is the usual iterator, and it load
  every element specified by the iterator ([start, end) with the given stride).

* `SimdIter`: The grammar for this is `simditer(x, ...)`. This loads all elements in chunks of the
  vector size, up until the point where elements would not fill the vector. As an example:

   `for(simditer([1,2,3,4,5]), ...)` will load a single `simd[i32]` with the elements `[1,2,3,4]`
   (assuming the vector length is 4).

* `FringeIter: The grammar for this is `fringeiter(x, ...)`. This loads the fringe of the
  corresponding `SimdIter`. From the example above, this would load the single scalar `5` (assuming
  a vector length of 4).

### Current Limitations and To Dos

* Only the `merger` builder is supported.
* Scatters and gathers are not supported (iterators must look at all elements), and index
  computations are disallowed in the for loop body).
* Nested loops not allowed. Indeed, only loops whose bodies contain the following expression kinds
  are allowed: `Literal`, `Ident`, `BinOp`, `Let`, `Merge`, `If`, `Select`, `GetField` (if its on
  the argument to the function).
* Unary math operators are not yet supported.
* Since the AST does not encode vector lengths, there is no way to express arbitrary vector literals
  at the moment. For example, there is no way to express a vector with the value `<1, 2, 3, 4>:simd[i32]`.
  Vectors composed of a single constant can be expressed using `broadcast`, however.
