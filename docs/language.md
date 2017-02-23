# NVL Language Overview

## Overview

NVL is a statically typed, referentially transparent language with built-in parallel constructs.
It has a number of operators similar to functional languages, but it is not truly functional, in that functions aren't first-class values and recursion is not allowed.
As a result, NVL programs have a finite call graph that is known in advance and are straightforward to analyze.
By "referentially transparent", we mean that each expression in NVL is a pure function of its inputs, and there are no side-effects.

The language lets users specify functions and expressions.
A function, such as `(a: int, b: int) => a + b`, consists of a list of named arguments and an expression for the result.
Variables in NVL are immutable, though there is a "let" statement for introducing new ones.
Some operators built into the language take functions (in fact closures), but functions are not first-class values (one cannot pass them to other functions or store them in variables).

NVL contains both a "core" language and higher-level "sugar" syntax for specifying common functional operators, such as `map` and `groupBy`.
The core language has only one parallel construct, the `for` expression, and a set of types called *builders* used to compute various types of results (e.g. sums, vectors, etc).
All of the sugar operators are translated to `for`s and builders through simple substitution rules.
We will begin by describing the core language, and then describe the currently supported sugar operators.

## Data Types

NVL contains "value" types that hold data, as well as "builder" types that are used to construct results in parallel from values that are "merged" into them.

The value types are:

* Scalars: `bit`, `char`, `int`, `long`, `double`.
* Vectors: `vec[T]` for some type `T`. These are variable-length.
* Dictionaries: `dict[K, V]` for types K, V.
* Structs: `{T1, T2, ...}` for field types T1, T2, etc.
* Fixed-length vectors: `fixed[T, n]` for a scalar type T and integer n. Right now fixed-length vectors cannot be stored in composite types (`vec`, `dict` and structs).

The builder types are:

* `vecBuilder[T]`: Builds a `vec[T]` from elements of type `T`.
* `groupBuilder[K, V]`: Merges `{K, V}` pairs to produce a `dict[K, vec[V]]`.
* `merger[zero, mergeValue, mergeResult]`: Combines results using associative "merge" functions. Its parameters are:
   * `zero`: An identity value of type `R` returned if there are no results
   * `mergeValue`: A function of type `(T, R) => R`
   * `mergeResult`: A function of type `(R, R) => R`.
* `dictMerger[K, mergeFunc]`: Combines `{K, V}` pairs by key into a dictionary. The parameters are:
   * `K`: Key type
   * `mergeFunc`: A function of type `(V, V) => V`, where `V` is the value type.
* `vecMerger[mergeFunc]`: Combines `{long, V}` pairs by key into a vector using `mergeFunc`. The builder is initialized with an initial vector to work with.
* Any struct whose fields are builders can also be used as a builder. This is used to build multiple results at the same time.

Note that among the builders that take functions and or values, two types are identical only if they're parameterized with the same functions / values.
Making functions part of the type is slightly unusual because the functions passed in could be closures over some variables that exist at that point in the program.
Right now the implementation doesn't allow this, and we have not tried to formalize what it means.

## Core Operations

The core language consists of the following expressions:

* Literals, e.g. `5.0`, `{6, 7}`, and `true`.
* Arithmetic expressions, e.g. `a + b`, `a - b`, `-a`, `a & b`, etc.
* `if(condition, onTrue, onFalse)`, which evaluates `onTrue` or `onFalse` based on the value of `condition`.
* `loop(initial, update)`, an expression used for sequential looping. `initial` is any type `T`, and `update` must be a function of type `T => {T, bit}` that computes a new value and also returns a bit specifying whether to continue looping. The value of `loop` is the last value returned by `update` (when it sets the bit to false).
* Let expressions, which introduce a new variable. The syntax for these is `name := expr; body`. This evaluates `expr`, assigns it to the variable `name`, and then evaluates `body` with that binding and returns its result.
* Collection expressions:
  * `lookup(dict, key)` and `lookup(vec, index)` return an element from a dictionary and vector respectively.
  * `len(dict)` or `len(vec)` return its length.
  * `struct.0`, `struct.1`, etc are used to access fields of a struct.
  * `sort(vec, keyFunc)` sorts a vector by a given key computed from each value.
  * `toVec(dict)` gets the entries of a dictionary as a vector.
* Three special expressions involving builders:
  * `merge(builder, value)` returns a new builder that incorporates `value` into the previous builder.
  * `res(builder)` computes the result of the builder given the values merged so far.
  * `for(vec, builder, update)` applies a function `update` to every element of a vector, possibly merging values into a builder for each one, and returns a final builder with all the merges incorporated. `vec` must be of type `vec[T]` for some `T`, `builder` can be any builder type `B`, and `update` must be a function of type `(B, T) => B` that possibly merges values into the `B` passed in.

The builder expressions are the only "interesting" ones, where parallelism comes in.
The basic idea is that a builder is a "write-only" data structure, and the `res` operation turns it into a read-only value.
Because builders are "write-only", it is okay to merge values into them from different iterations of a `for` loop in parallel.
Note that this requires that one does not call `res` on a builder inside a loop.
In reality, we can enforce this by making builders "linear types", and requiring that the `update` function in `for` return a builder derived from its argument.
Linear types are a concept in programming languages that we'll talk about below.
Our implementation does not statically enforce linearity, but it only works if `update` functions really do only return builders derived from their arguments, and if `res` is only called on each builder once.

### Example

Here are a few simple examples using builder expressions:

```
# basic use of vecBuilder
b := vecBuilder[int];
b2 := merge(b, 5);
b3 := merge(b, 6);
res(b3)    # returns [5, 6]
```

```
# basic use of merger
b := merger[int, 0, (a:int,b:int)=>a+b, (a:int,b:int)=>a+b];
b2 := merge(b, 5);
b3 := merge(b, 6);
res(b3)    # returns 11
```

```
# for expression with vecBuilder
b := vecBuilder[int];
data := [1, 2, 3];
b2 := for(data, b, (b: vecBuilder[int], n: int) => merge(b, 2 * n));
res(b2)    # returns [2, 4, 6]
```

```
# for expression with composite builder
b0 := vecBuilder[int];
b1 := vecBuilder[int];
data := [1, 2, 3];
bs := for(
  data,
  {b1, b2},
  (bs: {vecBuilder[int], vecBuilder[int]}, n: int) =>
    {merge(bs.0, n), merge(bs.1, 2 * n)}
);
res(bs)    # returns {[1, 2, 3], [2, 4, 6]}
```

### Linearity of Builder Types

We want to place a few constraints on builders to make them easier to implement and make their semantics clear.
First, for builders to have clear semantics in NVL, we need to make sure that `res` is not called on a builder while parallel work is still happening on it.
Otherwise, the language may have to be nondeterministic, which is not something we want for this version.
Second, for simplicity of implementation, we will also make sure that each builder is used in a *linear* sequence of operations (`merge`s and `for`s followed by at most one `res`), which will let us update the underlying memory in place instead of having to "fork" it if one derives two builders from it.
Likewise, we will enforce that the `update` function in a `for` always returns a builder derived from the one passed in as a parameter, and not, say, some kind of new builder it initialized inside.
This will help coordinate parallel execution and memory management for `for`s.

All these constraints can be enforced by making builders a [linear type](https://en.wikipedia.org/wiki/Substructural_type_system).
In particular, we will do the following:

* Every variable that represents a builder is passed as an argument to exactly one `merge`, `for`, `res`, or let expression (e.g. `b2 := b1`) in its scope, or is returned from its scope.
* In any `for` expression's `update` function, the builder returned by the function is *derived* from the one that it got as an argument. By *derived*, we mean that there is a sequence of `for` and `merge` expressions that produces the resulting builder from the argument on any control flow path through the function.

The one place where the situation is trickier is with structs of builders.
Here, we require that each field of the resulting struct is derived from the corresponding field of the struct passed as an argument, but differnet fields may pass through different expressions through the function.
It is less clear whether existing type systems capture this, but it should not be difficult to define one for it.

## Sugar Operations

To make programs easier to write, the NVL implementation also supports some "sugar" operations that translate into `for`s and builders.
These are currently represented by nodes in the AST, but are removed immediately by the `LoopConverter` program transformation, so none of the other transformations or code generation backends consider them.
The sugar operations are commonly used functional programming operations such as `map` and `filter`.
We list them below:

Signature | Notes
------------- | -------------
`map(v: vec[T], f: T => U): vec[U]` |
`filter(v: vec[T], f: T => bit): vec[T]` |
`flatten(v: vec[vec[T]]): vec[T]` |
`zip(v1: vec[T1], v2: vec[T2], ...): vec[{T1, T2, ...}]` |
`cross(v: vec[T], w: vec[U]): vec[{T, U}]` | Cartesian product
`agg(v: vec[T], zero: U, mergeValue: (U, T) => U, mergeRes: (U, U) => U): U` | Data will be split into partitions arbitrarily
`agg(v: vec[T], zero: T, merge: (T, T) => T)` | Simpler version of merge when `T = U`
`aggBy[v: vec[T], key: T => K, zero: U, build: (U, T) => U, merge: (U, U) => U]: dict[K, U]` |
`groupBy(v: vec[T], key: T => K, value: T => V): dict[K, vec[V]]` |
`update(d: dict[K, V], u: vec[{K, V}], func: (V, V) => V): dict[K, V])` | Update certain entries in a dict based on a list of pairs (returns a new dict). If a key is missing, just add the new value.
`update(v: vec[T], u: vec[{long, T}], func: (V, V) => V): vec[T])` | Update certain entries in a vec (returns a new vec).

All of these operations can straightforwardly be translated into `for` expressions.
For example, `map` and `filter` would be implemented as follows:

```
map(data: vec[T], f: T => U) :=
  res(for(data, vecBuilder[U], (b: vecBuilder[U], t: T) =>
    merge(b, f(t))))
```

```
filter(data: vec[T], f: T => bit) :=
  res(for(data, vecBuilder[T], (b: vecBuilder[T], t: T) =>
    if(f(t), merge(b, t), b)))
```

In future versions, a macro system that lets us express these operations may be useful.
