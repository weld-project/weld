# Weld Language Overview

## Overview

Weld is a statically typed, referentially transparent language with built-in parallel constructs.
It has a number of operators similar to functional languages, but it is not truly functional, in that functions aren't first-class values and recursion is not allowed.
As a result, Weld programs have a finite call graph that is known in advance and are straightforward to analyze.
By "referentially transparent", we mean that each expression in Weld is a pure function of its inputs, and there are no side-effects.

The language lets users specify functions and expressions.
A function, such as `|a: i32, b: i32| a + b`, consists of a list of named arguments and an expression for the result.
Variables in Weld are immutable, though there is a "let" statement for introducing new ones.
Some operators built into the language take functions (in fact closures), but functions are not first-class values (one cannot pass them to other functions or store them in variables). Types for the input arguments to a function are required, but can otherwise be inferred. 

Weld contains both a "core" language and higher-level "sugar" syntax for specifying common functional operators, such as `map` and `filter`.
The core language has only one parallel construct, the `for` expression, and a set of types called *builders* used to compute various types of results (e.g. sums, vectors, etc).
All of the sugar operators are translated to `for`s and builders through simple substitution rules.
We will begin by describing the core language, and then describe the currently supported sugar operators.

## Data Types

Weld contains "value" types that hold data, as well as "builder" types that are used to construct results in parallel from values that are "merged" into them.

The value types are:

* Scalars: `bool`, `i8`, `i32`, `i64`, `f32`, `f64`.
* Vectors: `vec[T]` for some type `T`. These are variable-length.
* Dictionaries: `dict[K, V]` for types K, V.
* Structs: `{T1, T2, ...}` for field types T1, T2, etc.

The builder types are:

* `appender[T]`: Builds a `vec[T]` from elements of type `T`.
* `merger[T,bin_op]`: Combines `T` values using a binary operation. Its parameters are:
   * `T`: The type of value this merger creates. Can be a scalar or a struct of scalars.
   * `bin_op`: A commutative binary operation (currently supports `+` and `*`). The operation is applied to structs elementwise.
* `dictmerger[K,V,bin_op]`: Combines `{K, V}` pairs by key into a dictionary. The parameters are:
   * `K`: Key type. Can be any value.
   * `V`: Value type. Can be a scalar or a struct of scalars.
   * `bin_op`: A commutative binary operation (currently supports `+` and `*`) for the value. The operation is applied to structs elementwise.
* `vecmerger[T,bin_op]`: Combines `{long, T}` pairs by key into a vector using `bin_op`. The builder is initialized with an initial vector to work with.
   * `T`: The vector element type of value this `vecmerger` creates. Can be a scalar or a struct of scalars.
   * `bin_op`: A commutative binary operation (currently supports `+` and `*`). The operation is applied to structs elementwise.

* Any struct whose fields are builders can also be used as a builder. This is used to build multiple results at the same time.

Note that among the builders that take functions and or values, two types are identical only if they're parameterized with the same operators / values.
Making functions part of the type is slightly unusual because the functions passed in could be closures over some variables that exist at that point in the program.
Right now the implementation doesn't allow this, and we have not tried to formalize what it means.

## Core Operations

The core language consists of the following expressions:

* Literals, e.g. `5.0`, `{6, 7}`, and `[1,2,3]`.

  Type | Syntax
  ------------- | -------------
  `i8` | `1c`, `1C`
  `i32` | `1`
  `i64` | `1l`, `1L`
  `f32` | `1.0f`, `1.0F`
  `f64` | `1.0`

* Arithmetic expressions, e.g. `a + b`, `a - b`, `-a`, `a & b`, etc.
* `if(condition, on_true, on_false)`, which evaluates `on_true` or `on_false` based on the value of `condition`.
* Let expressions, which introduce a new variable. The syntax for these is `let name = expr; body`. This evaluates `expr`, assigns it to the variable `name`, and then evaluates `body` with that binding and returns its result.
* `cudf[name,ty](args)` to call arbitrary C-style functions (see a discussion of UDFs [below](#user-defined-functions)).
* Collection expressions:
  * `lookup(dict, key)` and `lookup(vec, index)` return an element from a dictionary and vector respectively.
  * `len(vec)` return its length.
  * `struct.$0`, `struct.$1`, etc are used to access fields of a struct.
  * `tovec(dict)` gets the entries of a dictionary as a vector of `{K, V}` pairs.
* Three special expressions involving builders:
  * `merge(builder, value)` returns a new builder that incorporates `value` into the previous builder.
  * `result(builder)` computes the result of the builder given the values merged so far.
  * `for(vec, builder, update)` applies a function `update` to every element of a vector, possibly merging values into a builder for each one, and returns a final builder with all the merges incorporated. `vec` must be of type `vec[T]` for some `T`, `builder` can be any builder type `B`, and `update` must be a function of type `(B, I, T) => B` that possibly merges values into the `B` passed in. `I` is the index of the element being processed.
  * `zip(vec[T1], vec2[T2], ..)` returns a `vec[{T1, T2, ..}]`. This expression is special because it can only be used in the `for` loop.
  * `iter(data, start, end, stride)` returns a vector with certain elements skipped. `data` is a `vec[T]` with for some type `T`. `start`, `end`, and `stride` represent the start index, end index, and stride of the iteration respectively. This expression is special because it can only be used in the `for` loop.

The builder expressions are the only "interesting" ones, where parallelism comes in.
The basic idea is that a builder is a "write-only" data structure, and the `result` operation turns it into a read-only value.
Because builders are "write-only", it is okay to merge values into them from different iterations of a `for` loop in parallel.
Note that this requires that one does not call `result` on a builder inside a loop.
In reality, we can enforce this by making builders "linear types", and requiring that the `update` function in `for` return a builder derived from its argument.
Linear types are a concept in programming languages that we'll talk about below.
Our implementation does not statically enforce linearity, but it only works if `update` functions really do only return builders derived from their arguments, and if `result` is only called on each builder once.

### Example

Here are a few simple examples using builder expressions:

```
# basic use of appender
let b = appender[i32];
let b2 = merge(b, 5);
let b3 = merge(b2, 6);
result(b3)    # returns [5, 6]
```

```
# basic use of merger
let b = merger[i32,+];
let b2 = merge(b, 5);
let b3 = merge(b2, 6);
result(b3)    # returns 11
```

```
# for expression with appender
let b = appender[i32];
let data = [1, 2, 3];
let b2 = for(data, b, |b: appender[i32], i: i64, n: i32| merge(b, 2 * n));
result(b2)    # returns [2, 4, 6]
```

```
# for expression with appender and iter (only loop over first three elements)
let b = appender[i32];
let data = [1, 2, 3, 4, 5, 6];
let b2 = for(iter(data, 0L, 3L, 1L), b, |b: appender[i32], i: i64, n: i32| merge(b, 2 * n));
result(b2)    # returns [2, 4, 6]
```

```
# for expression with composite builder
let b0 = appender[i32];
let b1 = appender[i32];
let data = [1, 2, 3];
let bs = for(
  data,
  {b1, b2},
  (bs: {appender[i32], appender[i32]}, i: i64, n: i32) =>
    {merge(bs.$0, n), merge(bs.$1, 2 * n)}
);
result(bs)    # returns {[1, 2, 3], [2, 4, 6]}
```

### Linearity of Builder Types

We want to place a few constraints on builders to make them easier to implement and make their semantics clear.
First, for builders to have clear semantics in Weld, we need to make sure that `result` is not called on a builder while parallel work is still happening on it.
Otherwise, the language may have to be nondeterministic, which is not something we want for this version.
Second, for simplicity of implementation, we will also make sure that each builder is used in a *linear* sequence of operations (`merge`s and `for`s followed by at most one `result`), which will let us update the underlying memory in place instead of having to "fork" it if one derives two builders from it.
Likewise, we will enforce that the `update` function in a `for` always returns a builder derived from the one passed in as a parameter, and not, say, some kind of new builder it initialized inside.
This will help coordinate parallel execution and memory management for `for`s.

All these constraints can be enforced by making builders a [linear type](https://en.wikipedia.org/wiki/Substructural_type_system).
In particular, we will do the following:

* Every variable that represents a builder is passed as an argument to exactly one `merge`, `for`, `result`, or let expression (e.g. `let b2 = b1`) in its scope, or is returned from its scope.
* In any `for` expression's `update` function, the builder returned by the function is *derived* from the one that it got as an argument. By *derived*, we mean that there is a sequence of `for` and `merge` expressions that produces the resulting builder from the argument on any control flow path through the function.

The one place where the situation is trickier is with structs of builders.
Here, we require that each field of the resulting struct is derived from the corresponding field of the struct passed as an argument, but different fields may pass through different expressions through the function.
It is less clear whether existing type systems capture this, but it should not be difficult to define one for it.

## Sugar Operations

To make programs easier to write, the Weld implementation also supports some "sugar" operations that translate into `for`s and builders.
These are currently represented as *macros*, which are substitution rules whose definitions are not handled by the optimizer.
The sugar operations are commonly used functional programming operations such as `map` and `filter`.
We list them below:

Signature | Notes
------------- | -------------
`map(v: vec[T], f: T => U): vec[U]` |
`filter(v: vec[T], f: T => bit): vec[T]` |
`flatten(v: vec[vec[T]]): vec[T]` |
`zip(v1: vec[T1], v2: vec[T2], ...): vec[{T1, T2, ...}]` | Only allowed in the `vec` argument of the `for` loop.

All of these operations can straightforwardly be translated into `for` expressions.
For example, the macro rules for `map` and `filter` would be implemented as follows:

```
macro map(data, func) = (
  result(for(data, appender, |b, i, x| merge(b, func(x))))
);
```

```
macro filter(data, func) = (
  result(for(data, appender, |b, i, x| if(func(x), merge(b, x), b)))
);
```

## User Defined Functions

Weld supports invoking C-style UDFs from a Weld program. The `cudf[name,ty](arg1, arg2,...argN)` node enables this; `name` is a C symbol name which refers to a function in the same address space (e.g., a function in a dynamically loaded library), `ty` is the Weld return type of the UDF, and `arg1, arg2,...,argN` is a list of zero or more argument expressions.

C UDFs require a special format within C code. In particular, a valid C UDF must meet the following requirements:
 * The function has a `void` return type
 * Each argument passed to the C UDF is a pointer. For example, A UDF which takes one argument `arg1: T1` must have its first argument be `T1*`.
 * The last argument is a pointer to the return type. Weld allocates space for the return type struct; the UDF just needs to write data back to this pointer. However, buffers which the return type itself contains *are not managed by Weld*. For example, if UDF returns a vector, the `{T*, int64_t}` struct representing the vector is owned by Weld, but the `T*` buffer is not.
 
 Note that C UDFs must take as input types understood by the Weld runtime; see the [API documentation](https://github.com/weld-project/weld/master/docs/api.md) for how each type looks in memory.
 
 ### Examples
  
The UDF `cudf[add_five,i64](x:i64)` takes one argument of type `i64` and returns an `i64`:

```c
extern "C" void add_five(int64_t *x, int64_t *result) {
  *result = *x + 5;
}
```

The UDF `cudf[fast_matmul,vec[f32]](a:vec[f32], b:vec[f32])` takes two arguments of type `vec[f32]` and returns an `vec[f32]`:

```c
typedef struct float_vec {
  float *data;
  int64_t length;
} float_vec_t;

extern "C" void fast_matmul(float_vec_t *a, float_vec_t *b, float_vec_t *result) {
  // this malloc'd memory is owned by the caller, but can be passed to Weld.
  // Weld treats the memory as "read-only".
  result->data = malloc(sizeof(float) * a->length);
  result->length = a->length;
  // can call any arbitrary C code in a UDF.
  my_fast_matrix_multiply(a->data, b->data, result->data, a->length);
}
