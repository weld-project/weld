# Weld Language Overview

# Contents

- [Overview](#overview)
- [Data Types](#data-types)
  * [Value Types](#value-types)
  * [Builder Types](#builder-types)
    + [Commutative Binary Operations for Builders](#commutative-binary-operations-for-builders)
- [Core Operations](#core-operations)
  * [Basic Expressions](#basic-expressions)
  * [Expressions on Collections (Vectors, Dictionaries, Structs)](#expressions-on-collections-vectors-dictionaries-structs)
  * [Builder Expressions](#builder-expressions)
    + [Iterators in For Loops](#iterators-in-for-loops)
    + [About Builders](#about-builders)
    + [Examples of Builders](#examples-of-builders)
    + [Aside: Linearity of Builder Types](#aside-linearity-of-builder-types)
- [Comments](#comments)
- [Type Inference](#type-inference)
- [Sugar Operations](#sugar-operations)
- [User Defined Functions](#user-defined-functions)
    + [Examples](#examples)
- [Annotations](#annotations)

# Overview

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

This doc describes the types, operators, and basic features of the Weld language.

# Data Types

Weld contains "value" types that hold data, as well as "builder" types that are used to construct results in parallel from values that are "merged" into them.

## Value Types

* Scalars: `bool`, `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `f32`, `f64`. Scalars prefixed with `i` are signed, and ones prefixed with `u` are unsigned.
* SIMD values `simd[S]` for some *scalar type* `S`. The length of a SIMD value is currently platform dependent and chosen automatically.
* Vectors: `vec[T]` for some type `T`. These are variable-length (i.e., their length is not known at compile time).
* Dictionaries: `dict[K, V]` for types `K`, `V`.
* Structs: `{T1, T2, ...}` for field types `T1`, `T2`, etc.

Except from the SIMD type `simd[S]` (where `S` must be a scalar type), `T` in the types above can be any other type.

## Builder Types

* `appender[T]`: Builds a `vec[T]` from elements of type `T`.
* `merger[T,binop]`: Combines `T` values using a binary operation. Its parameters are:
   * `T`: The type of value this merger creates. Can be a scalar or a struct of scalars.
   * `binop`: [A commutative binary operation](#commutative-binary-operations-for-builders)
* `dictmerger[K,V,binop]`: Combines `{K, V}` pairs by key into a dictionary. The parameters are:
   * `K`: Key type. Can be any type.
   * `V`: Value type. Can be a scalar or a struct of scalars.
   * `binop`: [A commutative binary operation](#commutative-binary-operations-for-builders)
* `groupbuilder[K,V]`: Groups `{K, V}` by key in a dictionary. Used to produce a `dict[K,vec[V]]`.
   * `K`: Key type. Can be any type.
   * `V`: Value type. Can be any type.
* `vecmerger[T,binop]`: Combines `{i64, T}` pairs by key into a vector using `binop`. The builder is initialized with an initial vector to work with.
   * `T`: The vector element type of value this `vecmerger` creates. Can be a scalar or a struct of scalars.
   * `binop`: [A commutative binary operation](#commutative-binary-operations-for-builders)
* Any struct whose fields are builders can also be used as a builder. This is used to build multiple results at the same time.

### Commutative Binary Operations for Builders

For the builder types above, the supported commutative `binop` values are `+`,
`*`, `min`, and `max`. This operator is applied element-wise on
struct-of-scalar values. Builders are initialized with a default initial value based on the `binop`:

Binary Operator | Initial Value for Scalar or Each Struct Element
------------- | -------------
`+` | `0` |
`*` | `1` |
`min` | maximum value possible for scalar type |
`max` | minimum value possible for scalar type

Note that among the builders that take binary operators and merge-types, two
builder types are identical only if they're parameterized with the same
operators _and_ values.

# Core Operations

The core language consists of the following expressions. `E1` ... `En` refer to
a subexpression, which can be any of the operators below. `T`, `U`, and `V`
refer to types.

## Basic Expressions

* Literals, e.g. `5.0`, `{6,7}`, and `[1,2,3]`.

  Type | Syntax
  ------------- | -------------
  `bool` | `true`, `false`
  `i8` | `1c`, `1C`
  `i16` | `1si` (short int)
  `i32` | `1`
  `i64` | `1l`, `1L`
  `f32` | `1.0f`, `1.0F`
  `f64` | `1.0`
  `vec[T]` | `[ E1, E2, ...`
  structs | `{ E1, E2, ... }`

  Literals for other types are not supported. [Submit a pull request](https://github.com/weld-project/weld/pulls) if you see something missing that you would like supported!

* Binary operators expressed as `E1 + E2` or `op(E1, E2)`  The supported ones are:
  `+`, `-`, `*`, `/`, `>`, `<`, `>=`, `<=`, `==`, `!=`, `&&`, `&` (bitwise-and), `||`, `|` (bitwise-or), `^` (bitwise-xor), `min`, `max`, `pow`.
* Unary operators expressed as `op(E)`. The supported ones are:
  `exp`, `log`, `sqrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, and `erf`. These follow the behavior of the equivalent C function from `math.h`.
* Let expressions, which introduce a new variable. The syntax for these is `let name = E1; E2`.
  This first evaluates `E1`, assigns it to the variable `name`, and then evaluates `body` with that binding and returns its result.
* `if(condition, on_true, on_false)`, which evaluates `on_true` or `on_false` based on the value of `condition` (which must be of type `bool`).
* `select(condition, on_true, on_false)`, which evaluates `condition`, `on_true` and `on_false` unconditionally and returns `on_true` or `on_false` based on the result of `condition`.
* `iterate(initial_value, update_func)`, which performs a sequential loop.
  `initial_value`  can be any type `T`, and `update_func` must be a Weld
  function of type `T => {T, bool}`. We call `update_func` repeatedly on the
  value until the boolean it returns is `false`, and then return the `T` field in
  its output as the final value of the expression.
* `cudf[name,ty](args)` to call arbitrary C-style functions (see a discussion of UDFs [below](#user-defined-functions)).
* `serialize(data)` serializes `data` into a `vec[i8]`. The data in this vector can be written to disk, sent over the network, etc.
* `deserialize[T](data)` deserializes `data` (a `vec[i8]`) into a value of type `T`.
* Casting: `T(data)` implements a cast between scalar types if `T` is a scalar and `data` is also a scalar type.
* `broadcast(data)` takes a scalar value `data` and broadcasts the value into a SIMD type.

## Expressions on Collections (Vectors, Dictionaries, Structs)

* `lookup(dict, key)` and `lookup(vec, index)` return an element from a dictionary and vector respectively. `index` must be of type `i64`. It is an error to call `lookup` on a dictionary
  with a key that does not exist: see `keyexists`.
* `optlookup(dict, key)` batches `keyexists` and `lookup` into a single call. This can be more efficient since the key only needs to be hashed a single time. This operator returns `{bool, V}` (`V` is the value type) where the boolean indicates whether the key was present in the dictionary. If the boolean is false, it is an error to access `V`; although this is not enforced at the moment, the type system may be extended to support it eventually (e.g., by adding an `option` type).
* `keyexists(dict, key)` returns whether the `key` is in `dict`.
* `len(vec)` return its length as an `i64`.
* `slice(vec, index, size)` creates a view into a vector without allocating memory starting at `index` and containing `size` elements. Both must be of type `i64`.
* `sort(vec, func)` sorts a vector. `func` is of type `|T, T| => i32`, where `T` is the input vector's element type. The function returns a positive `i32` if `left > right`, a negative integer if `left < right`, and zero if `left == right`. By default, using the comparison binary operators, vectors are compared lexigraphically and structs are compared field-by-field from left to right. Sorting on vectors of dictionaries, builders, and SIMD values is currently disallowed.
* `struct.$0`, `struct.$1`, etc. are used to access fields of a struct.
* `tovec(dict)` gets the entries of a dictionary as a vector of `{K, V}` pairs.

## Builder Expressions
  * `merge(builder, value)` returns a new builder that incorporates `value` into the previous builder. This returns a new updated builder.
  * `result(builder)` computes the result of the builder given the values merged so far.
  * `for(vec, builder, update)` applies a function `update` to every element of a vector, possibly merging values into a builder for each one, and returns a final builder with all the merges incorporated. `vec` must be of type `vec[T]` for some `T` (see caveats in the section about [iterators](#iterators-in-for-loops), `builder` can be any builder type `B` (see [builder types](#builder-types), and `update` must be a function of type `(B, I, T) => B` that possibly merges values into the `B` passed in. `I` is the `i64` index of the element being processed.

### Iterators in For Loops

For loops support iteration over multiple vectors at once, ranges of vectors, vectors that are treated as N-dimensional tensors, and over ranges of indices without a vector). These features are
enabled via _iterators_, which are special expressions that can only be used in the first argument of a `for` loop. They are described below:

* `zip(vec[T1], vec2[T2], ..)` iterates over a `vec[{T1, T2, ..}]`. The vectors may be over other iterators (described below). Each iterator *must consume the same number of elements.*
* `iter(data, start, end, stride)` iterates over a vector with certain elements skipped. `data` is a `vec[T]` with for some type `T`. `start`, `end`, and `stride` represent the start index, end index, and stride of the iteration respectively.
* `simditer(data)` iterates until the last multiple of  `sizeof(simd[T])`. For example, in a vector with 13 elements, if a single SIMD type holds 4 elements, the `simditer` will consume elements 0-11.
* `fringeiter(data)` iterates over the portion of the vector that the `simditer` does not. From the above example, this iterator would consume only the last element.
* `rangeiter(start, end, stride)` iterates over a range of integers based on the `start`, `end`, and `stride` expressions. The `rangeiter` emits elements of type `i64`. In the for loop function, the second argument of the function when using a `rangeiter` is the
iteration number, while the third argument is the value produced by the iterator, so most programs will want to access the third argument.

### About Builders

A builder is a "write-only" data structure, and the `result` operation turns it
into a read-only value.  Because builders are "write-only", it is okay to merge
values into them from different iterations of a `for` loop in parallel.  Note
that this requires that one does not call `result` on a builder inside a loop.
In reality, we can enforce this by making builders "linear types", and
requiring that the `update` function in `for` return a builder derived from its
argument. Linear types are a concept in programming languages that we'll talk
about below. Our implementation does not statically enforce linearity, but it
only works if `update` functions really do only return builders derived from
their arguments, and if `result` is only called on each builder once.

### Examples of Builders

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
# for with composite builder
let b1 = appender[i32];
let b2 = appender[i32];
let data = [1, 2, 3];
let bs = for(
  data,
  {b1, b2},
  |bs: {appender[i32], appender[i32]}, i: i64, n: i32|
    {merge(bs.$0, n), merge(bs.$1, 2 * n)}
);
{result(bs.$0), result(bs.$1)} # returns {[1, 2, 3], [2, 4, 6]}
```

### Aside: Linearity of Builder Types

We want to place a few constraints on builders to make them easier to implement and make their semantics clear.
First, for builders to have clear semantics in Weld, we need to make sure that `result` is not called on a builder while parallel work is still happening on it.
Otherwise, the language may have to be non-deterministic, which is not something we want for this version.
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

# Comments

Weld supports Python-style one line comments with the `#` character. For example:

```
# A function that adds two vectors.
|v1 :vec[i32], v2: vec[i32]|
  map(zip(v1,v2),
    |e| # A struct of type {i32,i32}
      e.$0 + e.$1
  )
```


# Type Inference

Weld supports some basic type inference, so users do not need to specify a full type for each expression in the program (but may optionally choose to do so).
In particular, Weld only requires types for the top-level function arguments, and can generally infer types for most other expressions. For example:

```
|v: vec[i32]| # Define type here
  result(for(v,
            appender, # type of appender is inferred to be appender[i64]
            |b,i,e| # type of function arguments is inferred
              merge(b, i64(e))
        )
      )
```

# Sugar Operations

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
`compare(x: T, y: T)` | Implements a default comparator for `sort`. Expands to `if(x > y, 1, if(x < y, -1, 0))`.

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

# User Defined Functions

Weld supports invoking C-style UDFs from a Weld program. The `cudf[name,ty](arg1, arg2,...argN)` node enables this; `name` is a C symbol name which refers to a function in the same address space (e.g., a function in a dynamically loaded library), `ty` is the Weld return type of the UDF, and `arg1, arg2,...,argN` is a list of zero or more argument expressions.

C UDFs require a special format within C code. In particular, a valid C UDF must meet the following requirements:
 * The function has a `void` return type
 * Each argument passed to the C UDF is a pointer. For example, A UDF which takes one argument `arg1: T1` must have its first argument be `T1*`.
 * The last argument is a pointer to the return type. Weld allocates space for the return type struct; the UDF just needs to write data back to this pointer. However, buffers which the return type itself contains *are not managed by Weld*. For example, if UDF returns a vector, the `{T*, int64_t}` struct representing the vector is owned by Weld, but the `T*` buffer is not.
 
 Note that C UDFs must take as input types understood by the Weld runtime; see the [API documentation](https://github.com/weld-project/weld/blob/master/docs/api.md) for how each type looks in memory.
 
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
```
# Annotations

In addition, it's possible to specify annotations on both builder types and expressions: these could for example specify an implementation strategy for a builder. To specify an annotation on a `dictmerger`, one can use syntax like

```
@(name1:value1, name2:value2, ...) dictmerger[K,V,bin_op]
```

Annotations need to be specified before the builder type or expression, and
multiple annotations need to be comma-separated. Annotations are unstructured
string to string maps, and their definition and behavior is dependent on the
transforms and backends that use them.

In addition, we support the following annotations on generic expressions:
* `predicate`: Specifies whether the expression should be predicated or not -- value must be a `bool`.
* `vectorize`: Specifies whether the expression should be vectorized or not -- value must be a `bool`.
* `size`: Specifies the size of the expression -- value must be a `i64`.
