# WeldNumpy

## Contents

  * [Setup](#setup)
  * [Overview](#overview)
  * [Documentation](#documentation)
  * [Developer Documentation](#developer-documentation)
  * [Future Work](#future-work)
  * [Similar Projects](#similar-projects)

## Setup

### pip
TODO: Need to setup pip install 

### Installing from source
TODO: will need to install weld + grizzly first and set up environment
variables etc. before installing weldnumpy?

After installation, you should be able to verify it by running the tests from
the root directory with:

```bash
$ pytest 
```

## Overview

WeldNumpy is a library that provides a subclass of NumPy's ndarray module,
called weldarray, which 
supports automatic parallelization, lazy evaluation, and various other
optimizations for data science workloads based on the [Weld project](https://weld.rs).
This is achieved by implementing various NumPy operators in Weld's Intermediate
Representation (IR). Thus, as you operate on a weldarray, it will internally
build a graph of the operations, and pass them to weld's runtime system to
optimize and execute in parallel whenever required. 

In [examples](examples), you can see improvements of upto 5x on a single
thread on some NumPy workloads, essentially without changing any code from the
original NumPy implementations. Naturally, much bigger performance gains can be
got by using the parallelism provided by Weld. In general, Weld works well with
programs that operate on large NumPy arrays with compute operations that are
supported by Weld. 

NumPy features that WeldNumpy currently supports are:
* Supported operations include:
    * Unary Operations: np.exp, np.log, np.sqrt, np.sin, np.cos, np.tan,
    np.arccos, np.arcsin, np.arctan, np.sinh, np.cosh, np.tanh, scipy.special.erf
    * Binary Operations: np.add, np.subtract, np.multiply, np.divide
* Supported types are: np.float64, np.float32, np.int64, np.int32
* In general, the operations work over multi-dimensional arrays, including
non-contiguous arrays. But for inplace updates, there are more subtleties
involved, as described later.
* Supports reductions over 1d arrays
* Supports [broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
* All unsupported operations/methods on the weldarray are offloaded to NumPy -
so NumPy programs should still function correctly. After the NumPy methods
are executed, a weldarray is returned.

One of the goals of this library is to require minimal changes to your original
NumPy source code in order to harness the optimizations provided by Weld. Thus,
you can write NumPy programs in the same style, and WeldNumpy will only
evaluate those when neccessary.

In general, it may be more flexible to not always utilize weldarray - for
instance with relatively small sized NumPy arrays, the overhead of compiling
the Weld programs will be more expensive than the total NumPy execution times.

The biggest speed benefit from using WeldNumPy v/s NumPy is just that the Weld
code can be automatically parallelized. But even with single threaded code,
 there are various tricks that Weld uses in order to get the most out of
 the performance. Let us now look at a few basic usage examples that also
 highlight some of these benefits provided by WeldNumpy.

#### Materialization 

Because python is eagerly evaluated, it has to materialize every object in
memory even if it is only going to live for a short duration as part of other
calculations. By using lazy evaluation, WeldNumpy can avoid such costs. For
instance:

```python
import weldarray from weldnumpy
import numpy as np

a = weldarray(np.random.rand(1000000))
b = weldarray(np.random.rand(1000000))
c = weldarray(np.random.rand(1000000))

d = a*b + c
d += 100.00
# All the above operations on 'd' can again be fused together into a single
# loop which would avoid wasting memory on intermediate results like a*b etc.

# To cause the evaluation, one can do:
d = d.evaluate()
```

#### Loop Fusion

Again, because of lazy evaluation, WeldNumpy can look across the program, and
find multiple loops that go over the same ndarray - which can then be converted
into a single loop. Here is another contrived example:

```python
# Another way to use weldnumpy is to change the import statement at the top of
# the NumPy file. In general, the WeldNumpy class just serves a wrapper around
# NumPy functions - with the array creation routines modified to return 
# weldarray's instead. 

import weldnumpy as np

# These statements would return weldarray's, because the weldnumpy class
# provides wrapper functions for all array creation routines
a = np.random.rand(100000000)
b = np.random.rand(100000000)

for i in range(10):
    a += b

# Because weld is lazily computed, until this point, no computation has actually occurred. Now, if you choose to
# access the array, 'a' in any way, or call, a = a.evaluate(), then the stored
# operations on 'a' will be evaluated in weld. Looking at the complete program
# will let Weld apply optimizations like loop fusion, which would essentially
# convert the above program to:
#   a = ((((a+b)+b)+b)+ ... b)
# which clearly saves A LOT of loops compared to a traditional NumPy program.
# This will also be executed in parallel if specified.

print(a)    # since print accesses elements of a, internally it will call a.evalaute()
```

Changing the import statement as in the previous example, may serve as a quick
method to use weldnumpy, but in general for a large program, it makes sense to
only import weldarray and convert only the large arrays whose operations you
want to optimize, as in the first example above. More detailed examples with
comments are provided in [examples](examples).

#### Redundant Computations

```python

a = np.random.rand(10000)
# Here, NumPy would evaluate a**2 twice. But WeldNumpy will only need to
# evaluate it once - and then for the second a**2, the value would have been
# stored. 
b = a**2 + a**2 

# forces evaluation
b = b.evaluate()
```

Notice that in the example above, it seems like a very easy error to spot. But
in general, such redundant computations can get arbitrarily complicated, and it
is nice to be able to automatically eliminate these.

## Documentation

In general, the semantics for the operations on weldarray's are exactly the
same as the equivalent operations on NumPy arrays. The only extra operation on
a weldarray is: weldarray.evaluate() which forces the evaluation of all the
operations registered on the weldarray.

#### Differences with NumPy

##### Compilation Costs 
In general, there is a slight overhead for compiling the Weld IR to LLVM before
it can be executed. If the array sizes are large enough, then these compilation
costs add little overhead as compared to computations. But this also means that
if you are using NumPy with small arrays, then there would be little to no use
of WeldNumpy.

##### Lazy Evaluation

Weld is a lazily evaluated IR - i.e., when a program line is encountered, it is
not neccessarily executed. Instead, these operations are just stored as
metadata in the weldarray, so that they could be executed at a later time with
the intention that certain optimizations (like the ones described above) will
suddenly become possible. NumPy doesn't do this, so the challenge is to present
the same interface as NumPy without explicitly using lazy evaluation.

###### Implicit Evaluation

In general, if you print an array / or access it in some other way without
explicitly evaluating it, you will still see the correct results because
weldnumpy will implicitly evaluate it. For example:

```python
from weldnumpy import weldarray
import numpy as np
w = weldarray(np.random.rand(100000))
w2 = weldarray(np.random.rand(100000))
w += w2
# this will print latest value of w[0], and also cause evaluation of the whole
# w array.
print(w[0])
```

Another use case for implicit evaluation is when we offload operations to
NumPy. Since NumPy expects to operate on the memory of the ndarray like object
- we need to ensure that this memory represents the latest values of the array
object. In order to do this, we always implicitly evaluate the array when
offloading computations to NumPy. 

###### Evaluate

This is one of the two public functions on weld arrays that is different from
the NumPy operations. It will just evaluate all the operations registered with the
weldarray, and return an updated weldarray. In general, calling evaluate
multiple times on the same weldarray does not affect it.

```python
w = w + n
# assuming 'w' is a weldarray with stored operations, like "+ n", this will
# update the # weldarray to it's latest values.
w = w.evaluate()
```

As mentioned before, WeldNumPy also has a concept of implicit evaluation. But
these implicit evaluations are triggered based on heuristics - e.g., when we
are able to intercept a call to print. Thus, there can be many cases where the
user might want to force evaluation of all stored operations on a weldarray.
One crucial case that can lead to a subtle bug is when the weldarray is passed
to other unknown functions, in particular, some of NumPy's functions like
np.array_equal: 

```python
from weldnumpy import weldarray
import numpy as np
w = weldarray(np.random.rand(100))
n = np.random.rand(100)
w2 = w + 100.00
n2 = n + 100.00

# Now, clearly the two arrays n2, and w2, should be equal. And this is
# confirmed by:
np.allclose(n2, w2) # returns True

# But array_equal returns False, because array_equal, internally casts any array like
# object it gets into an array with its base memory. Since the w2 = w + 100.00
# operation above did not create any new memory for w2, the base array for w2
# is also w -- thus in array_equal NumPy actually downcasts w2 to this base
# array before doing the further calculations. 
np.array_equal(n2, w2)  # returns False!

# Instead if we do:
np.array_equal(n2, w2.evaluate())   # returns True!
# Or:
w2 = w2.evaluate()
np.array_equal(w2, n2)      # returns True

# Then in both these cases, the operations stored on w2 do get evaluated which
# makes Weld give it its own memory - thus casting it to its base array in
# array_equal only casts it to the 'correct' memory for this case.
```

We can also see similar things happening with other NumPy functions like
np.random_choice etc. Note: Downcasting to the base array is not the behaviour
for most NumPy functions, but there are a few that just do this. In general,
the simplest thing might be to remember to evaluate the arrays before sending
it into an unknown NumPy function.

###### Group Evaluate

TODO: Finalize syntax

##### Things that don't quite work

In general, most common functions on a ndarray are routed by NumPy through the
weldarray subclass - thus things like [universal
functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html), np.reshape,
np.T etc. just work as expected. And if there are some unsupported functions
(like reductions over multi-dimensional arrays), then these are just offloaded
to NumPy, and the result is converted to a weldarray before returning - thus
everything works as expected.

But we have not exhaustively looked through all the functions that NumPy
provides, so potentially, there could be some other issues. One example
scenario is is when you use import weldnumpy as np: This requires us to add
wrapper functions for various NumPy functions in WeldNumPy. We do this by using
a blunt import * - but there are functions that can be missed, for instance:

* np.max: This is just an alias for np.amax - but the alias' are not imported
with import * - thus np.max would not be found if you use import weldnumpy as
np in your project. Of course, np.max could be easily added as an alias - but
we believe that there are potentially many other such NumPy functions - and
rather than aliasing all of them, we suggest you use WeldNumpy by just using
weldarray: import weldarray from weldnumpy - as shown in many of the examples
above.

#### Making the most of the Weld model

We have designed this such that you can use it without really understanding
what is going on under the hood - but If you understand how a few things work,
you may be able to get the most out of your programs.

##### Fewer Evaluations = Better Performance

There are two main reasons why calling evaluate unneccessarily will cause
performance slowdowns:

* extra compilation costs: this is comparitively a minor point, but if you
evaluate without registering many ops, then these compilation costs can add up.
* Partially loses the benefits of lazy evaluation: In general, Weld should be
able to see the complete program in order to perform optimizations like loop
fusion, materialization etc. across the program. But whenever you call
evaluate, all the ops registered so far are evaluated - and future ops won't be
able to combine optimizations with the older ones.

Thus, besides avoiding explicit evaluate calls when you don't need them, you
should also avoid things that cause implicit evaluations: printing the
weldarray before the end of computations, using unsupported operations if you
can avoid them, and so on.

##### Inplace Ops and Views

Weld does not explicitly support Inplace Ops yet - thus we have a slightly more
complicated way to deal with these than expected - which could lead to some
surprising speed issues (but the functionality should still be correct). Since
the most common use case is around contiguous arrays - and here, we could
actually give the illusion as if in place ops are supported in Weld - we first
discuss that scenario:

* Contiguous Arrays:

* Non-Contiguous Arrays:

## Developer Documentation

Scenarios which cause Implicit Evaluation:

* printing the array
* unsupported operations which need to be offloaded to NumPy
* If the number of registered ops on the weldarray exceeds a threshold, like
50 ops, as beyond that, the Weld optimization passes start taking significant
time since they are quadratic. Also, if the number of ops gets around 1000,
 then the program can actually segfault because of some implementation
 details of the Weld optimization passes.
* operations between ndarrays and weldarrays: this does not neccessarily
require implicit evaluation of the weldarray, but there were some edge cases in
the non-contiguous arrays case which make us conservatively choose to
implicitly evaluate weldarrays in these cases.

## Future Work

## Similar Projects
