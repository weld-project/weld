# WeldNumpy

## Contents

  * [Setup](#setup)
  * [Overview](#overview)
  * [Documentation](#documentation)
  * [Developer Documentation](#developer-documentation)
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
optimizations for data science workloads based on the [Weld project](weld.rs).
This is achieved by implementing various NumPy operators in Weld's Intermediate
Representation (IR). Thus, as you operate on a weldarray, it will internally
build a graph of the operations, and pass them to weld's runtime system to
optimize and execute in parallel whenever required. 

In [examples](examples), you can see improvements of upto $5x$ on a single
thread on some NumPy workloads essentially without changing any code from the
original NumPy implementations. In general, Weld works well with programs that
operate on large NumPy arrays with compute operations that are supported by
Weld. 

(TODO: Need to actually add examples from the benchmarks repo)

NumPy features that WeldNumpy currently supports are:
* Supported operations include:
    * Unary Operations: np.exp, np.log, np.sqrt, np.sin, np.cos, np.tan,
    np.arccos, np.arcsin, np.arctan, np.sinh, np.cosh, np.tanh, scipy.special.erf
    * Binary Operations: np.add, np.subtract, np.multiply, np.divide
* Supported types are: np.float64, np.float32, np.int64, np.int32
* Supports reductions over 1d arrays
* Supports [broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
* All unsupported operations/methods on the weldarray are offloaded to NumPy -
so NumPy programs should still function correctly. After the NumPy methods
are executed, a weldarray is returned.

One of the goals of this library is to require minimal
changes to your original NumPy source code in order to harness the
optimizations provided by Weld. Thus, you can write NumPy programs in the same
style, and WeldNumpy will only evaluate those when neccessary.

In general, it may be more flexible to not always utilize weldarray - for
instance with relatively small sized NumPy arrays, the overhead of compiling
the weld programs will be more expensive than the total NumPy execution times.
A simple example of the usage is:

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

Another  way to use weldnumpy is to change the import statement at the top of
the NumPy file. In general, the WeldNumpy class just serves a wrapper around
NumPy functions - with the array creation routines modified to return
weldarray's instead. Here is another contrived example which shows some of the
benefits provided by weldnumpy:

```python
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

Changing the import statement may serve as a quick method to use weldnumpy, but
in general for a large program, it makes sense to only import weldarray and
convert only the large arrays whose operations you want to optimize, as in the
first example above. More detailed examples with comments are provided in
[examples](/examples).

## Documentation

In general, the semantics for the operations on weldarray's are exactly the
same as the equivalent operations on NumPy arrays. The only extra operation on
a weldarray is: weldarray.evaluate() which forces the evaluation of all the
operations registered on the weldarray.

#### Differences with NumPy

##### Compilation Costs 

##### Lazy Evaluation

TODO: Define Lazy evaluation. NumPy doesn't do this, so the challenge is to
present the same interface as NumPy without explicitly using lazy evaluation.

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

###### Evaluate

The only function on weld arrays that is different from the NumPy operations
is:

```python
# assuming 'w' is a weldarray with stored operations, this will update the
# weldarray to it's latest values.
w = w.evaluate()
```

* When passing to other functions, it may be safer to evaluate the arrays at
first
    * np.array_equal example
    * np.random_choice example

##### Views

##### Things that don't work

#### Making the most of the Weld model

Here, we highlight a few details about the way weld and weldnumpy work that
would be useful to get the most performance out of the system.

##### Reduce evaluations

##### Inplace ops

## Developer Documentation

## Similar Projects
