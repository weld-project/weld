# Hello, Weld!

In this tutorial, we will write a simple Python library using Weld over NumPy's `ndarray` type. Our library will support _elementwise_ operations over vectors. Specifically, we'll define a class `HelloWeldVector`, which acts as a wrapper around `numpy.ndarray`, and has four methods:

* `HelloWeldVector.add(number)`: Adds `number` to each element in the vector
* `HelloWeldVector.subtract(number)`: Subtracts `number` from each element in the vector
* `HelloWeldVector.multiply(number)`: Multiplies `number` with each element in the vector
* `HelloWeldVector.divide(number)`: Divides out `number` from each element in the vector

We will also support the Python `__str__` function, which will print out the vector.

For pedagogical reasons our library will have some limitations. First, we will only support 1-dimensional arrays (_i.e.,_ arrays whose `ndim` field is set to `1`). Next, we will only support `ndarray` objects whose `dtype='int32'`. This just prevents some checks we have to do in our library; at the end, we will discuss how to add support for other types too.

## Prerequisites

This tutorials assumes familiarity with Python and a Weld installation. It also requires Rust.

[Install Rust](https://www.rustup.rs/):

```
curl https://sh.rustup.rs -sSf | sh
```

The easiest way to install Weld is by cloning the git repository:

```bash
$ git clone https://www.github.com/weld-project/weld
$ cd weld
$ export WELD_HOME=`pwd` # Put this in your .rc file!
$ cargo build # Build
$ cargo test # Run tests
```

## Setting up the Project

1. Create a new file called `hello_weld.py`

2. Append the Python Path so it can find the Weld Python bindings:

  ```python
  import os
  import sys
  
  home = os.environ.get("WELD_HOME")
  libpath = home + "api/python"
  sys.path.append(libpath)
  ```
  
3. Import dependencies:

  ```python
  import numpy as np
  from weld.weldobject import *
  from weld.types import *
  from weld.encoders import NumpyArrayEncoder, NumpyArrayDecoder
  ```
  
  We will build our library on top of NumPy's `ndarray` type. The other imports come from the Weld library; we will revisit them later.
  
4. Set up a template for the `HelloWeldVector` class. We will fill this in as we go. Copy and paste the following, which defines the methods we will implement along with what they do, but provides no implementations:

  ```python
  class HelloWeldVector(object):
      def __init__(self, vector):
        """
        Create a new `HelloWeldVector`, initialized with an existing `numpy.ndarray` 'vector'.
        """
        pass

      def add(self, number):
        """
        Add `number` to each element in this vector.
        """
        pass

      def multiply(self, number):
        """
        Multiply each element in this vector by `number`.
        """
        pass

      def subtract(self, number):
        """
        Subtract `number` from each element in this vector.
        """
        pass

      def divide(self, number):
        """
        Divide each element in this vector by `number`.
        """
        pass

      def __str__(self):
        """
        Return a string representation of this vector.
        """
        return ""
  ```
  
## Weld Background

Weld uses a lazily evaluated API to build up a computation without actually executing. In other words, Weld does not actually execute any code until it absolutely needs to (_e.g.,_ when printing a result to the terminal). This allows Weld to perform optimizations like loop fusion, where multiple loops can be fused into a single one by analyzing a built-up expression.

The primary interface to Weld is `WeldObject`, which is a Python class which can keep track of a computation we have built so far. We will use `WeldObject` to make the operations in `HelloWeldVector` lazy, _i.e.,_ they do not produce a result except when printing the vector.

Weld registers computations using a special _intermediate representation_ (IR); think of it as a small functional programming language that can capture parallel programs. We won't discuss the IR in depth in this tutorial, but we will need to use it in order to express what we want to do with our vector.

## Initializing

Let's start off by initializing our vector. We need to do a few things here:

1. Track the vector the user passes in as the "initial vector"
2. Create a new `WeldObject`, which we will use to track the computations on the vector

Replace the `__init__` implementation with the following:

```python
  def __init__(self, vector):
    self.vector = vector
    self.weldobj = WeldObject(NumpyArrayEncoder(), NumpyArrayDecoder())
    name = self.weldobj.update(vector, WeldVec(WeldI32()))
    self.weldobj.weld_code = name
 ```
 
 Okay, there is a lot going on here, so let's take it line by line:
 
 ```python
 self.vector = vector
 ```
 
 This line is straightforward; we just track the vector the user passes in. Note that we might want to perform some checks here, like making sure the vector is a NumPy `ndarray` and the `dtype` is something we can support, but we'll skip that for now.
 
 ```python
 self.weldobj = WeldObject(NumpyArrayEncoder(), NumpyArrayDecoder())
 ```
 
 Let's break this line down. We're initializing a new `WeldObject` instance and setting it as a field in our object. `WeldObject` takes two parameters: an _encoder_ class and a _decoder_ class. These classes specify how a type in Python maps over to a type in Weld (Weld expects a certain in-memory format for each type). The Weld API contains some default encoders and decoders for common types in `weld.encoders`; here, we use the built in NumPy array encoder and decoder classes. In a different tutorial we will look at how to write encoders and decoders for custom objects.
 
 ```python
 name = self.weldobj.update(vector, WeldVec(WeldI32()))
 ```
 
 This is an important line. `WeldObject` instances have an `update` method which add a _dependency_ to the object. Dependencies are just values which will be passed into Weld when we actually want to compute something. The `update` method takes two parameters (a value and a type) and returns a string name.
 
Let's talk about these in more detail. The value is straightforward. The type warrants some discussion; it is the type the value will take on _in Weld_. Types supported by Weld are available in `weld.types`. In this example, because our value is a NumPy array of integers, the expected Weld type is a `WeldVec(WeldI32())` (a vector of `i32`, or 32-bit integer, values). The encoder object we discussed earlier is responsible for translating the NumPy array into this type in Weld.

The return type of the `update` function is a string name. The name is how we refer to this value in Weld code. Names are unique; no two values will ever be assigned the same name. Here, whenever we want to refer to the vector in our Weld code, we can just use this string as a placeholder to represent the value. `WeldObject` takes care of tracking which names are mapped to which values.

```python
self.weldobj.weld_code = name
```

Last line! Here, we're setting the actual Weld IR code of the `WeldObject`. The `weld_code` field is just a string which represents some Weld code (in our special Weld IR). Again, we won't discuss the IR itself here, but on this line our code is just the name we assigned to the vector. If we were to execute this code now, Weld would just return the vector we passed in as a result.

## Implementing an Operator

It may seem like we haven't done much, but we're already 80% of the way there! Now, we need to implement an operator.

Let's start with `add`. Copy and paste the following code, which implements the `add` operator:

```python
def add(self, number):
  template = "map({0}, |e| e + {1})"
  self.weldobj.weld_code = template.format(self.weldobj.weld_code, str(number))
```

Let's take it line by line again.

```python
template = "map({0}, |e| e + {1})"
```

This is some simple Python code. `template` represents some Weld IR, which performs a `map` function on some vector `{0}` (this is something we can use Python's `format` method to replace with another string). The map function adds `{1}` to each element; in short, `template` is some Weld code to implement an elementwise add operator).

```python
self.weldobj.weld_code = template.format(self.weldobj.weld_code, str(number))
```

This shold look familiar now. We are updating the `weld_code` field of our `WeldObject` instance using the `template`; we substitute `{0}` with the first argument of `format`, and `{1}` with the second argument.

The first argument is the `weld_code` we already had. What does this `weld_code` represent? Well, if each operation in our library produces a vector, the `weld_code` must represent a vector too! We are effectively passing in the computation we have done so far as the input to the `add` operator. If we haven't done any operations on the vector yet, recall that `weld_code` was initialized to the name of the initial vector from `__init__`, so we will do the `add` on that.

The second argument is just the number we want to add to each element.

And that's it! We've implemented the `add` operator. Note that we never actually compute a result here; rather, we just express what we want to do without actually doing it.

## Forcing Evaluation

Eventually, we do want to compute a result. In our library, when should this happen? On sensible time to do it is when a user wants to print out a result.

