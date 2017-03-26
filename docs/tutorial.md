# Hello, Weld!

In this tutorial, we will write a simple Python library using Weld over NumPy's `ndarray` type. Our library will support _elementwise_ operations over vectors. Specifically, we'll define a class `HelloWeldVector`, which acts as a wrapper around `numpy.ndarray`, and has four methods:

* `HelloWeldVector.add(number)`: Adds `number` to each element in the vector
* `HelloWeldVector.subtract(number)`: Subtracts `number` from each element in the vector
* `HelloWeldVector.multiply(number)`: Multiplies `number` with each element in the vector
* `HelloWeldVector.divide(number)`: Divides out `number` from each element in the vector

We will also support the Python `__str__` function, which will print out the vector.

For pedagogical reasons our library will have some limitations. First, we will only support 1-dimensional arrays (_i.e.,_ arrays whose `ndim` field is set to `1`). Next, we will only support `ndarray` objects whose `dtype='int32'`. This just prevents some checks we have to do in our library.

## Table of Contents

 * [Table of Contents](#table-of-contents)
 * [Prerequisites](#prerequisites)
 * [Setting up the Project](#setting-up-the-project)
 * [Weld Background](#weld-background)
 * [Initializing](#initializing)
 * [Implementing an Operator](#implementing-an-operator)
 * [Forcing Evaluation](#forcing-evaluation)
 * [Putting it all Together](#putting-it-all-together)
 * [Going From Here](#going-from-here)
 * [Common Issues](#common-issues)
     - [ImportError: No module named weld.weldobject](#importerror-no-module-named-weldweldobject)
     - [ValueError: Could not compile function ...](#valueerror-could-not-compile-function-)

## Prerequisites

This tutorial assumes you have a Weld installation and a familiarity of Python. See the [README](https://github.com/weld-project/weld/blob/master/README.md#building) for instructions on how to build Weld.

The tutorial also assumes you've installed the Weld Python package, as described [here](https://github.com/weld-project/weld/blob/master/docs/python.md). In particular, make sure you install the Weld packages by running:

```bash
$ python $WELD_HOME/python/setup.py install
```

## Setting up the Project

1. Create a new file called `hello_weld.py`.
  
2. Import dependencies:

  ```python
  import numpy as np
  from weld.weldobject import *
  from weld.types import *
  from weld.encoders import NumpyArrayEncoder, NumpyArrayDecoder
  ```
  
  We need NumPy because we will build our library on top of NumPy's `ndarray` type. The other imports come from the Weld library; we will revisit them later.
  
3. Set up a template for the `HelloWeldVector` class. We will fill this in as we go. Copy and paste the following, which defines the methods we will implement along with what they do, but provides no implementation:

  ```python
  class HelloWeldVector(object):
      def __init__(self, vector):
          """
          Create a new `HelloWeldVector`, initialized with an existing `numpy.ndarray` 'vector'.
          vector must have ndim=1 and dtype='int32'.
          """
          

      def add(self, number):
          """
          Add `number` to each element in this vector.
          """
          

      def multiply(self, number):
          """
          Multiply each element in this vector by `number`.
          """
          

      def subtract(self, number):
          """
          Subtract `number` from each element in this vector.
          """
          

      def divide(self, number):
          """
          Divide each element in this vector by `number`.
          """
          

      def __str__(self):
          """
          Return a string representation of this vector.
          """
  ```
  
## Weld Background

Before diving in further, let's discuss how Weld operates.

Weld uses a lazily evaluated API to build up a computation without actually executing. In other words, Weld does not actually execute any code until it absolutely needs to (_e.g.,_ when printing a result to the terminal). This allows Weld to perform optimizations like loop fusion, where multiple loops over some data can be fused into a single loop by analyzing a built-up expression.

The primary interface to Weld is `WeldObject`, which is a Python class which can keep track of a computation we have built so far. We will use `WeldObject` to make the operations in `HelloWeldVector` lazy, so they do not produce a result except when the a user prints the value of the vector.

Weld registers computations using a special _intermediate representation_ (IR); think of it as a small functional programming language that can capture parallel programs. We won't discuss the IR in depth in this tutorial, but we will need to use it in order to express computations on our vector.

## Initializing

We will start off by initializing our vector. We need to do a few things here:

1. Track the vector the user passes in as the "initial vector".
2. Create a new `WeldObject`, which we will use to track the computations on the vector.

Add each line described below to the `__init__` method.
  
 ```python
 self.vector = vector
 ```
 
This line tracks the vector the user passes in. Note that we might want to perform some checks here, like making sure the vector is a NumPy `ndarray` and the `dtype` is something we can support, but we'll skip that for now.
 
 ```python
 self.weldobj = WeldObject(NumpyArrayEncoder(), NumpyArrayDecoder())
 ```
 
Here, we are initializing a new `WeldObject` instance and setting it as a field. `WeldObject` takes two parameters: an _encoder_ class and a _decoder_ class. These classes specify how a type in Python maps over to a type in Weld (Weld expects a certain in-memory format for each type) and vice versa. The Weld API contains some default encoders and decoders for common types in the `weld.encoders` module; here, we use the included NumPy array encoder and decoder classes. In a different tutorial we will look at how to write encoders and decoders for custom objects.
 
 ```python
 name = self.weldobj.update(vector, WeldVec(WeldInt()))
 ```
 
`WeldObject` instances have an `update` method which add a _dependency_ to the object. Dependencies are just values which will be passed into Weld when we actually want to compute something. The `update` method takes two parameters (a value and a type) and returns a string name.
 
Let's talk about these in more detail. The value is the value to mark as a dependency. The type is the type the value will take on _in Weld_. Types supported by Weld are available in the `weld.types` module. In this example, because our value is a NumPy array of integers, the expected Weld type is a `WeldVec(WeldInt())` (a vector of 32-bit integer values). The encoder object we discussed earlier is responsible for translating the NumPy array into this type so Weld's execution engine understands it.

The return type of the `update` function is a string name. The name is how we refer to `value` in Weld code. Names are unique; no two values will ever be assigned the same name. Here, whenever we want to refer to the vector in our Weld code, we can just use this string as a placeholder to represent the value. `WeldObject` takes care of tracking which names are mapped to which values.

```python
self.weldobj.weld_code = name
```

Last line! Here, we're setting the actual Weld IR code of the `WeldObject`. The `weld_code` field is just a string which represents some Weld code (in our special Weld IR). Again, we won't discuss the IR itself here, but on this line our code is just the name we assigned to the vector. If we were to execute this code now, Weld would just return the vector we passed in as the result.

Here is what you should have at the end:

```python
  def __init__(self, vector):
      """
      Create a new `HelloWeldVector`, initialized with an existing `numpy.ndarray` 'vector'.
      vector must have ndim=1 and dtype='int32'.
      """
      self.vector = vector
      self.weldobj = WeldObject(NumpyArrayEncoder(), NumpyArrayDecoder())
      name = self.weldobj.update(vector, WeldVec(WeldInt()))
      self.weldobj.weld_code = name
 ```

## Implementing an Operator

We will now implement an operator. Let's start with `add`. As before, add each line below to the `add` method.

```python
template = "map({0}, |e| e + {1})"
```

`template` represents some Weld IR, which performs a `map` function on some vector `{0}` (this is something we can use Python's `format` method to replace with another string). The map function adds `{1}` to each element; in short, `template` is some Weld code to implement an elementwise add operator.

```python
self.weldobj.weld_code = template.format(self.weldobj.weld_code, str(number))
```

We are updating the `weld_code` field of our `WeldObject` instance using the `template`; we substitute `{0}` with the first argument of `format`, and `{1}` with the second argument.

The first argument is the `weld_code` we already had. What does this `weld_code` represent? Well, if each operation in our library produces a vector, the `weld_code` must represent a vector too! We are effectively passing in the computation we have done so far as the input to the `add` operator. If we haven't done any operations on the vector yet, recall that `weld_code` was initialized to the name of the initial vector from `__init__`, so we will do the `add` on that.

The second argument is just the number we want to add to each element.

And that's it! We've implemented the `add` operator. Note that we never actually compute a result here; rather, we just express what we want to do without actually doing it. Implementing the other operators is similar; we just change the `+` in the map function to the correct binary operator.

Here's what `add` looks like at the end:

```python
def add(self, number):
    """
    Add `number` to each element in this vector.
    """
    template = "map({0}, |e| e + {1})"
    self.weldobj.weld_code = template.format(self.weldobj.weld_code, str(number))
```

## Forcing Evaluation

Eventually, we do want to compute a result. In our library, where should this happen? One sensible place to do it is when a user wants to print the vector. Python allows defining custom behavior for how a result is printed by overriding the `__str__` method; that is exactly what we will do now.

Copy and paste the following into the `__str__` method:

```python
def __str__(self):
    v = self.weldobj.evaluate(WeldVec(WeldInt()))
    return str(v)
```

There is only one notable line here -- the call to `evaluate`. Calling evaluate on a `WeldObject` forces it's evaluation. In other words, calling evaluate will take the dependencies and Weld IR code registered with the `WeldObject`, generate a callable Weld function, compile it to fast parallel machine code, and run it. It then calls the _decoder_ we specified when creating the `WeldObject` to marshall Weld's return value into something Python understands; in our case, it will decode a Weld vector into a NumPy array. Note that we need to specify the Weld return type of the computation so the decoder knows what it should marshall; in our case, the return type will always be a Weld vector of 32-bit integers, since that's what our initial vector is and it is also what each of our operations returns.

`v` is thus a NumPy `ndarray`. To get a string representation of it, we just call and return `str(v)`.

## Putting it all Together

That's it! We now have a minimal implementation for a Weld-enabled library. Let's try it out. Open up a Python shell and import the code you just wrote (along with NumPy):

```python
>>> import numpy
>>> from hello_weld import *
```

Let's make a new vector initialized with all 0s, and then add some numbers to it:

```python
>>> my_vector = HelloWeldVector(numpy.array([0,0,0,0,0], dtype='int32'))
>>> my_vector.add(5)
>>> my_vector.add(100)
```

If you'd like, you can see what the Weld IR code looks like so far:

```python
>>> print my_vector.weldobj.weld_code
'map(map(e0, |e| e + 5), |e| e + 100))'
```

Now, let's print out the vector itself. This will compile the Weld code, pass in our NumPy array into Weld, and compute a result:

```python
>>> print my_vector
[105, 105, 105, 105, 105]
```

Nice work!

## Going From Here

We have a minimal working example of a Weld-enabled library now, but there is still a lot more we can do to make it more efficient/useful! Here are a few ideas: 

* __Caching:__ Right now, we perform the entire computation each time we print the vector; not exactly very efficient. The `HelloWeldVector` can be extended so computed values can be cached.
* __Supporting More Types:__ Supporting just the `'int32'` type is a bit limiting; extending this example to work with other types is not too difficult.
* __Supporting Other Operators:__ Elementwise operations are useful, but they're also just one class of operations over vectors. Weld supports all kinds of operations through its IR, and `HelloWeldVector` is a good starting point for them.

## Common Issues

#### ImportError: No module named weld.weldobject

Make sure the Weld modules are installed:

```bash
$ python $WELD_HOME/python/setup.py install
```

---

#### ValueError: Could not compile function ...

Take a look at the [language docs](https://github.com/sppalkia/weld/blob/master/docs/language.md); this is a compile error stating that the Weld code could not be compiled.

---



