# Hello, Weld!

In this tutorial, we will write a simple Python library using Weld over NumPy's `ndarray` type. Our library will support _elementwise_ operations over vectors. Specifically, we'll define a class `HelloWeldVector`, which acts as a wrapper around `numpy.ndarray`, and has four methods:

* `HelloWeldVector.add(number)`: Adds `number` to each element in the vector
* `HelloWeldVector.subtract(number)`: Subtracts `number` from each element in the vector
* `HelloWeldVector.multiply(number)`: Multiplies `number` with each element in the vector
* `HelloWeldVector.divide(number)`: Divides out `number` from each element in the vector

We will also support the Python `__str__` function, which will print out the vector.

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
  
4. Set up a template for the `HelloWeldVector` class. We will fill this in as we go. Copy and paste the following:

  ```python
  class HelloWeldVector(object):
      def __init__(self, vector):
        pass

      def add(self, number):
        pass

      def multiply(self, number):
        pass

      def subtract(self, number):
        pass

      def divide(self, number):
        pass

      def __str__(self):
        return ""
  ```
  






