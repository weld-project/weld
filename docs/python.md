# Python API

The Python API is found under the `python/` directory. It provides convinient wrapper objects for the low-level C API, as well as utilities for building Weld computations and composing Python libraries.

To use the Python API in 'development' mode, run the following from `$WELD_HOME/python`:

```bash
$ cd pyweld;  python setup.py develop; cd ..
$ cd grizzly; python setup.py develop; cd ..
```

Otherwise, run the following from `$WELD_HOME/python`:

```bash
$ cd pyweld;  python setup.py install; cd ..
$ cd grizzly; python setup.py install; cd ..
```

Alternatively, you can install our pre-built Python modules by running,

```bash
$ pip install pyweld
$ pip install pygrizzly
```

You should also follow the setup instructions [here](https://github.com/weld-project/weld/blob/master/README.md) (in particular, make sure `WELD_HOME` is set so the libraries Weld uses can be found). 
### Bindings

The `weld.bindings` module contains bindings for the [C API](https://github.com/weld-project/weld/blob/master/docs/api.md). Each type in the C API is wrapped as a Python object. Methods on the Python objects call the corresponding C API functions.

As an example, the code below creates a new `WeldConf`, sets a value on it, and then gets the value back:

```python
>>> import weld.bindings as bnd
>>> conf = bnd.WeldConf() # calls weld_conf_new()
>>> conf.set("myKey", "myValue") # calls weld_conf_set(...)
>>> print conf.get("myKey") # calls weld_conf_get(...)
"myValue"
```

### WeldObject API

The `WeldObject` API is included in the `weld.weldobject` module, and an example of how to use it is described [here](https://github.com/weld-project/weld/blob/master/docs/tutorial.md). 

This API provides an interface for _composing_ Python programs by building a lazily evaluated Weld computation. The `WeldObject` tracks values it operates over (called "dependencies") and builds and runs a runnable Weld module using a `evaluate` method. Dependencies are added using the `update` function. This function requires a Weld type for the dependency being added; types are available in the `weld.types` module.

The table below describes the `WeldObject` API in brief:

  Method/Field | Description
  ------------- | -------------
  `update(value, ty)` | Adds `value` (which has Weld type `ty`) in Weld as a dependency. Returns a string name which can be used in the object's Weld code to refer to this value.
  `evaluate(ty)` | Evaluates the object and returns a value. `ty` is the expected Weld type of the return value.
  `weld_code` | A string field representing the Weld IR for this object. This string is modified to register a computation with this object. See [this](https://github.com/weld-project/weld/blob/master/docs/language.md) document for a description of the language.


The general usage pattern for a WeldObject is to initialize it, add some dependencies and Weld code to register a computation, and then evaluate it to get a return value. Here's an example, where we add two numbers:

```python
>>> import weld.weldobject as wo
>>> import weld.encoders as enc
>>> obj = wo.WeldObject(enc.WeldScalarEncoder(), enc.WeldScalarDecoder()) # See more about encoders below
>>> name1 = obj.update(1, WeldI32())
>>> name2 = obj.update(2, WeldI32())
>>> obj.weld_code = name1 + " + " + name2 # Weld IR to add two numbers.
```

### Encoders and Decoders

When data is passed into Weld, it must be marshalled into a binary format which Weld understands (these formats are described in the [C API doc](https://github.com/weld-project/weld/blob/master/docs/api.md). In general, values are formatted using C scalars and structs; Python's `ctypes` module allows constructing these kinds of representations.

To support custom formats, the `WeldObject` API takes an encoder, which allows encoding a Python object as a Weld object, and a decoder, which allows decoding a Weld object into a Python object. These encoders and decoders are interfaces which must be implemented by a library writer.

Weld provides some commonly used encoders and decoders in the `weld.encoders` module. NumPy arrays, for example, are a common way to represent C-style arrays in Python. Weld thus includes a `WeldNumPyEncoder` and `WeldNumPyDecoder` class to marshall 1-dimensional NumPy arrays.
