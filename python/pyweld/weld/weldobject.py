#
# WeldObject
#
# Holds an object that can be evaluated.
#

import ctypes

import os
import time

import bindings as cweld
from types import *


class WeldObjectEncoder(object):
    """
    An abstract class that must be overwridden by libraries. This class
    is used to marshall objects from Python types to Weld types.
    """
    def encode(obj):
        """
        Encodes an object. All objects encodable by this encoder should return
        a valid Weld type using py_to_weld_type.
        """
        raise NotImplementedError

    def py_to_weld_type(obj):
        """
        Returns a WeldType corresponding to a Python object
        """
        raise NotImplementedError


class WeldObjectDecoder(object):
    """
    An abstract class that must be overwridden by libraries. This class
    is used to marshall objects from Weld types to Python types.
    """
    def decode(obj, restype):
        """
        Decodes obj, assuming object is of type `restype`. obj's Python
        type is ctypes.POINTER(restype.ctype_class).
        """
        raise NotImplementedError


class WeldObject(object):
    """
    Holds a Weld program to be lazily compiled and evaluated,
    along with any context required to evaluate the program.

    Libraries that use the Weld API return WeldObjects, which represent
    "lazy executions" of programs. WeldObjects can build on top of each other,
    i.e. libraries should be implemented so they can accept both their
    native types and WeldObjects.

    An WeldObject contains a Weld program as a string, along with a context.
    The context maps names in the Weld program to concrete values.

    When a WeldObject is evaluated, it uses its encode and decode functions to
    marshall native library types into types that Weld understands. The basic
    flow of evaluating a Weld expression is thus:

    1. "Finish" the Weld program by adding a function header
    2. Compile the Weld program and load it as a dynamic library
    3. For each argument, run object.encoder on each argument. See
       WeldObjectEncoder for details on how this works
    4. Pass the encoded arguments to Weld
    5. Run the decoder on the return value. See WeldObjectDecoder for details
       on how this works.
    6. Return the decoded value.
    """

    # Counter for assigning variable names
    _var_num = 0
    _obj_id = 100
    _registry = {}

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

        # Weld program
        self.weld_code = ""
        self.dependencies = {}

        # Assign a unique ID to the context
        self.obj_id = "obj%d" % WeldObject._obj_id
        WeldObject._obj_id += 1

        # Maps name -> input data
        self.context = {}
        # Maps name -> arg type (for arguments that don't need to be encoded)
        self.argtypes = {}

    def __repr__(self):
        return self.weld_code + " " + str(self.context)

    def update(self, value, tys=None, override=True):
        """
        Update this context. if value is another context,
        the names from that context are added into this one.
        Otherwise, a new name is assigned and returned.

        TODO tys for inputs.
        """
        if isinstance(value, WeldObject):
            self.context.update(value.context)
            self.dependencies.update(value.dependencies)
        else:
            # Ensure that the same inputs always have same names
            value_str = str(value)
            if value_str in WeldObject._registry:
                name = WeldObject._registry[value_str]
            else:
                name = "_inp%d" % WeldObject._var_num
                WeldObject._var_num += 1
                WeldObject._registry[value_str] = name
            self.context[name] = value
            if tys is not None and not override:
                self.argtypes[name] = tys
            return name

    def get_let_statements(self):
        queue = [self]
        visited = set()
        let_statements = []
        is_first = True
        while len(queue) > 0:
            cur_obj = queue.pop()
            cur_obj_id = cur_obj.obj_id
            if cur_obj_id in visited:
                continue
            if not is_first:
                let_statements.insert(0, "let %s = (%s);" % (cur_obj_id, cur_obj.weld_code))
            is_first = False
            for key in sorted(cur_obj.dependencies.keys()):
                queue.append(cur_obj.dependencies[key])
            visited.add(cur_obj_id)
        let_statements.sort()  # To ensure that let statements are in the right
                               # order in the final generated program
        return "\n".join(let_statements)

    def to_weld_func(self):
        names = self.context.keys()
        names.sort()
        arg_strs = ["{0}: {1}".format(str(name),
                                      str(self.encoder.py_to_weld_type(self.context[name])))
                    for name in names]
        header = "|" + ", ".join(arg_strs) + "|"
        keys = self.dependencies.keys()
        keys.sort()
        text = header + " " + self.get_let_statements() + "\n" + self.weld_code
        return text

    def evaluate(self, restype, verbose=True, decode=True, passes=None):
        function = self.to_weld_func()

        function = """
|vprice: vec[f32], vstrike: vec[f32], vt: vec[f32], vrate: vec[f32], vvol: vec[f32]|
  let c05 = 3.0f;
  let c10 = 1.5f;
  let r = for(
    zip(vprice, vstrike, vt, vrate, vvol),
    {appender[f32](len(vprice)) ,appender[f32](len(vprice))},
    |b,i,e|
      let rsig = e.$3 + e.$4 * e.$4 * c05;
      let vol_sqrt = e.$4 * sqrt(e.$2);
      let d1 = (log(e.$0 / e.$1) + rsig * e.$2) / vol_sqrt;
      let d2 = d1 - vol_sqrt;

      let d11 = c05 + c05 + erf(d1 * 1.0f);
      let d22 = c05 + c05 + erf(d2 * 1.0f);
      let e_rt = exp(e.$3 * e.$2);

      {
        merge(b.$0, e.$0 * d11 - e_rt * e.$1 * d22),
        merge(b.$1, e_rt * e.$1 * (c10 - d22) - e.$0 * (c10 - d11))
      }
    );
  {result(r.$0), result(r.$1)}
  """

        # Returns a wrapped ctypes Structure
        def args_factory(encoded):
            class Args(ctypes.Structure):
                _fields_ = [e for e in encoded]
            return Args

        # Encode each input argument. This is the positional argument list
        # which will be wrapped into a Weld struct and passed to the Weld API.
        names = self.context.keys()
        names.sort()

        start = time.time()
        encoded = []
        argtypes = []
        for name in names:
            if name in self.argtypes:
                argtypes.append(self.argtypes[name].ctype_class)
                encoded.append(self.context[name])
            else:
                argtypes.append(self.encoder.py_to_weld_type(
                    self.context[name]).ctype_class)
                encoded.append(self.encoder.encode(self.context[name]))
        end = time.time()
        if verbose:
            print "Python->Weld:", end - start

        start = time.time()
        Args = args_factory(zip(names, argtypes))
        weld_args = Args()
        for name, value in zip(names, encoded):
            setattr(weld_args, name, value)

        start = time.time()
        void_ptr = ctypes.cast(ctypes.byref(weld_args), ctypes.c_void_p)
        arg = cweld.WeldValue(void_ptr)
        conf = cweld.WeldConf()
        err = cweld.WeldError()

        if passes is not None:
            conf.set("weld.optimization.passes", ",".join(passes))

        conf.set("weld.compile.dumpCode", "true")
        conf.set("weld.compile.multithreadSupport", "false")

        module = cweld.WeldModule(function, conf, err)
        if err.code() != 0:
            raise ValueError("Could not compile function {}: {}".format(
                function, err.message()))
        end = time.time()
        if verbose:
            print "Weld compile time:", end - start

        start = time.time()
        conf = cweld.WeldConf()
        weld_num_threads = os.environ.get("WELD_NUM_THREADS", "1")
        conf.set("weld.threads", weld_num_threads)
        conf.set("weld.memory.limit", "100000000000")
        err = cweld.WeldError()
        weld_ret = module.run(conf, arg, err)
        if err.code() != 0:
            raise ValueError(("Error while running function,\n{}\n\n"
                              "Error message: {}").format(
                function, err.message()))
        ptrtype = POINTER(restype.ctype_class)
        data = ctypes.cast(weld_ret.data(), ptrtype)
        end = time.time()
        if verbose:
            print "Weld:", end - start

        start = time.time()
        if decode:
            result = self.decoder.decode(data, restype)
        else:
            data = cweld.WeldValue(weld_ret).data()
            result = ctypes.cast(data, ctypes.POINTER(
                ctypes.c_int64)).contents.value
        end = time.time()
        if verbose:
            print "Weld->Python:", end - start

        return result
