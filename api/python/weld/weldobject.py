#
# WeldObject
#
# Holds an object that can be evaluated.
#

import ctypes
import sys
import os

import time

import weld
from weld_types import *

class WeldObjectEncoder(object):
    """
    An abstract class that must be overwridden by libraries. This class
    is used to marshall objects from Python types to Weld types.
    """
    def encode(obj):
        """
        Encodes an object. All objects encodable by this encoder should return
        a valid Weld type using pyToWeldType.
        """
        raise NotImplementedError

    def pyToWeldType(obj):
        """
        Returns a WeldType corresponding to a Python object
        """
        raise NotImplementedError

class WeldObjectDecoder(object):
    """
    An abstract class that must be overwridden by libraries. This class
    is used to marshall objects from Weld types to Python types
    """
    def decode(obj, restype):
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
    marshall native library types into types that Weld understands. The basic flow
    of evaluating a Weld expression is thus:

    1. "Finish" the Weld program by adding a function header
    2. Compile the Weld program and load it as a dynamic library
    3. For each argument, run object.encoder on each argument. See WeldObjectEncoder
    for details on how this works
    4. Pass the encoded arguments to Weld
    5. Run the decoder on the return value. See WeldObjectDecoder for details on how
    this works.
    6. Return the decoded value.
    """

    # Counter for assigning variable names
    _var_num = 0
    _obj_id = 100

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

        # Weld program
        self.weld_code = ""

        # Weld type, which represents the return type of this Weld program.
        # decoder must support decoding the return type. TODO.
        self.dataType = None

        # Assign a unique ID to the context
        self.objectId = WeldObject._obj_id
        WeldObject._obj_id += 1

        # Maps name -> input data
        self.context = {}

    def __repr__(self):
        return self.weld_code + " " + str(self.context)

    def update(self, value, ty=None):
        """
        Update this context. if value is another context,
        the names from that context are added into this one.
        Otherwise, a new name is assigned and returned.

        TODO types for inputs.
        """
        if isinstance(value, WeldObject):
            self.context.update(value.context)
        else:
            name = "e" + str(WeldObject._var_num)
            WeldObject._var_num += 1
            self.context[name] = value
            return name

    def toWeldFunc(self):
        names = self.context.keys()
        names.sort()
        arg_strs = ["{0}: {1}".format(str(name), str(self.encoder.pyToWeldType(self.context[name])))
                for name in names]
        header = "|" + ", ".join(arg_strs) + "|"
        text = header + " " + self.weld_code
        return text

    def evaluate(self, restype):
        function = self.toWeldFunc()

        # Returns a wrapped ctypes Structure
        def args_factory(encoded):
            class Args(ctypes.Structure):
                _fields_ = [e for e in encoded]
            return Args

        # Encode each input argument. This is the positional argument list which will be
        # wrapped into a Weld struct and passed to the Weld API.
        names = self.context.keys()
        names.sort()
        encoded = [self.encoder.encode(self.context[name]) for name in names]
        argtypes = [self.encoder.pyToWeldType(self.context[name]).cTypeClass for name in names]
        Args = args_factory(zip(names, argtypes))
        weld_args = Args()
        for name, value in zip(names, encoded):
            setattr(weld_args, name, value)

        
        void_ptr = ctypes.cast(ctypes.byref(weld_args), ctypes.c_void_p)
        arg = weld.WeldValue(void_ptr)
        module = weld.WeldModule(function, weld.WeldConf(), weld.WeldError())
        weld_ret = module.run(weld.WeldConf(), arg, weld.WeldError())
        restype = POINTER(restype.cTypeClass)
        data = ctypes.cast(weld_ret.data(), restype)
        result = self.decoder.decode(data, restype)
        return result

