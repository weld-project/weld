"""Summary
"""
#
# NvlObject
#
# Holds an object that can be evaluated.
#

import ctypes
import numpy as np
import os
import subprocess
import sys

import time

from nvltypes import *
import nvlutils as utils

import weld


class NvlObjectEncoder(object):
    """
    An abstract class that must be overwridden by libraries. This class
    is used to marshall objects from Python types to NVL types.
    """
    def encode(obj):
        """
        Encodes an object. All objects encodable by this encoder should return
        a valid NVL type using pyToNvlType.

        Args:
            obj (TYPE): Description

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError

    def pyToNvlType(obj):
        """
        Returns an NvlType corresponding to a Python object

        Args:
            obj (TYPE): Description

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError


class NvlObjectDecoder(object):
    """
    An abstract class that must be overwridden by libraries. This class
    is used to marshall objects from NVL types to Python types
    """
    def decode(obj, restype):
        """Summary

        Args:
            obj (TYPE): Description
            restype (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError


class NvlObject(object):
    """
    Holds an NVL program to be lazily compiled and evaluated,
    along with any context required to evaluate the program.

    Libraries that use the NVL API return NvlObjects, which represent
    "lazy executions" of programs. NvlObjects can build on top of each other,
    i.e. libraries should be implemented so they can accept both their
    native types and NvlObjects.

    An NvlObject contains an NVL program as a string, along with a context.
    The context maps names in the NVL program to concrete values.

    When an NvlObject is evaluated, it uses its encode and decode functions to
    marshall native library types into types that NVL understands. The basic flow
    of evaluating an NVL expression is thus:

    1. "Finish" the NVL program by adding a function header
    2. Compile the NVL program and load it as a dynamic library
    3. For each argument, run object.encoder on each argument. See NvlObjectEncoder
    for details on how this works
    4. Pass the encoded arguments to NVL
    5. Run the decoder on the return value. See NvlObjectDecoder for details on how
    this works.
    6. Return the decoded value.

    Attributes:
        argtypes (dict): Description
        constants (list): Description
        context (dict): Description
        dataType (TYPE): Description
        decoder (TYPE): Description
        encoder (TYPE): Description
        nvl (str): Description
        objectId (TYPE): Description
    """

    # Counter for assigning variable names
    _var_num = 0
    _obj_id = 100

    def __init__(self, encoder, decoder):
        """Summary

        Args:
            encoder (TYPE): Description
            decoder (TYPE): Description
        """
        self.encoder = encoder
        self.decoder = decoder

        # NVL program
        self.nvl = ""
        self.constants = []

        # NVL type, which represents the return type of this NVL program.
        # decoder must support decoding the return type.
        self.dataType = None

        # Assign a unique ID to the context
        self.objectId = NvlObject._obj_id
        NvlObject._obj_id += 1

        # Maps name -> input data
        self.context = {}
        # Maps name -> arg type (for arguments that don't need to be encoded)
        self.argtypes = {}

    def __repr__(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return self.nvl + " " + str(self.context)

    def update(self, value, argtype=None):
        """
        Update this context. if value is another context,
        the names from that context are added into this one.
        Otherwise, a new name is assigned and returned.

        Args:
            value (TYPE): Description
            argtype (None, optional): Description

        """
        if isinstance(value, NvlObject):
            self.context.update(value.context)
            self.constants.extend(value.constants)
        else:
            name = "e" + str(NvlObject._var_num)
            NvlObject._var_num += 1
            self.context[name] = value
            if argtype is not None:
                self.argtypes[name] = argtype
            return name

    def _compileAndLoadRuntimeSO(self, text, verbose):
        """
        Load the runtime by compiling the NVL text and loading it as a dynamic library.

        Args:
            text (TYPE): Description
            verbose (TYPE): Description

        Raises:
            RuntimeError: Description
        """
        code = ctypes.c_char_p(text)
        conf = weld.weld_conf_new()
        err = ctypes.c_void_p()
        module = weld.weld_module_compile(code, conf, ctypes.byref(err))
        return module

    def toNvlFunc(self):
        """Summary

        Returns:
            TYPE: Description
        """
        names = sorted(self.context.keys())
        for name in names:
            if name in self.argtypes:
                arg_strs.append("{0}: {1}".format(
                    str(name), self.argtypes[name]))
            else:
                arg_strs.append("{0}: {1}".format(str(name), str(
                    self.encoder.pyToNvlType(self.context[name]))))
        header = "|" + ", ".join(arg_strs) + "|"
        text = header + " => " + \
            "\n".join(list(set(self.constants))) + "\n" + self.nvl
        return text

    def evaluate(self, restype, verbose=True, decode=True):
        """Summary

        Args:
            restype (TYPE): Description
            verbose (bool, optional): Description
            decode (bool, optional): Description

        Returns:
            TYPE: Description
        """

        names = self.context.keys()
        arg_strs = []
        for name in names:
            if name in self.argtypes:
                arg_strs.append("{0}: {1}".format(
                    str(name), self.argtypes[name]))
            else:
                arg_strs.append("{0}: {1}".format(str(name), str(
                    self.encoder.pyToNvlType(self.context[name]))))
        header = "|" + ", ".join(arg_strs) + "|"
        text = header + "\n".join(list(set(self.constants))) + "\n" + self.nvl

        # Returns a wrapped ctypes Structure
        def args_factory(encoded):
            """Summary

            Args:
                encoded (TYPE): Description

            Returns:
                TYPE: Description
            """
            class Args(ctypes.Structure):
                """Summary
                """
                _fields_ = [e for e in encoded]
            return Args

        module = self._compileAndLoadRuntimeSO(text, verbose)
        real_start = time.time()
        # Encode each input argument. This is the positional argument list which will be
        # wrapped into an NVL struct and passed to the NVL library.

        start = time.time()
        encoded = []
        argtypes = []
        for name in names:
            if name in self.argtypes:
                argtypes.append(self.argtypes[name].cTypeClass)
                encoded.append(self.context[name])
            else:
                argtypes.append(self.encoder.pyToNvlType(
                    self.context[name]).cTypeClass)
                encoded.append(self.encoder.encode(self.context[name]))
        end = time.time()
        if verbose:
            print "Total time encoding: " + str(end - start)

        Args = args_factory(zip(names, argtypes))
        nvl_args = Args()
        for name, value in zip(names, encoded):
            setattr(nvl_args, name, value)

        # Set up the runtime
        arg_obj = weld.weld_value_new(ctypes.byref(nvl_args))

        # The returned NVL object
        start = time.time()
        err2 = ctypes.c_void_p()
        conf = weld.weld_conf_new()
        nvl_ret = weld.weld_module_run(
            module, conf, arg_obj, ctypes.byref(err2))
        end = time.time()
        if verbose:
            print "NVL Runtime: " + str(end - start)
        start = time.time()
        if decode:
            result = self.decoder.decode(nvl_ret, restype)
        else:
            data = weld.weld_value_data(nvl_ret)
            result = ctypes.cast(data, ctypes.POINTER(
                ctypes.c_int64)).contents.value
        end = time.time()
        real_end = time.time()
        if verbose:
            print "Total time decoding: " + str(end - start)
            print "Total end-to-end time: " + str(real_end - real_start)

        return result
