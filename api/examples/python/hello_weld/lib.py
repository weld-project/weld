import os
import sys

home = os.environ.get("WELD_HOME")
if home is None:
    home = "../../../python/"
if home[-1] != "/":
    home += "/"

libpath = home + "api/python"
sys.path.append(libpath)

from weld.weldobject import *

import weld
import ctypes

import numpy as np

class HelloWeldVectorEncoder(WeldObjectEncoder):
    def _check(self, obj):
        assert isinstance(obj, np.ndarray)
        assert obj.dtype == 'int32'
        assert obj.ndim == 1

    def encode(self, obj):
        self._check(obj)
        c_class = WeldVec(WeldInt()).cTypeClass
        elem_class = WeldInt().cTypeClass
        data = obj.ctypes.data_as(POINTER(elem_class))
        size = ctypes.c_int64(len(obj))
        return c_class(data=data, size=size)

    def pyToWeldType(self, obj):
        self._check(obj)
        return WeldVec(WeldInt())


class HelloWeldVectorDecoder(WeldObjectDecoder):
    def decode(self, obj, restype):
        obj = obj.contents
        size = obj.size
        data = obj.data
        result = np.fromiter(data, dtype=np.int32, count=size)
        return result

_encoder = HelloWeldVectorEncoder()
_decoder = HelloWeldVectorDecoder()

def add(vector, number):
    template = "map({0}, |e| e + {1})"
    obj = WeldObject(_encoder, _decoder)
    if isinstance(vector, WeldObject):
        obj.update(vector, WeldVec(WeldInt()))
        obj.weld_code = template.format(vector.weld_code, str(number))
    else:
        name = obj.update(vector, WeldVec(WeldInt()))
        obj.weld_code = template.format(name, str(number))
    return obj

# Small Test program.

v = np.array([1,2,3,4,5], dtype='int32')
v = add(v, 1)
v = add(v, 5)
v = add(v, 10)
print v.weld_code

# Evaluate as a numpy array
print v.evaluate(WeldVec(WeldInt()))


