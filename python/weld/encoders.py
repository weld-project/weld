"""
Implements some common standard encoders and decoders mapping
Python types to Weld types.
"""

from types import *

from weldobject import WeldObjectEncoder, WeldObjectDecoder

import numpy as np
import ctypes

def dtype_to_weld_type(dtype):
    if dtype == 'int32':
        return WeldInt()
    elif dtype == 'int64':
        return WeldLong()
    elif dtype == 'float32':
        return WeldFloat()
    elif dtype == 'float64':
        return WeldDouble()
    else:
        raise ValueError("unsupported dtype {}".format(dtype))

class NumpyArrayEncoder(WeldObjectEncoder):
    def _check(self, obj):
        """
        Checks whether this NumPy array is supported by Weld.
        """
        assert isinstance(obj, np.ndarray)
        assert obj.ndim == 1

    def encode(self, obj):
        self._check(obj)
        elem_type = dtype_to_weld_type(obj.dtype)
        c_class = WeldVec(elem_type).cTypeClass
        elem_class = elem_type.cTypeClass
        ptr = obj.ctypes.data_as(POINTER(elem_class))
        size = ctypes.c_int64(len(obj))
        return c_class(ptr=ptr, size=size)

    def pyToWeldType(self, obj):
        self._check(obj)
        return WeldVec(dtype_to_weld_type(obj.dtype))


class NumpyArrayDecoder(WeldObjectDecoder):
    def decode(self, obj, restype):
        assert isinstance(restype, WeldVec)
        obj = obj.contents
        size = obj.size
        data = obj.data
        dtype = restype.elemType.cTypeClass
        result = np.fromiter(data, dtype=dtype, count=size)
        return result

class ScalarDecoder(WeldObjectDecoder):
     def decode(self, obj, restype):
         assert isinstance(restype, WeldLong)
         result = obj.contents.value
         return result
