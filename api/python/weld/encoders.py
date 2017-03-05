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
        return WeldI32()
    elif dtype == 'int64':
        return WeldI64()
    elif dtype == 'float32':
        return WeldF32()
    elif dtype == 'float64':
        return WeldF64()
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
        data = obj.ctypes.data_as(POINTER(elem_class))
        size = ctypes.c_int64(len(obj))
        return c_class(data=data, size=size)

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
         assert isinstance(restype, WeldI64)
         result = obj.contents.value
         return result
