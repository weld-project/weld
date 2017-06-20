"""Summary

Attributes:
    numpy_to_weld_type_mapping (TYPE): Description
"""

from weld.weldobject import *
import numpy as np
import pyarrow as pa
from weld import arrow_compat as ac
import os
import sys


numpy_to_weld_type_mapping = {
    'str': WeldVec(WeldChar()),
    'int32': WeldInt(),
    'int64': WeldLong(),
    'float32': WeldFloat(),
    'float64': WeldDouble(),
    'bool': WeldBit()
}

# The types below are those that Arrow and Weld layout the same way in memory
# on little endian systems.
arrow_to_weld_type_mapping = {
    'int32': WeldInt(),
    'int64': WeldLong(),
    'float32': WeldFloat(),
    'float64': WeldDouble()
}

# Weld bits are uint8, which allows Arrow support for functions which return
# boolean results.
weld_to_arrow_type_mapping = {
    WeldInt().cTypeClass: pa.int32(),
    WeldLong().cTypeClass: pa.int64(),
    WeldFloat().cTypeClass: pa.float32(),
    WeldDouble().cTypeClass: pa.float64(),
    WeldBit().cTypeClass: pa.uint8()
}

def to_shared_lib(name):
    """
    Returns the name with the platform dependent shared library extension.

    Args:
    name (TYPE): Description
    """
    if sys.platform.startswith('linux'):
        return name + ".so"
    elif sys.platform.startswith('darwin'):
        return name + ".dylib"
    else:
        sys.exit(1)


class NumPyEncoder(WeldObjectEncoder):
    """Summary

    Attributes:
        utils (TYPE): Description
    """

    def __init__(self):
        """Summary
        """
        # TODO: Remove dependency on WELD_HOME environment variable.
        if "WELD_HOME" not in os.environ:
            raise Exception("WELD_HOME environment variable not set!")
        self.utils = ctypes.PyDLL(to_shared_lib(
            os.path.join(os.environ["WELD_HOME"] + "/python/grizzly/",
                         "numpy_weld_convertor")))

    def pyToWeldType(self, obj):
        """Summary

        Args:
            obj (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            Exception: Description
        """
        if isinstance(obj, np.ndarray):
            dtype = str(obj.dtype)
            if dtype == 'int32':
                base = WeldInt()
            elif dtype == 'int64':
                base = WeldLong()
            elif dtype == 'float32':
                base = WeldFloat()
            elif dtype == 'float64':
                base = WeldDouble()
            elif dtype == 'bool':
                base = WeldBit()
            else:
                base = WeldVec(WeldChar())  # TODO: Fix this
            for i in xrange(obj.ndim):
                base = WeldVec(base)
        elif isinstance(obj, str):
            base = WeldVec(WeldChar())
        else:
            raise Exception("Invalid object type: unable to infer NVL type")
        return base

    def encode(self, obj):
        """Summary

        Args:
            obj (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            Exception: Description
        """
        if isinstance(obj, np.ndarray):
            if obj.ndim == 1 and obj.dtype == 'int32':
                numpy_to_weld = self.utils.numpy_to_weld_int_arr
            elif obj.ndim == 1 and obj.dtype == 'int64':
                numpy_to_weld = self.utils.numpy_to_weld_long_arr
            elif obj.ndim == 1 and obj.dtype == 'float64':
                numpy_to_weld = self.utils.numpy_to_weld_double_arr
            elif obj.ndim == 2 and obj.dtype == 'int32':
                numpy_to_weld = self.utils.numpy_to_weld_int_arr_arr
            elif obj.ndim == 2 and obj.dtype == 'int64':
                numpy_to_weld = self.utils.numpy_to_weld_long_arr_arr
            elif obj.ndim == 2 and obj.dtype == 'float64':
                numpy_to_weld = self.utils.numpy_to_weld_double_arr_arr
            elif obj.ndim == 1 and obj.dtype == 'bool':
                numpy_to_weld = self.utils.numpy_to_weld_bool_arr
            else:
                numpy_to_weld = self.utils.numpy_to_weld_char_arr_arr
        elif isinstance(obj, str):
            numpy_to_weld = self.utils.numpy_to_weld_char_arr
        else:
            raise Exception("Unable to encode; invalid object type")

        numpy_to_weld.restype = self.pyToWeldType(obj).cTypeClass
        numpy_to_weld.argtypes = [py_object]
        weld_vec = numpy_to_weld(obj)
        return weld_vec


class ArrowEncoder(WeldObjectEncoder):
    """Summary

    Attributes:
        utils (TYPE): Description
    """


    def pyToWeldType(self, obj):
        """Summary

        Args:
            obj (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            Exception: Description
        """
        if not isinstance(obj, pa.Array):
            raise Exception('ArrowEncoder does not support encoding %s type' % str(type(obj)))
        dtype = str(obj.type.to_pandas_dtype().__name__)
        if dtype not in arrow_to_weld_type_mapping:
            raise Exception('ArrowEncoder does not support encoding Arrays of ' + dtype)
        return WeldVec(arrow_to_weld_type_mapping[dtype])

    def encode(self, obj):
        """Summary

        Args:
            obj (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            Exception: Description
        """
        assert sys.byteorder == 'little'
        return ac.arrow_to_weld_arr(obj, self.pyToWeldType(obj).cTypeClass)


class NumPyDecoder(WeldObjectDecoder):
    """Summary

    Attributes:
        utils (TYPE): Description
    """

    def __init__(self):
        """Summary
        """
        # TODO: Remove dependency on WELD_HOME environment variable.
        if "WELD_HOME" not in os.environ:
            raise Exception("WELD_HOME environment variable not set!")
        self.utils = ctypes.PyDLL(to_shared_lib(
            os.path.join(os.environ["WELD_HOME"] + "/python/grizzly",
                         "numpy_weld_convertor")))

    def decode(self, obj, restype):
        """Summary

        Args:
            obj (TYPE): Description
            restype (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            Exception: Description
        """
        if restype == WeldInt():
            data = cweld.WeldValue(obj).data()
            result = ctypes.cast(data, ctypes.POINTER(c_int)).contents.value
            return result
        elif restype == WeldLong():
            data = cweld.WeldValue(obj).data()
            result = ctypes.cast(data, ctypes.POINTER(c_long)).contents.value
            return result
        elif restype == WeldFloat():
            data = cweld.WeldValue(obj).data()
            result = ctypes.cast(data, ctypes.POINTER(c_float)).contents.value
            return float(result)
        elif restype == WeldDouble():
            data = cweld.WeldValue(obj).data()
            result = ctypes.cast(data, ctypes.POINTER(c_double)).contents.value
            return float(result)

        # Obj is a WeldVec(WeldInt()).cTypeClass, which is a subclass of
        # ctypes._structure
        if restype == WeldVec(WeldBit()):
            weld_to_numpy = self.utils.weld_to_numpy_bool_arr
        elif restype == WeldVec(WeldDouble()):
            weld_to_numpy = self.utils.weld_to_numpy_double_arr
        elif restype == WeldVec(WeldVec(WeldChar())):
            weld_to_numpy = self.utils.weld_to_numpy_char_arr_arr
        elif restype == WeldVec(WeldVec(WeldInt())):
            weld_to_numpy = self.utils.weld_to_numpy_int_arr_arr
        elif restype == WeldVec(WeldVec(WeldLong())):
            weld_to_numpy = self.utils.weld_to_numpy_long_arr_arr
        elif restype == WeldVec(WeldVec(WeldDouble())):
            weld_to_numpy = self.utils.weld_to_numpy_double_arr_arr
        elif restype == WeldVec(WeldInt()):
            weld_to_numpy = self.utils.weld_to_numpy_int_arr
        else:
            raise Exception("Unable to decode; invalid return type")

        weld_to_numpy.restype = py_object
        weld_to_numpy.argtypes = [restype.cTypeClass]

        data = cweld.WeldValue(obj).data()
        result = ctypes.cast(data, ctypes.POINTER(restype.cTypeClass)).contents
        ret_vec = weld_to_numpy(result)
        return ret_vec


class ArrowDecoder(WeldObjectDecoder):
    """Summary

    Attributes:
        utils (TYPE): Description
    """
    def weldToArrowType(self, restype):
        """Summary

        Args:
            restype (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            Exception: Description
        """
        if not isinstance(restype, WeldVec):
            raise Exception('ArrowDecoder does not support decoding %s type' % str(type(restype)))
        dtype = restype.elemType.cTypeClass
        if dtype not in weld_to_arrow_type_mapping:
            raise Exception('ArrowDecoder does not support decoding Arrays of ' + str(dtype))
        return weld_to_arrow_type_mapping[dtype]


    def decode(self, obj, restype):
        """Summary

        Args:
            obj (TYPE): Description
            restype (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            Exception: Description
        """
        assert sys.byteorder == 'little'
        if restype == WeldInt():
            data = cweld.WeldValue(obj).data()
            result = ctypes.cast(data, ctypes.POINTER(c_int)).contents.value
            return result
        elif restype == WeldLong():
            data = cweld.WeldValue(obj).data()
            result = ctypes.cast(data, ctypes.POINTER(c_long)).contents.value
            return result
        elif restype == WeldFloat():
            data = cweld.WeldValue(obj).data()
            result = ctypes.cast(data, ctypes.POINTER(c_float)).contents.value
            return float(result)
        elif restype == WeldDouble():
            data = cweld.WeldValue(obj).data()
            result = ctypes.cast(data, ctypes.POINTER(c_double)).contents.value
            return float(result)
        data = cweld.WeldValue(obj).data()
        dtype = self.weldToArrowType(restype)
        result = ctypes.cast(data, ctypes.POINTER(restype.cTypeClass)).contents
        return ac.weld_arr_to_arrow(result, dtype)
