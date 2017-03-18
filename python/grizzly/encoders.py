"""Summary

Attributes:
    numpy_to_weld_type_mapping (TYPE): Description
"""

from weld.weldobject import *
import numpy as np
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
