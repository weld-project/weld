"""Summary

Attributes:
    numpy_to_nvl_type_mapping (TYPE): Description
"""
import subprocess

from weldobject import *
import weldutils as utils

numpy_to_weld_type_mapping = {
    'str': WeldVec(WeldChar()),
    'int32': WeldInt(),
    'int64': WeldLong(),
    'float32': WeldFloat(),
    'float64': WeldDouble(),
    'bool': WeldBit()
}


class NumPyEncoder(WeldObjectEncoder):
    """Summary

    Attributes:
        utils (TYPE): Description
    """

    def __init__(self):
        """Summary
        """
        subprocess.call(
            "cd $PANDAS_NVL_HOME; make convertor >/dev/null 2>/dev/null",
            shell=True)
        pandasNVLDir = os.environ.get("PANDAS_NVL_HOME")
        self.utils = ctypes.PyDLL(utils.to_shared_lib(
            os.path.join(pandasNVLDir, "numpy_nvl_convertor")))

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
                numpy_to_nvl = self.utils.numpy_to_nvl_int_arr
            elif obj.ndim == 1 and obj.dtype == 'int64':
                numpy_to_nvl = self.utils.numpy_to_nvl_long_arr
            elif obj.ndim == 1 and obj.dtype == 'float64':
                numpy_to_nvl = self.utils.numpy_to_nvl_double_arr
            elif obj.ndim == 2 and obj.dtype == 'int32':
                numpy_to_nvl = self.utils.numpy_to_nvl_int_arr_arr
            elif obj.ndim == 2 and obj.dtype == 'int64':
                numpy_to_nvl = self.utils.numpy_to_nvl_long_arr_arr
            elif obj.ndim == 2 and obj.dtype == 'float64':
                numpy_to_nvl = self.utils.numpy_to_nvl_double_arr_arr
            elif obj.ndim == 1 and obj.dtype == 'bool':
                numpy_to_nvl = self.utils.numpy_to_nvl_bool_arr
            else:
                numpy_to_nvl = self.utils.numpy_to_nvl_char_arr_arr
        elif isinstance(obj, str):
            numpy_to_nvl = self.utils.numpy_to_nvl_char_arr
        else:
            raise Exception("Unable to encode; invalid object type")

        numpy_to_nvl.restype = self.pyToWeldType(obj).cTypeClass
        numpy_to_nvl.argtypes = [py_object]
        nvl_vec = numpy_to_nvl(obj)
        return nvl_vec


class NumPyDecoder(WeldObjectDecoder):
    """Summary

    Attributes:
        utils (TYPE): Description
    """

    def __init__(self):
        """Summary
        """
        subprocess.call(
            "cd $PANDAS_NVL_HOME; make convertor >/dev/null 2>/dev/null",
            shell=True)
        pandasNVLDir = os.environ.get("PANDAS_NVL_HOME")
        self.utils = ctypes.PyDLL(utils.to_shared_lib(
            os.path.join(pandasNVLDir, "numpy_nvl_convertor")))

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
            nvl_to_numpy = self.utils.nvl_to_numpy_bool_arr
        elif restype == WeldVec(WeldDouble()):
            nvl_to_numpy = self.utils.nvl_to_numpy_double_arr
        elif restype == WeldVec(WeldVec(WeldChar())):
            nvl_to_numpy = self.utils.nvl_to_numpy_char_arr_arr
        elif restype == WeldVec(WeldVec(WeldInt())):
            nvl_to_numpy = self.utils.nvl_to_numpy_int_arr_arr
        elif restype == WeldVec(WeldVec(WeldLong())):
            nvl_to_numpy = self.utils.nvl_to_numpy_long_arr_arr
        elif restype == WeldVec(WeldInt()):
            nvl_to_numpy = self.utils.nvl_to_numpy_int_arr
        else:
            raise Exception("Unable to decode; invalid return type")

        nvl_to_numpy.restype = py_object
        nvl_to_numpy.argtypes = [restype.cTypeClass]

        data = cweld.WeldValue(obj).data()
        result = ctypes.cast(data, ctypes.POINTER(restype.cTypeClass)).contents
        ret_vec = nvl_to_numpy(result)
        return ret_vec
