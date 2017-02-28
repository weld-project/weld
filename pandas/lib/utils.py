"""Summary

Attributes:
    numpy_to_nvl_type_mapping (TYPE): Description
"""
from nvlobject import *


numpy_to_nvl_type_mapping = {
    'str': NvlVec(NvlChar()),
    'int32': NvlInt(),
    'int64': NvlLong(),
    'float32': NvlFloat(),
    'float64': NvlDouble(),
    'bool': NvlBit()
}

class NumPyEncoder(NvlObjectEncoder):
    """Summary

    Attributes:
        utils (TYPE): Description
    """

    def __init__(self):
        """Summary
        """        
        subprocess.call("cd $PANDAS_NVL_HOME; make convertor >/dev/null 2>/dev/null", shell=True)
        pandasNVLDir = os.environ.get("PANDAS_NVL_HOME")
        self.utils = ctypes.PyDLL(utils.to_shared_lib(os.path.join(pandasNVLDir, "numpy_nvl_convertor")))

    def pyToNvlType(self, obj):
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
                base = NvlInt()
            elif dtype == 'int64':
                base = NvlLong()
            elif dtype == 'float32':
                base = NvlFloat()
            elif dtype == 'float64':
                base = NvlDouble()
            elif dtype == 'bool':
                base = NvlBit()
            else:
                base = NvlVec(NvlChar())  # TODO: Fix this
            for i in xrange(obj.ndim):
                base = NvlVec(base)
        elif type(obj) == str:
            base = NvlVec(NvlChar())
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
        elif type(obj) == str:
            numpy_to_nvl = self.utils.numpy_to_nvl_char_arr
        else:
            raise Exception("Unable to encode; invalid object type")

        numpy_to_nvl.restype = self.pyToNvlType(obj).cTypeClass
        numpy_to_nvl.argtypes = [py_object]
        nvl_vec = numpy_to_nvl(obj)
        return nvl_vec

class NumPyDecoder(NvlObjectDecoder):
    """Summary

    Attributes:
        utils (TYPE): Description
    """

    def __init__(self):
        """Summary
        """
        subprocess.call("cd $PANDAS_NVL_HOME; make convertor >/dev/null 2>/dev/null", shell=True)
        pandasNVLDir = os.environ.get("PANDAS_NVL_HOME")
        self.utils = ctypes.PyDLL(utils.to_shared_lib(os.path.join(pandasNVLDir, "numpy_nvl_convertor")))

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
        if restype == NvlInt():
            data = weld.weld_value_data(obj)
            result = ctypes.cast(data, ctypes.POINTER(c_int)).contents.value
            return result
        if restype == NvlDouble():
            data = weld.weld_value_data(obj)
            result = ctypes.cast(data, ctypes.POINTER(c_float)).contents.value
            return float(result)

        # Obj is a NvlVec(NvlInt()).cTypeClass, which is a subclass of ctypes._structure
        if restype == NvlVec(NvlBit()):
            nvl_to_numpy = self.utils.nvl_to_numpy_bool_arr
        elif restype == NvlVec(NvlDouble()):
            nvl_to_numpy = self.utils.nvl_to_numpy_double_arr
        elif restype == NvlVec(NvlVec(NvlChar())):
            nvl_to_numpy = self.utils.nvl_to_numpy_char_arr_arr
        elif restype == NvlVec(NvlVec(NvlInt())):
            nvl_to_numpy = self.utils.nvl_to_numpy_int_arr_arr
        elif restype == NvlVec(NvlVec(NvlLong())):
            nvl_to_numpy = self.utils.nvl_to_numpy_long_arr_arr
        elif restype == NvlVec(NvlInt()):
            nvl_to_numpy = self.utils.nvl_to_numpy_int_arr
        else:
            raise Exception("Unable to decode; invalid return type")

        nvl_to_numpy.restype = py_object
        nvl_to_numpy.argtypes = [restype.cTypeClass]
        
        data = weld.weld_value_data(obj)
        result = ctypes.cast(data, ctypes.POINTER(restype.cTypeClass)).contents
        ret_vec = nvl_to_numpy(result)
        return ret_vec
