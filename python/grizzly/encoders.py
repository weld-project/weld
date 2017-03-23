"""
Implements some common standard encoders and decoders mapping
Python types to Weld types.
"""

from weld.types import *

from weld.weldobject import WeldObjectEncoder, WeldObjectDecoder

import numpy as np
import ctypes


def dtype_to_weld_type(dtype):
    if dtype == 'str':
        return WeldVec(WeldChar())
    if dtype == 'int32':
        return WeldInt()
    elif dtype == 'int64':
        return WeldLong()
    elif dtype == 'float32':
        return WeldFloat()
    elif dtype == 'float64':
        return WeldDouble()
    elif dtype == 'bool':
        return WeldBit()
    elif str(dtype).startswith('|S'):
        return WeldVec(WeldChar())
    else:
        raise ValueError("unsupported dtype {}".format(dtype))


class GrizzlyEncoder(WeldObjectEncoder):

    def _check(self, obj):
        """
        Checks whether this type is supported by the Grizzly encoder
        (either one or two dimensional NumPy arrays or strings).
        """
        if isinstance(obj, np.ndarray):
            assert obj.ndim == 1 or obj.ndim == 2
        elif not isinstance(obj, str):
            raise Exception("Obj to encode not an ndarray or string!")

    def _encode_1D_array(self, obj, elem_type):
        c_class = WeldVec(elem_type).cTypeClass
        elem_class = elem_type.cTypeClass
        ptr = obj.ctypes.data_as(POINTER(elem_class))
        size = ctypes.c_int64(len(obj))
        return c_class(ptr=ptr, size=size)

    def _encode_2D_array(self, obj, elem_type):
        elem_class = elem_type.cTypeClass
        row_c_class = WeldVec(elem_type).cTypeClass
        c_class = WeldVec(WeldVec(elem_type)).cTypeClass
        encoded_rows = ctypes.create_string_buffer(sizeof(row_c_class) * len(obj))
        i = 0
        for row in obj:
            ptr = row.ctypes.data_as(POINTER(elem_class))
            size = ctypes.c_int64(len(row))
            encoded_row = row_c_class(ptr=ptr, size=size)
            ctypes.memmove(byref(encoded_rows, sizeof(row_c_class) * i),
                           byref(encoded_row), sizeof(row_c_class))
            i += 1
        ptr = ctypes.cast(encoded_rows, POINTER(row_c_class))
        size = ctypes.c_int64(len(obj))
        return c_class(ptr=ptr, size=size)

    def _encode_str(self, obj):
        elem_class = WeldChar().cTypeClass
        c_class = WeldVec(WeldChar()).cTypeClass
        ptr = ctypes.cast(ctypes.create_string_buffer(obj), POINTER(elem_class))
        size = ctypes.c_int64(len(obj))
        return c_class(ptr=ptr, size=size)

    def _encode_str_array(self, obj):
        elem_class = WeldChar().cTypeClass
        row_c_class = WeldVec(WeldChar()).cTypeClass
        c_class = WeldVec(WeldVec(WeldChar())).cTypeClass
        encoded_rows = ctypes.create_string_buffer(sizeof(row_c_class) * len(obj))
        i = 0
        for row in obj:
            ptr = ctypes.cast(ctypes.create_string_buffer(row), POINTER(elem_class))
            size = ctypes.c_int64(len(row))
            encoded_row = row_c_class(ptr=ptr, size=size)
            ctypes.memmove(byref(encoded_rows, sizeof(row_c_class) * i),
                           byref(encoded_row), sizeof(row_c_class))
            i += 1
        ptr = ctypes.cast(encoded_rows, POINTER(row_c_class))
        size = ctypes.c_int64(len(obj))
        return c_class(ptr=ptr, size=size)

    def encode(self, obj):
        # TODO: Might need to transpose results in some cases.
        self._check(obj)
        if isinstance(obj, np.ndarray):
            elem_type = dtype_to_weld_type(obj.dtype)
            if not (elem_type == WeldVec(WeldChar())):
                if obj.ndim == 1:
                    return self._encode_1D_array(obj, elem_type)
                elif obj.ndim == 2:
                    return self._encode_2D_array(obj, elem_type)
            else:
                if obj.ndim == 1:
                    return self._encode_str_array(obj)
        elif isinstance(obj, str):
            return self._encode_str(obj)
        raise Exception("Encoding of %s not currently supported!" % obj)

    def pyToWeldType(self, obj):
        self._check(obj)
        if isinstance(obj, np.ndarray):
            elem_type = dtype_to_weld_type(obj.dtype)
            for i in xrange(obj.ndim):
                elem_type = WeldVec(elem_type)
            return elem_type
        elif isinstance(obj, str):
            return WeldVec(WeldChar())
        # Should not reach here.
        return None


class GrizzlyDecoder(WeldObjectDecoder):

    def decode(self, obj, restype):
        if isinstance(restype, WeldVec):
            obj = obj.contents
            size = obj.size
            data = obj.ptr
            elem_type = restype.elemType
            elem_class = elem_type.cTypeClass
            if elem_type == WeldVec(WeldChar()):
                output = []
                data_int = ctypes.cast(data, c_void_p).value
                for i in xrange(size):
                    row = ctypes.cast(data_int, POINTER(elem_class))
                    output.append(self.decode(row, elem_type))
                    data_int += sizeof(elem_class)
                return output
            elif elem_type == WeldChar():
                return ctypes.string_at(data, size)
            result = np.fromiter(data, dtype=elem_class, count=size)
            return result
        elif (isinstance(restype, WeldInt) or isinstance(restype, WeldLong) or
              isinstance(restype, WeldFloat) or isinstance(restype, WeldDouble)):
            result = obj.contents.value
            return result
        else:
            raise Exception("Unknown result type in decoder")
