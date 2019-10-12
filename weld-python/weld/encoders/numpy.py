"""
Implements encoders for NumPy values.

The Weld package includes native convertors for NumPy arrays because NumPy is
the standard way for interacting with C-like array data.

The encoder in this package accepts ndarray or its subclasses. The decoder in
this module returns a subclass of ndarray called `weldbasearray`, which may
hold a reference to a `WeldContext`. This prevents arrays backed by memory
allocated in Weld from being freed before the array's reference count drops to
0.

Zero-copy conversions (in particular, to 1D arrays) are implemented here
directly since they only involve a pointer copy.
"""

import ctypes
import numpy as np

from .encoder_base import *
from ..types import *

class weldbasearray(np.ndarray):
    """ A NumPy array possibly backed by a `WeldContext`.

    This class is a wrapper around the NumPy `ndarray` class, but it contains
    an additional `weld_context` attribute. This attribute references the
    memory that backs the array, if the array was returned by Weld (or created
    from another array that was returned by Weld). It prevents memory owned by
    the context from being freed before all references to the array are
    deleted.

    This class also contains an additional method, `copy2numpy`, which
    deep-copies the data referenced by this array to a regular `ndarray`. The
    resulting array does not hold a reference to the context or the original
    array.

    If the `weld_context` attribtue is `None`, this class acts like a regular
    `ndarray`, and the `copy2numpy` function simply copies this array.

    """

    def __new__(cls, input_array, weld_context=None):
        """ Instance initializer.

        Parameters
        ----------
        weld_context : WeldContext or None
            If this is not `None`, it should be the context that owns the
            memory for `input_array`.

        """
        obj = np.asarray(input_array).view(cls)
        obj.weld_context = weld_context
        return obj

    def __array_finalize__(self, obj):
        """ Finalizes array. See the NumPy documentation. """
        if obj is None:
            return
        self.weld_context = getattr(obj, 'weld_context', None)

    def copy2numpy(self):
        """ Copies this array's data into a new NumPy `ndarray`.

        This is an alias for `np.array(arr, copy=True)`

        Examples
        --------
        >>> arr = weldbasearray([1, 2, 3])
        >>> arr
        weldbasearray([1, 2, 3])
        >>> arr.copy2numpy()
        array([1, 2, 3])

        """
        return np.array(self, copy=True)

# Maps a string dtype representation to a Weld scalar type.
_known_types = {
        'int8': I8(),
        'int16': I16(),
        'int32': I32(),
        'int64': I64(),
        'uint8': U8(),
        'uint16': U16(),
        'uint32': U32(),
        'uint64': U64(),
        'float32': F32(),
        'float': F64(),
        'double': F64(),
        'float64': F64(),
        'bool': Bool()
        }

# Reverse of the above.
_known_types_weld2dtype = {v: k for k, v in _known_types.items()}

def weld_type_to_dtype(ty):
    """Converts a Weld type to a NumPy dtype.

    Examples
    --------
    >>> weld_type_to_dtype(I32())
    dtype('int32')
    >>> weld_type_to_dtype(F32())
    dtype('float32')
    >>> weld_type_to_dtype(F64())
    dtype('float64')

    Parameters
    ----------
    ty: WeldType
        The type to convert

    Returns
    -------
    dtype or None
        Returns None if the type is not recognized.
    
    """
    if ty in _known_types_weld2dtype:
        return np.dtype(_known_types_weld2dtype[ty])

def dtype_to_weld_type(ty):
    """Converts a NumPy data type to a Weld type.

    The data type can be a any type that can be converted to a NumPy dtype,
    e.g., a string (e.g., 'int32') or a NumPy scalar type (e.g., np.int32). The
    type chosen follows the rules specified by NumPy here:

    https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#dtype

    For example, 'i8' will map to an int64 in Weld, since it indicates a signed
    integer that has eight bytes.

    Examples
    --------
    >>> dtype_to_weld_type('int32')
    <weld.types.I32 object at 0x...>
    >>> dtype_to_weld_type('float')
    <weld.types.F64 object at 0x...>
    >>> dtype_to_weld_type('i8')
    <weld.types.I64 object at 0x...>
    >>> dtype_to_weld_type(np.int16)
    <weld.types.I16 object at 0x...>

    Parameters
    ----------
    ty : str or dtype or NumPy scalar type
        The NumPy type to convert

    Returns
    -------
    WeldType

    """
    if not isinstance(ty, np.dtype):
        ty = np.dtype(ty)

    ty = str(ty)
    if ty in _known_types:
        return _known_types.get(ty)

    if ty.startswith('S'):
        raise TypeError("Python 2 strings not supported -- use Unicode")
    if ty.find('U') != -1:
        raise NotImplementedError("Unicode strings not yet supported")

class NumPyWeldEncoder(WeldEncoder):

    @staticmethod
    def _convert_1d_array(array, check_type=None):
        """Converts a 1D NumPy array into a Weld vector.

        The vector holds a reference to the array.

        Examples
        --------
        >>> arr = np.array([1, 2, 3])
        >>> encoded = NumPyWeldEncoder._convert_1d_array(arr)
        >>> encoded.size
        3
        >>> encoded.data.contents
        c_long(1)

        Parameters
        ----------
        array : ndarray
            A one-dimensional NumPy array.
        check_type : WeldType, optional
            If this value is passed, this function will check whether the
            array's derived WeldType is equal to the passed type.  Defaults to
            None.

        Returns
        -------
        WeldVec

        """
        elem_type = dtype_to_weld_type(array.dtype)
        vec_type = WeldVec(elem_type)

        if check_type is not None:
            assert check_type == vec_type

        data = array.ctypes.data_as(ctypes.POINTER(elem_type.ctype_class))
        length = ctypes.c_int64(len(array))

        vec = vec_type.ctype_class()
        vec.data = data
        vec.size = length
        return vec

    def encode(self, obj, ty):
        if isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                return NumPyWeldEncoder._convert_1d_array(obj, check_type=ty)
            else:
                raise NotImplementedError
        else:
            raise TypeError("Unexpected type {} in NumPy encoder".format(type(obj)))


class NumPyWeldDecoder(WeldDecoder):
    """ Decodes an encoded Weld array into a NumPy array.

    >>> arr = np.array([1,2,3], dtype='int32')
    >>> encoded = NumPyWeldEncoder().encode(arr, WeldVec(I32()))
    >>> NumPyWeldDecoder().decode(ctypes.pointer(encoded), WeldVec(I32()))
    weldbasearray([1, 2, 3], dtype=int32)

    """

    @staticmethod
    def _memory_buffer(c_pointer, length, dtype):
        """Creates a Python memory buffer from the pointer.

        Parameters
        ----------

        c_pointer : ctypes pointer
            the pointer the buffer points to
        length : int
            the array length
        dtype : NumPy dtype
            the type of the elements in the buffer.

        Returns
        -------
        memory

        """
        arr_size = dtype.itemsize * length
        buf_from_mem = ctypes.pythonapi.PyMemoryView_FromMemory
        buf_from_mem.restype = ctypes.py_object
        buf_from_mem.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
        return buf_from_mem(c_pointer, arr_size, 0x100)


    def _numpy_type(weld_type):
        """Infers the ndarray dimensions and dtype from a Weld type.

        Throws a TypeError if the weld_type cannot be represented as an ndarray
        of some scalar type.

        Parameters
        ----------
        weld_type : WeldType
            The type to check

        Returns
        -------
        (int, dtype) tuple
            The first element is the nubmer of dimensions and the second
            element is the dtype.

        >>> NumPyWeldDecoder._numpy_type(WeldVec(I8()))
        (1, dtype('int8'))
        >>> NumPyWeldDecoder._numpy_type(WeldVec(WeldVec(F32())))
        (2, dtype('float32'))
        >>> NumPyWeldDecoder._numpy_type(I32())
        Traceback (most recent call last):
        ...
        TypeError: type cannot be represented as ndarray

        """
        if not isinstance(weld_type, WeldVec):
            raise TypeError("type cannot be represented as ndarray")

        dimension = 1
        elem_type = weld_type.elem_type
        if isinstance(elem_type, WeldVec):
            (inner_dims, inner_ty) = NumPyWeldDecoder._numpy_type(elem_type)
            dimension += inner_dims
        else:
            try:
                inner_ty = weld_type_to_dtype(elem_type)
            except:
                raise TypeError("unknown element type {}".format(elem_type))
        return (dimension, inner_ty)


    def decode(self, obj, restype, context=None):
        # A 1D NumPy array
        obj = obj.contents
        (dims, dtype) = NumPyWeldDecoder._numpy_type(restype)
        if dims == 1:
            elem_type = restype.elem_type
            buf = NumPyWeldDecoder._memory_buffer(obj.data, obj.size, dtype)
            array = np.frombuffer(buf, dtype=dtype, count=obj.size)
            return weldbasearray(array, weld_context=context)
        else:
            raise TypeError("Unsupported type {} in NumPy decoder".format(type(obj)))
