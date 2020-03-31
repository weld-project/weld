"""
Tests NumPy encoders and decoders.
"""

import ctypes
import numpy as np

from .helpers import encdec_factory

from weld import WeldConf, WeldContext
from weld.encoders.numpy import *
from weld.types import *

encdec = encdec_factory(NumPyWeldEncoder, NumPyWeldDecoder, eq=np.allclose)

def array(dtype, length=5):
    """Creates a 1D NumPy array with the given data type.

    The array is filled with data [1...length).

    >>> array('int8')
    array([0, 1, 2, 3, 4], dtype=int8)
    >>> array('float32')
    array([0., 1., 2., 3., 4.], dtype=float32)

    Parameters
    ----------
    dtype: np.dtype
        data type of array elements
    length: int
        elements in array

    Returns
    -------
    np.ndarray

    """
    return np.arange(start=0, stop=length, dtype=dtype)


# Tests for ensuring weldbasearrays propagate their contexts properly

def test_baseweldarray_basics():
    x = np.array([1, 2, 3, 4, 5], dtype="int8")

    ctx = WeldContext(WeldConf())

    welded = weldbasearray(x, weld_context=ctx)
    assert welded.dtype == "int8"
    assert welded.weld_context is ctx

    sliced = welded[1:]
    assert np.allclose(sliced, np.array([2,3,4,5]))
    assert sliced.base is welded
    assert sliced.weld_context is ctx

    copied = sliced.copy2numpy()
    assert copied.base is None
    try:
        copied.ctx
        assert False
    except AttributeError as e:
        pass

# Tests for encoding and decoding 1D arrays

def test_bool_vec():
    # Booleans in NumPy, like in Weld, are represented as bytes.
    encdec(np.array([True, True, False, False, True], dtype='bool'),
            WeldVec(Bool()))

def test_i8_vec():
    encdec(array('int8'), WeldVec(I8()))

def test_i16_vec():
    encdec(array('int16'), WeldVec(I16()))

def test_i32_vec():
    encdec(array('int32'), WeldVec(I32()))

def test_i64_vec():
    encdec(array('int64'), WeldVec(I64()))

def test_u8_vec():
    encdec(array('uint8'), WeldVec(U8()))

def test_u16_vec():
    encdec(array('uint16'), WeldVec(U16()))

def test_u32_vec():
    encdec(array('uint32'), WeldVec(U32()))

def test_u64_vec():
    encdec(array('uint64'), WeldVec(U64()))

def test_float32_vec():
    encdec(array('float32'), WeldVec(F32()))

def test_float64_vec():
    encdec(array('float64'), WeldVec(F64()))

def test_struct_of_vecs():
    arrays = (array('float32'), array('uint16'), array('uint32'))
    ty = WeldStruct([WeldVec(F32()), WeldVec(U16()), WeldVec(U32())])
    encdec(arrays, ty)

def test_type_conversions():
    types = ['bool', 'int8', 'uint8', 'int16', 'uint16',
            'int32', 'uint32', 'int64', 'uint64', 'float32', 'float64']
    for left in types:
        for right in types:
            print ("testing {}, {}".format(left, right))
            larr = np.ones(1, dtype=left)
            rarr = np.ones(1, dtype=right)
            left_weld_type = dtype_to_weld_type(left)
            right_weld_type = dtype_to_weld_type(right)
            got = binop_output_type(left_weld_type, right_weld_type)
            got = weld_type_to_dtype(got)
            expected = (larr + rarr).dtype
            assert expected == got, "input: {} {}, expected: {}, got: {}".format(
                    left, right, expected, got)
