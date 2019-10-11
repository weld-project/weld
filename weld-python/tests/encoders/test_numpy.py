"""
Tests NumPy encoders and decoders.
"""

import ctypes
import numpy as np

from .helpers import encdec_factory
from weld.encoders.numpy import NumPyWeldEncoder, NumPyWeldDecoder
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
