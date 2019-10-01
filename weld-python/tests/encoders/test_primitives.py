"""
Tests primitive encoders and decoders.
"""

import copy
import ctypes

from weld.encoders import PrimitiveWeldEncoder, PrimitiveWeldDecoder
from weld.types import *

def encdec(value, ty, assert_equal=True, err=False):
    """ Helper function that encodes a value and decodes it.

    The function asserts that the original value and the decoded value are
    equal.

    Parameters
    ----------
    value : any
        The value to encode and decode
    ty : WeldType
        the WeldType of the value
    assert_equal : bool (default True)
        Checks whether the original value and decoded value are equal.
    err :  bool (default False)
        If True, expects an error.

    """
    enc = PrimitiveWeldEncoder()
    dec = PrimitiveWeldDecoder()

    try:
        result = dec.decode(ctypes.pointer(enc.encode(value, ty)), ty)
    except Exception as e:
        if err:
            return
        else:
            raise e

    if err:
        raise RuntimeError("Expected error during encode/decode")

    if assert_equal:
        assert value == result

def test_i8_encode():
    encdec(-1, I8())

def test_i16_encode():
    encdec(-1, I16())

def test_i32_encode():
    encdec(-1, I32())

def test_i64_encode():
    encdec(-1, I64())

def test_u8_encode():
    encdec(1, U8())

def test_u16_encode():
    encdec(1, U16())

def test_u32_encode():
    encdec(1, U32())

def test_u64_encode():
    encdec(1, U64())

def test_f32_encode():
    encdec(1.0, F32())

def test_f64_encode():
    encdec(1.0, F64())

def test_float_encode_error():
    encdec(1.0, I32(), err=True)

def test_bool_encode():
    encdec(True, Bool())
    encdec(False, Bool())

def test_simple_struct():
    ty = WeldStruct((I32(), F32()))
    encdec((1, 1.0), ty)

def test_nested_struct():
    """ Tests nested structs. """
    inner = WeldStruct((I32(), I32()))
    outer = WeldStruct((inner, inner))
    encdec(((1, 1), (1, 1)), outer)

def test_padded_struct():
    """ Tests structs that will be padded. """
    inner = WeldStruct((I32(), I32()))
    outer = WeldStruct((Bool(), inner))
    encdec((True, (1, 1)), outer)


