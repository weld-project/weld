"""
Tests for encoding and decoding string data.

"""

import numpy as np
import pytest

from .helpers import encdec_factory

from weld.encoders.numpy import *
from weld.types import *

encdec_inner = encdec_factory(NumPyWeldEncoder, NumPyWeldDecoder, eq=lambda x, y: np.all(x == y))
weld_string_type = WeldVec(WeldVec(I8()))

def encdec(array):
    return encdec_inner(array, weld_string_type)

def strarray(string_list, dtype='S'):
    """
    Create a NumPy array of strings. If a different dtype is passed, the
    string will not be supported by the encoder.

    """
    return np.array(string_list, dtype=dtype)


def test_single_string():
    encdec(strarray(["hello"]))

def test_multiple_strings_with_same_length():
    encdec(strarray(["hello", "quite", "nice!"]))

def test_some_empty_strings():
    encdec(strarray(["hello", "", "", "nice!"]))

def test_all_empty_strings():
    encdec(strarray(["", "", "", ""]))

def test_strings_with_varying_lengths():
    encdec(strarray(["a", "ab", "abc", "abcd"]))

def test_strings_with_varying_lengths_2():
    encdec(strarray(["abcd", "abc", "ab", "a"]))

def test_utf8_encoded_string():
    encdec(strarray(["abcd", "∂å∂ƒ".encode("utf-8")]))

def test_unsupported_dtype():
    with pytest.raises(NotImplementedError):
        # Need to explicitly provide the 'S' dtype.
        encdec(np.array(["hello", "world"]))

def test_unsupported_dtype_2():
    with pytest.raises(NotImplementedError):
        # Need to explicitly provide the 'S' dtype.
        encdec(np.array(["hello", "world", 1234]))
