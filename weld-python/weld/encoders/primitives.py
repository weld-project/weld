"""
Implements encoders and decoders for primitive values.

The ``PrimitiveWeldEncoder`` and ``PrimitiveWeldDecoder`` classes are used as
the default encoders and decoders when one is not provided.

"""

from .encoder_base import *
from ..types import *

class PrimitiveWeldEncoder(WeldEncoder):
    """
    A primitive encoder for booleans, integers and floats.

    Eventually, this will also support encoding for tuples (structs) of other
    primitive types.

    Examples
    --------
    >>> encoder = PrimitiveWeldEncoder()
    >>> encoder.encode(1, I32())
    c_int(1)
    >>> encoder.encode(1, F32())
    c_float(1.0)
    >>> encoder.encode(1.0, I64())
    Traceback (most recent call last):
    ...
    TypeError: int expected instead of float

    """
    def encode(self, obj, target_type):
        encoder = target_type.ctype_class
        return encoder(obj)

class PrimitiveWeldDecoder(WeldDecoder):
    """
    A primitive encoder for booleans, integers, and floats.

    Eventually, this will also support decoding for structs (tuples) of other
    primitive types.

    Examples
    --------
    >>> import ctypes
    >>> encoder = PrimitiveWeldEncoder()
    >>> decoder = PrimitiveWeldDecoder()
    >>> x = encoder.encode(1, I32())
    >>> decoder.decode(ctypes.pointer(x), I32())
    1

    """
    def decode(self, obj, restype):
        value = obj.contents.value
        if isinstance(restype, Bool):
            return bool(value)
        return value
