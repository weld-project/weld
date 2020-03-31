"""
Implements encoders and decoders for primitive values.

The ``PrimitiveWeldEncoder`` and ``PrimitiveWeldDecoder`` classes are used as
the default encoders and decoders when one is not provided.

"""

from weld.encoders.struct import StructWeldEncoder, StructWeldDecoder
from weld.types import *

import ctypes

class PrimitiveWeldEncoder(StructWeldEncoder):
    """
    A primitive encoder for booleans, integers, floats, and tuples thereof.

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
    >>> s = encoder.encode((1.0, 1), WeldStruct((F32(), I32())))
    >>> s._0
    1.0
    >>> s._1
    1
    """
    def encode_element(self, obj, target_type):
        encoder = target_type.ctype_class
        return encoder(obj)

class PrimitiveWeldDecoder(StructWeldDecoder):
    """
    A primitive encoder for booleans, integers, floats, and tuples thereof.

    Examples
    --------
    >>> import ctypes
    >>> encoder = PrimitiveWeldEncoder()
    >>> decoder = PrimitiveWeldDecoder()
    >>> x = encoder.encode(1, I32())
    >>> decoder.decode(ctypes.pointer(x), I32())
    1
    >>> struct_type = WeldStruct((I32(), F32()))
    >>> x = encoder.encode((1, 1.0), struct_type)
    >>> decoder.decode(ctypes.pointer(x), struct_type)
    (1, 1.0)
    """

    def decode_element(self, obj, restype, context=None):
        if isinstance(restype, Bool):
            return bool(obj.contents.value)
        else:
            return obj.contents.value
