"""
Implements encoders and decoders for primitive values.

The ``PrimitiveWeldEncoder`` and ``PrimitiveWeldDecoder`` classes are used as
the default encoders and decoders when one is not provided.

"""

from .encoder_base import *
from ..types import *

def is_int_primitive(ty):
    """
    Returns whether the provided Weld type is an int primitive.
    """
    pass

def is_float_primitive(ty):
    """
    Returns whether the provided Weld type is an int primitive.
    """
    pass

class PrimitiveWeldEncoder(WeldEncoder):
    """
    A primitive encoder for boolean integers and floats.

    Eventually, this will also support encoding for tuples (structs) of other
    primitive types.
    """
    def encode(self, obj, target_type):
        encoder = target_type.ctype_class
        return encoder(obj)

class PrimitiveWeldDecoder(WeldDecoder):
    """
    A primitive encoder for booleans, integers, floats.

    Eventually, this will also support decoding for structs (tuples) of other
    primitive types.
    """
    def decode(self, obj, restype):
        value = obj.contents.value
        if isinstance(restype, Bool):
            return bool(value)
        return value
