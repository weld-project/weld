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
    >>> s = encoder.encode((1.0, 1), WeldStruct((F32(), I32())))
    >>> s._0
    1.0
    >>> s._1
    1
    """
    def encode(self, obj, target_type):
        encoder = target_type.ctype_class
        if isinstance(target_type, WeldStruct):
            struct = encoder()
            for (i, (field, weld_ty)) in enumerate(zip(\
                    obj, target_type.field_types)):
                encoded = self.encode(field, weld_ty)
                setattr(struct, "_" + str(i), encoded)
            return struct
        else:
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
    >>> struct_type = WeldStruct((I32(), F32()))
    >>> x = encoder.encode((1, 1.0), struct_type)
    >>> decoder.decode(ctypes.pointer(x), struct_type)
    (1, 1.0)
    """
    def decode(self, obj, restype):
        if isinstance(restype, Bool):
            return bool(obj.contents.value)
        elif isinstance(restype, WeldStruct):
            struct = obj.contents
            ctype_class = restype.ctype_class
            result = []
            for (i, (weld_ty, (cfield, cty))) in enumerate(zip(\
                    restype.field_types, ctype_class._fields_)):
                ofs = getattr(ctype_class, cfield).offset
                p = ctypes.pointer(cty.from_buffer(struct, ofs))
                result.append(self.decode(p, weld_ty))
            return tuple(result)
        else:
            return obj.contents.value
