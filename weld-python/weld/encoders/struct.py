"""
Implements an encoder mixin for tuples of values.

Tuples are encoded as Weld Structs. Weld structs are decoded back into
Python tuples. This encoder takes another encoder to encode or decode
individual elements.

"""

from abc import abstractmethod
from weld.encoders import WeldEncoder, WeldDecoder
from weld.types import WeldStruct

import ctypes

class StructWeldEncoder(WeldEncoder):
    """
    An encoder mixin for structs, to be subclassed by another encoders.

    Subclasses should encode their own types in 'encode_element()' instead
    of 'encode()'.

    """

    @abstractmethod
    def encode_element(self, obj, target_type):
        """
        Encodes a single non-struct element.

        Subclasses' 'encode()' implementation should go here.

        """
        pass

    def encode(self, obj, target_type):
        if isinstance(target_type, WeldStruct):
            encoder = target_type.ctype_class
            struct = encoder()
            for (i, (field, weld_ty)) in enumerate(zip(\
                    obj, target_type.field_types)):
                encoded = self.encode(field, weld_ty)
                setattr(struct, "_" + str(i), encoded)
            return struct
        else:
            return self.encode_element(obj, target_type)

class StructWeldDecoder(WeldDecoder):
    """
    A decoder abstract class for structs.

    Subclasses should encode their own types in 'decode_element()' instead
    of 'decode()'.

    """

    @abstractmethod
    def decode_element(self, obj, restype, context=None):
        """
        Decodes a single non-struct element.

        Subclasses' 'decode()' implementation should go here.

        """
        pass

    def decode(self, obj, restype, context=None):
        if isinstance(restype, WeldStruct):
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
            return self.decode_element(obj, restype, context)
