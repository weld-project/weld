"""
Defines the encoder and decoder interface for marshalling data between Python
objects and Weld.
"""

from abc import ABC, abstractmethod

class WeldEncoder(ABC):
    """
    An abstract class that must be overwridden by libraries. This class
    is used to marshall objects from Python types to Weld types.
    """

    @abstractmethod
    def encode(self, obj, target_type):
        """
	Encode an object.

	Parameters
        ----------
        obj: any
            The object to marshal.
        target_type : WeldType
            The type to which the encoder should encode this object.

        Returns
        -------
        any
            An object represented in the Weld ABI.

        """
        pass


class WeldDecoder(ABC):
    """
    This class is used to marshall objects from Weld types to Python types.
    """

    @abstractmethod
    def decode(obj, restype):
        """
        Decodes the object, assuming object has the WeldType restype.

        The object's Python type is ctypes.POINTER(restype.ctype_class).

        Parameters
        ----------

        obj : any
            An object encoded in the Weld ABI.
        restype : WeldType
            The WeldType of the object that is being decoded.

        Returns
        -------
        any
            The decoder can return any Python value.

        """
        pass
