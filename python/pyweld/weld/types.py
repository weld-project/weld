"""Summary
"""
#
# Wrappers for Types in NVL
#
#

from ctypes import *


class WeldType(object):
    """Summary
    """

    def __str__(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return "type"

    def __hash__(self):
        """Summary

        Returns:
            TYPE: Description
        """

        return hash(str(self))

    def __eq__(self, other):
        """Summary

        Args:
            other (TYPE): Description

        Returns:
            TYPE: Description
        """
        return hash(other) == hash(self)

    @property
    def cTypeClass(self):
        """
        Returns a class representing this type's ctype representation.

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError


class WeldChar(WeldType):
    """Summary
    """

    def __str__(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return "i8"

    @property
    def cTypeClass(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return c_wchar_p


class WeldBit(WeldType):
    """Summary
    """

    def __str__(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return "bool"

    @property
    def cTypeClass(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return c_bool


class WeldInt(WeldType):
    """Summary
    """

    def __str__(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return "i32"

    @property
    def cTypeClass(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return c_int


class WeldLong(WeldType):
    """Summary
    """

    def __str__(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return "i64"

    @property
    def cTypeClass(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return c_long


class WeldFloat(WeldType):
    """Summary
    """

    def __str__(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return "f32"

    @property
    def cTypeClass(self):
        return c_float


class WeldDouble(WeldType):
    """Summary
    """

    def __str__(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return "f64"

    @property
    def cTypeClass(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return c_double


class WeldVec(WeldType):
    """Summary

    Attributes:
        elemType (TYPE): Description
    """
    # Kind of a hack, but ctypes requires that the class instance returned is
    # the same object. Every time we create a new Vec instance (templatized by
    # type), we cache it here.
    _singletons = {}

    def __init__(self, elemType):
        """Summary

        Args:
            elemType (TYPE): Description
        """
        self.elemType = elemType

    def __str__(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return "vec[%s]" % str(self.elemType)

    @property
    def cTypeClass(self):
        """Summary

        Returns:
            TYPE: Description
        """
        def vec_factory(elemType):
            """Summary

            Args:
                elemType (TYPE): Description

            Returns:
                TYPE: Description
            """
            class Vec(Structure):
                """Summary
                """
                _fields_ = [
                    ("ptr", POINTER(elemType.cTypeClass)),
                    ("size", c_long),
                ]
            return Vec

        if self.elemType not in WeldVec._singletons:
            WeldVec._singletons[self.elemType] = vec_factory(self.elemType)
        return WeldVec._singletons[self.elemType]


class WeldStruct(WeldType):
    """Summary

    Attributes:
        fieldTypes (TYPE): Description
    """
    _singletons = {}

    def __init__(self, fieldTypes):
        """Summary

        Args:
            fieldTypes (TYPE): Description
        """
        assert False not in [isinstance(e, WeldType) for e in fieldTypes]
        self.fieldTypes = fieldTypes

    def __str__(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return "{" + ",".join([str(f) for f in self.fieldTypes]) + "}"

    @property
    def cTypeClass(self):
        """Summary

        Returns:
            TYPE: Description
        """
        def struct_factory(fieldTypes):
            """Summary

            Args:
                fieldTypes (TYPE): Description

            Returns:
                TYPE: Description
            """
            class Struct(Structure):
                """Summary
                """
                _fields_ = [(str(i), t.cTypeClass)
                            for i, t in enumerate(fieldTypes)]
            return Struct

        if frozenset(self.fieldTypes) not in WeldVec._singletons:
            WeldStruct._singletons[
                frozenset(self.fieldTypes)] = struct_factory(self.fieldTypes)
        return WeldStruct._singletons[frozenset(self.fieldTypes)]
