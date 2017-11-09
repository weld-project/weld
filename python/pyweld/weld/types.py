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
    def ctype_class(self):
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
    def ctype_class(self):
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
    def ctype_class(self):
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
    def ctype_class(self):
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
    def ctype_class(self):
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
    def ctype_class(self):
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
    def ctype_class(self):
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
    def ctype_class(self):
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
                    ("ptr", POINTER(elemType.ctype_class)),
                    ("size", c_long),
                ]
            return Vec

        if self.elemType not in WeldVec._singletons:
            WeldVec._singletons[self.elemType] = vec_factory(self.elemType)
        return WeldVec._singletons[self.elemType]


class WeldStruct(WeldType):
    """Summary

    Attributes:
        field_types (TYPE): Description
    """
    _singletons = {}

    def __init__(self, field_types):
        """Summary

        Args:
            field_types (TYPE): Description
        """
        assert False not in [isinstance(e, WeldType) for e in field_types]
        self.field_types = field_types

    def __str__(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return "{" + ",".join([str(f) for f in self.field_types]) + "}"

    @property
    def ctype_class(self):
        """Summary

        Returns:
            TYPE: Description
        """
        def struct_factory(field_types):
            """Summary

            Args:
                field_types (TYPE): Description

            Returns:
                TYPE: Description
            """
            class Struct(Structure):
                """Summary
                """
                _fields_ = [(str(i), t.ctype_class)
                            for i, t in enumerate(field_types)]
            return Struct

        if frozenset(self.field_types) not in WeldVec._singletons:
            WeldStruct._singletons[
                frozenset(self.field_types)] = struct_factory(self.field_types)
        return WeldStruct._singletons[frozenset(self.field_types)]
