#
# Wrappers for Types in NVL
#
#

from ctypes import *


class NvlType(object):

    def __str__(self):
        return "type"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return hash(other) == hash(self)

    @property
    def cTypeClass(self):
        """
        Returns a class representing this type's ctype representation.
        """
        raise NotImplementedError


class NvlChar(NvlType):

    def __str__(self):
        return "i8"

    @property
    def cTypeClass(self):
        return c_wchar_p


class NvlBit(NvlType):

    def __str__(self):
        return "bool"

    @property
    def cTypeClass(self):
        return c_bool


class NvlInt(NvlType):

    def __str__(self):
        return "i32"

    @property
    def cTypeClass(self):
        return c_int


class NvlLong(NvlType):

    def __str__(self):
        return "i64"

    @property
    def cTypeClass(self):
        return c_long


class NvlFloat(NvlType):

    def __str__(self):
        return "f32"

    @property
    def cTypeClass(self):
        return c_float


class NvlDouble(NvlType):

    def __str__(self):
        return "f64"

    @property
    def cTypeClass(self):
        return c_double


class NvlVec(NvlType):

    # Kind of a hack, but ctypes requires that the class instance returned is
    # the same object. Every time we create a new Vec instance (templatized by
    # type), we cache it here.
    _singletons = {}

    def __init__(self, elemType):
        self.elemType = elemType

    def __str__(self):
        return "vec[%s]" % str(self.elemType)

    @property
    def cTypeClass(self):
        def vec_factory(elemType):
            class Vec(Structure):
                _fields_ = [
                    ("ptr", POINTER(elemType.cTypeClass)),
                    ("size", c_long),
                ]
            return Vec

        if self.elemType not in NvlVec._singletons:
            NvlVec._singletons[self.elemType] = vec_factory(self.elemType)
        return NvlVec._singletons[self.elemType]


class NvlStruct(NvlType):

    _singletons = {}

    def __init__(self, fieldTypes):
        assert False not in [isinstance(e, NvlType) for e in fieldTypes]
        self.fieldTypes = fieldTypes

    def __str__(self):
        return "{" + ",".join([str(f) for f in self.fieldTypes]) + "}"

    @property
    def cTypeClass(self):
        def struct_factory(fieldTypes):
            class Struct(Structure):
                _fields_ = [(str(i), t.cTypeClass)
                            for i, t in enumerate(fieldTypes)]
            return Struct

        if frozenset(self.fieldTypes) not in NvlVec._singletons:
            NvlStruct._singletons[
                frozenset(self.fieldTypes)] = struct_factory(self.fieldTypes)
        return NvlStruct._singletons[frozenset(self.fieldTypes)]
