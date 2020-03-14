from ctypes import *
import numpy as np
import pkg_resources
import sys



# Wrappers for types in NVL
class WeldType(object):
    def __str__(self):
        return "type"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return hash(other) == hash(self)

    def __ne__(self, other):
        return hash(other) != hash(self)

    @property
    def ctype_class(self):
        """Returns a class representing this type's ctype representation."""
        raise NotImplementedError

class WeldChar(WeldType):
    def __str__(self):
        return "i8"

    @property
    def ctype_class(self):
        return c_wchar_p

class WeldVec(WeldType):
    # Kind of a hack, but ctypes requires that the class instance returned is
    # the same object. Every time we create a new Vec instance (templatized by
    # type), we cache it here.
    _singletons = {}

    def __init__(self, elem_type):
        self.elemType = elem_type

    def __str__(self):
        return "vec[%s]" % str(self.elemType)

    @property
    def ctype_class(self):
        def vec_factory(elem_type):
            class Vec(Structure):
                _fields_ = [
                    ("ptr", POINTER(elem_type.ctype_class)),
                    ("size", c_long),
                ]

            return Vec

        if self.elemType not in WeldVec._singletons:
            WeldVec._singletons[self.elemType] = vec_factory(self.elemType)

        return WeldVec._singletons[self.elemType]


utils = PyDLL("/Users/shoumikpalkar/Desktop/stringtest/build/lib.macosx-10.14-x86_64-3.7/strings.cpython-37m-darwin.so")

convertor = utils.NumpyArrayOfStringsToWeld
back = utils.WeldArrayOfStringsToNumPy
print(convertor)

x = np.array(["h", "hada".encode('utf-8')], dtype="S")
print(x)
print(x.dtype)

convertor.argtypes = [py_object]
convertor.restype = WeldVec(WeldChar()).ctype_class

back.argtypes = [convertor.restype]
back.restype = py_object

r = convertor(x)
print ("got r")
res = back(r)
print(res)
