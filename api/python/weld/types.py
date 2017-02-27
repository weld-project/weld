#
# Wrappers for Types in Weld
#
#

from ctypes import *

class WeldType(object):
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

class WeldI32(WeldType):
  def __str__(self):
    return "i32"

  @property
  def cTypeClass(self):
    return c_int32

class WeldI64(WeldType):
  def __str__(self):
    return "i64"

  @property
  def cTypeClass(self):
    return c_long

class WeldF32(WeldType):
  def __str__(self):
    return "f32"

  @property
  def cTypeClass(self):
    return c_float

class WeldF64(WeldType):
  def __str__(self):
    return "f64"

  @property
  def cTypeClass(self):
    return c_double

class WeldVec(WeldType):

  # ctypes requires that the class instance returned is
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
            ("data", POINTER(elemType.cTypeClass)),
            ("size", c_long),
          ]
      return Vec

    if self.elemType not in WeldVec._singletons:
      WeldVec._singletons[self.elemType] = vec_factory(self.elemType)
    return WeldVec._singletons[self.elemType]

class WeldStruct(WeldType):

  _singletons = {}

  def __init__(self, fieldTypes):
    assert False not in [isinstance(e, WeldType) for e in fieldTypes]
    self.fieldTypes = fieldTypes

  def __str__(self):
    return "{" + ",".join([str(f) for f in self.fieldTypes]) + "}"

  @property
  def cTypeClass(self):
    def struct_factory(fieldTypes):
      class Struct(Structure):
        _fields_ = [(str(i), t.cTypeClass()) for i, t in enumerate(fieldTypes)]
      return Struct

    if self.elemType not in WeldVec._singletons:
      WeldVec._singletons[self.elemType] = vec_factory(self.elemType)
    return WeldVec._singletons[self.elemType]

