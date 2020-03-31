"""
Types available in Weld.
"""

from abc import ABC, abstractmethod
import ctypes
import sys

class WeldType(ABC):
    """
    A Weld representation of a Python type.

    Other types should subclass this and provide an implementation of
    `ctype_class`, which describes what the Python type looks like in Weld.

    """

    @abstractmethod
    def __str__(self):
        """
        Returns a Weld IR string representing this type.
        """
        pass

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    @property
    @abstractmethod
    def ctype_class(self):
        """
        Returns a class representing this type's ctype representation.
        """
        pass

    @property
    def size(self):
        """
        Returns the size of this type in bytes.
        """
        return ctypes.sizeof(self.ctype_class)


def _define_primitive_type(typename, ir, ctype, classdoc=None, add_to_module=True):
    """
    A factory function that generates classes for different primitive weld types.

    Parameters
    ----------

    name : str
        The name of the class to be generated
    ir : str
        The Weld IR of the Weld type.
    ctype: ctype class
        The ctypes type class of the type.
    classdoc: str, optional [default=None]
        A docstring for the class.
    add_to_module: bool, optional [default=True]
        If true, adds the class definition to the current module.

    Returns
    -------
    class
        A class representing a WeldType.

    """

    template = """
class {typename}(WeldType):

    __slots__ = []

    def __str__(self):
        return "{ir}"

    @property
    def ctype_class(self):
        return {ctype}
    """
    class_definition = template.format(typename=typename, ir=ir, ctype=ctype)

    if add_to_module:
        namespace = globals()
    else:
        namespace = dict(__name__='weldtype_%s' % typename)
        namespace['WeldType'] = WeldType

    exec(class_definition, namespace)
    result = namespace[typename]
    result._source = class_definition

    # For pickling to work, the __module__ variable needs to be set to the frame
    # where the named tuple is created.  Bypass this step in enviroments where
    # sys._getframe is not defined (Jython for example) or sys._getframe is not
    # defined for arguments greater than 0 (IronPython).
    try:
        result.__module__ = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        pass

    return result

# Booleans and characters
_define_primitive_type('Bool', 'bool', "ctypes.c_byte", classdoc="A boolean type.")
_define_primitive_type('Char', 'i8',   "ctypes.c_byte",
        classdoc="An ASCII character, represented as a 8-bit signed integer.")

# Signed integers
_define_primitive_type('I8',  'i8',  "ctypes.c_int8",  classdoc="An 8-bit signed integer.")
_define_primitive_type('I16', 'i16', "ctypes.c_int16", classdoc="A 16-bit signed integer.")
_define_primitive_type('I32', 'i32', "ctypes.c_int32", classdoc="A 32-bit signed integer.")
_define_primitive_type('I64', 'i64', "ctypes.c_int64", classdoc="A 64-bit signed integer.")

# Unsigned integers
_define_primitive_type('U8',  'u8',  "ctypes.c_uint8",  classdoc="An 8-bit unsigned integer.")
_define_primitive_type('U16', 'u16', "ctypes.c_uint16", classdoc="A 16-bit unsigned integer.")
_define_primitive_type('U32', 'u32', "ctypes.c_uint32", classdoc="A 32-bit unsigned integer.")
_define_primitive_type('U64', 'u64', "ctypes.c_uint64", classdoc="A 64-bit unsigned integer.")

# Floating point.
_define_primitive_type('F32', 'f32', "ctypes.c_float",  classdoc="A 32-bit IEEE floating point.")
_define_primitive_type('F64', 'f64', "ctypes.c_double", classdoc="A 64-bit IEEE floating point.")

class WeldVec(WeldType):
    """
    A Weld vector, which is represented as a pointer followed by a 64-bit
    signed integer packed into a C-struct.

    Examples
    --------
    >>> v1 = WeldVec(I32())
    >>> str(v1)
    'vec[i32]'

    >>> v2 = WeldVec(F32())
    >>> str(v2)
    'vec[f32]'

    >>> v3 = WeldVec(I32())
    >>> assert v1.__class__ is v2.__class__

    """
    # ctypes requires that the class instance returned is
    # the same object. Every time we create a new Vec instance (templatized by
    # type), we cache it here.
    _singletons = {}

    def __init__(self, elem_type):
        """
        Create a new vector type.

        Parameters
        ----------
        elem_type : WeldType
            The type of the elements.

        """
        assert isinstance(elem_type, WeldType)
        self.elem_type = elem_type

    def __str__(self):
        return "vec[{}]".format(self.elem_type)

    @property
    def ctype_class(self):
        def vec_factory(elem_type):
            """
            Create a new vector class for the provided element type.

            If the vector class already exists, it is delivered via the
            _singletons dictionary.
            """
            class Vec(ctypes.Structure):
                _fields_ = [
                    ("data", ctypes.POINTER(elem_type.ctype_class)),
                    ("size", ctypes.c_int64),
                ]
            return Vec

        if self.elem_type not in WeldVec._singletons:
            WeldVec._singletons[self.elem_type] = vec_factory(self.elem_type)
        return WeldVec._singletons[self.elem_type]


class WeldStruct(WeldType):
    """
    A Weld struct, which is represented as a packed C struct.

    Examples
    --------
    >>> v1 = WeldStruct((I32(),))
    >>> str(v1)
    '{i32}'

    >>> v2 = WeldStruct((WeldVec(I32()), I64()))
    >>> str(v2)
    '{vec[i32],i64}'

    >>> v3 = WeldStruct((v1, v2))
    >>> str(v3)
    '{{i32},{vec[i32],i64}}'

    """
    _singletons = {}

    def __init__(self, field_types):
        """
        Create a new Weld struct type.

        Parameters
        ----------

        field_types : iteratable of WeldType
            The type of each field.

        """
        assert False not in [isinstance(e, WeldType) for e in field_types]
        self.field_types = field_types

    def __str__(self):
        return "{" + ",".join([str(f) for f in self.field_types]) + "}"

    @property
    def ctype_class(self):
        def struct_factory(field_types):
            """
            Create a new struct class for the provided field types.

            If the struct class already exists, it is delivered via the
            _singletons dictionary.
            """
            class Struct(ctypes.Structure):
                _fields_ = [("_" + str(i), t.ctype_class)
                            for i, t in enumerate(field_types)]
            return Struct

        if tuple(self.field_types) not in WeldStruct._singletons:
            WeldStruct._singletons[
                tuple(self.field_types)] = struct_factory(self.field_types)
        return WeldStruct._singletons[tuple(self.field_types)]
