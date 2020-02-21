"""
Basic element-wise operations supported in Grizzly.

"""

from abc import ABC, abstractmethod
import operator

class Op(ABC):

    @property
    @abstractmethod
    def name(self):
        pass

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class UnaryOp(Op):

    __slots__ = ['name', 'special', 'weld_name']

    def __init__(self, name, special=None, weld_name=None):
        if special is not None:
            special.startswith("__") and special.endswith("__")
        self.name = name
        self.special = special
        self.weld_name = weld_name

class BinaryOp(Op):

    __slots__ = ['name', "infix", 'special', 'weld_name']

    def __init__(self, name, infix, special=None, weld_name=None):
        """ Create a binary operator.

        Names when doing Weld code generation are chosen as follows:

        If infix is provided:
            LEFT <infix> RIGHT
        If weld_name is provided and infix is provided:
            LEFT <infix> RIGHT
        If weld_name is provided and infix is not provided:
            weld_name(LEFT, RIGHT)
        If weld_name is not provided and infix is not provided:
            not allowed.

        Parameters
        ----------
        name : str
            The API name of this operator.
        infix : str
            The infix string of this operator, or None if one
            does not exist.
        special : str
            A Python special method name if one exists for this op.
        weld_name : str
            The Weld operator name, if it doesn't exist.

        """
        assert infix is not None or weld_name is not None
        if special is not None:
            assert special.startswith("__") and special.endswith("__")
        self.name = name
        self.infix = infix
        self.special = special
        self.weld_name = weld_name

class CompareOp(BinaryOp):
    pass

class OpRegistry(object):
    """
    A singleton for holding the supported operations.

    Examples
    --------
    >>> OpRegistry.get('add')
    <weld.grizzly.ops.BinaryOp object at ...>
    >>> OpRegistry.get('lt')
    <weld.grizzly.ops.CompareOp object at ...>
    >>> OpRegistry.supports('add')
    True
    >>> OpRegistry.supports('fakeOperator')
    False

    """
    OPS = {
        # Binary operations
        "add": BinaryOp("add", "+", "__add__"),
        "sub": BinaryOp("sub", "-", "__sub__"),
        "mul": BinaryOp("mul", "*", "__mul__"),
        "div": BinaryOp("div", "/", "__truediv__"),
        "truediv": BinaryOp("truediv", "/", "__truediv__"),
        "mod": BinaryOp("mod", "%", "__mod__"),
        "pow": BinaryOp("pow", None, weld_name="pow"),
        "radd": BinaryOp("radd", "+", "__add__"),
        "rsub": BinaryOp("rsub", "-", "__sub__"),
        "and": BinaryOp("and", "&", "__and__"),
        "or": BinaryOp("or", "|", "__or__"),
        "xor": BinaryOp("xor", "^", "__xor__"),
        # Unary operations
        "sqrt": UnaryOp("sqrt", weld_name="sqrt"),
        "exp": UnaryOp("exp", weld_name="exp"),
        # Compare operations
        "lt": CompareOp("lt", "<", "__lt__"),
        "gt": CompareOp("gt", ">", "__gt__"),
        "le": CompareOp("le", "<=", "__le__"),
        "ge": CompareOp("ge", ">=", "__ge__"),
        "ne": CompareOp("ne", "!=", "__ne__"),
        "eq": CompareOp("eq", "==", "__eq__"),
        }

    @classmethod
    def get(cls, name):
        return OpRegistry.OPS[name]

    @classmethod
    def supports(cls, name):
        return name in OpRegistry.OPS

def unary_apply(op, value):
    """
    Constructs a Weld string to apply a unary function to a scalar.

    Examples
    --------
    >>> unary_apply("sqrt", "e")
    'sqrt(e)'
    """
    return "{op}({value})".format(op=op, value=value)

def unary_map(op, ty, value):
    """
    Constructs a Weld string to apply a unary function to a vector.

    Examples
    --------
    >>> unary_map("sqrt", "i32", "e")
    'map(e, |e: i32| sqrt(e))'
    """
    return "map({value}, |e: {ty}| {unary_apply})".format(
            value=value, unary_apply=unary_apply(op, "e"), ty=ty)

def binary_apply(op, leftval, rightval, cast_type, infix=True):
    """
    Applies the binary operator 'op' to 'leftval' and 'rightval'.
    The operands are cast to the type 'cast_type' first.
    """
    if infix:
        return "({cast_type}({leftval}) {op} {cast_type}({rightval}))".format(
                op=op, leftval=leftval, rightval=rightval, cast_type=cast_type)
    else:
        return "{op}({cast_type}({leftval}), {cast_type}({rightval}))".format(
                op=op, leftval=leftval, rightval=rightval, cast_type=cast_type)

def binary_map(op, left_type, right_type, leftval, rightval, cast_type, infix=True, scalararg=False):
    """
    Constructs a Weld string to apply a binary function to two vectors
    'leftval' and 'rightval' elementwise. Each element in the loop is cast to
    'cast_type' first.

    Examples
    --------
    >>> binary_map("+", "i32", "i32", "l", "r", "i32")
    'map(zip(l, r), |e: {i32,i32}| (i32(e.$0) + i32(e.$1)))'
    >>> binary_map("max", "i32", "i16", "l", "r", 'i64', infix=False)
    'map(zip(l, r), |e: {i32,i16}| max(i64(e.$0), i64(e.$1)))'
    >>> binary_map("+", "i32", "i16", "l", "1L", 'i64', scalararg=True)
    'map(l, |e: i32| (i64(e) + i64(1L)))'
    """
    if scalararg:
        return "map({leftval}, |e: {left_type}| {binary_apply})".format(
                 leftval=leftval,
                 left_type=left_type, right_type=right_type,
                 binary_apply=binary_apply(op, "e", rightval, cast_type, infix=infix))
    else:
        return "map(zip({leftval}, {rightval}), |e: {{{left_type},{right_type}}}| {binary_apply})".format(
                 leftval=leftval, rightval=rightval,
                 left_type=left_type, right_type=right_type,
                 binary_apply=binary_apply(op, "e.$0", "e.$1", cast_type, infix=infix))
