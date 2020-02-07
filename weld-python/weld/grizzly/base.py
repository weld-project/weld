
import weld

class GrizzlyBase(object):
    pass

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

def binary_apply(op, leftval, rightval, infix=True):
    if infix:
        return "({leftval} {op} {rightval})".format(
                op=op, leftval=leftval, rightval=rightval)
    else:
        return "{op}({leftval}, {rightval})".format(
                op=op, leftval=leftval, rightval=rightval)

def binary_map(op, ty, leftval, rightval, infix=True):
    """
    Constructs a Weld string to apply a binary function to two vectors
    elementwise.

    Examples
    --------
    >>> binary_map("+", "i32", "l", "r")
    'map(zip(l, r), |e: {i32,i32}| (e.$0 + e.$1))'
    >>> binary_map("max", "i32", "l", "r", infix=False)
    'map(zip(l, r), |e: {i32,i32}| max(e.$0, e.$1))'
    """
    return "map(zip({leftval}, {rightval}), |e: {{{ty},{ty}}}| {binary_apply})".format(
             leftval=leftval, rightval=rightval,
             ty=ty,
             binary_apply=binary_apply(op, "e.$0", "e.$1", infix=infix))
