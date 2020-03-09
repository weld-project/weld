"""
Basic Weld element-wise operations supported in Grizzly.

"""

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

def lookup_expr(collection, key):
    """
    Lookup a value in a Weld vector. This will add a cast for the key to an `I64`.

    Examples
    --------
    >>> lookup_expr("v", "i64(1.0f)")
    'lookup(v, i64(i64(1.0f)))'
    >>> lookup_expr("[1,2,3]", "1.0f")
    'lookup([1,2,3], i64(1.0f))'
    >>> lookup_expr("[1,2,3]", 1)
    'lookup([1,2,3], i64(1))'

    """
    return "lookup({collection}, i64({key}))".format(
            collection=collection,
            key=key)

def slice_expr(collection, start, count):
    """
    Lookup a value in a Weld vector. This will add a cast the start and stop to 'I64'.

    Examples
    --------
    >>> slice_expr("v", 1, 2)
    'slice(v, i64(1), i64(2))'

    """
    return "slice({collection}, i64({start}), i64({count}))".format(
            collection=collection, start=start, count=count)

def mask(collection, collection_ty, booleans):
    """
    Returns a masking operation that filters values from 'collection' using
    the bitvector 'booleans'.

    Examples
    --------
    >>> mask("v", "i64", "mask")
    'map(filter(zip(v, mask), |e: {i64,bool}| e.$1), |e: {i64,bool}| e.$0)'

    """
    struct_ty = "{{{collection_ty},bool}}".format(collection_ty=collection_ty)
    template = "map(filter(zip({collection}, {mask}), |e: {struct_ty}| e.$1), |e: {struct_ty}| e.$0)"
    return template.format(collection=collection, mask=booleans, struct_ty=struct_ty)
