"""Contains implementations for each ported operation in Pandas.

Attributes:
    decoder_ (NumPyEncoder): Description
    encoder_ (NumPyDecoder): Description
"""
from encoders import *
from weld.weldobject import *


encoder_ = NumPyEncoder()
decoder_ = NumPyDecoder()


def unique(array, ty):
    """
    Returns a new array-of-arrays with all duplicate arrays removed.

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        ty (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    weld_template = """
       map(
         tovec(
           result(
             for(
               map(
                 %(array)s,
                 |p: %(ty)s| {p,0}
               ),
               dictmerger[%(ty)s,i32,+],
               |b, i, e| merge(b,e)
             )
           )
         ),
         |p: {%(ty)s, i32}| p.$0
       )
    """
    weld_obj.weld_code = weld_template % {"array": array_var, "ty": ty}
    return weld_obj


def aggr(array, op, initial_value, ty):
    """
    Returns sum of elements in the array.

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        op (str): Op string used to aggregate the array (+ / *)
        initial_value (int): Initial value for aggregation
        ty (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    weld_template = """
      result(
        for(
          %(array)s,
          merger[%(ty)s,%(op)s],
          |b, i, e| merge(b, e)
        )
      )
    """
    weld_obj.weld_code = weld_template % {
        "array": array_var, "ty": ty, "op": op}
    return weld_obj


def mask(array, predicates, new_value, ty):
    """
    Returns a new array, with each element in the original array satisfying the
    passed-in predicate set to `new_value`

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        predicates (WeldObject / Numpy.ndarray<bool>): Predicate set
        new_value (WeldObject / Numpy.ndarray / str): mask value
        ty (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    predicates_var = weld_obj.update(predicates)
    if isinstance(predicates, WeldObject):
        predicates_var = predicates.obj_id
        weld_obj.dependencies[predicates_var] = predicates

    if str(ty).startswith("vec"):
        new_value_var = weld_obj.update(new_value)
        if isinstance(new_value, WeldObject):
            new_value_var = new_value.obj_id
            weld_obj.dependencies[new_value_var] = new_value
    else:
        new_value_var = "%s(%s)" % (ty, str(new_value))

    weld_template = """
       map(
         zip(%(array)s, %(predicates)s),
         |p: {%(ty)s, bool}| if (p.$1, %(new_value)s, p.$0)
       )
    """
    weld_obj.weld_code = weld_template % {
        "array": array_var,
        "predicates": predicates_var,
        "new_value": new_value_var,
        "ty": ty}

    return weld_obj


def filter(array, predicates, ty):
    """
    Returns a new array, with each element in the original array satisfying the
    passed-in predicate set to `new_value`

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        predicates (WeldObject / Numpy.ndarray<bool>): Predicate set
        ty (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    predicates_var = weld_obj.update(predicates)
    if isinstance(predicates, WeldObject):
        predicates_var = predicates.obj_id
        weld_obj.dependencies[predicates_var] = predicates

    weld_template = """
       result(
         for(
           zip(%(array)s, %(predicates)s),
           appender,
           |b, i, e| if (e.$1, merge(b, e.$0), b)
         )
       )
    """
    weld_obj.weld_code = weld_template % {
        "array": array_var,
        "predicates": predicates_var,
        "ty": ty}

    return weld_obj


def element_wise_op(array, other, op, ty):
    """
    Operation of series and other, element-wise (binary operator add)

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        other (WeldObject / Numpy.ndarray): Second Input array
        op (str): Op string used to compute element-wise operation (+ / *)
        ty (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    other_var = weld_obj.update(other)
    if isinstance(other, WeldObject):
        other_var = other.obj_id
        weld_obj.dependencies[other_var] = other

    weld_template = """
       map(
         zip(%(array)s, %(other)s),
         |a| a.$0 %(op)s a.$1
       )
    """

    weld_obj.weld_code = weld_template % {"array": array_var,
                                          "other": other_var,
                                          "ty": ty, "op": op}
    return weld_obj


def compare(array, other, op, ty_str):
    """
    Performs passed-in comparison op between every element in the passed-in
    array and other, and returns an array of booleans.

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        other (WeldObject / Numpy.ndarray): Second input array
        op (str): Op string used for element-wise comparison (== >= <= !=)
        ty (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    # Strings need to be encoded into vec[char] array.
    # Constants can be added directly to NVL snippet.
    if isinstance(other, str) or isinstance(other, WeldObject):
        other_var = weld_obj.update(other)
        if isinstance(other, WeldObject):
            other_var = tmp.obj_id
            weld_obj.dependencies[other_var] = other
    else:
        other_var = "%s(%s)" % (ty_str, str(other))

    weld_template = """
       map(
         %(array)s,
         |a: %(ty)s| a %(op)s %(other)s
       )
    """
    weld_obj.weld_code = weld_template % {"array": array_var,
                                          "other": other_var,
                                          "op": op, "ty": ty_str}

    return weld_obj


def slice(array, start, size, ty):
    """
    Returns a new array-of-arrays with each array truncated, starting at
    index `start` for `length` characters.

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        start (int): starting index
        size (int): length to truncate at
        ty (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    weld_template = """
       map(
         %(array)s,
         |array: %(ty)s| slice(array, %(start)dL, %(size)dL)
       )
    """
    weld_obj.weld_code = weld_template % {"array": array_var, "start": start,
                                          "ty": ty, "size": size}

    return weld_obj


def count(array, ty):
    """
    Return number of non-NA/null observations in the Series
    TODO : Filter out NaN's once Weld supports NaN's

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        ty (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    weld_template = """
     len(%(array)s)
  """

    weld_obj.weld_code = weld_template % {"array": array_var}

    return weld_obj


def groupby_sum(columns, column_tys, grouping_column):
    """
    Groups the given columns by the corresponding grouping column
    value, and aggregate by summing values.

    Args:
        columns (List<WeldObject>): List of columns as WeldObjects
        column_tys (List<str>): List of each column data ty
        grouping_column (WeldObject): Column to group rest of columns by

    Returns:
          A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    grouping_column_var = weld_obj.update(grouping_column)
    if isinstance(grouping_column, WeldObject):
        grouping_column_var = grouping_column.obj_id
        weld_obj.dependencies[grouping_column_var] = grouping_column

    columns_var_list = []
    for column in columns:
        column_var = weld_obj.update(column)
        if isinstance(column, WeldObject):
            column_var = column.obj_id
            weld_obj.dependencies[column_var] = column
        columns_var_list.append(column_var)

    if len(columns_var_list) == 1:
        columns_var = columns_var_list[0]
        tys_str = column_tys[0]
        result_str = "merge(b, e)"
    else:
        # TODO Support Multicolumn
        columns_var = "zip(%s)" % ", ".join(columns_var_list)
        tys_str = "{%s}" % ", ".join(column_tys)
        result_str_list = []
        for i in xrange(len(columns)):
            result_str_list.append("elem1.%d + elem2.%d" % (i, i))
        result_str = "{%s}" % ", ".join(result_str_list)

    weld_template = """
     tovec(
       result(
         for(
           zip(%(grouping_column)s, %(columns)s),
           dictmerger[vec[i8], %(ty)s, +],
           |b, i, e| %(result)s
         )
       )
     )
  """

    weld_obj.weld_code = weld_template % {"grouping_column": grouping_column_var,
                                          "columns": columns_var,
                                          "result": result_str,
                                          "ty": tys_str}
    return weld_obj


def get_column(columns, column_tys, index):
    """
    Get column corresponding to passed-in index from ptr returned
    by groupBySum.

    Args:
        columns (List<WeldObject>): List of columns as WeldObjects
        column_tys (List<str>): List of each column data ty
        index (int): index of selected column

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)
    columns_var = weld_obj.update(columns, tys=WeldVec(column_tys), override=False)
    if isinstance(columns, WeldObject):
        columns_var = columns.obj_id
        weld_obj.dependencies[columns_var] = columns

    weld_template = """
     map(
       %(columns)s,
       |elem: %(ty)s| elem.$%(index)s
     )
  """

    weld_obj.weld_code = weld_template % {"columns": columns_var,
                                          "ty": column_tys,
                                          "index": index}
    return weld_obj
