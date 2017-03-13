"""Contains implementations for each ported operation in Pandas.

Attributes:
    decoder_ (NumPyEncoder): Description
    encoder_ (NumPyDecoder): Description
"""
from utils import *
from weldobject import *


encoder_ = NumPyEncoder()
decoder_ = NumPyDecoder()


def unique(array, type):
    """
    Returns a new array-of-arrays with all duplicate arrays removed.

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        type (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_str = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_str = array.weld_code

    weld_template = """
       map(
         tovec(
           result(
             for(
               map(
                 %(array)s,
                 |p: %(type)s| {p,0}
               ),
               dictmerger[%(type)s,i32,+],
               |b, i, e| merge(b,e)
             )
           )
         ),
         |p: {%(type)s, i32}| p.$0
       )
    """
    weld_obj.weld_code = weld_template % {"array": array_str, "type": type}
    return weld_obj


def aggr(array, op, initial_value, type):
    """
    Returns sum of elements in the array.

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        op (str): Op string used to aggregate the array (+ / *)
        initial_value (int): Initial value for aggregation
        type (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_str = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_str = array.weld_code

    weld_template = """
      result(
        for(
          %(array)s,
          merger[%(type)s,%(op)s],
          |b, i, e| merge(b, e)
        )
      )
    """
    weld_obj.weld_code = weld_template % {"array": array_str, "type": type, "op": op}
    return weld_obj


def mask(array, predicates, new_value, type):
    """
    Returns a new array, with each element in the original array satisfying the
    passed-in predicate set to `new_value`

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        predicates (WeldObject / Numpy.ndarray<bool>): Predicate set
        new_value (WeldObject / Numpy.ndarray / str): mask value
        type (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_str = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_str = array.weld_code

    predicates_str = weld_obj.update(predicates)
    if isinstance(predicates, WeldObject):
        predicates_str = predicates.weld_code

    if str(type).startswith("vec"):
        new_value_str = weld_obj.update(new_value)
        if isinstance(new_value, WeldObject):
            new_value_str = new_value.weld
    else:
        new_value_str = "%s(%s)" % (type, str(new_value))

    weld_template = """
       map(
         zip(%(array)s, %(predicates)s),
         |p: {%(type)s, bool}| if (p.$1, %(new_value)s, p.$0)
       )
    """
    weld_obj.weld_code = weld_template % {
        "array": array_str,
        "predicates": predicates_str,
        "new_value": new_value_str,
        "type": type}

    return weld_obj


def filter(array, predicates, type):
    """
    Returns a new array, with each element in the original array satisfying the
    passed-in predicate set to `new_value`

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        predicates (WeldObject / Numpy.ndarray<bool>): Predicate set
        type (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_str = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_str = array.weld_code

    predicates_str = weld_obj.update(predicates)
    if isinstance(predicates, WeldObject):
        predicates_str = predicates.weld_code

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
        "array": array_str,
        "predicates": predicates_str,
        "type": type}

    return weld_obj


def element_wise_op(array, other, op, type):
    """
    Operation of series and other, element-wise (binary operator add)

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        other (WeldObject / Numpy.ndarray): Second Input array
        op (str): Op string used to compute element-wise operation (+ / *)
        type (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_str = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_str = array.weld_code

    other_str = weld_obj.update(other)
    if isinstance(other, WeldObject):
        other_str = other.weld_code

    weld_template = """
       map(
         zip(%(array)s, %(other)s),
         |a| a.$0 %(op)s a.$1
       )
    """

    weld_obj.weld_code = weld_template % {"array": array_str, "other": other_str,
                                  "type": type, "op": op}
    return weld_obj


def compare(array, other, op, type_str):
    """
    Performs passed-in comparison op between every element in the passed-in array and other,
    and returns an array of booleans.

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        other (WeldObject / Numpy.ndarray): Second input array
        op (str): Op string used for element-wise comparison (== >= <= !=)
        type (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_str = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_str = array.weld_code

    # Strings need to be encoded into vec[char] array.
    # Constants can be added directly to NVL snippet.
    if isinstance(other, str) or isinstance(other, WeldObject):
        other_str = weld_obj.update(other)
        if isinstance(other, WeldObject):
            other_str = other.weld_code
    else:
        other_str = "%s(%s)" % (type_str, str(other))

    weld_template = """
       map(
         %(array)s,
         |a: %(type)s| a %(op)s %(other)s
       )
    """
    weld_obj.weld_code = weld_template % {"array": array_str, "other": other_str,
                                  "op": op, "type": type_str}

    return weld_obj


def slice(array, start, size, type):
    """
    Returns a new array-of-arrays with each array truncated, starting at
    index `start` for `length` characters.

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        start (int): starting index
        size (int): length to truncate at
        type (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_str = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_str = array.weld_code

    weld_template = """
       map(
         %(array)s,
         |array: %(type)s| slice(array, %(start)dL, %(size)dL)
       )
    """
    weld_obj.weld_code = weld_template % {"array": array_str, "start": start,
                                  "type": type, "size": size}

    return weld_obj


def count(array, type):
    """
    Return number of non-NA/null observations in the Series
    TODO : Filter out NaN's once Weld supports NaN's

    Args:
        array (WeldObject / Numpy.ndarray): Input array
        type (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_str = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_str = array.weld_code

    weld_template = """
     len(%(array)s)
  """

    weld_obj.weld_code = weld_template % {"array": array_str}

    return weld_obj


def groupby_sum(columns, column_types, grouping_column):
    """
    Groups the given columns by the corresponding grouping column
    value, and aggregate by summing values.

    Args:
        columns (List<WeldObject>): List of columns as WeldObjects
        column_types (List<str>): List of each column data type
        grouping_column (WeldObject): Column to group rest of columns by

    Returns:
          A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    grouping_column_str = weld_obj.update(grouping_column)
    if isinstance(grouping_column, WeldObject):
        grouping_column_str = grouping_column.weld_code

    columns_str_list = []
    for column in columns:
        column_str = weld_obj.update(column)
        if isinstance(column, WeldObject):
            column_str = column.weld_code
        columns_str_list.append(column_str)

    if len(columns_str_list) == 1:
        columns_str = columns_str_list[0]
        types_str = column_types[0]
        result_str = "merge(b, e)"
    else:
        # TODO Support Multicolumn
        columns_str = "zip(%s)" % ", ".join(columns_str_list)
        types_str = "{%s}" % ", ".join(column_types)
        result_str_list = []
        for i in xrange(len(columns)):
            result_str_list.append("elem1.%d + elem2.%d" % (i, i))
        result_str = "{%s}" % ", ".join(result_str_list)

    weld_template = """
     tovec(
       result(
         for(
           zip(%(grouping_column)s, %(columns)s),
           dictmerger[vec[i8], %(type)s, +],
           |b, i, e| %(result)s
         )
       )
     )
  """

    weld_obj.weld_code = weld_template % {"grouping_column": grouping_column_str,
                                  "columns": columns_str, "result": result_str,
                                  "type": types_str}
    return weld_obj


def get_column(columns, column_types, index):
    """
    Get column corresponding to passed-in index from ptr returned
    by groupBySum.

    Args:
        columns (List<WeldObject>): List of columns as WeldObjects
        column_types (List<str>): List of each column data type
        index (int): index of selected column

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)
    columns_str = weld_obj.update(columns, types=WeldVec(column_types))
    if isinstance(columns, WeldObject):
        columns_str = columns.weld_code

    weld_template = """
     map(
       %(columns)s,
       |elem: %(type)s| elem.$%(index)s
     )
  """

    weld_obj.weld_code = weld_template % {"columns": columns_str, "type": column_types,
                                          "index": index}
    return weld_obj
