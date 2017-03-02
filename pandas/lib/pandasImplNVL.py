"""Contains implementations for each ported operation in Pandas.

Attributes:
    decoder_ (NumPyEncoder): Description
    encoder_ (NumPyDecoder): Description
"""
from utils import *
from nvlobject import *


encoder_ = NumPyEncoder()
decoder_ = NumPyDecoder()

def unique(array, type):
    """
    Returns a new array-of-arrays with all duplicate arrays removed.

    Args:
        array (NvlObject / Numpy.ndarray): Input array
        type (NvlType): Type of each element in the input array

    Returns:
        A NvlObject representing this computation
    """
    nvl_obj = NvlObject(encoder_, decoder_)

    array_str = nvl_obj.update(array)
    if isinstance(array, NvlObject):
        array_str = array.nvl

    nvl_template = \
    """
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
    nvl_obj.nvl = nvl_template % {"array": array_str, "type": type}
    return nvl_obj

def aggr(array, op, initial_value, type):
    """
    Returns sum of elements in the array.

    Args:
        array (NvlObject / Numpy.ndarray): Input array
        op (str): Op string used to aggregate the array (+ / *)
        initial_value (int): Initial value for aggregation
        type (NvlType): Type of each element in the input array

    Returns:
        A NvlObject representing this computation
    """
    nvl_obj = NvlObject(encoder_, decoder_)

    array_str = nvl_obj.update(array)
    if isinstance(array, NvlObject):
        array_str = array.nvl

    nvl_template = \
    """
      result(
        for(
          %(array)s,
          merger[%(type)s,%(op)s],
          |b, i, e| merge(b, e)
        )
      )
    """
    nvl_obj.nvl = nvl_template % {"array": array_str, "type": type, "op": op}
    return nvl_obj

def mask(array, predicates, new_value, type):
    """
    Returns a new array, with each element in the original array satisfying the
    passed-in predicate set to `new_value`

    Args:
        array (NvlObject / Numpy.ndarray): Input array
        predicates (NvlObject / Numpy.ndarray<bool>): Predicate set
        new_value (NvlObject / Numpy.ndarray / str): mask value
        type (NvlType): Type of each element in the input array

    Returns:
        A NvlObject representing this computation
    """
    nvl_obj = NvlObject(encoder_, decoder_)

    array_str = nvl_obj.update(array)
    if isinstance(array, NvlObject):
        array_str = array.nvl

    predicates_str = nvl_obj.update(predicates)
    if isinstance(predicates, NvlObject):
        predicates_str = predicates.nvl

    if str(type).startswith("vec"):
        new_value_str = nvl_obj.update(new_value)
        if isinstance(new_value, NvlObject):
            new_value_str = new_value.nvl
    else:
        new_value_str = "%s(%s)" % (type, str(new_value))

    nvl_template = \
    """
       map(
         zip(%(array)s, %(predicates)s),
         |p: {%(type)s, bool}| if (p.$1, %(new_value)s, p.$0)
       )
    """
    nvl_obj.nvl = nvl_template % {"array": array_str, "predicates": predicates_str,
                                  "new_value": new_value_str, "type": type}

    return nvl_obj

def filter(array, predicates, type):
    """
    Returns a new array, with each element in the original array satisfying the
    passed-in predicate set to `new_value`

    Args:
        array (NvlObject / Numpy.ndarray): Input array
        predicates (NvlObject / Numpy.ndarray<bool>): Predicate set
        type (NvlType): Type of each element in the input array

    Returns:
        A NvlObject representing this computation
    """
    nvl_obj = NvlObject(encoder_, decoder_)

    array_str = nvl_obj.update(array)
    if isinstance(array, NvlObject):
        array_str = array.nvl

    predicates_str = nvl_obj.update(predicates)
    if isinstance(predicates, NvlObject):
        predicates_str = predicates.nvl

    nvl_template = \
    """
       res(
         for(
           zip(%(array)s, %(predicates)s),
           appender[%(type)s],
           |b, i, e| if (element.$1, merge(b, element.$0), b)
         )
       )
    """
    nvl_obj.nvl = nvl_template % {"array": array_str, "predicates": predicates_str,
                                  "type": type}

    return nvl_obj

def element_wise_op(array, other, op, type):
    """
    Operation of series and other, element-wise (binary operator add)

    Args:
        array (NvlObject / Numpy.ndarray): Input array
        other (NvlObject / Numpy.ndarray): Second Input array
        op (str): Op string used to compute element-wise operation (+ / *) 
        type (NvlType): Type of each element in the input array

    Returns:
        A NvlObject representing this computation
    """
    nvl_obj = NvlObject(encoder_, decoder_)

    array_str = nvl_obj.update(array)
    if isinstance(array, NvlObject):
        array_str = array.nvl

    other_str = nvl_obj.update(other)
    if isinstance(other, NvlObject):
        other_str = other.nvl

    nvl_template = \
    """
       map(
         zip(%(array)s, %(other)s),
         |a| a.$0 %(op)s a.$1
       )
    """

    nvl_obj.nvl = nvl_template % {"array": array_str, "other": other_str,
                                  "type": type, "op": op}
    return nvl_obj
        
def compare(array, other, op, type_str):
    """
    Performs passed-in comparison op between every element in the passed-in array and other,
    and returns an array of booleans.

    Args:
        array (NvlObject / Numpy.ndarray): Input array
        other (NvlObject / Numpy.ndarray): Second input array
        op (str): Op string used for element-wise comparison (== >= <= !=)
        type (NvlType): Type of each element in the input array

    Returns:
        A NvlObject representing this computation
    """
    nvl_obj = NvlObject(encoder_, decoder_)

    array_str = nvl_obj.update(array)
    if isinstance(array, NvlObject):
        array_str = array.nvl

    # Strings need to be encoded into vec[char] array.
    # Constants can be added directly to NVL snippet.
    if type(other) == str or isinstance(other, NvlObject):
        other_str = nvl_obj.update(other)
        if isinstance(other, NvlObject):
            other_str = other.nvl
    else:
        other_str = "%s(%s)" % (type_str, str(other))

    nvl_template = \
    """
       map(
         %(array)s,
         |a: %(type)s| a %(op)s %(other)s
       )
    """
    nvl_obj.nvl = nvl_template % {"array": array_str, "other": other_str,
                                  "op": op, "type": type_str}

    return nvl_obj

def slice(array, start, size, type):
    """
    Returns a new array-of-arrays with each array truncated, starting at
    index `start` for `length` characters.

    Args:
        array (NvlObject / Numpy.ndarray): Input array
        start (int): starting index
        size (int): length to truncate at
        type (NvlType): Type of each element in the input array

    Returns:
        A NvlObject representing this computation
    """
    nvl_obj = NvlObject(encoder_, decoder_)

    array_str = nvl_obj.update(array)
    if isinstance(array, NvlObject):
        array_str = array.nvl

    nvl_template = \
    """
       map(
         %(array)s,
         |array: %(type)s| slice(array, %(start)dL, %(size)dL)
       )
    """
    nvl_obj.nvl = nvl_template % {"array": array_str, "start": start,
                                  "type": type, "size": size}

    return nvl_obj

def count(array, type):
  """
  Return number of non-NA/null observations in the Series
  TODO : Filter out NaN's once Weld supports NaN's

  Args:
      array (NvlObject / Numpy.ndarray): Input array
      type (NvlType): Type of each element in the input array

  Returns:
      A NvlObject representing this computation
  """
  nvl_obj = NvlObject(encoder_, decoder_)

  array_str = nvl_obj.update(array)
  if isinstance(array, NvlObject):
    array_str = array.nvl

  nvl_template = \
  """
     len(%(array)s)
  """

  nvl_obj.nvl = nvl_template % {"array": array_str}

  return nvl_obj

def groupby_sum(columns, column_types, grouping_column):
  """
  Groups the given columns by the corresponding grouping column
  value, and aggregate by summing values.

  Args:
      columns (List<NvlObject>): List of columns as NvlObjects
      column_types (List<str>): List of each column data type
      grouping_column (NvlObject): Column to group rest of columns by

  Returns:
        A NvlObject representing this computation
  """
  nvl_obj = NvlObject(encoder_, decoder_)

  grouping_column_str = nvl_obj.update(grouping_column)
  if isinstance(grouping_column, NvlObject):
    grouping_column_str = grouping_column.nvl

  columns_str_list = []
  for column in columns:
    column_str = nvl_obj.update(column)
    if isinstance(column, NvlObject):
      column_str = column.nvl
    columns_str_list.append(column_str)

  if len(columns_str_list) == 1:
    columns_str = columns_str_list[0]
    types_str = column_types[0]
    result_str = "elem1 + elem2"
  else:
    columns_str = "zip(%s)" % ", ".join(columns_str_list)
    types_str = "{%s}" % ", ".join(column_types)
    result_str_list = []
    for i in xrange(len(columns)):
      result_str_list.append("elem1.%d + elem2.%d" % (i, i))
    result_str = "{%s}" % ", ".join(result_str_list)

  # TODO: Fix this. Horribly broken.
  nvl_template = \
  """
     tovec(
       result(
         for(
           zip(%(grouping_column)s, %(columns)s),
           dictmerger[%(key_type)s, %(type)s, +],
           |b, i, e| => %(result)s
         )
       )
     )
  """

  nvl_obj.nvl = nvl_template % {"grouping_column": grouping_column_str,
                                "columns": columns_str, "result": result_str,
                                "types": types_str}
  return nvl_obj

def get_column(columns, column_types, index):
  """
  Get column corresponding to passed-in index from ptr returned
  by groupBySum.

  Args:
      columns (List<NvlObject>): List of columns as NvlObjects
      column_types (List<str>): List of each column data type
      index (int): index of selected column

  Returns:
      A NvlObject representing this computation
  """
  nvl_obj = NvlObject(encoder_, decoder_)

  columns_str = nvl_obj.update(columns, argtype=NvlVec(column_types))
  if isinstance(columns, NvlObject):
    columns_str = columns.nvl

  # TODO: Fix this. Horribly broken.
  nvl_template = \
  """
     map(
       %(columns)s,
       |elem: %(type)s| => elem.%(index)s
     )
  """

  nvl_obj.nvl = nvl_template % {"columns": columns_str, "type": column_types,
                                "index": index}
  return nvl_obj
