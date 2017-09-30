"""Contains implementations for each ported operation in NumPy.

Attributes:
    decoder_ (TYPE): Description
    encoder_ (TYPE): Description
    norm_factor_id_ (int): Unique IDs given to norm factor literals
"""
from encoders import *
from weld.weldobject import *

norm_factor_id_ = 1
encoder_ = NumPyEncoder()
decoder_ = NumPyDecoder()


def div(array, other, type):
    """
    Normalizes the passed-in array by the passed in quantity.

    Args:
        array (WeldObject / Numpy.ndarray): Input array to normalize
        other (WeldObject / float): Normalization factor
        type (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_str = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_str = array.weld_code

    other_str = None
    if isinstance(other, WeldObject):
        weld_obj.update(other)
        global norm_factor_id_
        constant_name = "norm_factor%d" % norm_factor_id_
        constant = "let %s = (%s);" % (constant_name, other.weld_code)
        norm_factor_id_ += 1
        weld_obj.constants.append(constant)
        other_str = constant_name
    else:
        other_str = ("%.2f" % other)

    weld_template = """
       map(
         %(array)s,
         |value| value / %(type)s(%(other)s)
       )
    """
    weld_obj.weld_code = weld_template % {"array": array_str,
                                          "other": other_str,
                                          "type": type}
    return weld_obj


def aggr(array, op, initial_value, type):
    """
    Computes the aggregate of elements in the array.

    Args:
        array (WeldObject / Numpy.ndarray): Input array to aggregate
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
    weld_obj.weld_code = weld_template % {
        "array": array_str, "type": type, "op": op}
    return weld_obj


def dot(matrix, vector, matrix_type, vector_type):
    """
    Computes the dot product between a matrix and a vector.

    Args:
        matrix (WeldObject / Numpy.ndarray): 2-d input matrix
        vector (WeldObject / Numpy.ndarray): 1-d input vector
        type (WeldType): Type of each element in the input matrix and vector

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    matrix_str = weld_obj.update(matrix)
    if isinstance(matrix, WeldObject):
        matrix_str = matrix.weld_code

    vector_str = weld_obj.update(vector)
    if isinstance(vector, WeldObject):
        vector_str = vector.weld_code

    weld_template = """
       map(
         %(matrix)s,
         |row: vec[%(matrix_type)s]|
           result(
             for(
               map(
                 zip(row, %(vector)s),
                 |ele: {%(matrix_type)s, %(vector_type)s}|
                   f64(ele.$0 * %(matrix_type)s(ele.$1))
               ),
               merger[f64,+],
               |b, i, e| merge(b, e)
             )
           )
       )
    """
    weld_obj.weld_code = weld_template % {"matrix": matrix_str,
                                          "vector": vector_str,
                                          "matrix_type": matrix_type,
                                          "vector_type": vector_type}
    return weld_obj


def exp(array, type):
    """
    Computes the per-element exponenet of the passed-in array.

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
         %(array)s,
         |ele: %(type)s| exp(ele)
       )
    """
    weld_obj.weld_code = weld_template % {"array": array_str, "type": type}
    return weld_obj
