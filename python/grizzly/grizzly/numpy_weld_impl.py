"""Contains implementations for each ported operation in NumPy.

Attributes:
    decoder_ (TYPE): Description
    encoder_ (TYPE): Description
    norm_factor_id_ (int): Unique IDs given to norm factor literals
"""
from encoders import *
from weld.weldobject import *

encoder_ = NumPyEncoder()
decoder_ = NumPyDecoder()


def div(array, other, ty):
    """
    Normalizes the passed-in array by the passed in quantity.

    Args:
        array (WeldObject / Numpy.ndarray): Input array to normalize
        other (WeldObject / float): Normalization factor
        ty (WeldType): Type of each element in the input array

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    other_var = None
    if isinstance(other, WeldObject):
        other_var = other.obj_id
        weld_obj.dependencies[other_var] = other
    else:
        other_var = ("%.2f" % other)

    weld_template = """
       map(
         %(array)s,
         |value| value / %(ty)s(%(other)s)
       )
    """
    weld_obj.weld_code = weld_template % {"array": array_var,
                                          "other": other_var,
                                          "ty": ty}
    return weld_obj


def aggr(array, op, initial_value, ty):
    """
    Computes the aggregate of elements in the array.

    Args:
        array (WeldObject / Numpy.ndarray): Input array to aggregate
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


def dot(matrix, vector, matrix_ty, vector_ty):
    """
    Computes the dot product between a matrix and a vector.

    Args:
        matrix (WeldObject / Numpy.ndarray): 2-d input matrix
        vector (WeldObject / Numpy.ndarray): 1-d input vector
        ty (WeldType): Type of each element in the input matrix and vector

    Returns:
        A WeldObject representing this computation
    """
    weld_obj = WeldObject(encoder_, decoder_)

    matrix_var = weld_obj.update(matrix)
    if isinstance(matrix, WeldObject):
        matrix_var = matrix.obj_id
        weld_obj.dependencies[matrix_var] = matrix

    vector_var = weld_obj.update(vector)
    loopsize_annotation = ""
    if isinstance(vector, WeldObject):
        vector_var = vector.obj_id
        weld_obj.dependencies[vector_var] = vector
    if isinstance(vector, np.ndarray):
        loopsize_annotation = "@(loopsize: %dL)" % len(vector)

    weld_template = """
       map(
         %(matrix)s,
         |row: vec[%(matrix_ty)s]|
           result(
             %(loopsize_annotation)s
             for(
               result(
                 %(loopsize_annotation)s
                 for(
                   zip(row, %(vector)s),
                   appender,
                   |b2, i2, e2: {%(matrix_ty)s, %(vector_ty)s}|
                     merge(b2, f64(e2.$0 * %(matrix_ty)s(e2.$1)))
                 )
               ),
               merger[f64,+],
               |b, i, e| merge(b, e)
             )
           )
       )
    """
    weld_obj.weld_code = weld_template % {"matrix": matrix_var,
                                          "vector": vector_var,
                                          "matrix_ty": matrix_ty,
                                          "vector_ty": vector_ty,
                                          "loopsize_annotation": loopsize_annotation}
    return weld_obj


def exp(array, ty):
    """
    Computes the per-element exponenet of the passed-in array.

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
         %(array)s,
         |ele: %(ty)s| exp(ele)
       )
    """
    weld_obj.weld_code = weld_template % {"array": array_var, "ty": ty}
    return weld_obj
