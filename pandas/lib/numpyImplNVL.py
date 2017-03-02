"""Contains implementations for each ported operation in NumPy.

Attributes:
    decoder_ (TYPE): Description
    encoder_ (TYPE): Description
    norm_factor_id_ (int): Unique IDs given to norm factor literals
"""
from utils import *
from nvlobject import *

norm_factor_id_ = 1
encoder_ = NumPyEncoder()
decoder_ = NumPyDecoder()


def div(array, other, type):
    """
    Normalizes the passed-in array by the passed in quantity.

    Args:
        array (NvlObject / Numpy.ndarray): Input array to normalize
        other (NvlObject / float): Normalization factor
        type (NvlType): Type of each element in the input array

    Returns:
        A NvlObject representing this computation
    """
    nvl_obj = NvlObject(encoder_, decoder_)

    array_str = nvl_obj.update(array)
    if isinstance(array, NvlObject):
        array_str = array.nvl

    other_str = None
    if isinstance(other, NvlObject):
        nvl_obj.update(other)
        global norm_factor_id_
        constant_name = "norm_factor%d" % norm_factor_id_
        constant = "%s := (%s);" % (constant_name, other.nvl)
        norm_factor_id_ += 1
        nvl_obj.constants.append(constant)
        other_str = constant_name
    else:
        other_str = ("%.2f" % other)

    nvl_template = """
       map(
         %(array)s,
         |value| value / %(type)s(%(other)s)
       )
    """
    nvl_obj.nvl = nvl_template % {"array": array_str, "other": other_str,
                                  "type": type}
    return nvl_obj


def aggr(array, op, initial_value, type):
    """
    Computes the aggregate of elements in the array.

    Args:
        array (NvlObject / Numpy.ndarray): Input array to aggregate
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

    nvl_template = """
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


def dot(matrix, vector, matrix_type, vector_type):
    """
    Computes the dot product between a matrix and a vector.

    Args:
        matrix (NvlObject / Numpy.ndarray): 2-d input matrix
        vector (NvlObject / Numpy.ndarray): 1-d input vector
        type (NvlType): Type of each element in the input matrix and vector

    Returns:
        A NvlObject representing this computation
    """
    nvl_obj = NvlObject(encoder_, decoder_)

    matrix_str = nvl_obj.update(matrix)
    if isinstance(matrix, NvlObject):
        matrix_str = matrix.nvl

    vector_str = nvl_obj.update(vector)
    if isinstance(vector, NvlObject):
        vector_str = vector.nvl

    nvl_template = """
       map(
         %(matrix)s,
         |row: vec[%(matrix_type)s]|
           result(
             for(
               map(
                 zip(row, %(vector)s),
                 |ele: {%(matrix_type)s, %(vector_type)s}| f64(ele.$0 * %(matrix_type)s(ele.$1))
               ),
               merger[f64,+],
               |b, i, e| merge(b, e)
             )
           )
       )
    """
    nvl_obj.nvl = nvl_template % {"matrix": matrix_str, "vector": vector_str,
                                  "matrix_type": matrix_type, "vector_type": vector_type}
    return nvl_obj


def exp(array, type):
    """
    Computes the per-element exponenet of the passed-in array.

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

    nvl_template = """
       map(
         %(array)s,
         |ele: %(type)s| exp(ele)
       )
    """
    nvl_obj.nvl = nvl_template % {"array": array_str, "type": type}
    return nvl_obj
