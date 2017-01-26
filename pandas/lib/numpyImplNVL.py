from utils import *
from nvlobject import *

norm_factor_id_ = 1
encoder_ = NumPyEncoder()
decoder_ = NumPyDecoder()

def div(array, other, type):
    """
    Normalizes the passed-in array by the passed in quantity.
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

    nvl_template = \
    """
       map(
         %(array)s,
         (value: %(type)s) =>
           value / %(type)s(%(other)s)
       )
    """
    nvl_obj.nvl = nvl_template % {"array": array_str, "other": other_str,
                                  "type": type}
    return nvl_obj

def aggr(array, op, initial_value, type):
    """
    Returns sum of elements in the array.
    """
    nvl_obj = NvlObject(encoder_, decoder_)

    array_str = nvl_obj.update(array)
    if isinstance(array, NvlObject):
        array_str = array.nvl

    nvl_template = \
    """
       agg(
         %(array)s,
         %(type)s(%(initial_value)d),
         (x: %(type)s, y: %(type)s) => x %(op)s y,
         (x: %(type)s, y: %(type)s) => x %(op)s y
       )
    """
    nvl_obj.nvl = nvl_template % {"array": array_str, "type": type, "op": op,
                                  "initial_value": initial_value}
    return nvl_obj

def dot(matrix, vector, type):
    """
    Computes the dot product between a matrix and a vector.
    """
    nvl_obj = NvlObject(encoder_, decoder_)

    matrix_str = nvl_obj.update(matrix)
    if isinstance(matrix, NvlObject):
        matrix_str = matrix.nvl

    vector_str = nvl_obj.update(vector)
    if isinstance(vector, NvlObject):
        vector_str = vector.nvl

    nvl_template = \
    """
       map(
         %(matrix)s,
         (row: vec[%(type)s]) =>
           agg(
             map(
               zip(row, %(vector)s),
               (ele: {%(type)s, %(type)s}) => double(ele.0 * ele.1)
             ),
             0.0,
             (a: double, b: double) => a + b
           )
       )
    """
    nvl_obj.nvl = nvl_template % {"matrix": matrix_str, "vector": vector_str,
                                  "type": type}
    return nvl_obj

def exp(array, type):
    """
    Computes the per-element exponenet of the passed-in array.
    """
    nvl_obj = NvlObject(encoder_, decoder_)

    array_str = nvl_obj.update(array)
    if isinstance(array, NvlObject):
        array_str = array.nvl

    nvl_template = \
    """
       map(
         %(array)s,
         (ele: %(type)s) => exp(ele)
       )
    """
    nvl_obj.nvl = nvl_template % {"array": array_str, "type": type}
    return nvl_obj
