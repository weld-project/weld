import numpy as np

import numpy_weld_impl
from lazy_op import LazyOpResult
from weld.weldobject import *


class NumpyArrayWeld(LazyOpResult):
    """Summary

    Attributes:
        dim (TYPE): Description
        expr (TYPE): Description
        weld_type (TYPE): Description
    """

    def __init__(self, expr, weld_type, dim=1):
        """Summary

        Args:
            expr (TYPE): Description
            weld_type (TYPE): Description
            dim (int, optional): Description
        """
        self.expr = expr
        self.weld_type = weld_type
        self.dim = dim

    def __div__(self, other):
        """Summary

        Args:
            other (TYPE): Description

        Returns:
            TYPE: Description
        """
        if isinstance(other, LazyOpResult):
            other = other.expr
        return NumpyArrayWeld(
            numpy_weld_impl.div(
                self.expr,
                other,
                self.weld_type
            ),
            self.weld_type
        )

    def sum(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return NumpyArrayWeld(
            numpy_weld_impl.aggr(
                self.expr,
                "+",
                0,
                self.weld_type
            ),
            self.weld_type,
            0
        )


def dot(matrix, vector):
    """
    Computes the dot product between a matrix and a vector.
    TODO: Make this more generic

    Args:
        matrix (TYPE): Description
        vector (TYPE): Description
    """
    matrix_weld_type = None
    vector_weld_type = None

    if isinstance(matrix, LazyOpResult):
        matrix_weld_type = matrix.weld_type
        matrix = matrix.expr
    elif isinstance(matrix, np.ndarray):
        matrix_weld_type = numpy_weld_impl.numpy_to_weld_type_mapping[
            str(matrix.dtype)]

    if isinstance(vector, LazyOpResult):
        vector_weld_type = vector.weld_type
        vector = vector.expr
    elif isinstance(vector, np.ndarray):
        vector_weld_type = numpy_weld_impl.numpy_to_weld_type_mapping[
            str(vector.dtype)]

    return NumpyArrayWeld(
        numpy_weld_impl.dot(
            matrix,
            vector,
            matrix_weld_type,
            vector_weld_type),
        WeldDouble())


def exp(vector):
    """
    Computes a per-element exponent of the passed-in vector.

    Args:
        vector (TYPE): Description
    """
    weld_type = None
    if isinstance(vector, LazyOpResult):
        weld_type = vector.weld_type
        vector = vector.expr
    elif isinstance(vector, np.ndarray):
        weld_type = numpy_weld_impl.numpy_to_weld_type_mapping[
            str(vector.dtype)]
    return NumpyArrayWeld(numpy_weld_impl.exp(vector, weld_type), WeldDouble())
