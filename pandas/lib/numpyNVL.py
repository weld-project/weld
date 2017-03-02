import numpy as np

import numpyImplNVL
from lazyOp import LazyOpResult
from nvlobject import *


class NumpyArrayNVL(LazyOpResult):
    """Summary

    Attributes:
        dim (TYPE): Description
        expr (TYPE): Description
        nvl_type (TYPE): Description
    """

    def __init__(self, expr, nvl_type, dim=1):
        """Summary

        Args:
            expr (TYPE): Description
            nvl_type (TYPE): Description
            dim (int, optional): Description
        """
        self.expr = expr
        self.nvl_type = nvl_type
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
        return NumpyArrayNVL(
            numpyImplNVL.div(
                self.expr,
                other,
                self.nvl_type
            ),
            self.nvl_type
        )

    def sum(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return NumpyArrayNVL(
            numpyImplNVL.aggr(
                self.expr,
                "+",
                0,
                self.nvl_type
            ),
            self.nvl_type,
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
    matrix_nvl_type = None
    vector_nvl_type = None

    if isinstance(matrix, LazyOpResult):
        matrix_nvl_type = matrix.nvl_type
        matrix = matrix.expr
    elif isinstance(matrix, np.ndarray):
        matrix_nvl_type = numpyImplNVL.numpy_to_nvl_type_mapping[str(matrix.dtype)]

    if isinstance(vector, LazyOpResult):
        vector_nvl_type = vector.nvl_type
        vector = vector.expr
    elif isinstance(vector, np.ndarray):
        vector_nvl_type = numpyImplNVL.numpy_to_nvl_type_mapping[str(vector.dtype)]

    return NumpyArrayNVL(
        numpyImplNVL.dot(
            matrix,
            vector,
            matrix_nvl_type,
            vector_nvl_type),
        NvlDouble())


def exp(vector):
    """
    Computes a per-element exponent of the passed-in vector.

    Args:
        vector (TYPE): Description
    """
    nvl_type = None
    if isinstance(vector, LazyOpResult):
        nvl_type = vector.nvl_type
        vector = vector.expr
    elif isinstance(vector, np.ndarray):
        nvl_type = numpyImplNVL.numpy_to_nvl_type_mapping[str(vector.dtype)]
    return NumpyArrayNVL(numpyImplNVL.exp(vector, nvl_type), NvlDouble())
