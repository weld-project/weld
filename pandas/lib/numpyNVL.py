import numpy as np

import numpyImplNVL
from lazyOp import LazyOpResult
from nvlobject import *


class NumpyArrayNVL(LazyOpResult):

    def __init__(self, expr, nvl_type, dim=1):
        self.expr = expr
        self.nvl_type = nvl_type
        self.dim = dim

    def __div__(self, other):
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
    """
    nvl_type = None
    if isinstance(matrix, LazyOpResult):
        matrix = matrix.expr
    if isinstance(vector, LazyOpResult):
        nvl_type = vector.nvl_type
        vector = vector.expr
    elif isinstance(vector, np.ndarray):
        nvl_type = numpyImplNVL.numpy_to_nvl_type_mapping[str(vector.dtype)]
    return NumpyArrayNVL(
        numpyImplNVL.dot(
            matrix,
            vector,
            nvl_type),
        NvlDouble())


def exp(vector):
    """
    Computes a per-element exponent of the passed-in vector.
    """
    nvl_type = None
    if isinstance(vector, LazyOpResult):
        nvl_type = vector.nvl_type
        vector = vector.expr
    elif isinstance(vector, np.ndarray):
        nvl_type = numpyImplNVL.numpy_to_nvl_type_mapping[str(vector.dtype)]
    return NumpyArrayNVL(numpyImplNVL.exp(vector, nvl_type), NvlDouble())
