"""Summary
"""
from nvlobject import *


def to_nvl_type(nvl_type, dim):
    """Summary

    Args:
        nvl_type (TYPE): Description
        dim (TYPE): Description

    Returns:
        TYPE: Description
    """
    for i in xrange(dim):
        nvl_type = NvlVec(nvl_type)
    return nvl_type


class LazyOpResult:
    """Wrapper class around as yet un-evaluated Weld computation results

    Attributes:
        dim (int): Dimensionality of the output
        expr (NvlObject / Numpy.ndarray): The expression that needs to be
            evaluated
        nvl_type (NvlType): Type of the output object
    """

    def __init__(self, expr, nvl_type, dim):
        """Summary

        Args:
            expr (TYPE): Description
            nvl_type (TYPE): Description
            dim (TYPE): Description
        """
        self.expr = expr
        self.nvl_type = nvl_type
        self.dim = dim

    def evaluate(self, verbose=True, decode=True):
        """Summary

        Args:
            verbose (bool, optional): Description
            decode (bool, optional): Description

        Returns:
            TYPE: Description
        """
        if isinstance(self.expr, NvlObject):
            return self.expr.evaluate(
                to_nvl_type(
                    self.nvl_type,
                    self.dim),
                verbose,
                decode)
        return self.expr
