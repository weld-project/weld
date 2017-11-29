"""Summary
"""
from weld.weldobject import *


def to_weld_type(weld_type, dim):
    """Summary

    Args:
        weld_type (TYPE): Description
        dim (TYPE): Description

    Returns:
        TYPE: Description
    """
    for i in xrange(dim):
        weld_type = WeldVec(weld_type)
    return weld_type


class LazyOpResult:
    """Wrapper class around as yet un-evaluated Weld computation results

    Attributes:
        dim (int): Dimensionality of the output
        expr (WeldObject / Numpy.ndarray): The expression that needs to be
            evaluated
        weld_type (WeldType): Type of the output object
    """

    def __init__(self, expr, weld_type, dim):
        """Summary

        Args:
            expr (TYPE): Description
            weld_type (TYPE): Description
            dim (TYPE): Description
        """
        self.expr = expr
        self.weld_type = weld_type
        self.dim = dim

    def evaluate(self, verbose=True, decode=True, passes=None, num_threads=1,
                 apply_experimental_transforms=False):
        """Summary

        Args:
            verbose (bool, optional): Description
            decode (bool, optional): Description

        Returns:
            TYPE: Description
        """
        if isinstance(self.expr, WeldObject):
            return self.expr.evaluate(
                to_weld_type(
                    self.weld_type,
                    self.dim),
                verbose,
                decode,
                passes=passes,
                num_threads=num_threads,
                apply_experimental_transforms=apply_experimental_transforms)
        return self.expr
