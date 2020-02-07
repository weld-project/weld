"""
A Weld wrapper for pandas.Series.
"""

import numpy as np

from pandas import Series
from pandas.core.internals import SingleBlockManager

from weld.encoders.numpy import dtype_to_weld_type

def _weldseries_constructor_with_fallback(data=None, index=None, **kwargs):
    """
    A flexible constructor for WeldSeries._constructor, which needs to be able
    to fall back to a Series (if a certain operation does not produce
    geometries)
    """
    try:
        return WeldSeries(data=data, index=index, **kwargs)
    except TypeError:
        return Series(data=data, index=index, **kwargs)

class WeldSeries(GrizzlyBase, Series):
    """ A lazy Series object backed by a Weld computation. """

    @property
    def _constructor(self):
        return _weldseries_constructor_with_fallback

    @property
    def _constructor_expanddim(self):
        return NotImplemented

    def __init__(self, *args, **kwargs):
        # Everything important is done in __new__.
        pass

    def __new__(cls, data=None, index=None, **kwargs):
        s = None
        if not isinstance(data, np.ndarray):
            s = Series(data, index=index, **kwargs)
            data = s.values
        if index is not None or dtype_to_weld_type(data.dtype) is not None:
            self = super(WeldSeries, cls).__new__(cls)
            super(WeldSeries, self).__init__(data, index=index, **kwargs)
            return self
        return s if s is not None else Series(data, index=index, **kwargs)

    def add(self, other, **unsupported_kwargs):
        # TODO: This should be code-gen'd.
        if len(unsupported_kwargs) > 0:
            return NotImplemented
        op = OpRegistry.get("add")
        op.build_map(self.dtype, other.dtype)
        code = binary_op(op.infix, self.resolve_dtype(self.dtype, other.dtype)