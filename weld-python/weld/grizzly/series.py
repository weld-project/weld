"""
A Weld wrapper for pandas.Series.
"""

import numpy as np

from pandas import Series
from pandas.core.internals import SingleBlockManager

from weld.encoders.numpy import dtype_to_weld_type
from weld.lazy import WeldLazy

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

class WeldSeries(WeldLazy, Series):
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

    @classmethod
    def resolve_weld_type(self, data):
        return NotImplemented

    def __new__(cls, data=None, index=None, **kwargs):
        s = None
        if isinstance(data, WeldLazy):
            # TODO(shoumik): Create a 1-1 dependency
            return NotImplemented
        elif not isinstance(data, np.ndarray):
            # First, convert the input into a Series backed by an ndarray.
            s = Series(data, index=index, **kwargs)
            data = s.values
        if _supports_grizzly(data) and index is None:
            data = PhysicalValue(data)
            self = super(WeldSeries, cls).__new__(cls)
            super(WeldSeries, self).__init__(data, **kwargs)
            return self
        return s if s is not None else Series(data, index=index, **kwargs)
