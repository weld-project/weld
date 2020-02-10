"""
A Weld wrapper for pandas.Series.
"""

import numpy as np
import pandas as pd

import weld.encoders.numpy as wenp

from pandas.core.internals import SingleBlockManager
from weld.lazy import PhysicalValue, WeldLazy, WeldNode

def _grizzlyseries_constructor_with_fallback(data=None, index=None, **kwargs):
    """
    A flexible constructor for Series._constructor, which needs to be able
    to fall back to a Series (if a certain operation does not produce
    geometries)
    """
    try:
        return GrizzlySeries(data=data, index=index, **kwargs)
    except TypeError:
        return pd.Series(data=data, index=index, **kwargs)

class GrizzlySeries(WeldNode, pd.Series):
    """
    A lazy `Series` object backed by a Weld computation.

    A `GrizzlySeries` can either represent to a normal `pandas.Series` object, or it
    can represent to a lazy computation by calling Weld-supported methods.
    `GrizzlySeries` that do not support Weld computation automatically fall back to
    regular pandas behavior. Similarly, `GrizzlySeries` that are initialized with
    features that Grizzly does not understand are initialized as `pd.Series`.

    Parameters
    ----------

    data : array-like, Iteratble, dict, or scalar value
        Contains data stored in the series.

    Examples
    --------
    >>> x = GrizzlySeries([1,2,3]) 
    >>> x
    0    1
    1    2
    2    3
    dtype: int64
    """

    # TODO(shoumik): Does this need to be in _metadata instead?
    _internal_names = pd.Series._internal_names + ['weld_value']
    _internal_names_set = set(_internal_names)

    # ---------------------- WeldNode abstract methods ---------------------

    @property
    def children(self):
        raise NotImplementedError

    @property
    def output_type(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    # ---------------------- pd.Series methods ------------------------------

    @property
    def _constructor(self):
        return _grizzlyseries_constructor_with_fallback

    @property
    def _constructor_expanddim(self):
        return NotImplemented

    # ---------------------- Class methods ------------------------------

    @classmethod
    def _supports_grizzly(cls, data):
        """
        Returns the Weld type if data supports Grizzly operation, or returns
        `None` otherwise.

        Parameters
        ----------
        data : np.array

        """
        if not isinstance(data, np.ndarray) or data.ndim != 1:
            return None
        return wenp.dtype_to_weld_type(data.dtype)

    # ---------------------- Implementation ------------------------------

    def __init__(self, data, **kwargs):
        # Everything important is done in __new__.
        pass

    def __new__(cls, data, **kwargs):
        """
        Internal initialization. Tests below are for internal visibility only.

        >>> x = GrizzlySeries([1,2,3]) 
        >>> x
        0    1
        1    2
        2    3
        dtype: int64
        >>> x.__class__
        <class 'weld.grizzly.series.GrizzlySeries'>
        >>> x = GrizzlySeries(np.ones(5))
        >>> x.__class__
        <class 'weld.grizzly.series.GrizzlySeries'>
        >>> y = GrizzlySeries(['hi', 'bye']) # Unsupported
        >>> y.__class__
        <class 'pandas.core.series.Series'>
        >>> y = GrizzlySeries([1, 2, 3], index=[1, 0, 2]) # Unsupported
        >>> y.__class__
        <class 'pandas.core.series.Series'>
        """

        s = None
        if isinstance(data, WeldNode):
            # TODO(shoumik): Create a 1-1 dependency
            return NotImplemented
        elif len(kwargs) != 0:
            return pd.Series(data, **kwargs)
        elif not isinstance(data, np.ndarray):
            # First, convert the input into a Series backed by an ndarray.
            s = pd.Series(data, **kwargs)
            data = s.values
        
        # Try to create a Weld type for the input.
        weld_type = GrizzlySeries._supports_grizzly(data)
        if weld_type is not None:
            self = super(GrizzlySeries, cls).__new__(cls)
            super(GrizzlySeries, self).__init__(data, **kwargs)
            self.weld_value = PhysicalValue(data, weld_type, wenp.NumPyWeldEncoder())
            return self
        return s if s is not None else pd.Series(data, **kwargs)
