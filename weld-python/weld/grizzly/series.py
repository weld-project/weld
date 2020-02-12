"""
A Weld wrapper for pandas.Series.
"""

import inspect
import numpy as np
import pandas as pd

import weld.encoders.numpy as wenp

from pandas.core.internals import SingleBlockManager
from weld.lazy import PhysicalValue, WeldLazy, WeldNode, identity
from weld.types import *

from .ops import *

def _grizzlyseries_constructor_with_fallback(data=None, **kwargs):
    """
    A flexible constructor for Series._constructor, which needs to be able
    to fall back to a Series (if a certain operation does not produce
    geometries)
    """
    try:
        return GrizzlySeries(data=data, **kwargs)
    except TypeError:
        return pd.Series(data=data, **kwargs)

class GrizzlySeries(pd.Series):
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
    dtype : numpy.dtype or str
        Data type of the values.

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
    _internal_names = pd.Series._internal_names + ['weld_value_', 'cached_']
    _internal_names_set = set(_internal_names)

    # The encoders and decoders are stateless, so no need to instantiate them
    # for each computation.
    _encoder = wenp.NumPyWeldEncoder()
    _decoder = wenp.NumPyWeldDecoder()

    # ---------------------- WeldNode abstract methods ---------------------

    @property
    def children(self):
        """
        Returns the Weld children of this `GrizzlySeries.
        """
        return self.weld_value_.children

    @property
    def output_type(self):
        """
        Returns the Weld output type of this `GrizzlySeries`.
        """
        return self.weld_value_.output_type

    @property
    def is_value(self):
        """
        Returns whether this `GrizzlySeries` wraps a physical value rather than
        a computation.
        """
        return self.weld_value_.is_identity

    def evaluate(self):
        """
        Evaluates this `GrizzlySeries` and returns itself.

        Evaluation reduces the currently stored computation to a physical value
        by compiling and running a Weld program. If this `GrizzlySeries` refers
        to a physical value and no computation, no program is compiled, and this
        method does nothing.

        Evaluation results are cached. Subsequent calls to `evaluate()` on this
        `GrizzlySeries` will return `self` unmodified.

        """
        if self.cached_ is None:
            result = self.weld_value_.evaluate()
            # TODO(shoumik): it's unfortunate that this copy is needed, but
            # things are breaking without it (even if we hold a reference to
            # the WeldContext). DEBUG ME!
            if isinstance(result[0], wenp.weldbasearray):
                #super(GrizzlySeries, self).__init__(result[0].copy2numpy())
                self.cached_ = pd.Series(result[0].copy2numpy())
            else:
                self.cached_ = pd.Series(result[0])
            self.weld_value_ = identity(PhysicalValue(self.cached_.values,\
                    self.output_type, GrizzlySeries._encoder),
                    GrizzlySeries._decoder)
        return self

    def to_pandas(self):
        self.evaluate()
        assert self.cached_ is not None
        return self.cached_

    # ------------- pd.Series methods required for subclassing ----------

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
        elem_type = wenp.dtype_to_weld_type(data.dtype)
        return WeldVec(elem_type) if elem_type is not None else None

    # ---------------------- Initialization ------------------------------

    def __init__(self, data, dtype=None, **kwargs):
        # Everything important is done in __new__.
        pass

    def __new__(cls, data, dtype=None, **kwargs):
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
        >>> z = GrizzlySeries([1, 2, 3], dtype='int16') # Unsupported
        >>> z
        0    1
        1    2
        2    3
        dtype: int16
        >>> z.__class__
        <class 'weld.grizzly.series.GrizzlySeries'>
        >>> y = GrizzlySeries(['hi', 'bye']) # Unsupported
        >>> y.__class__
        <class 'pandas.core.series.Series'>
        >>> y = GrizzlySeries([1, 2, 3], index=[1, 0, 2]) # Unsupported
        >>> y.__class__
        <class 'pandas.core.series.Series'>
        """
        s = None
        if isinstance(data, WeldLazy):
            self = super(GrizzlySeries, cls).__new__(cls)
            super(GrizzlySeries, self).__init__(None, dtype=dtype, **kwargs)
            self.weld_value_ = data
            self.cached_ = None
            return self
        elif len(kwargs) != 0:
            return pd.Series(data, **kwargs)
        elif not isinstance(data, np.ndarray):
            # First, convert the input into a Series backed by an ndarray.
            s = pd.Series(data, dtype=dtype, **kwargs)
            data = s.values
        
        # Try to create a Weld type for the input.
        weld_type = GrizzlySeries._supports_grizzly(data)
        if weld_type is not None:
            self = super(GrizzlySeries, cls).__new__(cls)
            super(GrizzlySeries, self).__init__(data, dtype=dtype, **kwargs)
            self.weld_value_ = identity(
                    PhysicalValue(data, weld_type, GrizzlySeries._encoder),
                    GrizzlySeries._decoder)
            self.cached_ = None
            return self
        # Don't re-convert values if we did it once already -- it's expensive.
        return s if s is not None else pd.Series(data, **kwargs)

    # ---------------------- Operators ------------------------------
    #
    # TODO(shoumik): this can probably be refactored a bit, so common arguments
    # added to each of these functions will propagate automatically, etc.
    @classmethod
    def _get_class_that_defined_method(cls, meth):
        """
        Returns the class that defines the requested method. For methods that are
        defined outside of a particular set of Grizzly-defined classes, Grizzly will
        first evaluate lazy results before forwarding the data to `pandas.Series`.
        """
        if inspect.ismethod(meth):
            for cls in inspect.getmro(meth.__self__.__class__):
                if cls.__dict__.get(meth.__name__) is meth:
                    return cls
        if inspect.isfunction(meth):
            return getattr(inspect.getmodule(meth),
                           meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0])

    @classmethod
    def _requires_forwarding(cls, meth):
        defined_in = cls._get_class_that_defined_method(meth)
        if defined_in is not None and defined_in is not cls:
            return True
        else:
            return False

    @classmethod
    def forward(cls):
        from functools import wraps
        def forward_decorator(func):
            @wraps(func)
            def forwarding_wrapper(self, *args, **kwargs):
                print("forwarding {}".format(func.__name__))
                self.evaluate()
                return func(self.cached_, *args, **kwargs)
            return forwarding_wrapper
        return forward_decorator

    @classmethod
    def _add_forwarding_methods(cls):
        methods = dir(cls) 
        for meth in methods:
            if meth.startswith("_"):
                # We only want to do this for API methods.
                continue
            attr = getattr(cls, meth)
            if cls._requires_forwarding(attr):
                setattr(cls, meth, cls.forward()(attr))

    def _arithmetic_binop_impl(self, other, op, truediv=False):
        """
        Performs the operation on two `Series` elementwise.
        """
        left_ty = self.output_type.elem_type
        right_ty = other.output_type.elem_type
        output_type = wenp.binop_output_type(left_ty, right_ty, truediv)
        code = binary_map(op,
                left_type=str(left_ty),
                right_type=str(right_ty),
                leftval=str(self.weld_value_.id),
                rightval=str(other.weld_value_.id),
                cast_type=output_type)
        lazy = WeldLazy(code, [self.weld_value_, other.weld_value_],
                WeldVec(output_type), GrizzlySeries._decoder)
        return GrizzlySeries(lazy, dtype=wenp.weld_type_to_dtype(output_type))

    def add(self, other):
        return self._arithmetic_binop_impl(other, '+')

    def sub(self, other):
        return self._arithmetic_binop_impl(other, '-')

    def mul(self, other):
        return self._arithmetic_binop_impl(other, '*')

    def truediv(self, other):
        return self._arithmetic_binop_impl(other, '/', truediv=True)

    def divide(self, other):
        return self.truediv(other)

    def div(self, other):
        return self.truediv(other)

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    def __mul__(self, other):
        return self.mul(other)

    def __truediv__(self, other):
        return self.truediv(other)

    def __divmod__(self, other):
        return self.divmod(other)

    def __str__(self):
        if self.is_value:
            self.evaluate()
        if self.cached_ is not None:
            return str(self.cached_)
        return "GrizzlySeries({}, dtype: {}, deps: [{}])".format(
                self.weld_value_.expression,
                str(wenp.weld_type_to_dtype(self.output_type.elem_type)),
                ", ".join([str(child.id) for child in self.children]))

    def __repr__(self):
        return str(self)

GrizzlySeries._add_forwarding_methods()
