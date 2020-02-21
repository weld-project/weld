"""
A Weld wrapper for pandas.Series.
"""

import inspect
import numpy as np
import pandas as pd
import warnings

import weld.encoders.numpy as wenp

from pandas.core.internals import SingleBlockManager
from weld.lazy import PhysicalValue, WeldLazy, WeldNode, identity
from weld.types import *

from .ops import *
from .error import *

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
    _internal_names = pd.Series._internal_names + ['weld_value_']
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
        method returns `self` unmodified.

        """
        if not self.is_value:
            result = self.weld_value_.evaluate()
            # TODO(shoumik): it's unfortunate that this copy is needed, but
            # things are breaking without it (even if we hold a reference to
            # the WeldContext). DEBUG ME!
            if isinstance(result[0], wenp.weldbasearray):
                super(GrizzlySeries, self).__init__(result[0].copy2numpy())
            else:
                super(GrizzlySeries, self).__init__(result[0])
            self.weld_value_ = identity(PhysicalValue(self.values,\
                    self.output_type, GrizzlySeries._encoder),
                    GrizzlySeries._decoder)
        return self

    def to_pandas(self, copy=False):
        """
        Evaluates this computation and coerces it into a pandas `Series`.

        This is guaranteed to be 0-copy if `self.is_value == True`. Otherwise,
        some allocation may occur. Regardless, `self` and the returned `Series`
        will always have the same underlying data unless `copy = True`.

        Parameters
        ----------
        copy : boolean, optional
            Specifies whether the new `Series` should copy data from self

        Examples
        --------
        >>> x = GrizzlySeries([1,2,3])
        >>> res = x + x
        >>> pandas_res = res.to_pandas()
        >>> pandas_res
        0    2
        1    4
        2    6
        dtype: int64
        >>> res # Also evaluated now
        0    2
        1    4
        2    6
        dtype: int64
        >>> res.values is pandas_res.values
        True
        >>> pandas_res = res.to_pandas(copy=True)
        >>> res.values is pandas_res.values
        False
        """
        self.evaluate()
        assert self.is_value
        return pd.Series(data=self.array,
                index=self.index,
                copy=copy,
                fastpath=True)

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

    def __init__(self, data, dtype=None, index=None, **kwargs):
        # Everything important is done in __new__.
        pass

    def __new__(cls, data, dtype=None, index=None, **kwargs):
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
        if isinstance(data, WeldLazy):
            self = super(GrizzlySeries, cls).__new__(cls)
            super(GrizzlySeries, self).__init__(None, dtype=dtype, **kwargs)
            self.weld_value_ = data
            return self
        elif index is not None and not isinstance(index, pd.RangeIndex):
            # TODO(shoumik): This is probably incomplete, since we could have a
            # RangeIndex that does not capture the full span of the data, has a
            # non-zero step, etc.
            return pd.Series(data, dtype=dtype, index=index, **kwargs)
        elif len(kwargs) != 0:
            return pd.Series(data, dtype=dtype, index=index, **kwargs)
        elif not isinstance(data, np.ndarray):
            # First, convert the input into a Series backed by an ndarray.
            s = pd.Series(data, dtype=dtype, index=index, **kwargs)
            data = s.values

        # Try to create a Weld type for the input.
        weld_type = GrizzlySeries._supports_grizzly(data)
        if weld_type is not None:
            self = super(GrizzlySeries, cls).__new__(cls)
            super(GrizzlySeries, self).__init__(data, dtype=dtype, **kwargs)
            self.weld_value_ = identity(
                    PhysicalValue(data, weld_type, GrizzlySeries._encoder),
                    GrizzlySeries._decoder)
            return self
        # Don't re-convert values if we did it once already -- it's expensive.
        return s if s is not None else pd.Series(data, dtype=dtype, index=index, **kwargs)

    # ---------------------- Operators ------------------------------

    @classmethod
    def _scalar_ty(cls, value, cast_ty):
        """
        Returns the scalar type of a scalar value. If the value is not a scalar,
        returns None. For primitive 'int' values, returns 'cast_ty'.

        This returns 'None' if the value type is not supported.

        Parameters
        ----------
        value : any
            value whose dtype to obtain.

        Returns
        -------
        np.dtype

        """
        if hasattr(value, 'dtype') and hasattr(value, 'shape') and value.shape == ():
            return wenp.dtype_to_weld_type(value.dtype)
        if isinstance(value, int):
            return cast_ty
        if isinstance(value, float):
            return F64()
        if isinstance(value, bool):
            return Bool()

    def _arithmetic_binop_impl(self, other, op, truediv=False, dtype=None):
        """
        Performs the operation on two `Series` elementwise.
        """
        left_ty = self.output_type.elem_type
        scalar_ty = GrizzlySeries._scalar_ty(other, left_ty)
        if scalar_ty is not None:
            # Inline scalars directly instead of adding them
            # as dependencies.
            right_ty = scalar_ty
            rightval = str(other)
            print(rightval)
            dependencies = [self.weld_value_]
        else:
            # Value is not a scalar -- for now, we require collection types to be
            # GrizzlySeries.
            if not isinstance(other, GrizzlySeries):
                raise GrizzlyError("RHS of binary operator must be a GrizzlySeries")
            right_ty = other.output_type.elem_type
            rightval = str(other.weld_value_.id)
            dependencies = [self.weld_value_, other.weld_value_]

        cast_type = wenp.binop_output_type(left_ty, right_ty, truediv)
        output_type = cast_type if dtype is None else dtype
        code = binary_map(op,
                left_type=str(left_ty),
                right_type=str(right_ty),
                leftval=str(self.weld_value_.id),
                rightval=rightval,
                scalararg=scalar_ty is not None,
                cast_type=cast_type)
        lazy = WeldLazy(code, dependencies,
                WeldVec(output_type), GrizzlySeries._decoder)
        return GrizzlySeries(lazy, dtype=wenp.weld_type_to_dtype(output_type))

    def _compare_binop_impl(self, other, op):
        """
        Performs the comparison operation on two `Series` elementwise.
        """
        return self._arithmetic_binop_impl(other, op, dtype=Bool())

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

    def eq(self, other):
        return self._compare_binop_impl(other, '==')

    def ge(self, other):
        return self._compare_binop_impl(other, '>=')

    def gt(self, other):
        return self._compare_binop_impl(other, '>')

    def le(self, other):
        return self._compare_binop_impl(other, '<=')

    def lt(self, other):
        return self._compare_binop_impl(other, '<')

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

    def __eq__(self, other):
        return self.eq(other)

    def __ge__(self, other):
        return self.ge(other)

    def __gt__(self, other):
        return self.gt(other)

    def __le__(self, other):
        return self.le(other)

    def __lt__(self, other):
        return self.lt(other)

    def __str__(self):
        if self.is_value:
            # This is guaranteed to be 0-copy.
            return str(self.to_pandas())
        return "GrizzlySeries({}, dtype: {}, deps: [{}])".format(
                self.weld_value_.expression,
                str(wenp.weld_type_to_dtype(self.output_type.elem_type)),
                ", ".join([str(child.id) for child in self.children]))

    def __repr__(self):
        return str(self)

    # ---------------------- Method forwarding setup ------------------------------

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
                self.evaluate()
                result = func(self, *args, **kwargs)
                # Unsupported functions will return Series -- try to
                # switch back to GrizzlySeries.
                if not isinstance(result, GrizzlySeries) and isinstance(result, pd.Series):
                    try_convert = GrizzlySeries(data=result.values, index=result.index)
                    if not isinstance(try_convert, GrizzlySeries):
                        warnings.warn("Unsupported operation '{}' produced unsupported Series: falling back to Pandas".format(
                            func.__name__))
                    return try_convert
                return result
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

# Wrap public API methods for forwarding
GrizzlySeries._add_forwarding_methods()
