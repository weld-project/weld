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
        return self.weld_value_.is_identity or hasattr(self, "evaluating_")

    @property
    def code(self):
        """
        Returns the Weld code for this computation.
        """
        return self.weld_value_.code

    @property
    def values(self):
        """
        Returns the raw data values of this `GrizzlySeries`. If `self.is_value`
        is `False`, this property raises an exception.

        Examples
        --------
        >>> x = GrizzlySeries([1,2,3])
        >>> x.is_value
        True
        >>> x.values
        array([1, 2, 3])
        >>> x = x + x
        >>> x.is_value
        False
        >>> x.values
        Traceback (most recent call last):
        ...
        weld.grizzly.error.GrizzlyError: GrizzlySeries is not evaluated and does not have values. Try calling 'evaluate()' first.
        """
        if not self.is_value:
            raise GrizzlyError("GrizzlySeries is not evaluated and does not have values. Try calling 'evaluate()' first.")
        return super(GrizzlySeries, self).values

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
            setattr(self, "evaluating_", 0)
            self.weld_value_ = identity(PhysicalValue(self.values,\
                    self.output_type, GrizzlySeries._encoder),
                    GrizzlySeries._decoder)
            delattr(self, "evaluating_")
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

    # ---------------------- Indexing ------------------------------

    def __setitem__(self, key, value):
        """
        Point access is not allowed in Grizzly -- require conversion to
        Pandas first.

        """
        raise GrizzlyError("Grizzly does not support point access: use 'to_pandas()' first.")

    def __getitem__(self, key):
        """
        Access elements in a `GrizzlySeries`.

        The Grizzly accessor supports scalar access (return a single element
        from a `GrizzlySeries`) as well as masked access. Masked access returns
        a new filtered `GrizzlySeries`, where the key is another series-like
        object of dtype `bool` and the same length as `self`: each element
        corresponding to `True` is kept, and each element corresponding to
        `False` is discarded.

        Note that `__getitem__()` will call `evaluate()` when a scalar key is passed,
        but will produce a lazy value when a non-scalar key is passed (i.e., when a
        non-scalar value will be returned).

        Differences from Pandas
        -----------------------

        * In Pandas, many indexing operations just modify the index on the Series. In
        Grizzly, we instead register a lazy computation and return a new GrizzlySeries
        that always effectively has a RangeIndex (note that Grizzly does not actually
        support indexes at the moment). This means that some indexing information is
        lost in Grizzly. Here is an example of where a difference arises:

        >>> p1 = pd.Series([1,2,3])
        >>> p2 = pd.Series([4,5])
        >>> p2 + p1[1:3] # Aligns indexes
        0    NaN
        1    7.0
        2    NaN
        dtype: float64
        >>> p1 = GrizzlySeries([1,2,3])
        >>> p2 = GrizzlySeries([4,5])
        >>> (p2 + p1[1:3]).evaluate() # No alignment
        0    6
        1    8
        dtype: int64

        Note that this behavior may change in the future when we add support for indexes.

        Basic Examples
        --------
        >>> x = GrizzlySeries([1,2,3])
        >>> x[1]
        2
        >>> y = x + x
        >>> y[1] # Causes evaluation
        4
        >>> y = x + x
        >>> z = y[0:2]
        >>> z.evaluate()
        0    2
        1    4
        dtype: int64
        >>> y = x + x
        >>> z = y[:2]
        >>> z.evaluate()
        0    2
        1    4
        dtype: int64

        Examples with Masking
        ---------------------
        >>> x = GrizzlySeries([1,2,3,4,5])
        >>> y = x[GrizzlySeries([True, False, False, False, False])]
        >>> y.evaluate()
        0    1
        dtype: int64
        >>> y = x[x % 2 == 0]
        >>> y.evaluate()
        0    2
        1    4
        dtype: int64

        """
        # If the key is a scalar
        scalar_key = GrizzlySeries._scalar_ty(key, I64())
        if isinstance(scalar_key, I64):
            self.evaluate()
            return self.values[key]

        def normalize_slice_arg(arg, default=None):
            """
            Returns the slice argument as a Python integer, and
            throws an error if the argument cannot be represented as such.

            """
            if arg is None:
                if default is not None:
                    arg = default
                else:
                    raise GrizzlyError("slice got 'None' when value where expected")
            arg_ty = GrizzlySeries._scalar_ty(arg, I64())
            if not isinstance(arg_ty, I64):
                raise GrizzlyError("slices in __getitem__() must be integers")
            return int(arg)

        if isinstance(key, slice):
            if key.step is not None:
                # Don't support step yet.
                self.evaluate()
                return self.values[key]
            start = normalize_slice_arg(key.start, default=0)
            stop = normalize_slice_arg(key.stop, default=None)
            # Weld's slice operator takes a size instead of a stopping index.
            code = slice_expr(self.weld_value_.id, start, stop - start)
            dependencies = [self.weld_value_]
            lazy = WeldLazy(code, dependencies, self.output_type, GrizzlySeries._decoder)
            return GrizzlySeries(lazy, dtype=self.dtype)

        if not isinstance(key, GrizzlySeries):
            raise GrizzlyError("array-like key in __getitem__ must be a GrizzlySeries.")

        if isinstance(key.output_type.elem_type, Bool):
            return self.mask(key)

    def mask(self, bool_mask, other=None):
        """
        Mask out values from `self` using the `GrizzlySeries` `bool_mask`.

        Parameters
        ----------
        bool_mask : GrizzlySeries
            A boolean GrizzlySeries used for the masking
        other : unused
            A value to replace values that evaluate to `False` in the mask.
            Currently unsupported -- this method will throw an error if it is not None.

        Examples
        --------

        >>> x = GrizzlySeries([1, 2, 3, 4, 5])
        >>> x.mask(GrizzlySeries([True, False, True, False, True])).evaluate()
        0    1
        1    3
        2    5
        dtype: int64

        """
        assert other is None
        if not isinstance(bool_mask, GrizzlySeries) or\
                not isinstance(bool_mask.output_type.elem_type, Bool):
                    raise GrizzlyError("bool_mask must be a GrizzlySeries with dtype=bool")

        code = mask(self.weld_value_.id, self.output_type.elem_type, bool_mask.weld_value_.id)
        dependencies = [self.weld_value_, bool_mask.weld_value_]
        lazy = WeldLazy(code, dependencies, self.output_type, GrizzlySeries._decoder)
        return GrizzlySeries(lazy, dtype=self.dtype)

    # ---------------------- Operators ------------------------------

    @classmethod
    def _scalar_ty(cls, value, cast_ty):
        """
        Returns the scalar Weld type of a scalar Python value. If the value is not a scalar,
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

    def _arithmetic_binop_impl(self, other, op, truediv=False, weld_elem_type=None):
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
        output_type = cast_type if weld_elem_type is None else weld_elem_type
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
        return self._arithmetic_binop_impl(other, op, weld_elem_type=Bool())

    def add(self, other):
        return self._arithmetic_binop_impl(other, '+')

    def sub(self, other):
        return self._arithmetic_binop_impl(other, '-')

    def mod(self, other):
        return self._arithmetic_binop_impl(other, '%')

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

    def __mod__(self, other):
        return self.mod(other)

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
