"""
A Weld wrapper for pandas.Series.

"""

import numpy as np
import pandas as pd

import weld.encoders.numpy as wenp
import weld.grizzly.weld.agg as weldagg

from weld.lazy import PhysicalValue, WeldLazy, WeldNode, identity
from weld.grizzly.weld.ops import *
from weld.grizzly.core.error import GrizzlyError
from weld.grizzly.core.forwarding import Forwarding
from weld.grizzly.core.generic import GrizzlyBase
from weld.grizzly.core.strings import StringMethods
from weld.types import *

class GrizzlySeries(Forwarding, GrizzlyBase):
    """
    A lazy `Series` object backed by a Weld computation.

    A `GrizzlySeries` can either represent to a normal `pandas.Series` object, or it
    can represent to a lazy computation by calling Weld-supported methods.
    `GrizzlySeries` that do not support Weld computation automatically fall back to
    regular pandas behavior. Similarly, `GrizzlySeries` that are initialized with
    features that Grizzly does not understand are initialized as `pd.Series`.

    Implementation Notes
    --------------------

    A Series in Grizzly is a 1D array of a single data type. Transformations on
    Series can result in scalar values, which are directly evaluated and
    returned as scalars, as other Series that encapsulate lazy Weld
    computations, or as DataFrames (currently unsupported).

    A Series always has a known type: unlike pandas Series, which are wrappers
    around Python object, Series with heterogeneous types (i.e., Series with
    `dtype=object`) are disallowed. Series that are not evaluated/store a lazy
    compuations also always have a known type.

    Internally, most Series are backed by NumPy arrays, which themselves are
    stored as contiguous C arrays. The main exception is string data. Grizzly
    operates exclusively over UTF-8 encoded data, and most string operations
    will create a copy of the input data upon evaluation. Strings are currently
    encoded as NumPy arrays with the 'S' dtype: this may change in the future
    (the 'S' dtype forces memory usage per string to match the length of the
    largest string in the array, which is clearly suboptimal).

    Unlike pandas Series, GrizzlySeries do not support alignment of index
    values for performance reasons. To align the indices of two Series, consider
    using a DataFrame with a join operator.

    Parameters
    ----------

    data : array-like, Iteratble, dict, or scalar value
        Contains data stored in the series.
    dtype : numpy.dtype or str
        Data type of the values.
    name : str
        A name to give the series.

    Examples
    --------
    >>> x = GrizzlySeries([1,2,3], name="numbers")
    >>> x
    0    1
    1    2
    2    3
    Name: numbers, dtype: int64
    """

    # The encoders and decoders are stateless, so no need to instantiate them
    # for each computation.
    _encoder = wenp.NumPyWeldEncoder()
    _decoder = wenp.NumPyWeldDecoder()

    @property
    def weld_value(self):
        return self.weld_value_

    @weld_value.setter
    def weld_value(self, value):
        self.weld_value_ = value

    @property
    def is_value(self):
        """
        Returns whether this collection wraps a physical value rather than
        a computation.
        """
        return self.weld_value.is_identity or hasattr(self, "evaluating_")

    @property
    def elem_type(self):
        """
        Returns the element type of this `GrizzlySeries`.
        """
        return self.output_type.elem_type

    @property
    def dtype(self):
        """
        Returns the NumPy dtype of this Series.

        """
        elem_type = self.elem_type
        if elem_type == WeldVec(I8()):
            return np.dtype('S')
        return wenp.weld_type_to_dtype(elem_type)

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
        weld.grizzly.core.error.GrizzlyError: GrizzlySeries is not evaluated and does not have values. Try calling 'evaluate()' first.
        """
        if not self.is_value:
            raise GrizzlyError("GrizzlySeries is not evaluated and does not have values. Try calling 'evaluate()' first.")
        return self.values_

    def evaluate(self):
        """
        Evaluates this `GrizzlySeries` and returns itself.

        Evaluation reduces the currently stored computation to a physical value
        by compiling and running a Weld program. If this `GrizzlySeries` refers
        to a physical value and no computation, no program is compiled, and this
        method returns `self` unmodified.

        """
        if not self.is_value:
            result = self.weld_value.evaluate()
            # TODO(shoumik): it's unfortunate that this copy is needed, but
            # things are breaking without it (even if we hold a reference to
            # the WeldContext). DEBUG ME!
            if isinstance(result[0], wenp.weldbasearray):
                self.__init__(result[0].copy2numpy(), name=self.name)
            else:
                self.__init__(result[0], name=self.name)
            setattr(self, "evaluating_", 0)
            self.weld_value = identity(PhysicalValue(self.values,\
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
        return pd.Series(data=self.values_, name=self.name, copy=copy)

    # ---------------------- Class methods ------------------------------

    @classmethod
    def supported(cls, data):
        """
        Returns whether the given ndarray supports Grizzly operation.

        Parameters
        ----------
        data : np.array

        Examples
        -------
        >>> GrizzlySeries.supported(np.array([1,2,3], dtype='int32'))
        vec[i32]
        >>> GrizzlySeries.supported(np.array(["foo","bar"], dtype='object'))
        >>> GrizzlySeries.supported(np.array(["foo","bar"], dtype='S'))
        vec[vec[i8]]

        """
        if not isinstance(data, np.ndarray) or data.ndim != 1:
            return None
        if data.dtype.char == 'S':
            elem_type = WeldVec(I8())
        else:
            elem_type = wenp.dtype_to_weld_type(data.dtype)
        return WeldVec(elem_type) if elem_type is not None else None

    # ---------------------- Initialization ------------------------------

    def __init__(self, data, dtype=None, name=None):
        """
        Initialize a new GrizzlySeries.

        >>> x = GrizzlySeries([1,2,3])
        >>> x
        0    1
        1    2
        2    3
        dtype: int64

        """

        s = None
        if isinstance(data, WeldLazy):
            self.name = name
            self.values_ = None
            self.weld_value = data
            return

        self.name = name
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
            # Try to convert a list of strings into a supported Numpy array.
            self.values_ = np.array(data, dtype='S')
        elif isinstance(data, pd.Series):
            if self.name is None:
                self.name = data.name
            if data.values.dtype == 'object' and len(data) > 0 and isinstance(data[0], str):
                self.values_ = np.array(data, dtype='S')
            else:
                self.values_ = data.values
        elif not isinstance(data, np.ndarray):
            # First, convert the input into a Numpy array.
            self.values_ = np.array(data, dtype=dtype)
        else:
            self.values_ = data

        # Try to create a Weld type for the input.
        weld_type = GrizzlySeries.supported(self.values_)
        if weld_type:
            self.weld_value = identity(
                    PhysicalValue(self.values_, weld_type, GrizzlySeries._encoder),
                    GrizzlySeries._decoder)
        else:
            raise GrizzlyError("unsupported data type '{}'".format(self.values_.dtype))

    # ---------------------- StringMethods ------------------------------

    @property
    def str(self):
        return StringMethods(self)

    # ---------------------- Aggregation ------------------------------

    def agg(self, funcs):
        """
        Apply the provided aggregation functions.

        If a single function is provided, this returns a scalar. If multiple
        functions are provided, this returns a `GrizzlySeries`, where the Nth
        element in the series is the result of the Nth aggregation.

        Examples
        --------
        >>> s = GrizzlySeries([1,2,3,4])
        >>> s.agg('sum')
        10
        >>> s.agg(['sum', 'mean']).evaluate()
        0    10.0
        1     2.5
        dtype: float64

        """
        if isinstance(funcs, str):
            funcs = [funcs]
        if isinstance(funcs, list):
            assert all([weldagg.supported(agg) for agg in funcs])

        output_elem_type = weldagg.result_elem_type(self.elem_type, funcs)
        if len(funcs) > 1:
            output_type = WeldVec(output_elem_type)
            decoder = GrizzlySeries._decoder
        else:
            output_type = output_elem_type
            # Result is a primitive, so we can use the default primitive decoder.
            decoder = None

        result_weld_value = weldagg.agg(self.weld_value, self.elem_type, funcs)(output_type, decoder)
        if decoder is not None:
            return GrizzlySeries(result_weld_value, dtype=wenp.weld_type_to_dtype(output_elem_type), name=self.name)
        else:
            # TODO(shoumik.palkar): Do we want to evaluate here? For now, we will, but eventually it may
            # be advantageous to have a lazy scalar value that can be used elsewhere.
            return result_weld_value.evaluate()[0]

    def count(self):
        """
        Computes the count.

        Examples
        --------
        >>> s = GrizzlySeries([1,2,3,4])
        >>> s.count()
        4

        """
        return self.agg('count')

    def sum(self):
        """
        Computes the sum.

        Examples
        --------
        >>> s = GrizzlySeries([1,2,3,4])
        >>> s.sum()
        10

        """
        return self.agg('sum')

    def prod(self):
        """
        Computes the product.

        Examples
        --------
        >>> s = GrizzlySeries([1,2,3,4])
        >>> s.prod()
        24

        """
        return self.agg('prod')

    def min(self):
        """
        Finds the min element.

        Examples
        --------
        >>> s = GrizzlySeries([1,2,3,4])
        >>> s.min()
        1

        """
        return self.agg('min')

    def max(self):
        """
        Returns the max element.

        Examples
        --------
        >>> s = GrizzlySeries([1,2,3,4])
        >>> s.max()
        4

        """
        return self.agg('max')

    def mean(self):
        """
        Computes the mean.

        Examples
        --------
        >>> s = GrizzlySeries([1,2,3,4])
        >>> s.mean()
        2.5

        """
        return self.agg('mean')

    def var(self):
        """
        Computes the variance.

        Examples
        --------
        >>> s = GrizzlySeries([1,2,3,4])
        >>> s.var()
        1.6666666666666667

        """
        return self.agg('var')

    def std(self):
        """
        Computes the standard deviation.

        Examples
        --------
        >>> s = GrizzlySeries([1,2,3,4])
        >>> s.std()
        1.2909944487358056

        """
        return self.agg('std')

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
        support indexes at the moment). This means that the index is currently **always
        dropped** in Grizzly. Here is an example of where a difference arises:

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

        This is equivalent to always calling `Series.reset_index(drop=True)` in Pandas.
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
        scalar_key = GrizzlySeries.scalar_ty(key, I64())
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
            arg_ty = GrizzlySeries.scalar_ty(arg, I64())
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
            lazy = slice_expr(self.weld_value, start, stop - start)(self.output_type, GrizzlySeries._decoder)
            return GrizzlySeries(lazy, dtype=self.dtype, name=self.name)

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
                not isinstance(bool_mask.elem_type, Bool):
                    raise GrizzlyError("bool_mask must be a GrizzlySeries with dtype=bool")

        lazy = mask(self.weld_value, self.elem_type, bool_mask.weld_value)(self.output_type, GrizzlySeries._decoder)
        return GrizzlySeries(lazy, dtype=self.dtype, name=self.name)

    # ---------------------- Operators ------------------------------

    def _arithmetic_binop_impl(self, other, op, truediv=False, weld_elem_type=None):
        """
        Performs the operation on two `Series` elementwise.

        Parameters
        ----------
        other: scalar or GrizzlySeries
            the RHS operand
        op: str
            the operator to apply.
        truediv: bool, optional
            is this a truediv?
        weld_elem_type: WeldType or None
            the element type produced by this operation. None
            means it is inferred based on Numpy's type conversion rules.

        Returns
        -------
        GrizzlySeries

        """
        left_ty = self.elem_type
        scalar_ty = GrizzlySeries.scalar_ty(other, left_ty)
        if scalar_ty is not None:
            # Inline scalars directly instead of adding them
            # as dependencies.
            right_ty = scalar_ty
            rightval = str(other)
        else:
            # Value is not a scalar -- for now, we require collection types to be
            # GrizzlySeries.
            if not isinstance(other, GrizzlySeries):
                raise GrizzlyError("RHS of binary operator must be a GrizzlySeries")
            right_ty = other.elem_type
            rightval = other.weld_value

        is_string = isinstance(left_ty, WeldVec)
        if op != "==" and op != "!=" and is_string:
            raise TypeError("Unsupported operand type(s) for '{}': {} and {}".format(op, left_ty, right_ty))

        # Don't cast if we're dealing with strings.
        if is_string:
            cast_type = ""
        else:
            cast_type = wenp.binop_output_type(left_ty, right_ty, truediv)

        output_type = cast_type if weld_elem_type is None else weld_elem_type
        lazy = binary_map(op,
                left_type=str(left_ty),
                right_type=str(right_ty),
                leftval=self.weld_value,
                rightval=rightval,
                scalararg=scalar_ty is not None,
                cast_type=cast_type)(WeldVec(output_type), GrizzlySeries._decoder)
        return GrizzlySeries(lazy, dtype=wenp.weld_type_to_dtype(output_type), name=self.name)

    def _compare_binop_impl(self, other, op):
        """
        Performs the comparison operation on two `Series` elementwise.
        """
        return self._arithmetic_binop_impl(other, op, weld_elem_type=Bool())

    def add(self, other):
        """
        Performs an element-wise addition.

        Examples
        --------
        >>> s = GrizzlySeries([10, 20, 30])
        >>> s.add(10).evaluate()
        0    20
        1    30
        2    40
        dtype: int64

        """
        return self._arithmetic_binop_impl(other, '+')

    def sub(self, other):
        """
        Performs an element-wise subtraction.

        Examples
        --------
        >>> s = GrizzlySeries([10, 20, 30])
        >>> s.sub(10).evaluate()
        0     0
        1    10
        2    20
        dtype: int64

        """
        return self._arithmetic_binop_impl(other, '-')

    def mod(self, other):
        """
        Performs an element-wise modulo.

        Examples
        --------
        >>> s = GrizzlySeries([10, 25, 31])
        >>> s.mod(10).evaluate()
        0    0
        1    5
        2    1
        dtype: int64

        """
        return self._arithmetic_binop_impl(other, '%')

    def mul(self, other):
        """
        Performs an element-wise multiplication.

        Examples
        --------
        >>> s = GrizzlySeries([10, 25, 31])
        >>> s.mul(10).evaluate()
        0    100
        1    250
        2    310
        dtype: int64

        """
        return self._arithmetic_binop_impl(other, '*')

    def truediv(self, other):
        """
        Performs an element-wise "true" division.

        Examples
        --------
        >>> s = GrizzlySeries([10, 20, 30])
        >>> s.truediv(10).evaluate()
        0    1.0
        1    2.0
        2    3.0
        dtype: float64

        """
        return self._arithmetic_binop_impl(other, '/', truediv=True)

    def divide(self, other):
        """
        Performs an element-wise "true" division.

        Examples
        --------
        >>> s = GrizzlySeries([10, 20, 30])
        >>> s.divide(10).evaluate()
        0    1.0
        1    2.0
        2    3.0
        dtype: float64

        """
        return self.truediv(other)

    def div(self, other):
        """
        Performs an element-wise "true" division.

        Examples
        --------
        >>> s = GrizzlySeries([10, 20, 30])
        >>> s.div(10).evaluate()
        0    1.0
        1    2.0
        2    3.0
        dtype: float64

        """
        return self.truediv(other)

    def eq(self, other):
        """
        Performs an element-wise equality comparison.

        Examples
        --------
        >>> s = GrizzlySeries([10, 20, 30])
        >>> s.eq(10).evaluate()
        0     True
        1    False
        2    False
        dtype: bool

        """
        return self._compare_binop_impl(other, '==')

    def ne(self, other):
        """
        Performs an element-wise not-equal comparison.

        Examples
        --------
        >>> s = GrizzlySeries([10, 20, 30])
        >>> s.ne(10).evaluate()
        0    False
        1     True
        2     True
        dtype: bool

        """
        return self._compare_binop_impl(other, '!=')

    def ge(self, other):
        """
        Performs an element-wise '>=' comparison.

        Examples
        --------
        >>> s = GrizzlySeries([10, 20, 30])
        >>> s.ge(20).evaluate()
        0    False
        1     True
        2     True
        dtype: bool

        """
        return self._compare_binop_impl(other, '>=')

    def gt(self, other):
        """
        Performs an element-wise '>' comparison.

        Examples
        --------
        >>> s = GrizzlySeries([10, 20, 30])
        >>> s.gt(20).evaluate()
        0    False
        1    False
        2     True
        dtype: bool

        """
        return self._compare_binop_impl(other, '>')

    def le(self, other):
        """
        Performs an element-wise '<=' comparison.

        Examples
        --------
        >>> s = GrizzlySeries([10, 20, 30])
        >>> s.le(20).evaluate()
        0     True
        1     True
        2    False
        dtype: bool

        """
        return self._compare_binop_impl(other, '<=')

    def lt(self, other):
        """
        Performs an element-wise '<' comparison.

        Examples
        --------
        >>> s = GrizzlySeries([10, 20, 30])
        >>> s.lt(20).evaluate()
        0     True
        1    False
        2    False
        dtype: bool

        """
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

    def __ne__(self, other):
        return self.ne(other)

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
            return str(self.to_pandas(copy=False))
        return "GrizzlySeries({}, name: {}, dtype: {}, deps: [{}])".format(
                self.weld_value.expression,
                self.name,
                str(self.dtype),
                ", ".join([str(child.id) for child in self.children]))

    def __repr__(self):
        return str(self)
