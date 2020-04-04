"""
A Weld wrapper for pandas.DataFrame.

"""

import numpy as np
import pandas as pd

import weld.encoders.numpy as wenp
import weld.grizzly.weld.ops as weldops

from weld.lazy import WeldLazy, identity
from weld.grizzly.core.forwarding import Forwarding
from weld.grizzly.core.generic import GrizzlyBase
from weld.grizzly.core.indexes import ColumnIndex
from weld.grizzly.core.series import GrizzlySeries
from weld.types import *

class GrizzlyDataFrame(Forwarding, GrizzlyBase):
    """
    An API-compatible DataFrame backed by Weld.

    DataFrames are dictionary-like containers of Series of the same length.
    Each Series can be a different data type. Operations on DataFrames align on
    column name. Unlike pandas, DataFrame operations do not align on row
    indexes.

    Examples
    --------
    >>> df = GrizzlyDataFrame({'name': ['mike', 'sam', 'sally'], 'age': [20, 22, 56]})
    >>> df
        name  age
    0   mike   20
    1    sam   22
    2  sally   56
    >>> df2 = GrizzlyDataFrame({'nom': ['jacques', 'kelly', 'marie'], 'age': [50, 60, 70]})
    >>> df.add(df2).to_pandas()
       age  name  nom
    0   70   NaN  NaN
    1   82   NaN  NaN
    2  126   NaN  NaN

    """

    # Indicates that the length of a DataFrame is not known. Certain operations on DataFrames
    # with this length are disallowed, since Grizzly (like Pandas) assumes each column in a DataFrame
    # is of the same length.
    UNKNOWN_LENGTH = -1

    _encoder = wenp.NumPyWeldEncoder()
    _decoder = wenp.NumPyWeldDecoder()

    # ------------------- GrizzlyBase methods ----------------------

    @property
    def weld_value(self):
        if hasattr(self, "weld_value_"):
            return self.weld_value_

        if len(self.data) == 0:
            raise GrizzlyError("weld_value cannot be accessed on DataFrame with no data")
        output_type = WeldStruct([col.weld_value.output_type for col in self.data])
        self.weld_value_ = weldops.make_struct(*[col.weld_value for col in self.data])(output_type, GrizzlyDataFrame._decoder)
        return self.weld_value_

    @property
    def is_value(self):
        """
        Returns whether this DataFrame is a physical value.

        If this is True, evaluate() is guaranteed to be a no-op.

        Examples
        --------
        >>> df = GrizzlyDataFrame({'name': ['sam', 'sally'], 'age': [25, 50]})
        >>> df.is_value
        True
        >>> df = df.eq(df)
        >>> df.is_value
        False

        """
        return self.pandas_df is not None or\
                all([child.is_identity for child in self.children])

    def evaluate(self):
        """
        Evaluates this `GrizzlyDataFrame` and returns itself.

        Evaluation reduces the currently stored computation to a physical value
        by compiling and running a Weld program. If this `GrizzlyDataFrame` refers
        to a physical value and no computation, no program is compiled, and this
        method returns `self` unmodified.

        Returns
        -------
        GrizzlyDataFrame

        """
        if not self.is_value:
            if len(self.data) == 0:
                # We're an empty DataFrame
                self.pandas_df = pd.DataFrame()
                return
            # Collect each vector into a struct rather than evaluating each Series individually:
            # this is more efficient so computations shared among vectors can be optimized.
            result = self.weld_value.evaluate()
            columns = result[0]
            new_data = []
            length = None
            for column in columns:
                data = column.copy2numpy()
                column_length = len(data)
                if length is not None:
                    assert column_length == length, "invalid DataFrame produced after evaluation"
                else:
                    length = column_length
                series = GrizzlySeries(data)
                new_data.append(series)

            # Columns is unchanged
            self.pandas_df = None
            self.data = new_data
            self.length = length
            # Reset the weld representation.
            delattr(self, "weld_value_")
        assert self.is_value
        return self

    def to_pandas(self, copy=False):
        """
        Evaluate and convert this GrizzlyDataFrame to a pandas DataFrame.

        Parameters
        ----------
        copy : bool
            whether to copy the data into the new DataFrame. This only guarantees
            that a copy _will_ occur; it does not guarantee that a copy will not.

        Returns
        -------
        pandas.DataFrame

        """
        self.evaluate()
        if self.pandas_df is not None:
            return self.pandas_df
        col_to_data = dict()
        for col in self.columns:
            col_to_data[col] = self._col(col).values
        self.pandas_df = pd.DataFrame(data=col_to_data, copy=copy)
        return self.pandas_df

    # ------------------- Initialization ----------------------

    def __init__(self, data, columns=None, _length=UNKNOWN_LENGTH, _fastpath=False):
        if _fastpath:
            assert all([isinstance(d, GrizzlySeries) for d in data])
            assert isinstance(columns, ColumnIndex)
            self.data = data
            self.columns = columns
            self.length = _length
            self.pandas_df = None
            return

        self.data = []
        column_index = []
        # Keep a reference to the Pandas DataFrame. This should not consume any extra memory, since GrizzlySeries
        # just keep references ot the same values. The only exception is for string data, where a copy will be created
        # to force use of the 'S' dtype.
        #
        # TODO(shoumik): Convert string data to dtype 'S' here so it doesn't get copied when wrapping the Series as
        # GrizzlySeries.
        if isinstance(data, pd.DataFrame):
            self.pandas_df = data
        else:
            self.pandas_df = pd.DataFrame(data, columns=columns)
        self.length = len(self.pandas_df)
        for (i, col) in enumerate(self.pandas_df):
            grizzly_series = GrizzlySeries(self.pandas_df[col], name=self.pandas_df[col].name)
            if not isinstance(grizzly_series, GrizzlySeries):
                raise TypeError("Unsupported Series in DataFrame: {}".format(self.pandas_df[col]))
            self.data.append(grizzly_series)
            column_index.append(col)

        self.columns = ColumnIndex(column_index)

    def _col(self, col):
        """
        Returns the column associated with the column name 'col'.

        """
        return self.data[self.columns[col]]

    # ------------------- Getting and Setting Items ----------------------

    # TODO!

    # ------------------- Ops ----------------------

    def explain(self):
        """
        Prints a string that describes the operations to compute each column.

        If this DataFrame is a value, prints the data.

        """
        if self.pandas_df is not None:
            print(self.pandas_df)
        else:
            for col in self.columns:
                code = self._col(col).code
                code = code.replace("\n", "\n\t")
                print("{}: {}".format(col, code))


    def _arithmetic_binop_impl(self, other, series_op, fill=np.nan):
        """
        Apply the binary operation between this DataFrame and other.

        Parameters
        ----------
        other : DataFrame, scalar, GrizzlySeries
            If other is a DataFrame, aligns on column name. If the binary
            operator is not supported on any column, raises a TypeError.
        series_op : func
            The binary operation to apply.
        compare : bool
            whether this is a comparison operation.

        Returns
        -------
        GrizzlyDataFrame

        """
        new_data = []
        if not isinstance(other, GrizzlyDataFrame):
            for data in self.data:
                new_data.append(series_op(data, other))
            return GrizzlyDataFrame(new_data,
                    columns=copy.deepcopy(self.columns),
                    _length=self.length,
                    _fastpath=True)

        new_cols = []
        for (col, left_slot, right_slot) in self.columns.zip(other.columns):
            new_cols.append(col)
            if left_slot is None or right_slot is None:
                # TODO(shoumik): Handle this case by making a lazy computation.
                assert self.length != GrizzlyDataFrame.UNKNOWN_LENGTH
                dtype = np.array([fill]).dtype
                vals = np.empty(self.length, dtype=dtype)
                vals[:] = fill
                new_data.append(GrizzlySeries(vals))
            else:
                new_data.append(series_op(self.data[left_slot], other.data[right_slot]))
        return GrizzlyDataFrame(new_data,
                columns=ColumnIndex(new_cols),
                _length=self.length,
                _fastpath=True)

    def add(self, other):
        return self._arithmetic_binop_impl(other, GrizzlySeries.add)

    def sub(self, other):
        return self._arithmetic_binop_impl(other, GrizzlySeries.sub)

    def mod(self, other):
        return self._arithmetic_binop_impl(other, GrizzlySeries.mod)

    def mul(self, other):
        return self._arithmetic_binop_impl(other, GrizzlySeries.mul)

    def truediv(self, other):
        return self._arithmetic_binop_impl(other, GrizzlySeries.truediv)

    def divide(self, other):
        return self.truediv(other)

    def div(self, other):
        return self.truediv(other)

    def eq(self, other):
        return self._arithmetic_binop_impl(other, GrizzlySeries.eq, fill=False)

    def ne(self, other):
        # Fill with True on this one.
        return self._arithmetic_binop_impl(other, GrizzlySeries.ne, fill=True)

    def ge(self, other):
        return self._arithmetic_binop_impl(other, GrizzlySeries.ge, fill=False)

    def gt(self, other):
        return self._arithmetic_binop_impl(other, GrizzlySeries.gt, fill=False)

    def le(self, other):
        return self._arithmetic_binop_impl(other, GrizzlySeries.le, fill=False)

    def lt(self, other):
        return self._arithmetic_binop_impl(other, GrizzlySeries.lt, fill=False)

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
        if self.pandas_df is not None:
            return str(self.pandas_df)
        else:
            return repr(self)

    def __repr__(self):
        if self.pandas_df is not None:
            return repr(self.pandas_df)
        else:
            return "GrizzlyDataFrame(lazy, {})".format([name for name in self.columns.columns])

