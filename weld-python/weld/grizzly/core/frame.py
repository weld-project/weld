"""
A Weld wrapper for pandas.DataFrame.


    List of things this needs to support:

    1. Access to each column as a Series.
    2. Access to each row as a vector-of-structs
    3. Alignment of operations along columns

    E.g, how do we implement element-wise ops over DataFrames?

    Each DataFrame has a Weld type of Struct(Vec, Vec, ...)

    Each operation will update one or more vectors.

    let col1 = ...;
    let col2 = ...;
    ...
    {col1, col2, ...}

    Binary ops:
    DataFrame + Series: match column name, if no name, align to first column
        # This is different from pandas, it seems
    DataFrame + DataFrame: match column name
    DataFrame + scalar: apply to each value.

    Name -> Slot (Name can be anything, slot is an int refering to the slot in the struct)

    merge indices
        names = Name -> (SlotLeft, SlotRight)
        names = sortbykey(names)

    for (name, i) in enumerate(names):
        new_data[i] = Op(data[SlotLeft], data[SlotRight])
        new_index[name] = i

    ^ indexing logic is implemented in Python, processing logic is implemented
    in Weld. Combines flexibility with speed.

    What about indexing along rows? Grizzly doesn't support this. Too hard to
    support arbitrary row indexing with good performance. Also hard to parallelize
    things when you allow random row-based indexing (Why? Because need
    to do a lookup to match indexes, so every operation becomes a join)

    - a: Int64Index[0, 1, 2, 4], 0, 1, 2, 3
    - b: Int64Index[0, 1, 3, 4], 0, 1, 2, 3
    Need to do a join on these indices...

    Row index alignment instead happens with explicit joins.
    1) Convert to DataFrame with an index as a column
    2) Apply filters to each DataFrame
    3) Join the dataframes on the index
    4) Perform task.

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

    Examples
    --------

    >>> df = GrizzlyDataFrame({'name': ['mike', 'sam', 'sally'], 'age': [20, 22, 56]})
    >>> df
        name  age
    0   mike   20
    1    sam   22
    2  sally   56

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
        >>> df = df.add(df)
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
        self.pandas_df = pd.DataFrame(data, columns=columns)
        self.length = len(self.pandas_df)
        for (i, col) in enumerate(self.pandas_df):
            grizzly_series = GrizzlySeries(self.pandas_df[col], name=self.pandas_df[col].name)
            if not isinstance(grizzly_series, GrizzlySeries):
                raise TypeError("Unsupported Series in DataFrame: {}".format(self.pandas_df[col]))
            self.data.append(grizzly_series)
            column_index.append(col)

        self.columns = ColumnIndex(column_index)

    def _require_known_length(self):
        if self.length == GrizzlyDataFrame.UNKNOWN_LENGTH:
            raise GrizzlyError("function {} disallowed on DataFrame of unknown length: try calling 'evaluate()' first".format(
                func))

    def _col(self, col):
        """
        Returns the column associated with the column name 'col'.

        """
        return self.data[self.columns[col]]

    # ------------------- Getting and Setting Items ----------------------

    def __getitem__(self, key):
        # TODO(shoumik): Fuller implementation of this.
        return self._col(key)

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


    def add(self, other):
        """
        Add this DataFrame with another one, aligning on columns.

        """
        new_data = []
        if not isinstance(other, GrizzlyDataFrame):
            for data in self.data:
                new_data.append(data + other)
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
                nans = np.empty(self.length)
                nans[:] = np.nan
                new_data.append(GrizzlySeries(nans))
            else:
                new_data.append(self.data[left_slot] + other.data[right_slot])
        return GrizzlyDataFrame(new_data,
                columns=ColumnIndex(new_cols),
                _length=self.length,
                _fastpath=True)

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

