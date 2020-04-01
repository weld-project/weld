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
import weld.grizzly.weld.agg as weldagg

from weld.lazy import PhysicalValue, WeldLazy, WeldNode, identity
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

    @property
    def weld_value(self):
        # TODO: Construct the Weld value. This is a 'MakeStruct' of each series in slot order.
        pass

    def is_value(self):
        return self.pandas_df is not None or\
                all([child.is_identity for child in self.children]) or hasattr(self, "_evaluating")

    def evaluate(self):
        pass

    def to_pandas(self, copy=False):
        self.evaluate()
        pass

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
        if self.length == UNKNOWN_LENGTH:
            raise GrizzlyError("function {} disallowed on DataFrame of unknown length: try calling 'evaluate()' first".format(
                func))

    def __getitem__(self, key):
        return self.data[self.columns[key]]

    """
    Disabling the below because we need to know the length of a Series before adding it to a DataFrame.
    def __setitem__(self, key, value):
        self.require_known_length(self, __setitem__)
        if not isinstance(value, GrizzlySeries):
            value = GrizzlySeries(value)
            if not isinstance(value, GrizzlySeries):
                raise TypeError("Unsupported Series in DataFrame: {}".format(value))

        self.columns.append(key)
        assert self.columns[key] == len(self.data)
        self.data.append(value)
    """

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
                assert self.length != UNKNOWN_LENGTH
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

