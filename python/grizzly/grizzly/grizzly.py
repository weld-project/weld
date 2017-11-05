import numpy as np
import pandas as pd

import grizzly_impl
from lazy_op import LazyOpResult, to_weld_type
from weld.weldobject import *


def group(exprs):
    weld_type = [to_weld_type(expr.weld_type, expr.dim) for expr in exprs]
    exprs = [expr.expr for expr in exprs]
    weld_obj = WeldObject(grizzly_impl.encoder_, grizzly_impl.decoder_)
    weld_type = WeldStruct(weld_type)
    dim = 0

    expr_names = [expr.obj_id for expr in exprs]
    for expr in exprs:
        weld_obj.update(expr)
    weld_obj.weld_code = "{%s}" % ", ".join(expr_names)
    for expr in exprs:
        weld_obj.dependencies[expr.obj_id] = expr

    return LazyOpResult(weld_obj, weld_type, dim)


class DataFrameWeld:
    """Summary

    Attributes:
        df (TYPE): Description
        predicates (TYPE): Description
        unmaterialized_cols (TYPE): Description
    """

    def __init__(self, df, predicates=None):
        self.df = df
        self.unmaterialized_cols = dict()
        self.predicates = predicates
        self.raw_columns = dict()
        for key in self.df:
            raw_column = self.df[key].values
            if raw_column.dtype == object:
                raw_column = np.array(self.df[key], dtype=str)
            self.raw_columns[key] = raw_column

    def __getitem__(self, key):
        """Summary

        Args:
            key (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            Exception: Description
        """
        if isinstance(key, str):  # Single-key get
            # First check if key corresponds to an un-materialized column
            if key in self.unmaterialized_cols:
                return self.unmaterialized_cols[key]
            raw_column = self.df[key].values
            dtype = str(raw_column.dtype)
            # If column type is "object", then cast as "vec[char]" in Weld
            if dtype == 'object':
                raw_column = self.raw_columns[key]
                weld_type = WeldVec(WeldChar())
            else:
                weld_type = grizzly_impl.numpy_to_weld_type_mapping[dtype]
            if self.predicates is None:
                return SeriesWeld(raw_column, weld_type, self, key)
            return SeriesWeld(
                grizzly_impl.filter(
                    raw_column,
                    self.predicates.expr,
                    weld_type
                ),
                weld_type,
                self,
                key
            )
        elif isinstance(key, list):
            # For multi-key get, return type is a dataframe
            return DataFrameWeld(self.df[key], self.predicates)
        elif isinstance(key, SeriesWeld):
            # Can also apply predicate to a dataframe
            return DataFrameWeld(self.df, key)
        raise Exception("Invalid type in __getitem__")

    def __setitem__(self, key, value):
        """Summary

        Args:
            key (TYPE): Description
            value (TYPE): Description

        Returns:
            TYPE: Description
        """
        if isinstance(value, np.ndarray):
            dtype = str(value.dtype)
            weld_type = grizzly_impl.numpy_to_weld_type_mapping[dtype]
            self.unmaterialized_cols[key] = SeriesWeld(
                value,
                weld_type,
                self,
                key
            )
        elif isinstance(value, SeriesWeld):
            self.unmaterialized_cols[key] = value
        elif isinstance(value, LazyOpResult):
            self.unmaterialized_cols[key] = SeriesWeld(
                value.expr,
                value.weld_type,
                self,
                key
            )

    def __getattr__(self, key):
        """Summary

        Args:
            key (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            Exception: Description
        """
        if key == 'values':
            if self.predicates is None:
                return self.df.values
            if isinstance(self.df.values, np.ndarray):
                weld_type = grizzly_impl.numpy_to_weld_type_mapping[
                    str(self.df.values.dtype)]
                dim = self.df.values.ndim
                return LazyOpResult(
                    grizzly_impl.filter(
                        self.df.values,
                        self.predicates.expr,
                        weld_type
                    ),
                    weld_type,
                    dim
                )
        raise AttributeError("Attr %s does not exist" % key)

    def _get_column_names(self):
        """Summary

        Returns:
            TYPE: Description
        """
        column_names = set()
        for column in self.df:
            column_names.add(column)
        for column in self.unmaterialized_cols:
            column_names.add(column)
        return list(column_names)

    def groupby(self, grouping_column_name):
        """Summary

        Args:
            grouping_column_name (TYPE): Description

        Returns:
            TYPE: Description
        """
        return GroupByWeld(
            self,
            grouping_column_name
        )

    def to_pandas(self):
        """Summary

        Returns:
            TYPE: Description
        """
        # TODO: Do more work here (need to materialize all columns as needed)
        return self.df


class GroupedDataFrameWeld(LazyOpResult):
    """Summary

    Attributes:
        column_names (TYPE): Description
        column_types (TYPE): Description
        dim (int): Description
        expr (TYPE): Description
        grouping_column_name (TYPE): Description
        grouping_column_type (TYPE): Description
        weld_type (TYPE): Description
        ptr (TYPE): Description
    """

    def __init__(
            self,
            expr,
            grouping_column_names,
            column_names,
            grouping_column_types,
            column_types):
        """Summary

        Args:
            expr (TYPE): Description
            grouping_column_name (TYPE): Description
            column_names (TYPE): Description
            grouping_column_type (TYPE): Description
            column_types (TYPE): Description
        """
        self.expr = expr
        self.grouping_column_name = grouping_column_names
        self.column_names = column_names
        self.grouping_column_types = grouping_column_types
        self.column_types = column_types
        self.ptr = None

        self.dim = 1
        if isinstance(self.column_types, list):
            if len(self.column_types) == 1:
                column_types = self.column_types[0]
            else:
                column_types = WeldStruct(self.column_types)

        if len(self.grouping_column_types) == 1:
            grouping_column_types = self.grouping_column_types[0]
        else:
            grouping_column_types = WeldStruct(self.grouping_column_types)
        self.weld_type = WeldStruct([grouping_column_types, column_types])

    def slice(self, start, size):
        return GroupedDataFrameWeld(
            grizzly_impl.grouped_slice(
                self.expr,
                self.weld_type,
                start,
                size
            ),
            self.grouping_column_name,
            self.column_names,
            self.grouping_column_types,
            self.column_types
        )

    def get_column(self, column_name, column_type, index, verbose=True):
        """Summary

        Args:
            column_name (TYPE): Description
            column_type (TYPE): Description
            index (TYPE): Description

        Returns:
            TYPE: Description
        """
        return LazyOpResult(
            grizzly_impl.get_column(
                self.expr,
                self.weld_type,
                index
            ),
            column_type,
            1
        ).evaluate(verbose=verbose)

    def reset_index(self, inplace=True, drop=True):
        """ Flattens the grouped data structure.

        Flattens the grouped data structure.
        What is returned is a DataFrameWeld object.
        """
        if len(self.column_types) == 1:
            vectype = self.column_types[0]
            if isinstance(vectype, WeldVec):
                elem_type = vectype.elemType
                if isinstance(elem_type, WeldStruct):
                    self.column_types = elem_type.fieldTypes
                else:
                    self.column_types = elem_type
                group_type = WeldStruct(self.grouping_column_types)
                value_type = WeldStruct(self.column_types)
                self.weld_type = WeldStruct([group_type, value_type])
                self.expr = grizzly_impl.flatten_group(
                    self.expr,
                    self.column_types,
                    self.grouping_column_types
                )

    def evaluate(self, verbose=True):
        """Summary

        Returns:
            TYPE: Description
        """
        df = pd.DataFrame(columns=[])
        i = 0
        for column_name in self.column_names:
            if len(self.column_names) > 1:
                index = "1.$%d" % i
            else:
                index = "1"
            df[column_name] = self.get_column(
                column_name,
                self.column_types[i],
                index,
                verbose=verbose
            )
            i += 1

        i = 0
        for column_name in self.grouping_column_name:
            if len(self.column_names) > 1:
                index = "0.$%d" % i
            else:
                index = "0"
            df[column_name] = self.get_column(
                column_name,
                self.grouping_column_types[i],
                index,
                verbose=verbose
            )
            i += 1
        return DataFrameWeld(df)


class SeriesWeld(LazyOpResult):
    """Summary

    Attributes:
        column_name (TYPE): Description
        df (TYPE): Description
        dim (int): Description
        expr (TYPE): Description
        weld_type (TYPE): Description
    """

    def __init__(self, expr, weld_type, df=None, column_name=None):
        """Summary

        Args:
            expr (TYPE): Description
            weld_type (TYPE): Description
            df (None, optional): Description
            column_name (None, optional): Description
        """
        self.expr = expr
        self.weld_type = weld_type
        self.dim = 1
        self.df = df
        self.column_name = column_name

    def __setitem__(self, predicates, new_value):
        """Summary

        Args:
            predicates (TYPE): Description
            new_value (TYPE): Description

        Returns:
            TYPE: Description
        """
        if self.df is not None and self.column_name is not None:
            self.df[self.column_name] = self.mask(predicates, new_value)

    def __getattr__(self, key):
        """Summary

        Args:
            key (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            Exception: Description
        """
        if key == 'str' and self.weld_type == WeldVec(WeldChar()):
            return StringSeriesWeld(
                self.expr,
                self.weld_type,
                self.df,
                self.column_name
            )
        raise AttributeError("Attr %s does not exist" % key)

    def unique(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return LazyOpResult(
            grizzly_impl.unique(
                self.expr,
                self.weld_type
            ),
            self.weld_type,
            self.dim
        )

    def prod(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return LazyOpResult(
            grizzly_impl.aggr(
                self.expr,
                "*",
                1,
                self.weld_type
            ),
            self.weld_type,
            0
        )

    def sum(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return LazyOpResult(
            grizzly_impl.aggr(
                self.expr,
                "+",
                0,
                self.weld_type
            ),
            self.weld_type,
            0
        )

    def max(self):
        """Summary

        Returns:
            TYPE: Description
        """
        pass

    def min(self):
        """Summary

        Returns:
            TYPE: Description
        """
        pass

    def count(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return LazyOpResult(
            grizzly_impl.count(
                self.expr,
                self.weld_type
            ),
            WeldInt(),
            0
        )

    def mask(self, predicates, new_value):
        """Summary

        Args:
            predicates (TYPE): Description
            new_value (TYPE): Description

        Returns:
            TYPE: Description
        """
        if isinstance(predicates, SeriesWeld):
            predicates = predicates.expr
        return SeriesWeld(
            grizzly_impl.mask(
                self.expr,
                predicates,
                new_value,
                self.weld_type
            ),
            self.weld_type,
            self.df,
            self.column_name
        )

    def filter(self, predicates):
        if isinstance(predicates, SeriesWeld):
            predicates = predicates.expr
        return SeriesWeld(
            grizzly_impl.filter(
                self.expr,
                predicates,
                self.weld_type
            ),
            self.weld_type,
            self.df,
            self.column_name
        )

    def add(self, other):
        """Summary

        Args:
            other (TYPE): Description

        Returns:
            TYPE: Description
        """
        if isinstance(other, SeriesWeld):
            other = other.expr
        return SeriesWeld(
            grizzly_impl.element_wise_op(
                self.expr,
                other,
                "+",
                self.weld_type
            ),
            self.weld_type,
            self.df,
            self.column_name
        )

    def sub(self, other):
        """Summary

        Args:
            other (TYPE): Description

        Returns:
            TYPE: Description
        """
        if isinstance(other, SeriesWeld):
            other = other.expr
        return SeriesWeld(
            grizzly_impl.element_wise_op(
                self.expr,
                other,
                "-",
                self.weld_type
            ),
            self.weld_type,
            self.df,
            self.column_name
        )

    def mul(self, other):
        """Summary

        Args:
            other (TYPE): Description

        Returns:
            TYPE: Description
        """
        if isinstance(other, SeriesWeld):
            other = other.expr
        return SeriesWeld(
            grizzly_impl.element_wise_op(
                self.expr,
                other,
                "*",
                self.weld_type
            ),
            self.weld_type,
            self.df,
            self.column_name
        )

    def div(self, other):
        """Summary

        Args:
            other (TYPE): Description

        Returns:
            TYPE: Description
        """
        if isinstance(other, SeriesWeld):
            other = other.expr
        return SeriesWeld(
            grizzly_impl.element_wise_op(
                self.expr,
                other,
                "/",
                self.weld_type
            ),
            self.weld_type,
            self.df,
            self.column_name
        )

    def mod(self, other):
        """Summary

        Args:
            other (TYPE): Description

        Returns:
            TYPE: Description
        """
        if isinstance(other, SeriesWeld):
            other = other.expr
        return SeriesWeld(
            grizzly_impl.element_wise_op(
                self.expr,
                other,
                "%",
                self.weld_type
            ),
            self.weld_type,
            self.df,
            self.column_name
        )

    def __eq__(self, other):
        """Summary

        Args:
            other (TYPE): Description

        Returns:
            TYPE: Description
        """
        return SeriesWeld(
            grizzly_impl.compare(
                self.expr,
                other,
                "==",
                self.weld_type
            ),
            WeldBit(),
            self.df,
            self.column_name
        )

    def __ne__(self, other):
        """Summary

        Args:
            other (TYPE): Description

        Returns:
            TYPE: Description
        """
        return SeriesWeld(
            grizzly_impl.compare(
                self.expr,
                other,
                "!=",
                self.weld_type
            ),
            WeldBit(),
            self.df,
            self.column_name
        )

    def __gt__(self, other):
        """Summary

        Args:
            other (TYPE): Description

        Returns:
            TYPE: Description
        """
        return SeriesWeld(
            grizzly_impl.compare(
                self.expr,
                other,
                ">",
                self.weld_type
            ),
            WeldBit(),
            self.df,
            self.column_name
        )

    def __ge__(self, other):
        """Summary

        Args:
            other (TYPE): Description

        Returns:
            TYPE: Description
        """
        return SeriesWeld(
            grizzly_impl.compare(
                self.expr,
                other,
                ">=",
                self.weld_type
            ),
            WeldBit(),
            self.df,
            self.column_name
        )

    def __lt__(self, other):
        """Summary

        Args:
            other (TYPE): Description

        Returns:
            TYPE: Description
        """
        return SeriesWeld(
            grizzly_impl.compare(
                self.expr,
                other,
                "<",
                self.weld_type
            ),
            WeldBit(),
            self.df,
            self.column_name
        )

    def __le__(self, other):
        """Summary

        Args:
            other (TYPE): Description

        Returns:
            TYPE: Description
        """
        return SeriesWeld(
            grizzly_impl.compare(
                self.expr,
                other,
                "<=",
                self.weld_type
            ),
            WeldBit(),
            self.df,
            self.column_name
        )


class StringSeriesWeld:
    """Summary

    Attributes:
        column_name (TYPE): Description
        df (TYPE): Description
        dim (int): Description
        expr (TYPE): Description
        weld_type (TYPE): Description
    """

    def __init__(self, expr, weld_type, df=None, column_name=None):
        """Summary

        Args:
            expr (TYPE): Description
            weld_type (TYPE): Description
            df (None, optional): Description
            column_name (None, optional): Description
        """
        self.expr = expr
        self.weld_type = weld_type
        self.dim = 1
        self.df = df
        self.column_name = column_name

    def slice(self, start, size):
        """Summary

        Args:
            start (TYPE): Description
            size (TYPE): Description

        Returns:
            TYPE: Description
        """
        return SeriesWeld(
            grizzly_impl.slice(
                self.expr,
                start,
                size,
                self.weld_type
            ),
            self.weld_type,
            self.df,
            self.column_name
        )

class GroupByWeld:
    """Summary

    Attributes:
        column_names (TYPE): Description
        column_types (list): Description
        columns (list): Description
        grouping_column (TYPE): Description
        grouping_column_name (TYPE): Description
        grouping_column_type (TYPE): Description
    """

    def __init__(self, df, grouping_column_names):
        """Summary

        Args:
            df (TYPE): Description
            grouping_column_name (TYPE): Description
        """
        self.grouping_columns = []
        self.grouping_column_types = []

        if isinstance(grouping_column_names, str):
            grouping_column_names = [grouping_column_names]
        for column_name in grouping_column_names:
            column = df[column_name]
            if isinstance(column, LazyOpResult):
                self.grouping_column_types.append(column.weld_type)
                self.grouping_columns.append(column.expr)
            elif isinstance(column, np.ndarray):
                column_type = numpyImpl.numpy_to_weld_type_mapping[
                    str(column.dtype)]
                self.grouping_column_types.append(column_type)
                self.grouping_columns.append(column)

        self.grouping_column_names = grouping_column_names
        self.column_names = []
        for x in df._get_column_names():
            if x not in self.grouping_column_names:
                self.column_names.append(x)

        self.columns = []
        self.column_types = []
        for column_name in self.column_names:
            column = df[column_name]
            column_type = None
            if isinstance(column, LazyOpResult):
                column_type = column.weld_type
                column = column.expr
            elif isinstance(column, np.ndarray):
                column_type = numpyImpl.numpy_to_weld_type_mapping[
                    str(column.dtype)]

            self.columns.append(column)
            self.column_types.append(column_type)

    def sum(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return GroupedDataFrameWeld(
            grizzly_impl.groupby_sum(
                self.columns,
                self.column_types,
                self.grouping_columns,
                self.grouping_column_types
            ),
            self.grouping_column_names,
            self.column_names,
            self.grouping_column_types,
            self.column_types
        )

    def sort_values(self, by, ascending=True):
        """Summary

        Returns:
            TYPE: Description
        """
        if len(self.column_types) == 1:
            vec_type = [WeldVec(self.column_types[0])]
        else:
            vec_type = [WeldVec(WeldStruct(self.column_types))]

        if len(self.column_names) > 1:
            key_index = self.column_names.index(by)
        else :
            key_index = None

        return GroupedDataFrameWeld(
            grizzly_impl.groupby_sort(
                self.columns,
                self.column_types,
                self.grouping_columns,
                self.grouping_column_types,
                key_index,
                ascending
            ),
            self.grouping_column_names,
            self.column_names,
            self.grouping_column_types,
            vec_type
        )

    def mean(self):
        """Summary

        Returns:
            TYPE: Description
        """
        pass

    def count(self):
        """Summary

        Returns:
            TYPE: Description
        """
        pass

    def apply(self, func):
        return func(self)
