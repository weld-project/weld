import numpy as np
import pandas as pd

import pandasImplNVL
from lazyOp import LazyOpResult
from nvlobject import *


class DataFrameNVL:
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

    def __getitem__(self, key):
         """Summary

        Args:
            key (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            Exception: Description
        """
        if type(key) == str:  # Single-key get
            # First check if key corresponds to an un-materialized column
            if key in self.unmaterialized_cols:
                return self.unmaterialized_cols[key]
            raw_column = self.df[key].values
            dtype = str(raw_column.dtype)
            # If column type is "object", then cast as "vec[char]" in NVL
            if raw_column.dtype == object:
                raw_column = np.array(self.df[key], dtype=str)
                nvl_type = NvlVec(NvlChar())
            else:
                nvl_type = pandasImplNVL.numpy_to_nvl_type_mapping[dtype]
            if self.predicates is None:
                return SeriesNVL(raw_column, nvl_type, self, key)
            return SeriesNVL(
                pandasImplNVL.filter(
                    raw_column,
                    self.predicates.expr,
                    nvl_type
                ),
                nvl_type,
                self,
                key
            )
        elif type(key) == list:  # For multi-key get, return type is a dataframe
            return DataFrameNVL(self.df[key], self.predicates)
        elif isinstance(key, SeriesNVL):  # Can also apply predicate to a dataframe
            return DataFrameNVL(self.df, key)
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
            nvl_type = pandasImplNVL.numpy_to_nvl_type_mapping[dtype]
            self.unmaterialized_cols[key] = SeriesNVL(
                value,
                nvl_type,
                self,
                key
            )
        elif isinstance(value, SeriesNVL):
            self.unmaterialized_cols[key] = value
        elif isinstance(value, LazyOpResult):
            self.unmaterialized_cols[key] = SeriesNVL(
                value.expr,
                value.nvl_type,
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
                nvl_type = pandasImplNVL.numpy_to_nvl_type_mapping[str(self.df.values.dtype)]
                dim = self.df.values.ndim
                for i in xrange(dim-1):
                    nvl_type = NvlVec(nvl_type)
                return LazyOpResult(
                    pandasImplNVL.filter(
                        self.df.values,
                        self.predicates.expr,
                        nvl_type
                    ),
                    nvl_type,
                    1
                )
        raise Exception("Attr %s does not exist" % key)

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
        return GroupByNVL(
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


class GroupedDataFrameNVL(LazyOpResult):
    """Summary

    Attributes:
        column_names (TYPE): Description
        column_types (TYPE): Description
        dim (int): Description
        expr (TYPE): Description
        grouping_column_name (TYPE): Description
        grouping_column_type (TYPE): Description
        nvl_type (TYPE): Description
        ptr (TYPE): Description
    """

    def __init__(self, expr, grouping_column_name, column_names, grouping_column_type, column_types):
        """Summary

        Args:
            expr (TYPE): Description
            grouping_column_name (TYPE): Description
            column_names (TYPE): Description
            grouping_column_type (TYPE): Description
            column_types (TYPE): Description
        """
        self.expr = expr
        self.grouping_column_name = grouping_column_name
        self.column_names = column_names
        self.grouping_column_type = grouping_column_type
        self.column_types = column_types
        self.ptr = None

        self.dim = 1
        column_types = self.column_types[0]
        if len(self.column_types) > 1:
            column_types = NvlStruct(self.column_types)
        self.nvl_type = NvlStruct([self.grouping_column_type, column_types])

    def get_column(self, column_name, column_type, index):
        """Summary

        Args:
            column_name (TYPE): Description
            column_type (TYPE): Description
            index (TYPE): Description

        Returns:
            TYPE: Description
        """
        return LazyOpResult(
            pandasImplNVL.get_column(
                self.ptr,
                self.nvl_type,
                index
            ),
            column_type,
            1
        ).evaluate()

    def evaluate(self):
        """Summary

        Returns:
            TYPE: Description
        """
        self.ptr = LazyOpResult.evaluate(self, decode=False)
        df = pd.DataFrame(columns=[])
        df[self.grouping_column_name] = self.get_column(
            self.grouping_column_name,
            self.grouping_column_type,
            "0"
        )
        i = 0
        for column_name in self.column_names:
            if len(self.column_names) > 1:
                index = "1.%d" % i
            else:
                index = "1"
            df[column_name] = self.get_column(
                column_name,
                self.column_types[i],
                index
            )
            i += 1
        return DataFrameNVL(df)


class SeriesNVL(LazyOpResult):
    """Summary

    Attributes:
        column_name (TYPE): Description
        df (TYPE): Description
        dim (int): Description
        expr (TYPE): Description
        nvl_type (TYPE): Description
    """
    
    def __init__(self, expr, nvl_type, df=None, column_name=None):
        """Summary

        Args:
            expr (TYPE): Description
            nvl_type (TYPE): Description
            df (None, optional): Description
            column_name (None, optional): Description
        """
        self.expr = expr
        self.nvl_type = nvl_type
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
        if key == 'str' and self.nvl_type == NvlVec(NvlChar()):
            return StringSeriesNVL(
                self.expr,
                self.nvl_type,
                self.df,
                self.column_name
            )
        raise Exception("Attr %s does not exist" % key)

    def unique(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return LazyOpResult(
            pandasImplNVL.unique(
                self.expr,
                self.nvl_type
            ),
            self.nvl_type,
            self.dim
        )

    def prod(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return LazyOpResult(
            pandasImplNVL.aggr(
                self.expr,
                "*",
                1,
                self.nvl_type
            ),
            self.nvl_type,
            0
        )

    def sum(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return LazyOpResult(
            pandasImplNVL.aggr(
                self.expr,
                "+",
                0,
                self.nvl_type
            ),
            self.nvl_type,
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
            pandasImplNVL.count(
                self.expr,
                self.nvl_type
            ),
            NvlInt(),
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
        if isinstance(predicates, SeriesNVL):
            predicates = predicates.expr
        return SeriesNVL(
            pandasImplNVL.mask(
                self.expr,
                predicates,
                new_value,
                self.nvl_type
            ),
            self.nvl_type,
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
        if isinstance(other, SeriesNVL):
            other = other.expr
        return SeriesNVL(
            pandasImplNVL.element_wise_op(
                self.expr,
                other,
                "+",
                self.nvl_type
            ),
            self.nvl_type,
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
        if isinstance(other, SeriesNVL):
            other = other.expr
        return SeriesNVL(
            pandasImplNVL.element_wise_op(
                self.expr,
                other,
                "-",
                self.nvl_type
            ),
            self.nvl_type,
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
        if isinstance(other, SeriesNVL):
            other = other.expr
        return SeriesNVL(
            pandasImplNVL.element_wise_op(
                self.expr,
                other,
                "*",
                self.nvl_type
            ),
            self.nvl_type,
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
        if isinstance(other, SeriesNVL):
            other = other.expr
        return SeriesNVL(
            pandasImplNVL.element_wise_op(
                self.expr,
                other,
                "/",
                self.nvl_type
            ),
            self.nvl_type,
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
        if isinstance(other, SeriesNVL):
            other = other.expr
        return SeriesNVL(
            pandasImplNVL.element_wise_op(
                self.expr,
                other,
                "%",
                self.nvl_type
            ),
            self.nvl_type,
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
        return SeriesNVL(
            pandasImplNVL.compare(
                self.expr,
                other,
                "==",
                self.nvl_type
            ),
            NvlBit(),
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
        return SeriesNVL(
            pandasImplNVL.compare(
                self.expr,
                other,
                "!=",
                self.nvl_type
            ),
            NvlBit(),
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
        return SeriesNVL(
            pandasImplNVL.compare(
                self.expr,
                other,
                ">",
                self.nvl_type
            ),
            NvlBit(),
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
        return SeriesNVL(
            pandasImplNVL.compare(
                self.expr,
                other,
                ">=",
                self.nvl_type
            ),
            NvlBit(),
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
        return SeriesNVL(
            pandasImplNVL.compare(
                self.expr,
                other,
                "<",
                self.nvl_type
            ),
            NvlBit(),
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
        return SeriesNVL(
            pandasImplNVL.compare(
                self.expr,
                other,
                "<=",
                self.nvl_type
            ),
            NvlBit(),
            self.df,
            self.column_name
        )


class StringSeriesNVL:
    """Summary

    Attributes:
        column_name (TYPE): Description
        df (TYPE): Description
        dim (int): Description
        expr (TYPE): Description
        nvl_type (TYPE): Description
    """
    def __init__(self, expr, nvl_type, df=None, column_name=None):
        """Summary

        Args:
            expr (TYPE): Description
            nvl_type (TYPE): Description
            df (None, optional): Description
            column_name (None, optional): Description
        """
        self.expr = expr
        self.nvl_type = nvl_type
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
        return SeriesNVL(
            pandasImplNVL.slice(
                self.expr,
                start,
                size,
                self.nvl_type
            ),
            self.nvl_type,
            self.df,
            self.column_name
        )


class GroupByNVL:
    """Summary

    Attributes:
        column_names (TYPE): Description
        column_types (list): Description
        columns (list): Description
        grouping_column (TYPE): Description
        grouping_column_name (TYPE): Description
        grouping_column_type (TYPE): Description
    """
    
    def __init__(self, df, grouping_column_name):
        """Summary

        Args:
            df (TYPE): Description
            grouping_column_name (TYPE): Description
        """
        self.grouping_column = None
        self.grouping_column_type = None
        column = df[grouping_column_name]
        if isinstance(column, LazyOpResult):
            self.grouping_column_type = column.nvl_type
            self.grouping_column = column.expr
        elif isinstance(column, np.ndarray):
            self.grouping_column_type = numpyImpl.numpy_to_nvl_type_mapping[str(column.dtype)]
            self.grouping_column = column

        self.column_names = df._get_column_names()
        self.column_names.remove(grouping_column_name)
        self.grouping_column_name = grouping_column_name

        self.columns = []
        self.column_types = []
        for column_name in self.column_names:
            column = df[column_name]
            column_type = None
            if isinstance(column, LazyOpResult):
                column_type = column.nvl_type
                column = column.expr
            elif isinstance(column, np.ndarray):
                column_type = numpyImpl.numpy_to_nvl_type_mapping[str(column.dtype)]
            self.columns.append(column)
            self.column_types.append(column_type)

    def sum(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return GroupedDataFrameNVL(
            pandasImplNVL.groupby_sum(
                self.columns,
                self.column_types,
                self.grouping_column
            ),
            self.grouping_column_name,
            self.column_names,
            self.grouping_column_type,
            self.column_types
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

