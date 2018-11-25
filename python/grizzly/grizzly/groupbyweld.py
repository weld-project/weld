import numpy as np
import pandas as pd

import grizzly_impl
from lazy_op import LazyOpResult, to_weld_type
from weld.weldobject import *
import utils

import dataframeweld
from seriesweld import SeriesWeld

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
    
        self.df = df
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

    def __getitem__(self, item):
        item_index = self.column_names.index(item)
        return GroupByWeldSeries(
            item,
            self.columns[item_index],
            self.column_types[item_index],
            self.grouping_column_names,
            self.grouping_columns,
            self.grouping_column_types
        )

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

    def size(self):
        """Returns the sizes of the groups as series.

        Returns:
            TYPE: Description
        """
        if len(self.grouping_column_types) > 1:
            index_type = WeldStruct([self.grouping_column_types])
            # Figure out what to use for multi-key index name
            # index_name = ??
        else:
            index_type = self.grouping_column_types[0]
            index_name = self.grouping_column_names[0]
        return SeriesWeld(
            grizzly_impl.groupby_size(
                self.columns,
                self.column_types,
                self.grouping_columns,
                self.grouping_column_types
            ),
            WeldLong(),
            index_type=index_type,
            index_name=index_name
        )

    def count(self):
        """Summary

        Returns:
            TYPE: Description
        """
        pass

    def apply(self, func):
        return func(self)

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
        )

    def reset_index(self, inplace=True, drop=True):
        """ Flattens the grouped data structure.

        #TODO: The parameters here are meaningless

        Flattens the grouped data structure.
        What is returned is a DataFrameWeld object.
        """
        if len(self.column_types) == 1:
            vectype = self.column_types[0]
            if isinstance(vectype, WeldVec):
                elem_type = vectype.elemType
                if isinstance(elem_type, WeldStruct):
                    self.column_types = elem_type.field_types
                    value_type = WeldStruct(self.column_types)
                else:
                    self.column_types = elem_type
                    value_type = elem_type
                if len(self.grouping_column_types) == 1:
                    group_type = self.grouping_column_types[0]
                else:
                    group_type = WeldStruct(self.grouping_column_types)

                self.weld_type = WeldStruct([group_type, value_type])
                self.expr = grizzly_impl.flatten_group(
                    self.expr,
                    self.column_types,
                    self.grouping_column_types
                )

    def evaluate(self, verbose=True, passes=None):
        """Summary

        Returns:
            TYPE: Description
        """
        i = 0
        exprs = []
        for column_name in self.column_names:
            if len(self.column_names) > 1:
                index = "1.$%d" % i
            else:
                index = "1"
            expr = self.get_column(
                column_name,
                self.column_types[i],
                index,
                verbose=verbose
            )
            i += 1
            exprs.append(expr)

        i = 0
        for column_name in self.grouping_column_name:
            if len(self.grouping_column_name) > 1:
                index = "0.$%d" % i
            else:
                index = "0"
            expr = self.get_column(
                column_name,
                self.grouping_column_types[i],
                index,
                verbose=verbose
            )
            exprs.append(expr)
            i += 1

        result = utils.group(exprs).evaluate(verbose=verbose, passes=passes)
        df = pd.DataFrame(columns=[])
        all_columns = self.column_names + self.grouping_column_name
        for i, column_name in enumerate(all_columns):
            df[column_name] = result[i]
        return dataframeweld.DataFrameWeld(df)

class GroupByWeldSeries:
    """Summary

    Attributes:
        column_name
        column_type
        column,
        grouping_column
    """

    def __init__(self, name, column, column_type, grouping_column_names, grouping_columns, grouping_column_types):
        self.name = name
        self.column = column
        self.column_type = column_type
        self.grouping_column_names = grouping_column_names
        self.grouping_columns = grouping_columns
        self.grouping_column_types = grouping_column_types

    def std(self):
        """Standard deviation

        Note that is by default normalizd by n - 1
        # TODO, what does pandas do for multiple grouping columns?
        # Currently we are just going to use one grouping column
        """
        std_expr = grizzly_impl.groupby_std(
            [self.column],
            [self.column_type],
            self.grouping_columns,
            self.grouping_column_types
        )
        unzipped_columns = grizzly_impl.unzip_columns(
            std_expr,
            self.grouping_column_types + [WeldDouble()],
        )
        index_expr = LazyOpResult(
            grizzly_impl.get_field(unzipped_columns, 0),
            self.grouping_column_types[0],
            1
        )
        column_expr = LazyOpResult(
            grizzly_impl.get_field(unzipped_columns, 1),
            self.grouping_column_types[0],
            1
        )
        group_expr = utils.group([index_expr, column_expr])
        return SeriesWeld(
            group_expr.expr,
            WeldDouble(),
            index_type=self.grouping_column_types[0],
            index_name=self.grouping_column_names[0]
        )
