import numpy as np
import pandas as pd

import grizzly_impl
from lazy_op import LazyOpResult, to_weld_type
from weld.weldobject import *
from utils import *

from groupbyweld import GroupByWeld
from seriesweld import SeriesWeld

class DataFrameWeld:
    """Summary

    Attributes:
        df (TYPE): Description
        predicates (TYPE): Description
        unmaterialized_cols (TYPE): Description
        expr (TYPE): Description
    """

    def __init__(self, df, predicates=None, expr=None):
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
            if self.predicates is not None:
                return DataFrameWeld(self.df, key.per_element_and(self.predicates))
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

    @property
    def values(self):
        if self.predicates is None:
            return self.df.values
        else:
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

    def filter(self, predicates):
        """Summary

        Args:
            grouping_column_name (TYPE): Description

        Returns:
            TYPE: Description
        """
        tys = []
        for col_name, raw_column in self.raw_columns.items():
            dtype = str(raw_column.dtype)
            if dtype == 'object' or dtype == '|S64':
                weld_type = WeldVec(WeldChar())
            else:
                weld_type = grizzly_impl.numpy_to_weld_type_mapping[dtype]
            tys.append(weld_type)

        if len(tys) == 1:
            weld_type = tys[0]
        else:
            weld_type = WeldStruct(tys)

        if isinstance(predicates, SeriesWeld):
            predicates = predicates.expr

        return DataFrameWeldExpr(
            grizzly_impl.filter(
                grizzly_impl.zip_columns(
                    self.raw_columns.values(),
                ),
                predicates
            ),
            self.raw_columns.keys(),
            weld_type
        )

    def pivot_table(self, values, index, columns, aggfunc='sum'):
        tys = []
        for col_name, raw_column in self.raw_columns.items():
            dtype = str(raw_column.dtype)
            if dtype == 'object' or dtype == '|S64':
                weld_type = WeldVec(WeldChar())
            else:
                weld_type = grizzly_impl.numpy_to_weld_type_mapping[dtype]
            tys.append(weld_type)

        if len(tys) == 1:
            weld_type = tys[0]
        else:
            weld_type = WeldStruct(tys)

        return DataFrameWeldExpr(
            grizzly_impl.zip_columns(
                self.raw_columns.values(),
            ),
            self.raw_columns.keys(),
            weld_type
        ).pivot_table(values, index, columns, aggfunc)

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

class DataFrameWeldExpr:
    # TODO We need to merge this with the original DataFrameWeld class
    def __init__(self, expr, column_names, weld_type, is_pivot=False):
        if isinstance(weld_type, WeldStruct):
           self.column_types = weld_type.field_types
           self.weld_type = weld_type
        else:
            raise Exception("DataFrameWeldExpr can only except struct types")

        self.expr = expr
        self.column_names = column_names
        self.colindex_map = {name:i for i, name in enumerate(column_names)}
        self.is_pivot = is_pivot

    def __setitem__(self, key, item):
        if self.is_pivot:
            # Note that if this is a pivot table,
            # We have to modify the structure of the pivot table
            # that is append item to nested vector of vectors
            # and update the col_vec field
            # Also setitem appends a new item to the pivot table
            # Modifying an existing item is not implemented yet
            # TODO if pivot table check that the column being added
            # is same type
            if isinstance(item, SeriesWeld):
                if item.index_type is not None:
                    item_expr = grizzly_impl.get_field(item.expr, 1)
                else:
                    item_expr = item.expr
                self.expr = grizzly_impl.set_pivot_column(
                    self.expr,
                    key,
                    item_expr,
                    self.column_types[1].elemType,
                    self.column_types[2].elemType
                )
        else:
            raise Exception("Setitem not implemented for non-pivot table")

    def __getitem__(self, item):
        if self.is_pivot:
            # Note that if this is a pivot table,
            # then we can't check if the item is in the column
            # since we have not materialized the unique column values
            # Note if
            return SeriesWeld(
                grizzly_impl.get_pivot_column(
                    self.expr,
                    item,
                    self.column_types[2].elemType
                ),
                self.column_types[1].elemType.elemType,
                df=None,
                column_name=None,
                index_type=self.column_types[0].elemType,
                index_name=self.column_names[0]
            )

    @property
    def loc(self):
        return DataFrameWeldLoc(self)

    def merge(self, df2):
        if not isinstance(df2, DataFrameWeldExpr):
            raise Exception("df2 must be of type DataFrameWeldExpr")

        keys_d1 = set(self.colindex_map.keys())
        keys_d2 = set(df2.colindex_map.keys())
        join_keys = keys_d1 & keys_d2
        key_index_d1 = [self.colindex_map[key] for key in join_keys]
        key_index_d2 = [df2.colindex_map[key] for key in join_keys]
        rest_keys_d1 = keys_d1.difference(join_keys)
        rest_keys_d2 = keys_d2.difference(join_keys)
        rest_index_d1 = [self.colindex_map[key] for key in rest_keys_d1]
        rest_index_d2 = [df2.colindex_map[key] for key in rest_keys_d2]

        new_column_names = list(join_keys) + list(rest_keys_d1) + list(rest_keys_d2)

        # We add the key column first, followed by those in self, and finally df2

        key_index_types = []
        for i in key_index_d1:
            key_index_types.append(self.column_types[i])

        if len(key_index_types) > 1:
            join_keys_type = WeldStruct(key_index_types)
        else:
            join_keys_type = key_index_types[0]

        rest_types_d1 = []
        for i in rest_index_d1:
            rest_types_d1.append(self.column_types[i])

        rest_types_d2 = []
        for i in rest_index_d2:
            rest_types_d2.append(df2.column_types[i])

        new_types = key_index_types + rest_types_d1 + rest_types_d2
        return DataFrameWeldExpr(
            grizzly_impl.join(
                self.expr,
                df2.expr,
                key_index_d1,
                key_index_d2,
                join_keys_type,
                rest_index_d1,
                WeldStruct(rest_types_d1),
                rest_index_d2,
                WeldStruct(rest_types_d2)
            ),
            new_column_names,
            WeldStruct(new_types)
        )

    def pivot_table(self, values, index, columns, aggfunc='sum'):
        value_index = self.colindex_map[values]
        index_index = self.colindex_map[index]
        columns_index = self.colindex_map[columns]

        ind_ty = to_weld_type(self.column_types[index_index], 1)
        piv_ty = to_weld_type(WeldDouble(), 2)
        col_ty = to_weld_type(self.column_types[columns_index], 1)
        return DataFrameWeldExpr(
            grizzly_impl.pivot_table(
                self.expr,
                value_index,
                self.column_types[value_index],
                index_index,
                self.column_types[index_index],
                columns_index,
                self.column_types[columns_index],
                aggfunc
            ),
            [index, columns, values],
            WeldStruct([ind_ty, piv_ty, col_ty]),
            is_pivot=True
        )

    def sum(self, axis):
        if axis == 1:
            if self.is_pivot:
                if isinstance(self.column_types[1], WeldVec):
                    elem_type = self.column_types[1].elemType
                    value_type = elem_type.elemType
                    return SeriesWeld(
                        grizzly_impl.pivot_sum(
                            self.expr,
                            value_type
                        ),
                        value_type
                    )
            else:
                raise Exception("Sum for non-pivot table data frames not supported")
        elif axis == 0:
            raise Exception("Sum not implemented yet for axis = 0")

    def div(self, series, axis):
        if axis == 0:
            if self.is_pivot:
                if isinstance(self.column_types[1], WeldVec):
                    elem_type = self.column_types[1].elemType
                    value_type = elem_type.elemType
                    ind_ty = self.column_types[0]
                    piv_ty = to_weld_type(WeldDouble(), 2)
                    col_ty = self.column_types[2]
                    return DataFrameWeldExpr(
                        grizzly_impl.pivot_div(
                            self.expr,
                            series.expr,
                            elem_type,
                            value_type,
                        ),
                        self.column_names,
                        WeldStruct([ind_ty, piv_ty, col_ty]),
                        is_pivot=True
                    )
            else:
                raise Exception("Div for non-pivot table data frames not supported")
        elif axis == 1:
            raise Exception("Div not implemented yet for axis = 0")

    def sort_values(self, by):
        if self.is_pivot:
            return DataFrameWeldExpr(
                grizzly_impl.pivot_sort(
                    self.expr,
                    by,
                    self.column_types[0].elemType, # The index type
                    self.column_types[2].elemType, # The column name types (usually string)
                    self.column_types[1].elemType.elemType # The column value type
                ),
                self.column_names,
                self.weld_type,
                is_pivot=True
            )
        else:
            raise Expcetion("sort_values needs to be implemented for non pivot tables")

    def evaluate(self, verbose=True, passes=None):
        """Summary

        Returns:
            TYPE: Description
        """
        if self.is_pivot:
            index, pivot, columns = LazyOpResult(
                self.expr,
                self.weld_type,
                0
            ).evaluate(verbose=verbose, passes=passes)
            df_dict = {}
            for i, column_name in enumerate(columns):
                df_dict[column_name] = pivot[i]
            return DataFrameWeld(pd.DataFrame(df_dict, index=index))
        else:
            df = pd.DataFrame(columns=[])
            weldvec_type_list = []
            for type in self.column_types:
                weldvec_type_list.append(WeldVec(type))

            columns = LazyOpResult(
                grizzly_impl.unzip_columns(
                    self.expr,
                    self.column_types
                ),
                WeldStruct(weldvec_type_list),
                0
            ).evaluate(verbose=verbose, passes=passes)

            for i, column_name in enumerate(self.column_names):
                df[column_name] = columns[i]

            return DataFrameWeld(df)

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

class DataFrameWeldLoc:
    """
    Label location based indexer for selection by label for dataframe objects.

    Attributes:
        df (TYPE): The DataFrame  being indexed into.
    """
    def __init__(self, df):
        self.df = df

    def __getitem__(self, key):
        if isinstance(key, SeriesWeld):
            # We're going to assume that the first column in these dataframes
            # is an index column. This assumption does not hold throughout grizzly,
            # so we should fix that moving forward.
            index_expr = grizzly_impl.get_field(self.df.expr, 0)
            if self.df.is_pivot:
                index_type, pivot_type, column_type = self.df.column_types
                index_elem_type = index_type.elemType
                index_expr_predicate = grizzly_impl.isin(index_expr, key.expr, index_elem_type)
                return DataFrameWeldExpr(
                    grizzly_impl.pivot_filter(
                        self.df.expr,
                        index_expr_predicate
                    ),
                    self.df.column_names,
                    self.df.weld_type,
                    is_pivot=True
                )
            # TODO : Need to implement for non-pivot tables
        raise Exception("Cannot invoke getitem on an object that is not SeriesWeld")
