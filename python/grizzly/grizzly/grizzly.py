import numpy as np
import pandas as pd

import grizzly_impl
from lazy_op import LazyOpResult, to_weld_type
from weld.weldobject import *

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

class WeldLocIndexer:
    """
    Label location based indexer for selection by label for Series objects.

    Attributes:
        grizzly_obj (TYPE): The Series being indexed into.
    """
    def __init__(self, grizzly_obj):
        # If index_type field of grizzly_obj is None
        # then we assume normal 0 - 1 indexing
        self.grizzly_obj = grizzly_obj

    def __getitem__(self, key):
        if isinstance(self.grizzly_obj, SeriesWeld):
            series = self.grizzly_obj
            if isinstance(key, SeriesWeld):
                if series.index_type is not None:
                    index_expr = grizzly_impl.get_field(series.expr, 0)
                    column_expr = grizzly_impl.get_field(series.expr, 1)
                    zip_expr = grizzly_impl.zip_columns([index_expr, column_expr])
                    predicate_expr = grizzly_impl.isin(index_expr, key.expr, series.index_type)
                    filtered_expr = grizzly_impl.filter(
                        zip_expr,
                        predicate_expr
                    )
                    unzip_expr = grizzly_impl.unzip_columns(
                        filtered_expr,
                        [series.index_type, series.weld_type]
                    )
                    return SeriesWeld(
                        unzip_expr,
                        series.weld_type,
                        series.df,
                        series.column_name,
                        series.index_type,
                        series.index_name
                    )
            # TODO : Need to implement for non-pivot tables
        raise Exception("Cannot invoke getitem on non SeriesWeld object")

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

def merge(df1, df2):
    if isinstance(df1, DataFrameWeld):
        # TODO need to passin filtering by predicates as well here
        tys = []
        for col_name, raw_column in df1.raw_columns.items():
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

        df1 = DataFrameWeldExpr(
            grizzly_impl.zip_columns(
                df1.raw_columns.values()
            ),
            df1.raw_columns.keys(),
            weld_type
        )
    if isinstance(df2, DataFrameWeld):
        tys = []
        for col_name, raw_column in df2.raw_columns.items():
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
        df2 = DataFrameWeldExpr(
            grizzly_impl.zip_columns(
                df2.raw_columns.values()
            ),
            df2.raw_columns.keys(),
            weld_type
        )
    return df1.merge(df2)

def group_eval(objs, passes=None):
    LazyOpResults = []
    for ob in objs:
        if isinstance(ob, SeriesWeld):
            if ob.index_type is not None:
                weld_type = WeldStruct([WeldVec(ob.index_type), WeldVec(ob.weld_type)])
                LazyOpResults.append(LazyOpResult(ob.expr, weld_type, 0))
        else:
            LazyOpResults.append(LazyOpResult(ob.expr, ob.weld_type, 0))
    
    results = group(LazyOpResults).evaluate((True, -1), passes=passes)
    pd_results = []
    for i, result in enumerate(results):
        ob = objs[i]
        if isinstance(ob, SeriesWeld):
            if ob.index_type is not None:
                index, column = result
                series = pd.Series(column, index)
                series.index.rename(ob.index_name, True)
                pd_results.append(series)
            else:
                pd_results.append(series)
        if isinstance(ob, DataFrameWeldExpr):
            if ob.is_pivot:
                index, pivot, columns = result
                df_dict = {}
                for i, column_name in enumerate(columns):
                    df_dict[column_name] = pivot[i]
                pd_results.append(pd.DataFrame(df_dict, index=index))
            else:
                columns = result
                df_dict = {}
                for i, column_name in enumerate(ob.column_names):
                    df_dict[column_name] = columns[i]
                pd_results.append(pd.DataFrame(df_dict))
    return pd_results

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

        result = group(exprs).evaluate(verbose=True, passes=passes)
        df = pd.DataFrame(columns=[])
        all_columns = self.column_names + self.grouping_column_name
        for i, column_name in enumerate(all_columns):
            df[column_name] = result[i]
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

    def __init__(self, expr, weld_type, df=None, column_name=None, index_type=None, index_name=None):
        """Summary

        TODO: Implement an actual Index Object like how Pandas does
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
        self.index_type = index_type
        self.index_name = index_name

    def __getitem__(self, key):
        """Summary

        Args:
            predicates (TYPE): Description
            new_value (TYPE): Description

        Returns:
            TYPE: Description
        """
        if isinstance(key, slice):
            start = key.start
            # TODO : We currently do nothing with step
            step = key.step
            stop = key.stop
            if self.index_type is not None:
                index_expr = grizzly_impl.get_field(self.expr, 0)
                column_expr = grizzly_impl.get_field(self.expr, 1)
                zip_expr = grizzly_impl.zip_columns([index_expr, column_expr])
                sliced_expr = grizzly_impl.slice_vec(zip_expr, start, stop)
                unzip_expr = grizzly_impl.unzip_columns(
                    sliced_expr,
                    [self.index_type, self.weld_type]
                )
                return SeriesWeld(
                    unzip_expr,
                    self.weld_type,
                    self.df,
                    self.column_name,
                    self.index_type,
                    self.index_name
                )
            else:
                return SeriesWeld(
                    grizzly_impl.slice_vec(
                        self.expr,
                        start,
                        stop
                    )
                )
        else:
            # By default we return as if the key were predicates to filter by
            return self.filter(key)

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

    @property
    def loc(self):
        return WeldLocIndexer(
            self
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
        if key == 'str' and self.weld_type == WeldVec(WeldChar()):
            return StringSeriesWeld(
                self.expr,
                self.weld_type,
                self.df,
                self.column_name
            )
        raise AttributeError("Attr %s does not exist" % key)

    @property
    def index(self):
        if self.index_type is not None:
            return SeriesWeld(
                grizzly_impl.get_field(
                    self.expr,
                    0
                ),
                self.index_type,
                self.df,
                self.index_name
            )
        # TODO : Make all series have a series attribute
        raise Exception("No index present")

    def evaluate(self, verbose=False, passes=None):
        if self.index_type is not None:
            index, column = LazyOpResult(
                self.expr,
                WeldStruct([WeldVec(self.index_type), WeldVec(self.weld_type)]),
                0
            ).evaluate(verbose=verbose, passes=passes)
            series = pd.Series(column, index)
            series.index.rename(self.index_name, True)
            return series
        else:
            column = LazyOpResult.evaluate(self, verbose=verbose, passes=passes)
            return pd.Series(column)

    def sort_values(self, ascending=False):
        """ Sorts the values of this series

        """
        if self.index_type is not None:
            index_expr = grizzly_impl.get_field(self.expr, 0)
            column_expr = grizzly_impl.get_field(self.expr, 1)
            zip_expr = grizzly_impl.zip_columns([index_expr, column_expr])
            result_expr = grizzly_impl.sort(zip_expr, 1, self.weld_type, ascending)
            unzip_expr = grizzly_impl.unzip_columns(
                result_expr,
                [self.index_type, self.weld_type]
            )
            return SeriesWeld(
                unzip_expr,
                self.weld_type,
                self.df,
                self.column_name,
                self.index_type,
                self.index_name
            )
        else:
            result_expr = grizzly_impl.sort(self.expr)
            # TODO need to finish this

    def unique(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return SeriesWeld(
            grizzly_impl.unique(
                self.expr,
                self.weld_type
            ),
            self.weld_type,
            self.df,
            self.column_name
        )

    def lower(self):
        """Summary

        Returns:
            TYPE: Description
        """
        # TODO : Bug in nested map operating on strings
        # TODO : Check that self.weld_type is a string type
        vectype = self.weld_type
        if isinstance(vectype, WeldVec):
            elem_type = vectype.elemType
            if isinstance(elem_type, WeldChar):
                        return SeriesWeld(
                            grizzly_impl.to_lower(
                                self.expr,
                                elem_type
                            ),
                            self.weld_type,
                            self.df,
                            self.column_name
                        )
        raise Exception("Cannot call to_lower on non string type")

    def contains(self, string):
        """Summary

        Returns:
        TYPE: Description
        """
        # Check that self.weld_type is a string type
        vectype = self.weld_type
        if isinstance(vectype, WeldVec):
            elem_type = vectype.elemType
            if isinstance(elem_type, WeldChar):
                return SeriesWeld(
                    grizzly_impl.contains(
                        self.expr,
                        elem_type,
                        string
                    ),
                    WeldBit(),
                    self.df,
                    self.column_name
                )
        raise Exception("Cannot call to_lower on non string type")

    def isin(self, ls):
        if isinstance(ls, SeriesWeld):
            if self.weld_type == ls.weld_type:
                return SeriesWeld(
                    grizzly_impl.isin(self.expr,
                                      ls.expr,
                                      self.weld_type),
                    WeldBit(),
                    self.df,
                    self.column_name
                )
        raise Exception("Cannot call isin on different typed list")

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

    def __sub__(self, other):
        # TODO subtractionw without index variables
        if self.index_type is not None:
            index = grizzly_impl.get_field(self.expr, 0)
            expr1 = grizzly_impl.get_field(self.expr, 1)
        else:
            expr1 = self.expr
        if other.index_type is not None:
            index2 = grizzly_impl.get_field(other.expr, 0)
            expr2 = grizzly_impl.get_field(other.expr, 1)
        else:
            expr2 = other.expr
        index_expr = LazyOpResult(index, self.index_type, 0)
        sub_expr = SeriesWeld(
            grizzly_impl.element_wise_op(
                expr1,
                expr2,
                "-",
                self.weld_type
            ),
            self.weld_type,
            self.df,
            self.column_name
        )

        index_sub_expr = group([index_expr, sub_expr])
        return SeriesWeld(
            index_sub_expr.expr,
            self.weld_type,
            self.df,
            self.column_name,
            self.index_type,
            self.index_name
        )
        # We also need to ensure that both indexes of the subtracted
        # columns are compatible

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

    def per_element_and(self, other):
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
                "&&",
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
        if self.index_type is not None:
            expr = grizzly_impl.get_field(self.expr, 1)
        else:
            expr = self.expr
        return SeriesWeld(
            grizzly_impl.compare(
                expr,
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
        group_expr = group([index_expr, column_expr])
        return SeriesWeld(
            group_expr.expr,
            WeldDouble(),
            index_type=self.grouping_column_types[0],
            index_name=self.grouping_column_names[0]
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
