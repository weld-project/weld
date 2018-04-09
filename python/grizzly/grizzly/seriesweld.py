import pandas as pd

import grizzly_impl
from lazy_op import LazyOpResult, to_weld_type
from weld.weldobject import *

import utils

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

        index_sub_expr = utils.group([index_expr, sub_expr])
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
