"""
Test basic Series functionality.

"""

import numpy as np
import pandas as pd
import weld.grizzly.series as gr

types_ = ['int8', 'uint8', 'int16', 'uint16', 'int32',\
        'uint32', 'int64', 'uint64', 'float32', 'float64']

def _test_binop(grizzly_op, pandas_op, name):
    """
    Test binary operators, ensuring that their output/data type
    matches Pandas.

    """
    types = ['int8', 'uint8', 'int16', 'uint16', 'int32',\
            'uint32', 'int64', 'uint64', 'float32', 'float64']
    for left in types:
        for right in types:
            a = gr.GrizzlySeries([1, 2, 3], dtype=left)
            b = gr.GrizzlySeries([1, 2, 3], dtype=right)
            result = grizzly_op(a, b).to_pandas()

            a = pd.Series([1, 2, 3], dtype=left)
            b = pd.Series([1, 2, 3], dtype=right)

            expect = pandas_op(a, b)
            assert result.equals(expect), "{}, {} (op={})".format(left, right, name)

def test_add():
    _test_binop(gr.GrizzlySeries.add, pd.Series.add, "add")
    _test_binop(gr.GrizzlySeries.__add__, pd.Series.__add__, "__add__")

def test_sub():
    _test_binop(gr.GrizzlySeries.sub, pd.Series.sub, "sub")
    _test_binop(gr.GrizzlySeries.__sub__, pd.Series.__sub__, "__sub__")

def test_mul():
    _test_binop(gr.GrizzlySeries.mul, pd.Series.mul, "mul")
    _test_binop(gr.GrizzlySeries.__mul__, pd.Series.__mul__, "__mul__")

def test_truediv():
    _test_binop(gr.GrizzlySeries.truediv, pd.Series.truediv, "truediv")
    _test_binop(gr.GrizzlySeries.div, pd.Series.div, "div")
    _test_binop(gr.GrizzlySeries.divide, pd.Series.divide, "divide")
    _test_binop(gr.GrizzlySeries.__truediv__, pd.Series.__truediv__, "__truediv__")

def _compare_vs_pandas(func):
    """
    Helper to compare Pandas and Grizzly output. `func`
    is a generator that can yield one or more values to test
    for equality.
    """
    expect = func(pd.Series)
    got = func(gr.GrizzlySeries)
    for (expect, got) in zip(func(pd.Series), func(gr.GrizzlySeries)):
        # Make sure we actually used Grizzly for the whole comptuation.
        assert isinstance(got, gr.GrizzlySeries)
        got = got.to_pandas()
        assert got.equals(expect)

def test_arithmetic_expression():
    def eval_expression(cls):
        a = cls([1, 2, 3], dtype='int32')
        b = cls([4, 5, 6], dtype='int32')
        c = a + b * b - a
        d = (c + a) * (c + b)
        e = (d / a) - (d / b)
        yield a + b + c * d - e
    _compare_vs_pandas(eval_expression)

def test_compare_ops():
    def eval_expression(cls):
        a = cls([1, np.nan, 3, 4, 6])
        b = cls([1, np.nan, 2, 5, np.nan])
        yield a == b
        yield a > b
        yield a >= b
        yield a <= b
        yield a < b
    _compare_vs_pandas(eval_expression)

def test_float_nan():
    def eval_expression(cls):
        a = cls([1, 2, np.nan])
        b = cls([np.nan, 5, 6])
        c = a + b * b - a
        d = (c + a) * (c + b)
        e = (d / a) - (d / b)
        yield a + b + c * d - e
    _compare_vs_pandas(eval_expression)

def test_basic_fallback():
    # Tests basic unsupported functionality.
    # NOTE: This test will need to change as more features are added...
    def eval_expression(cls):
        a = cls([1, 2, 3])
        b = cls([-4, 5, -6])
        # Test 1: abs()
        c = a + b
        yield (c.abs() + a)
        # Test 2: agg()
        c = a + b
        yield cls(c.agg(np.sum)) # wrap with cls for type checking
        # Test 3: argmin()
        c = a + b
        yield cls(c.argmin())
        # Test 4: reindex()
        c = a + b
        res = c.reindex(index=[2, 0, 1])
        # Falls back to Pandas, since we don't support indices.
        assert isinstance(res, pd.Series)
    _compare_vs_pandas(eval_expression)
