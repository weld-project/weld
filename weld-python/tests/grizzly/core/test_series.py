"""
Test basic Series functionality.

"""

import numpy as np
import pandas as pd
import pytest
import weld.grizzly as gr

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
    # Exhaustive type-to-type test.
    _test_binop(gr.GrizzlySeries.add, pd.Series.add, "add")

def test_div():
    # Exhaustive type-to-type test.
    _test_binop(gr.GrizzlySeries.truediv, pd.Series.truediv, "truediv")
    _test_binop(gr.GrizzlySeries.div, pd.Series.div, "div")

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

def test_scalar():
    types = ['int8', 'uint8', 'int16', 'uint16', 'int32',\
            'uint32', 'int64', 'uint64', 'float32', 'float64']
    for left in types:
        for right in types:
            a = gr.GrizzlySeries([1, 2, 3], dtype=left)
            b = 123
            result = (a + b).to_pandas()

            a = pd.Series([1, 2, 3], dtype=left)
            expect = a + b
            assert result.equals(expect), "{}, {} (op={})".format(left, right, "scalar")

def test_indexing():
    # We don't compare with Pandas in these tests because the output
    # doesn't always match (this is because we don't currently support indexes).
    x = gr.GrizzlySeries(list(range(100)), dtype='int64')
    assert x[0] == 0
    assert x[50] == 50
    assert np.array_equal(x[10:50].evaluate().values, np.arange(10, 50, dtype='int64'))
    assert np.array_equal(x[:50].evaluate().values, np.arange(50, dtype='int64'))
    assert np.array_equal(x[x > 50].evaluate().values, np.arange(51, 100, dtype='int64'))
    assert np.array_equal(x[x == 2].evaluate().values, np.array([2], dtype='int64'))
    assert np.array_equal(x[x < 0].evaluate().values, np.array([], dtype='int64'))

def test_name():
    # Test that names propagate after operations.
    x = gr.GrizzlySeries([1,2,3], name="testname")
    y = x + x
    assert y.evaluate().name == "testname"
    y = x.agg(['sum', 'count'])
    assert y.evaluate().name == "testname"
    y = x[:2]
    assert y.evaluate().name == "testname"
    y = x[x == 1]
    assert y.evaluate().name == "testname"


def test_unsupported_binop_error():
    # Test unsupported
    from weld.grizzly.core.error import GrizzlyError
    with pytest.raises(GrizzlyError):
        a = gr.GrizzlySeries([1,2,3])
        b = pd.Series([1,2,3])
        a.add(b)

    with pytest.raises(TypeError):
        a = gr.GrizzlySeries(["hello", "world"])
        b = gr.GrizzlySeries(["hello", "world"])
        a.divide(b)
