"""
Test basic Series functionality.

"""

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

def test_arithmetic_expression():
    def eval_expression(cls):
        a = cls([1, 2, 3], dtype='int32')
        b = cls([4, 5, 6], dtype='int32')
        c = a + b * b - a
        d = (c + a) * (c + b)
        e = (d / a) - (d / b)
        return a + b + c * d - e

    expect = eval_expression(pd.Series)
    got = eval_expression(gr.GrizzlySeries)
    # Make sure we actually used Grizzly for the whole comptuation.
    assert isinstance(got, gr.GrizzlySeries)
    got = got.to_pandas()
    assert got.equals(expect)
