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

def test_sub():
    _test_binop(gr.GrizzlySeries.sub, pd.Series.sub, "sub")

def test_mul():
    _test_binop(gr.GrizzlySeries.mul, pd.Series.mul, "mul")

def test_div():
    _test_binop(gr.GrizzlySeries.div, pd.Series.div, "div")
