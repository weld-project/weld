"""
Test basic DataFrame functionality.

"""

import pandas as pd
import pytest
import weld.grizzly as gr

def get_frames(cls, strings):
    """
    Returns two DataFrames for testing binary operators.

    The DataFrames have columns of overlapping/different names, types, etc.

    """
    df1 = pd.DataFrame({
        'name': ['Bob', 'Sally', 'Kunal', 'Deepak', 'James', 'Pratiksha'],
        'lastName': ['Kahn', 'Lopez', 'Smith', 'Narayanan', 'Thomas', 'Thaker'],
        'age': [20, 30, 35, 20, 50, 35],
        'score': [20.0, 30.0, 35.0, 50.0, 35.0, 25.0]
        })
    df2 = pd.DataFrame({
        'firstName': ['Bob', 'Sally', 'Kunal', 'Deepak', 'James', 'Pratiksha'],
        'lastName': ['Kahn', 'Lopez', 'smith', 'narayanan', 'Thomas', 'thaker'],
        'age': [25, 30, 45, 20, 60, 35],
        'scores': [20.0, 30.0, 35.0, 50.0, 35.0, 25.0]
        })
    if not strings:
        df1 = df1.drop(['name', 'lastName'], axis=1)
        df2 = df2.drop(['firstName', 'lastName'], axis=1)
    return (cls(df1), cls(df2))

def _test_binop(pd_op, gr_op, strings=True):
    """
    Test a binary operator.

    Binary operators align on column name. For columns that don't exist in both
    DataFrames, the column is filled with NaN (for non-comparison operations) and
    or False (for comparison operations).

    If the RHS is a Series, the Series should be added to all columns.

    """
    df1, df2 = get_frames(pd.DataFrame, strings)
    gdf1, gdf2 = get_frames(gr.GrizzlyDataFrame, strings)

    expect = pd_op(df1, df2)
    result = gr_op(gdf1, gdf2).to_pandas()
    assert expect.equals(result)

def test_evaluation():
    # Test to make sure that evaluating a DataFrame once caches the result/
    # doesn't cause another evaluation.
    df1 = gr.GrizzlyDataFrame({
        'age': [20, 30, 35, 20, 50, 35],
        'score': [20.0, 30.0, 35.0, 50.0, 35.0, 25.0]
        })
    df2 = gr.GrizzlyDataFrame({
        'age': [20, 30, 35, 20, 50, 35],
        'scores': [20.0, 30.0, 35.0, 50.0, 35.0, 25.0]
        })
    df3 = (df1 + df2) * df2 + df1 / df2
    assert not df3.is_value
    df3.evaluate()
    assert df3.is_value
    weld_value = df3.weld_value
    df3.evaluate()
    # The same weld_value should be returned.
    assert weld_value is df3.weld_value

def test_add():
    _test_binop(pd.DataFrame.add, gr.GrizzlyDataFrame.add, strings=False)

def test_sub():
    _test_binop(pd.DataFrame.sub, gr.GrizzlyDataFrame.sub, strings=False)

def test_mul():
    _test_binop(pd.DataFrame.mul, gr.GrizzlyDataFrame.mul, strings=False)

def test_div():
    _test_binop(pd.DataFrame.div, gr.GrizzlyDataFrame.div, strings=False)

def test_eq():
    _test_binop(pd.DataFrame.eq, gr.GrizzlyDataFrame.eq, strings=True)

def test_ne():
    _test_binop(pd.DataFrame.ne, gr.GrizzlyDataFrame.ne, strings=True)

def test_le():
    _test_binop(pd.DataFrame.le, gr.GrizzlyDataFrame.le, strings=False)

def test_lt():
    _test_binop(pd.DataFrame.lt, gr.GrizzlyDataFrame.lt, strings=False)

def test_ge():
    _test_binop(pd.DataFrame.ge, gr.GrizzlyDataFrame.ge, strings=False)

def test_gt():
    _test_binop(pd.DataFrame.gt, gr.GrizzlyDataFrame.gt, strings=False)
