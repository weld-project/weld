"""
Tests for aggregation functions in Grizzly.

"""

import numpy as np
import pandas as pd
import pytest
import weld.grizzly as gr

def _compare_vs_pandas(aggs, data=None):
    """
    Compare the result of aggregations vs. Pandas.

    Returns the code used to compute the result if the result
    is a `GrizzlySeries`.

    """
    if data is None:
        data = list(range(-10, 25))

    pandas_result = pd.Series(data).agg(aggs)
    grizzly_result = gr.GrizzlySeries(data).agg(aggs)

    if isinstance(pandas_result, pd.Series):
        assert isinstance(grizzly_result, gr.GrizzlySeries)
        code = grizzly_result.code
        grizzly_result = grizzly_result.to_pandas()
        # Need to reset index since labels in Pandas becoem the aggregation name.
        # Grizzly doesn't support indices right now.
        assert pandas_result.reset_index(drop=True).equals(grizzly_result)
        return code
    else:
        assert isinstance(pandas_result, (int, float, np.float64, np.int64))
        assert isinstance(grizzly_result, (int, float, np.float64, np.int64))
        assert pandas_result == grizzly_result
        return None

    return grizzly_result

def test_count():
    _compare_vs_pandas('count')

def test_sum():
    _compare_vs_pandas('sum')

def test_prod():
    _compare_vs_pandas('prod')

def test_prod():
    _compare_vs_pandas('mean')

def test_min():
    _compare_vs_pandas('min')

def test_max():
    _compare_vs_pandas('max')

def test_std():
    _compare_vs_pandas('std')

def test_var():
    _compare_vs_pandas('var')

def test_recursive_dependencies():
    code = _compare_vs_pandas(['count', 'std'])
    # Each recursive dependency should appear exactly once.
    assert code.count("let std_result") == 1
    assert code.count("let var_result") == 1
    assert code.count("let mean_result") == 1
    assert code.count("let sum_result") == 1
    assert code.count("let count_result") == 1

def test_multiple_in_priority_order():
    _compare_vs_pandas(['count', 'sum', 'mean'])

def test_multiple_out_of_order():
    # Mean is computed last, since it depends on count and sum. It should
    # still appear first in the result.
    _compare_vs_pandas(['mean', 'count', 'sum'])

def test_duplicate():
    # The mean and its dependencies should only be computed once.
    code = _compare_vs_pandas(['mean', 'mean', 'mean', 'var'])
    # Each agg and dependency should only appear one time.
    assert code.count("let var_result") == 1
    assert code.count("let mean_result") == 1
    assert code.count("let sum_result") == 1
    assert code.count("let count_result") == 1
