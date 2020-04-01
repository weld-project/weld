"""
Test string functionality.

The behavior is tested against Pandas unless noted otherwise.

"""

import numpy as np
import pandas as pd
import pytest
import weld.grizzly as gr

# To check whether the output is a string.
# TODO(shoumik): There should be a better way to do this, another reason
# to use ExtensionArray and a custom dtype for Weldified string arrays.
from weld.types import WeldVec, I8

def compare_vs_pandas(func, strings, *args, **kwargs):
    pandas_series = pd.Series(strings)
    grizzly_series = gr.GrizzlySeries(strings)

    pandas_result = getattr(pandas_series.str, func)(*args, **kwargs)
    grizzly_result = getattr(grizzly_series.str, func)(*args, **kwargs)
    if grizzly_result.output_type.elem_type != WeldVec(I8()):
        grizzly_result = grizzly_result.to_pandas()
    else:
        # Perform UTF-8 decoding.
        grizzly_result = grizzly_result.str.to_pandas()
    assert pandas_result.equals(grizzly_result)

# Strings to test capitalization functions.
capitals_strings = [
        "hello",  "HELLO", "LonGHelLO", "",
        "3.141592, it's pi!", "many words in this one"]

def test_lower():
    compare_vs_pandas('lower', capitals_strings)

def test_upper():
    compare_vs_pandas('upper', capitals_strings)

def test_capitalize():
    compare_vs_pandas('capitalize', capitals_strings)

def test_get():
    """
    Behavior of get is different in Grizzly -- it currently returns empty strings
    in cases where Pandas returns NaN. This will be changed in a later patch.

    """
    inp = ["hello", "world", "test", "me", '']
    expect = ['l', 'l', 't', '', '']
    grizzly_result = gr.GrizzlySeries(inp).str.get(3).str.to_pandas()
    pandas_result = pd.Series(expect)
    assert pandas_result.equals(grizzly_result)

    expect = ['o', 'd', 't', 'e', '']
    grizzly_result = gr.GrizzlySeries(inp).str.get(-1).str.to_pandas()
    pandas_result = pd.Series(expect)
    assert pandas_result.equals(grizzly_result)

    expect = ['', '', '', '', '']
    grizzly_result = gr.GrizzlySeries(inp).str.get(-50).str.to_pandas()
    pandas_result = pd.Series(expect)
    assert pandas_result.equals(grizzly_result)

def test_eq():
    left = ["hello", "world", "strings", "morestrings"]
    right = ["hel", "world", "string", "morestrings"]
    x = gr.GrizzlySeries(left)
    y = gr.GrizzlySeries(right)
    assert list(x.eq(y).evaluate().values) == [False, True, False, True]
    assert list(x.ne(y).evaluate().values) == [True, False, True, False]

def test_strip():
    compare_vs_pandas('strip', ["",
    "   hi   ",
    "\t\thi\n",
    """

    hello

    """,
    "    \t goodbye",
    "goodbye again    ",
    "   \n hi \n bye \n ",
    """

    hi

    bye

    """])

def test_contains():
    compare_vs_pandas('contains', ["abc", "abcdefg", "gfedcbaabcabcdef", ""], "abc")

def test_startswith():
    compare_vs_pandas('startswith', ["abc", "abcdefg", "gfedcba", "", "defabc"], "abc")

def test_endswith():
    compare_vs_pandas('endswith', ["abc", "abcdefg", "gfedabc", "", "defabc"], "abc")

def test_find():
    compare_vs_pandas('find', ["abc", "abcdefg", "gfedcbaabcabcdef", ""], "abc")
    compare_vs_pandas('find', ["abc", "abcdefg", "gfedcbaabcabcdef", ""], "abc", 2)
    compare_vs_pandas('find', ["abc", "abcdefg", "gfedcbaabcabcdef", ""], "abc", 3)
    compare_vs_pandas('find', ["abc", "abcdefg", "gfedcbaabcabcdef", ""], "abc", end=2)
    compare_vs_pandas('find', ["abc", "abcdefg", "gfedcbaabcabcdef", ""], "abc", end=3)
    compare_vs_pandas('find', ["abc", "abcdefg", "gfedcbaabcabcdef", ""], "abc", 3, end=7)
    compare_vs_pandas('find', ["abc", "abcdefg", "gfedcbaabcabcdef", ""], "abc", 100, end=105)

def test_replace():
    """
    Behavior of replace is different in Grizzly -- it currently only replaces the *first*
    occurrance. This will be changed in a later patch.

    """
    import copy
    inp = ["abc", "abcdefg", "abcabcabc", "gfedcbaabcabcdef", "", "XYZ"]
    expect = [s.replace("abc", "XYZ", 1) for s in copy.copy(inp)]
    grizzly_result = gr.GrizzlySeries(inp).str.replace("abc", "XYZ").str.to_pandas()
    pandas_result = pd.Series(expect)
    assert pandas_result.equals(grizzly_result)

