
from weld import *
from weld.lazy import WeldLazy, weldfunc
from weld.types import *

def test_weldfunc_simple():
    @weldfunc
    def exp(a, b):
        return "exp({}, {})".format(a, b)

    result = exp("1", "2")(F64(), None)
    assert isinstance(result, WeldLazy)
    assert result.expression == "exp(1, 2)"

    result2 = exp("1", result)(F64(), None)
    assert isinstance(result, WeldLazy)
    assert result2.expression == "exp(1, {})".format(result.id)
    assert result2.num_dependencies == 1
    assert result in result2.children

def test_weldfunc_non_string_args():
    @weldfunc
    def exp(a, b, use_pow):
        operator = "pow" if use_pow else "exp"
        return "{}({}, {})".format(operator, a, b)

    result = exp("1", "2", True)(F64(), None)
    assert isinstance(result, WeldLazy)
    assert result.expression == "pow(1, 2)"

    result2 = exp("1", result, False)(F64(), None)
    assert isinstance(result, WeldLazy)
    assert result2.expression == "exp(1, {})".format(result.id)
    assert result2.num_dependencies == 1
    assert result in result2.children

def test_weldfunc_non_string_kwargs():
    @weldfunc
    def exp(a, b, use_pow=True):
        operator = "pow" if use_pow else "exp"
        return "{}({}, {})".format(operator, a, b)

    result = exp("1", "2")(F64(), None)
    assert isinstance(result, WeldLazy)
    assert result.expression == "pow(1, 2)"

    result2 = exp("1", result, use_pow=False)(F64(), None)
    assert isinstance(result, WeldLazy)
    assert result2.expression == "exp(1, {})".format(result.id)
    assert result2.num_dependencies == 1
    assert result in result2.children
