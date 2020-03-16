
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

def test_weldfunc_many_children():
    @weldfunc
    def exp(a, b):
        return "exp({}, {})".format(a, b)

    result = exp("1", "2")(F64(), None)
    result2 = exp("1", result)(F64(), None)
    result3 = exp(result2, result)(F64(), None)

    assert result in result3.children
    assert result2 in result3.children
    assert result3.num_dependencies == 2
    assert result3.expression == "exp({}, {})".format(result2.id, result.id)

def test_weldfunc_chain():
    @weldfunc
    def sqrt(a):
        return "sqrt({})".format(a)

    result = sqrt(1)(F64(), None)
    assert result.num_dependencies == 0
    assert result.expression == "sqrt(1)"

    chain_length = 10
    for i in range(chain_length):
        result = sqrt(result)(F64(), None)
    code = result.code
    # Make sure the final code has chain_length + 1 square roots.
    count = 0
    loc = code.find("sqrt")
    while loc > 0:
        count += 1
        code = code[loc + len("sqrt"):]
        loc = code.find("sqrt")
    assert count == chain_length + 1

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
