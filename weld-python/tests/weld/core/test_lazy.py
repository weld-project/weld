"""
Tests for constructing and evaluating lazy operations.
"""

from weld.encoders.primitives import PrimitiveWeldEncoder, PrimitiveWeldDecoder
from weld.types import *
from weld.lazy import *

def test_simple():
    a = 1
    b = 2
    a = PhysicalValue(1, I32(), PrimitiveWeldEncoder())
    b = PhysicalValue(2, I32(), PrimitiveWeldEncoder())
    comp1 = WeldLazy(
            "{0} + {1}".format(a.id, b.id),
            [a, b],
            I32(),
            PrimitiveWeldDecoder())
    x, _ = comp1.evaluate()

    assert comp1.num_dependencies == 2
    assert x == 3

def test_dependencies():
    a = 1
    b = 2
    a = PhysicalValue(1, I32(), PrimitiveWeldEncoder())
    b = PhysicalValue(2, I32(), PrimitiveWeldEncoder())
    comp1 = WeldLazy(
            "{0} + {1}".format(a.id, b.id),
            [a, b],
            I32(),
            PrimitiveWeldDecoder())
    comp2 = WeldLazy(
            "{0} + {1}".format(comp1.id, comp1.id),
            # These should be de-duplicated.
            [comp1, comp1],
            I32(),
            PrimitiveWeldDecoder())

    assert comp2.num_dependencies == 3
    x, _ = comp2.evaluate()
    assert x == 6

def test_long_chain():
    a = 1
    b = 2
    a = PhysicalValue(1, I32(), PrimitiveWeldEncoder())
    b = PhysicalValue(2, I32(), PrimitiveWeldEncoder())
    def add_previous(prev1, prev2):
        # Returns an expression representing prev1 + prev2
        return WeldLazy(
            "{0} + {1}".format(prev1.id, prev2.id),
            [prev1, prev2],
            I32(),
            PrimitiveWeldDecoder())

    length = 100
    expr = add_previous(a, b)
    for i in range(length):
        expr = add_previous(expr, a)

    # 2 physical values and length sub-expressions.
    assert expr.num_dependencies == length + 2
    x, _ = expr.evaluate()
    assert x == (length +  3)
