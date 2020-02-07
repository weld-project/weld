"""
Tests for constructing and evaluating lazy operations.
"""

from weld.encoders import PrimitiveWeldEncoder, PrimitiveWeldDecoder
from weld.types import *
from weld.lazy import *

def test_simple():
    a = 1
    b = 2
    comp1 = "a + b"

    a = PhysicalValue(1, I32(), PrimitiveWeldEncoder())
    b = PhysicalValue(2, I32(), PrimitiveWeldEncoder())
    comp1 = WeldLazy(
            "{0} + {1}".format(a.id, b.id),
            [a, b],
            I32(),
            PrimitiveWeldDecoder())
    x, _ = comp1.evaluate()
    assert x == 3
