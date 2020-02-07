"""
Tests for constructing and evaluating lazy operations.
"""

from weld.encoders import PrimitiveWeldEncoder, PrimitiveWeldDecoder
from weld.types import *
from weld.lazy import *

class LazyObject(WeldLazyMixin):
    """ A thin wrapper around a lazy Weld value. """
    pass

def test_simple():
    a = 1
    b = 2
    comp1 = "a + b"

    a = Dependency(1, I32(), PrimitiveWeldEncoder())
    b = Dependency(2, I32(), PrimitiveWeldEncoder())
    comp1 = WeldLazyMixin(
            "{0} + {1}".format(a.expr_id, b.expr_id),
            [a, b],
            I32(),
            PrimitiveWeldDecoder())
    x, _ = comp1.evaluate()
    assert x == 3
