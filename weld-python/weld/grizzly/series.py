"""
A Weld wrapper for pandas.Series.
"""

import numpy as np

from pandas import Series
from pandas.core.internals import SingleBlockManager

from weld.encoders.numpy import dtype_to_weld_type

# WELD

import weld.compile

from abc import ABC

Subexpr = namedtuple('Subexpr', ['expr_id', 'value'])
ExprId = namedtuple('ExprId', ['name', 'ty'])
Dependency = namedtuple('Dependency', ['value', 'ty', 'encoder'])

class WeldLazyMixin(ABC):
    """
    """

    # Counter for generating unique IDs.
    COUNTER = -1
    # Prefix for names of values.
    PREFIX = "obj"

    @classmethod
    def generate_id(cls, ty):
        cls.COUNTER += 1
        return ExprId("{0}{1}".format(cls.PREFIX, cls.COUNTER), ty)

    def __init__(self, expression, dependencies, ty, decoder):
        # An ordered list of children that this computation depends on.  Each
        # child is either another WeldLazyMixin, or a Dependency, which
        # represents a physical value that will be passed into Weld upon
        # compilation.
        self.children = list(dependencies)
        self.values = {}
        for (i, child) in enumerate(self.children):
            if not isinstance(value, WeldLazyMixin):
                assert isinstance(value, Dependency)
                # Create a new name for the non-Weld dependency.
                self.values[i] = WeldLazyMixin.generate_id(value.ty)

        # Name of this computation
        self.expr_id = WeldLazyMixin.generate_id(ty)
        # Expression this object evaluates. This should depend only on names that
        # also appear as dependencies.
        self.expression =  expression
    
    def _collect_subexprs(self):
        """
        """
        exprs = []
        for (index, child) in enumerate(self.children):
            if not isinstance(child, WeldLazyMixin):
                exprs.append(Subexpr(expr_id=values[index], value=child))
            exprs += child._collect_subexprs())
        exprs.append(Subexpr(expr_id=self.expr_id, value=self.expression))
        return exprs

    def _create_function_header(self, inputs):
        arguments = ["{0}: {1}".format(inp.name, str(inp.ty)) for inp in inputs]
        return "|" + ", ".join(arguments) + "|"

    def _assemble_program(self):
        subexprs = self.collect_subexprs()
        # Assemble the inputs.
        inputs = [expr for expr in exprs if isinstance(expr, Dependency)]
        # Sort the inputs so they appear in a consistent order.
        inputs.sort(key=lambda e: e.expr_id.name)
        arg_types = [inp.ty for inp in inputs]
        encoders = [inp.encoder for inp in inputs]
        # Assemble the Weld program.
        exprs = [expr for expr in exprs if not isinstance(expr, Dependency)]
        program = self._create_function_header(inputs) + "\n".join(exprs)
        program = weld.compile.compile(program, arg_types, encoders, self.ty, self.decoder)

# end WELD

def _weldseries_constructor_with_fallback(data=None, index=None, **kwargs):
    """
    A flexible constructor for WeldSeries._constructor, which needs to be able
    to fall back to a Series (if a certain operation does not produce
    geometries)
    """
    try:
        return WeldSeries(data=data, index=index, **kwargs)
    except TypeError:
        return Series(data=data, index=index, **kwargs)

class WeldSeries(GrizzlyBase, Series):
    """ A lazy Series object backed by a Weld computation. """

    @property
    def _constructor(self):
        return _weldseries_constructor_with_fallback

    @property
    def _constructor_expanddim(self):
        return NotImplemented

    def __init__(self, *args, **kwargs):
        # Everything important is done in __new__.
        pass

    def __new__(cls, data=None, index=None, **kwargs):
        s = None
        if not isinstance(data, np.ndarray):
            s = Series(data, index=index, **kwargs)
            data = s.values
        if index is not None or dtype_to_weld_type(data.dtype) is not None:
            self = super(WeldSeries, cls).__new__(cls)
            super(WeldSeries, self).__init__(data, index=index, **kwargs)
            return self
        return s if s is not None else Series(data, index=index, **kwargs)

    def add(self, other, **unsupported_kwargs):
        # TODO: This should be code-gen'd.
        if len(unsupported_kwargs) > 0:
            return NotImplemented
        op = OpRegistry.get("add")
        op.build_map(self.dtype, other.dtype)
        code = binary_op(op.infix, self.resolve_dtype(self.dtype, other.dtype)
