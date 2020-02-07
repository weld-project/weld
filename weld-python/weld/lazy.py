
import weld.compile

from abc import ABC, abstractmethod
from collections import namedtuple

class ExprId(object):
    
    __slots__ = ['name', 'ty']

    def __init__(self, name, ty):
        self.name = name
        self.ty = ty

    def __str__(self):
        return self.name

class WeldLazyBase(ABC):
    """
    Base class for Weld lazy computations
    """
    # Counter for generating unique IDs.
    COUNTER = -1
    # Prefix for names of values.
    PREFIX = "obj"

    @classmethod
    def generate_id(cls, ty):
        cls.COUNTER += 1
        return ExprId("{0}{1}".format(cls.PREFIX, cls.COUNTER), ty)

class Dependency(WeldLazyBase):
    """
    A dependency representing a physical value that a lazy computation depends on.
    """
    def __init__(self, value, ty, encoder):
        self.value = value
        self.encoder = encoder
        # Name of this computation
        self.expr_id = Dependency.generate_id(ty)

    @property
    def ty(self):
        return self.expr_id.ty

Subexpr = namedtuple('Subexpr', ['expr_id', 'value'])

class WeldLazyMixin(WeldLazyBase):
    """
    """

    # Counter for generating unique IDs.
    COUNTER = -1
    # Prefix for names of values.
    PREFIX = "obj"

    def __init__(self, expression, dependencies, ty, decoder):
        # An ordered list of children that this computation depends on.  Each
        # child is another WeldLazyMixin or a Dependency that represents a
        # physical value that will be passed into Weld upon compilation.
        self.children = list(dependencies)
        # Name of this computation
        self.expr_id = WeldLazyMixin.generate_id(ty)
        # Expression this object evaluates. This should depend only on names that
        # also appear as dependencies.
        self.expression =  expression
        self.decoder = decoder

    @property
    def ty(self):
        return self.expr_id.ty
    
    def _collect_subexprs(self):
        """
        """
        exprs = []
        for (index, child) in enumerate(self.children):
            if isinstance(child, WeldLazyMixin):
                exprs += child._collect_subexprs()
            elif isinstance(child, Dependency):
                exprs.append(Subexpr(expr_id=child.expr_id, value=child))
            else:
                raise ValueError("Unexpected value {} during evaluate".format(child))
        exprs.append(Subexpr(expr_id=self.expr_id, value=self.expression))
        return exprs

    def _create_function_header(self, inputs):
        arguments = ["{0}: {1}".format(inp.expr_id.name, str(inp.expr_id.ty)) for inp in inputs]
        return "|" + ", ".join(arguments) + "|"

    def evaluate(self):
        exprs = self._collect_subexprs()
        # Assemble the inputs.
        inputs = [expr.value for expr in exprs if isinstance(expr.value, Dependency)]
        # Sort the inputs so they appear in a consistent order.
        inputs.sort(key=lambda e: e.expr_id.name)
        arg_types = [inp.ty for inp in inputs]
        encoders = [inp.encoder for inp in inputs]
        # Assemble the Weld program.
        expressions = [expr.value for expr in exprs if not isinstance(expr.value, Dependency)]
        program = self._create_function_header(inputs) + " " + "\n".join(expressions)
        program = weld.compile.compile(program, arg_types, encoders, self.ty, self.decoder)

        values = [inp.value for inp in inputs]
        return program(*values)
