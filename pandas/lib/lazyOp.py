from nvlobject import *


def to_nvl_type(nvl_type, dim):
    for i in xrange(dim):
        nvl_type = NvlVec(nvl_type)
    return nvl_type


class LazyOpResult:
    def __init__(self, expr, nvl_type, dim):
        self.expr = expr
        self.nvl_type = nvl_type
        self.dim = dim

    def evaluate(self, verbose=True, decode=True):
        if isinstance(self.expr, NvlObject):
            return self.expr.evaluate(to_nvl_type(self.nvl_type, self.dim), verbose, decode)
        return self.expr
