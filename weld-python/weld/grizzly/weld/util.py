"""
Utilities to help construct Weld computations.

"""

import functools

from weld.lazy import WeldLazy

def weldfunc():
    """
    An annotation for functions that return Weld strings.

    A function annotated with `weldfunc` takes zero or more `str` arguments and
    returns a Weld string as a result. The annotation converts this function
    into a one that returns a _partial function_ that returns a `WeldLazy`. The
    `WeldLazy` represents the computation constructed by the annotated
    function. The partial function takes two arguments: an output Weld type,
    and a Weld decoder.

    Functions annotated with `weldfunc` can accept `WeldLazy` as input: the
    annotation will unwrap `WeldLazy.id` and pass it to the annotated function.

    Examples
    --------

    TODO!

    """
    def wrapper(self, func):
        @functools.wraps(func)
        def _decorated(*args, **kwargs):
            new_args = list()
            new_kwargs = dict()
            dependencies = []
            for arg in args:
                if isinstance(arg, WeldLazy):
                    new_args.append(str(arg.id))
                    dependencies.append(arg)
                else:
                    new_args.append(arg)
            for (k, v) in kwargs:
                if isinstance(v, WeldLazy):
                    new_kwargs[k] = str(v.id)
                    dependencies.append(v)
                else:
                    new_kwargs[k] = v

            assert len(new_args) == len(args)
            assert len(new_kwargs) == len(kwargs)

            code = func(*new_args, **new_kwargs)
            def partial_func(weld_output_type, decoder):
                WeldLazy(code, dependencies, weld_output_type, decoder)

