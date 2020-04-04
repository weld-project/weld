"""
Implements method forwarding from one class to another.

The `Forwarding` class can be used as a mixin.

"""

import inspect
import warnings

class Forwarding(object):
    @classmethod
    def _get_class_that_defined_method(cls, meth):
        """
        Returns the class that defines the requested method.

        For methods that are defined outside of a particular set of
        Grizzly-defined classes, Grizzly will first evaluate lazy results
        before forwarding the data to the requested class.

        """
        if inspect.ismethod(meth):
            for cls in inspect.getmro(meth.__self__.__class__):
                if cls.__dict__.get(meth.__name__) is meth:
                    return cls
        if inspect.isfunction(meth):
            return getattr(inspect.getmodule(meth),
                           meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0])

    @classmethod
    def _requires_forwarding(cls, meth):
        defined_in = cls._get_class_that_defined_method(meth)
        if defined_in is not None and defined_in is not cls:
            return True
        else:
            return False

    @classmethod
    def _forward(cls, to_cls):
        from functools import wraps
        def forward_decorator(func):
            @wraps(func)
            def forwarding_wrapper(self, *args, **kwargs):
                self.evaluate()
                result = func(self, *args, **kwargs)
                # Unsupported functions will return Series -- try to
                # switch back to GrizzlySeries.
                if not isinstance(result, cls) and isinstance(result, to_cls):
                    try_convert = cls(data=result.values, index=result.index)
                    if not isinstance(try_convert, cls):
                        warnings.warn("Unsupported operation '{}' produced unsupported Series: falling back to Pandas".format(
                            func.__name__))
                    return try_convert
                return result
            return forwarding_wrapper
        return forward_decorator

    @classmethod
    def add_forwarding_methods(cls, to_cls):
        """
        Add forwarding methods from this class to `to_cls`.

        """
        methods = dir(cls)
        for meth in methods:
            if meth.startswith("_"):
                # We only want to do this for API methods.
                continue
            attr = getattr(cls, meth)
            if cls._requires_forwarding(attr):
                setattr(cls, meth, cls._forward(to_cls)(attr))
