"""
Weld bindings.
"""

class WeldError(RuntimeError):
    """
    An error raised by Weld.

    All Weld compilation and runtime errors are converted to a `WeldError`.
    """
    pass

from weld.weld import *
