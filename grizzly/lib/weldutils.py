"""Summary
"""
#
# Utilities for the Weld API.
#
#

import sys


def to_shared_lib(name):
    """
    Returns the name with the platform dependent shared library extension.

    Args:
    name (TYPE): Description
    """
    if sys.platform.startswith('linux'):
        return name + ".so"
    elif sys.platform.startswith('darwin'):
        return name + ".dylib"
    else:
        sys.exit(1)
