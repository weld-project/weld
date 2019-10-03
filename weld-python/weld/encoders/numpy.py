"""
Implements encoders for NumPy values.

Zero-copy conversions (in particular, to 1D arrays) are implemented here
directly since they only involve a pointer copy. Conversions of types that
currently require copies are implemented in Rust.
"""

import numpy as np

from .encoder_base import *
from ..types import *

# Maps a string dtype representation to a Weld scalar type.
_known_types = {
        'int8': I8(),
        'int16': I16(),
        'int32': I32(),
        'int64': I64(),
        'uint8': U8(),
        'uint16': U16(),
        'uint32': U32(),
        'uint64': U64(),
        'float': F32(),
        'float32': F32(),
        'double': F64(),
        'float64': F64(),
        'bool': Bool()
        }

def dtype_to_weld_type(ty):
    """Converts a NumPy data type to a Weld type.

    The data type can be a any type that can be converted to a NumPy dtype,
    e.g., a string (e.g., 'int32') or a NumPy scalar type (e.g., np.int32). The
    type chosen follows the rules specified by NumPy here:

    https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#dtype

    For example, 'i8' will map to an int64 in Weld, since it indicates a signed
    integer that has eight bytes.

    Examples
    --------
    >>> dtype_to_weld_type('int32')
    <weld.types.I32 object at 0x...>
    >>> dtype_to_weld_type('float')
    <weld.types.F64 object at 0x...>
    >>> dtype_to_weld_type('i8')
    <weld.types.I64 object at 0x...>
    >>> dtype_to_weld_type(np.int16)
    <weld.types.I16 object at 0x...>

    Parameters
    ----------
    ty : str or dtype or NumPy scalar type
        The NumPy type to convert

    Returns
    -------
    WeldType

    """
    if not isinstance(ty, np.dtype):
        ty = np.dtype(ty)

    ty = str(ty)
    if ty in _known_types:
        return _known_types.get(ty)

    if ty.startswith('S'):
        raise TypeError("Python 2 strings not supported -- use Unicode")
    if ty.find('U') != -1:
        raise NotImplementedError("Unicode strings not yet supported")

class NumPyWeldEncoder(WeldEncoder):
    pass

class NumPyWeldDecoder(WeldDecoder):
    pass
