"""
Custom arrays that wrap values in ways that Pandas doesn't understand natively.
Transformation between Pandas and Grizzly arrays is generally expensive.

"""

import from pandas.api.extensions import ExtensionArray

class GrizzlyStringArray(ExtensionArray):
    """
    An array that wraps string data accessible via Weld.

    A string array internally represents a C-style array that holds ASCII
    strings (UTF-8 support coming soon). In particular, the array internally
    has a memory layout that looks as follows:

    ```
    {i64, char*}
    {i64, char*}
    {i64, char*}
    {i64, char*}
    ...
    ```

    where each tuple represents a string length and a pointer to the string's buffer.
    Each string pointer can also be NULL, in which case the length of the string will
    be exactly -1. This indicates a missing value.

    `GrizzlyStringArray` has 0-copy interoperability with Weld, but will
    generally require copies when constructed from Python strings.

    """

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return NotImplemented

    @classmethod
    def _from_sequence_of_strings(cls, strings, dtype=None, copy=False):
        return NotImplemented

    @classmethod
    def _from_factorized(cls, values, original):
        return NotImplemented

    def __getitem__(self, key):
        return NotImplemented

    def __len__(self):
        return NotImplemented

    @property
    def dtype(self):
        return NotImplemented

    @property
    def nbytes(self) -> int:
        return NotImplemented

    def isna(self):
        return NotImplemented

    def take(self, indices, allow_fill=False, fill_value=None):
        return NotImplemented

    def copy(self):
        return NotImplemented

    @classmethod
    def _concat_same_type(cls, to_concat):
        return NotImplemented
