"""
String methods supported by Series.

"""

import weld.encoders.numpy as wenp
import weld.grizzly.weld.str as weldstr

from weld.types import *

class StringMethods(object):
    """
    String methods for Grizzly. Currently, string methods only apply to ASCII
    strings; while users can pass UTF-8 strings into Grizzly, their codepoints
    will be ignored by the below operations and will be returned unmodified.

    """

    __slots__ = [ "series", "constructor" ]

    def __init__(self, series):
        if series.dtype.char != 'S':
            raise ValueError("StringMethods only available for Series with dtype 'S'")
        self.series = series
        # TODO(shoumik): This is a hack: we should define an abstract class that captures
        # the interface additional functionality needs.
        self.constructor = self.series.__class__

    def to_pandas(self):
        """
        Convert an array of strings to a Pandas series.

        We provide a specialized implementation of `to_pandas` here that will perform UTF-8 decoding
        of the raw bytestrings that Grizzly series operate over.

        Examples
        --------
        >>> x = gr.GrizzlySeries(["Welcome", "to", "Grizzly!"])
        >>> x
        0     b'Welcome'
        1          b'to'
        2    b'Grizzly!'
        dtype: bytes64
        >>> x.str.to_pandas()
        0     Welcome
        1          to
        2    Grizzly!
        dtype: object

        """
        return self.series.evaluate().to_pandas().str.decode("utf-8")

    def _apply(self, func, *args, return_weld_elem_type=None):
        """
        Apply the given weldfunc to `self.series` and return a new GrizzlySeries.

        If the return type of the result is not a string GrizzlySeries, pass
        'return_weld_elem_type' to specify the element type of the result.

        """
        output_type = self.series.output_type if return_weld_elem_type is None else WeldVec(return_weld_elem_type)
        dtype = 'S' if return_weld_elem_type is None else wenp.weld_type_to_dtype(return_weld_elem_type)
        lazy = func(self.series.weld_value_, *args)(output_type, self.constructor._decoder)
        return (self.constructor)(lazy, dtype=dtype)

    def lower(self):
        """
        Lowercase strings.

        Examples
        --------
        >>> x = gr.GrizzlySeries(["HELLO", "WorLD"])
        >>> x.str.lower().str.to_pandas()
        0    hello
        1    world
        dtype: object

        """
        return self._apply(weldstr.lower)

    def upper(self):
        """
        Uppercase strings.

        Examples
        --------
        >>> x = gr.GrizzlySeries(["hello", "WorlD"])
        >>> x.str.upper().str.to_pandas()
        0    HELLO
        1    WORLD
        dtype: object

        """
        return self._apply(weldstr.upper)

    def capitalize(self):
        """
        Capitalize the first character in each string.

        Examples
        --------
        >>> x = gr.GrizzlySeries(["hello", "worlD"])
        >>> x.str.capitalize().str.to_pandas()
        0    Hello
        1    World
        dtype: object

        """
        return self._apply(weldstr.capitalize)

    def get(self, index):
        """
        Get the character at index 'i' from each string. If 'index' is greater than
        the string length, this returns an empty string. If 'index' is less than 0,
        this wraps around, using Python's indexing behavior.

        Examples
        --------
        >>> x = gr.GrizzlySeries(["hello", "worlD"])
        >>> x.str.get(4).str.to_pandas()
        0    o
        1    D
        dtype: object
        >>> x.str.get(-3).str.to_pandas()
        0    l
        1    r
        dtype: object

        """
        return self._apply(weldstr.get, index)

    def strip(self):
        """
        Strip whitespace from the string.

        Examples
        --------
        >>> x = gr.GrizzlySeries(["     hello   ", "   world    \t  "])
        >>> x.str.strip().str.to_pandas()
        0    hello
        1    world
        dtype: object

        """
        return self._apply(weldstr.strip)

    def contains(self, pat):
        """
        Returns whether each string contains the provided pattern.

        Pattern must be a Python string.

        Examples
        --------
        >>> x = gr.GrizzlySeries(["hello", "world"])
        >>> x.str.contains('wor').evaluate()
        0    False
        1     True
        dtype: bool

        """
        if not isinstance(pat, str):
            raise TypeError("pattern in contains must be a Python 'str'")
        return self._apply(weldstr.contains, pat, return_weld_elem_type=Bool())

    def startswith(self, pat):
        """
        Returns whether each string starts with the provided pattern.

        Pattern must be a Python string.

        Examples
        --------
        >>> x = gr.GrizzlySeries(["hello", "world"])
        >>> x.str.startswith('wo').evaluate()
        0    False
        1     True
        dtype: bool

        """
        if not isinstance(pat, str):
            raise TypeError("pattern in startswith must be a Python 'str'")
        return self._apply(weldstr.startswith, pat, return_weld_elem_type=Bool())

    def endswith(self, pat):
        """
        Returns whether each string starts with the provided pattern.

        Pattern must be a Python string.

        Examples
        --------
        >>> x = gr.GrizzlySeries(["hello", "world"])
        >>> x.str.endswith('rld').evaluate()
        0    False
        1     True
        dtype: bool

        """
        if not isinstance(pat, str):
            raise TypeError("pattern in endswith must be a Python 'str'")
        return self._apply(weldstr.endswith, pat, return_weld_elem_type=Bool())

    def find(self, sub, start=0, end=None):
        """
        Find 'sub' in each string. Each string is searched in the range [start,end).

        'sub' must be a Python string, and 'start' and 'end' must be Python integers.

        Examples
        --------
        >>> x = gr.GrizzlySeries(["bigfatcat", "fatcatbig", "reallybigcat"])
        >>> x.str.find('fat').evaluate()
        0    3
        1    0
        2   -1
        dtype: int64
        >>> x.str.find('big', end=2).evaluate()
        0   -1
        1   -1
        2   -1
        dtype: int64

        """
        if not isinstance(sub, str):
            raise TypeError("sub in find must be a Python 'str'")
        if not isinstance(start, int):
            raise TypeError("start in find must be a Python 'int'")
        if end is not None and not isinstance(end, int):
            raise TypeError("end in find must be a Python 'int'")
        return self._apply(weldstr.find, sub, start, end, return_weld_elem_type=I64())

    def replace(self, pat, rep):
        """
        Replaces the first occurrence of 'pat' with 'rep' in each string.

        Pattern and replacement must be Python strings.

        Examples
        --------
        >>> x = gr.GrizzlySeries(["hello", "world"])
        >>> x.str.replace('o', 'lalala').str.to_pandas()
        0    helllalala
        1    wlalalarld
        dtype: object

        """
        if not isinstance(pat, str):
            raise TypeError("pattern in replace must be a Python 'str'")
        if not isinstance(rep, str):
            raise TypeError("replacement in replace must be a Python 'str'")
        return self._apply(weldstr.replace, pat, rep)
