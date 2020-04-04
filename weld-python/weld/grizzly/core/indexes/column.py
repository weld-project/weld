"""
Index used for access columns in Grizzly.

"""

from weld.grizzly.core.indexes.base import Index

class ColumnIndex(Index):
    """
    An index used for columns in a Grizzly DataFrame.

    Each index value is a Python object. For operations between two DataFrames
    with the same ColumnIndex, the result will also have the same index. For
    operations between two DataFrames with different ColumnIndex, the output
    will have a join of the two ColumnIndex, sorted by the index values.

    Two ColumnIndex are equal if their index names are equal and have the same
    order.

    Parameters
    ----------
    columns : iterable
        column names.
    slots : iterable of int or None
        slots associated with each column. If provided, the length must be
        len(columns). This is used for underlying data access only; index
        equality depends only on the column names and ordering.


    Examples
    --------
    >>> ColumnIndex(["name", "age"])
    ColumnIndex(['name', 'age'], [0, 1])
    >>> ColumnIndex(["name", "age"], slots=[1, 0])
    ColumnIndex(['name', 'age'], [1, 0])
    >>> ColumnIndex(["name", "age"], slots=[1, 2])
    Traceback (most recent call last):
    ...
    ValueError: slots must be contiguous starting at 0

    """

    def __init__(self, columns, slots=None):
        if not isinstance(columns, list):
            columns = list(columns)
        if slots is not None:
            assert len(columns) == len(slots)
            sorted_slots = sorted(slots)
            # Make sure each slot is occupied/there are no "holes".
            if not sorted_slots == list(range(len(slots))):
                raise ValueError("slots must be contiguous starting at 0")
        else:
            slots = range(len(columns))

        # The original column order.
        self.columns = columns
        # The mapping from columns to slots.
        self.index = dict(zip(columns, slots))

    def __iter__(self):
        """
        Iterates over columns in the order in which they appear in a DataFrame.

        Examples
        --------
        >>> x = ColumnIndex(["name", "age"], slots=[1, 0])
        >>> [name for name in x]
        ['name', 'age']

        """
        for col in self.columns:
            yield col

    def zip(self, other):
        """
        Zips this index with 'other', returning an iterator of `(name,
        slot_in_self, slot_in_other)`. The slot may be `None` if the name does
        not appear in either column.

        The result columns are ordered in a way consistent with how DataFrame
        columns should be be ordered (i.e., same order `self` if `self ==
        other`, and sorted by the union of columns from `self` and `other`
        otherwise).

        Examples
        --------
        >>> a = ColumnIndex(["name", "age"])
        >>> b = ColumnIndex(["name", "age"])
        >>> list(a.zip(b))
        [('name', 0, 0), ('age', 1, 1)]
        >>> b = ColumnIndex(["income", "age", "name"])
        >>> list(a.zip(b))
        [('age', 1, 1), ('income', None, 0), ('name', 0, 2)]

        """
        if self == other:
            for name in self.columns:
                yield (name, self.index[name], other.index[name])
        else:
            columns = sorted(list(set(self.columns).union(other.columns)))
            for name in columns:
                yield (name, self.index.get(name), other.index.get(name))

    def __getitem__(self, key):
        """
        Get the slot for a paritcular column name.

        Examples
        --------
        >>> a = ColumnIndex(["name", "age"])
        >>> a["age"]
        1

        """
        return self.index[key]

    def append(self, key):
        """
        Add a new column to the index. The slot is set to be `len(columns) - 1`.

        Examples
        --------
        >>> a = ColumnIndex(["name", "age"])
        >>> a.append("income")
        >>> a["income"]
        2

        """
        self.index[key] = len(self.columns)
        self.columns.append(key)

    def __eq__(self, other):
        """
        Compare equality depending on column names.

        Examples
        --------
        >>> a = ColumnIndex(["name", "age"])
        >>> a == ColumnIndex(["name", "age"])
        True
        >>> a == ColumnIndex(["age", "name"])
        False
        >>> a == ColumnIndex(["name", "age", "income"])
        False

        """
        return isinstance(other, ColumnIndex) and self.columns == other.columns

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "ColumnIndex({}, {})".format(self.columns, [self.index[col] for col in self.columns])
