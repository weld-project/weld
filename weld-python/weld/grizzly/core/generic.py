"""
Base class for Grizzly collection types.

"""

from abc import ABC, abstractmethod

class GrizzlyBase(ABC):

    @property
    @abstractmethod
    def weld_value(self):
        """
        Returns the WeldLazy represention of this object.

        """
        pass

    @property
    @abstractmethod
    def is_value(self):
        """
        Returns whether this collection wraps a physical value rather than
        a computation.
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Evaluates this collection and returns itself.

        Evaluation reduces the currently stored computation to a physical value
        by compiling and running a Weld program. If this collection refers
        to a physical value and no computation, no program is compiled, and this
        method returns `self` unmodified.

        """
        pass

    @abstractmethod
    def to_pandas(self, copy=False):
        """
        Evaluates this computation and coerces it into a pandas object.

        This is guaranteed to be 0-copy if `self.is_value == True`. Otherwise,
        some allocation may occur. Regardless, `self` and the returned value
        will always have the same underlying data unless `copy = True`.

        Parameters
        ----------
        copy : boolean, optional
            Specifies whether the new collection should copy data from self

        """
        pass

    @property
    def children(self):
        """
        Returns the Weld children of this value.
        """
        return self.weld_value.children

    @property
    def output_type(self):
        """
        Returns the Weld output type of this collection.

        The output type is always a `WeldVec` of some type.
        """
        return self.weld_value.output_type

    @property
    def code(self):
        """
        Returns the Weld code for this computation.
        """
        return self.weld_value.code



