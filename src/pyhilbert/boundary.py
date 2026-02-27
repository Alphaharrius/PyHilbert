from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from sympy import ImmutableDenseMatrix


class BoundaryCondition(ABC):
    """
    Abstract base class for boundary conditions in lattice systems.

    Subclasses must define how to "wrap" an index that falls outside the allowed region
    and provide their extent (size) along their dimension. "size" may sometimes be None,
    e.g., for infinite boundaries.
    """

    basis: ImmutableDenseMatrix

    @abstractmethod
    def wrap(self, index: int) -> int:
        """
        Wrap an index into the valid region of this boundary.
        For periodic boundaries, this means modulo arithmetic.
        For open boundaries, this may raise an exception when out of bounds.
        """
        pass


@dataclass(frozen=True)
class PeriodicBoundary(BoundaryCondition):
    """
    Periodic boundary: wraps indices using modulo arithmetic.

    Attributes
    ----------
    basis : ImmutableDenseMatrix
        Number of unit cells along this dimension (periodicity).
    """

    def wrap(self, index: ImmutableDenseMatrix) -> ImmutableDenseMatrix:
        """
        Wrap the index within [0, basis-1] using modulo operation.

        Parameters
        ----------
        index : int
            The index to be wrapped.

        Returns
        -------
        int
            The wrapped index.
        """
        return index % self.basis
