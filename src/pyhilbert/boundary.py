from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

class BoundaryCondition(ABC):
    @property
    @abstractmethod
    def size(self) -> Optional[int]:
        """Returns the number of sites along this dimension, or None if infinite."""
        pass
    
    @abstractmethod
    def wrap(self, index: int) -> int:
        """
        Wraps an index back into the valid unit cell according to the boundary.
        Should raise an exception (e.g., `OutOfBoundsError`) if the index crosses
        an impassable boundary.
        """
        pass

@dataclass(frozen=True)
class PeriodicBoundary(BoundaryCondition):
    length: int

    @property
    def size(self) -> int:
        return self.length

    def wrap(self, index: int) -> int:
        return index % self.length

@dataclass(frozen=True)
class OpenBoundary(BoundaryCondition):
    length: int

    @property
    def size(self) -> int:
        return self.length

    def wrap(self, index: int) -> int:
        if 0 <= index < self.length:
            return index
        raise IndexError(f"Index {index} out of bounds for OpenBoundary of length {self.length}")

@dataclass(frozen=True)
class InfiniteBoundary(BoundaryCondition):
    @property
    def size(self) -> Optional[int]:
        return None

    def wrap(self, index: int) -> int:
        return index # No boundaries, all indices are valid

@dataclass(frozen=True)
class TwistedBoundary(PeriodicBoundary):
    # E.g., for magnetic fluxes or specialized Hamiltonians
    phase: float 