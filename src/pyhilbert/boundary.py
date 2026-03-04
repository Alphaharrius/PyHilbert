from abc import ABC, abstractmethod
from dataclasses import dataclass
from sympy import ImmutableDenseMatrix

class BoundaryCondition(ABC):
    """
    Abstract base class for boundary conditions in lattice systems.
    """

    @property
    @abstractmethod
    def basis(self) -> ImmutableDenseMatrix:
        """The size of the boundary along each dimension."""
        pass

    @abstractmethod
    def wrap(self, index: ImmutableDenseMatrix) -> ImmutableDenseMatrix:
        """Wrap an index into the valid region of this boundary."""
        pass


@dataclass(frozen=True)
class PeriodicBoundary(BoundaryCondition):
    """
    Periodic boundary: wraps indices using modulo arithmetic.
    """
    
    # Dataclass automatically implements the basis property required by the ABC
    basis: ImmutableDenseMatrix

    def wrap(self, index: ImmutableDenseMatrix) -> ImmutableDenseMatrix:
        """
        Wrap the index within [0, basis-1] using element-wise modulo operation.
        """
        if index.shape != self.basis.shape:
            raise ValueError("Index and basis must have the same shape.")

        # Perform element-wise modulo for SymPy matrices
        wrapped_elements = [i % b for i, b in zip(index, self.basis)]
        
        return ImmutableDenseMatrix(wrapped_elements).reshape(index.rows, index.cols)