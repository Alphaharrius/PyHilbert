from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict
from abc import ABC, abstractmethod
from multipledispatch import dispatch  # type: ignore[import-untyped]
from itertools import product
from functools import lru_cache
import sympy as sy
import numpy as np
import torch
from .precision import get_precision_config
from sympy import ImmutableDenseMatrix, sympify
from .utils import FrozenDict, validate_matrix
from .abstracts import Operable, HasDual, HasBase, Plottable


@dataclass(frozen=True)
class Spatial(Operable, Plottable, ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError()


@dataclass(frozen=True)
class AffineSpace(Spatial):
    basis: ImmutableDenseMatrix

    def __post_init__(self):
        validate_matrix(self.basis, "AffineSpace.basis")

    @property
    def dim(self) -> int:
        return self.basis.rows

    def __str__(self):
        data = [[str(sympify(x)) for x in row] for row in self.basis.tolist()]
        return f"AffineSpace(basis={data})"

    def __repr__(self):
        return str(self)


@dataclass(frozen=True)
class AbstractLattice(AffineSpace, HasDual):
    shape: Tuple[int, ...]

    @property
    def affine(self) -> AffineSpace:
        return AffineSpace(basis=self.basis)


@dataclass(frozen=True)
class Lattice(AbstractLattice):
    unit_cell: FrozenDict = field(
        default_factory=FrozenDict
    )  # TODO : Any way to improve the init

    def __post_init__(self):
        unit_cell_source = self.unit_cell
        if len(unit_cell_source) == 0:
            unit_cell_source = FrozenDict(
                {
                    "r": Offset(
                        rep=ImmutableDenseMatrix([0] * self.dim), space=self.affine
                    )
                }
            )
        processed_cell = {}
        for key, value in unit_cell_source.items():
            if not isinstance(key, str):
                raise TypeError(f"unit_cell keys must be strings, but got {type(key)}")
            if isinstance(value, Offset):
                processed_cell[key] = value
            else:
                try:
                    rep = ImmutableDenseMatrix(value)
                    if rep.shape != (self.dim, 1):
                        rep = rep.reshape(self.dim, 1)
                    processed_cell[key] = Offset(rep=rep, space=self.affine)
                except Exception as e:
                    raise TypeError(
                        f"Could not convert unit_cell value {value} for key '{key}' to an Offset."
                    ) from e

        object.__setattr__(self, "unit_cell", FrozenDict(processed_cell))

    @property
    @lru_cache
    def dual(self) -> "ReciprocalLattice":
        reciprocal_basis = 2 * sy.pi * self.basis.inv().T
        return ReciprocalLattice(basis=reciprocal_basis, shape=self.shape, lattice=self)

    def coords(
        self,
        subs: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Vectorized calculation of all site coordinates.
        Avoids running SymPy substitution inside the loop.
        """
        precision = get_precision_config()
        basis_sym = self.basis
        if subs:
            basis_eval = basis_sym.subs(subs)
        else:
            basis_eval = basis_sym.subs({s: 1.0 for s in basis_sym.free_symbols})

        try:
            basis_mat = torch.tensor(
                np.array(basis_eval).astype(precision.np_float),
                dtype=precision.torch_float,
            )
        except Exception as e:
            raise ValueError(
                f"Basis matrix contains unresolved symbols: {basis_eval.free_symbols}"
            ) from e

        lat_offsets = cartes(self)

        lat_reps = []
        for off in lat_offsets:
            lat_reps.append(np.array(off.rep).flatten().astype(precision.np_float))

        if not lat_reps:
            return torch.empty((0, self.dim))

        lat_tensor = torch.tensor(
            np.array(lat_reps), dtype=precision.torch_float
        )  # (N_cells, Dim)

        basis_reps = []
        if not self.unit_cell:
            basis_reps.append(np.zeros(self.dim, dtype=precision.np_float))
        else:
            sorted_unit_cell = sorted(self.unit_cell.items(), key=lambda x: str(x[0]))
            for _, site_offset in sorted_unit_cell:
                site_vec = site_offset.rep
                if subs:
                    site_vec = site_vec.subs(subs)
                basis_reps.append(
                    np.array(site_vec).flatten().astype(precision.np_float)
                )

        basis_tensor = torch.tensor(
            np.array(basis_reps), dtype=precision.torch_float
        )  # (N_basis, Dim)

        total_crystal = lat_tensor.unsqueeze(1) + basis_tensor.unsqueeze(0)

        total_crystal_flat = total_crystal.view(-1, self.dim)

        coords = total_crystal_flat @ basis_mat

        return coords


@dataclass(frozen=True)
class ReciprocalLattice(AbstractLattice):
    lattice: Lattice

    @property
    @lru_cache
    def dual(self) -> Lattice:
        return self.lattice


@dataclass(frozen=True)
class Offset(Spatial, HasBase[AffineSpace]):
    rep: ImmutableDenseMatrix
    space: AffineSpace

    def __post_init__(self):
        if self.rep.shape != (self.space.dim, 1):
            raise ValueError("Invalid Shape")
        validate_matrix(self.rep, "Offset.rep")

    @property
    def dim(self) -> int:
        return self.rep.rows

    def fractional(self) -> "Offset":
        """
        Return the fractional coordinates of this Offset within its lattice space.
        """
        n = sy.Matrix([sy.floor(x) for x in self.rep])
        s = self.rep - n
        return Offset(rep=sy.ImmutableDenseMatrix(s), space=self.space)

    fractional = lru_cache(fractional)

    def base(self) -> AffineSpace:
        """Get the `AffineSpace` this `Offset` is expressed in."""
        return self.space

    def rebase(self, space: AffineSpace) -> "Offset":
        """
        Re-express this Offset in a different AffineSpace.

        Parameters
        ----------
        `space` : `AffineSpace`
            The new affine space to express this Offset in.

        Returns
        -------
        `Offset`
            New Offset expressed in the given affine space.
        """
        rebase_transform_mat = space.basis.inv() @ self.space.basis
        new_rep = rebase_transform_mat @ self.rep
        return Offset(rep=ImmutableDenseMatrix(new_rep), space=space)

    def __str__(self):
        # If it's a column vector, flatten to 1D python list
        if self.rep.shape[1] == 1:
            vec = [str(sympify(v)) for v in list(self.rep)]
        else:
            vec = [[str(sympify(x)) for x in row] for row in self.rep.tolist()]
        basis = [[str(sympify(x)) for x in row] for row in self.space.basis.tolist()]
        return f"Offset({vec} ∈ {basis})"

    def __repr__(self):
        return str(self)


@dataclass(frozen=True)
class Momentum(Offset, HasBase[ReciprocalLattice]):
    def fractional(self) -> "Momentum":
        """
        Return the fractional coordinates of this Offset within its lattice space.
        """
        n = sy.Matrix([sy.floor(x) for x in self.rep])
        s = self.rep - n
        return Momentum(rep=sy.ImmutableDenseMatrix(s), space=self.space)

    fractional = lru_cache(fractional)

    def base(self) -> ReciprocalLattice:
        """Get the `ReciprocalLattice` this `Momentum` is expressed in."""
        assert isinstance(self.space, ReciprocalLattice), (
            "Momentum.space must be a ReciprocalLattice"
        )
        return self.space

    def rebase(self, space: ReciprocalLattice) -> "Momentum":  # type: ignore[override]
        """
        Re-express this Momentum in a different ReciprocalLattice.

        Parameters
        ----------
        `space` : `AffineSpace`
            The new affine space (must be a ReciprocalLattice) to express this Momentum in.

        Returns
        -------
        `Momentum`
            New Momentum expressed in the given reciprocal lattice.
        """

        rebase_transform_mat = space.basis.inv() @ self.space.basis
        new_rep = rebase_transform_mat @ self.rep
        return Momentum(rep=ImmutableDenseMatrix(new_rep), space=space)


@dispatch(Lattice)  # type: ignore[no-redef]
@lru_cache
def cartes(lattice: Lattice) -> Tuple[Offset, ...]:
    elements = product(*tuple(range(n) for n in lattice.shape))
    return tuple(
        Offset(rep=ImmutableDenseMatrix(el), space=lattice.affine) for el in elements
    )


@dispatch(ReciprocalLattice)  # type: ignore[no-redef]
def cartes(lattice: ReciprocalLattice) -> Tuple[Momentum, ...]:
    elements = product(*tuple(range(n) for n in lattice.shape))
    sizes = ImmutableDenseMatrix(tuple(sy.Rational(1, n) for n in lattice.shape))
    elements = (ImmutableDenseMatrix(el).multiply_elementwise(sizes) for el in elements)
    return tuple(
        Momentum(rep=ImmutableDenseMatrix(el), space=lattice) for el in elements
    )
