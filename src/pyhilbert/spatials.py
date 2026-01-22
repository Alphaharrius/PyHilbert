from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
from abc import ABC, abstractmethod
from multipledispatch import dispatch  # type: ignore[import-untyped]
from itertools import product
from functools import lru_cache
from collections import OrderedDict
from functools import reduce

import sympy as sy
import numpy as np
import torch
from sympy import ImmutableDenseMatrix, sympify

from .utils import FrozenDict
from .abstracts import Operable, HasDual, Plottable


@dataclass(frozen=True)
class Spatial(Operable, Plottable, ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError()


@dataclass(frozen=True)
class AffineSpace(Spatial):
    basis: ImmutableDenseMatrix

    @property
    def dim(self) -> int:
        return self.basis.rows

    def __str__(self):
        data = [[str(sympify(x)) for x in row] for row in self.basis.tolist()]
        return f"AffineSpace(basis={data})"

    def __repr__(self):
        return str(self)


@dataclass(frozen=True)
class Lattice(AffineSpace, HasDual):
    shape: Tuple[int, ...]

    @property
    def affine(self) -> AffineSpace:
        return AffineSpace(basis=self.basis)

    @property
    @lru_cache
    def dual(self) -> "ReciprocalLattice":
        reciprocal_basis = 2 * sy.pi * self.basis.inv().T
        return ReciprocalLattice(basis=reciprocal_basis, shape=self.shape)

    def compute_coords(
        self,
        basis_offsets: Optional[List["Offset"]] = None,
        subs: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Vectorized calculation of all site coordinates.
        Avoids running SymPy substitution inside the loop.
        """
        basis_sym = self.basis
        if subs:
            basis_eval = basis_sym.subs(subs)
        else:
            basis_eval = basis_sym.subs({s: 1.0 for s in basis_sym.free_symbols})

        try:
            basis_mat = torch.tensor(
                np.array(basis_eval).astype(np.float64), dtype=torch.float64
            )
        except Exception as e:
            raise ValueError(
                f"Basis matrix contains unresolved symbols: {basis_eval.free_symbols}"
            ) from e

        lat_offsets = cartes(self)

        lat_reps = []
        for off in lat_offsets:
            lat_reps.append(np.array(off.rep).flatten().astype(np.float64))

        if not lat_reps:
            return torch.empty((0, self.dim))

        lat_tensor = torch.tensor(
            np.array(lat_reps), dtype=torch.float64
        )  # (N_cells, Dim)

        if basis_offsets is None:
            basis_offsets = [
                Offset(rep=sy.ImmutableDenseMatrix([0] * self.dim), space=self.affine)
            ]

        basis_reps = []
        for off in basis_offsets:
            rep = off.rep
            if subs:
                rep = rep.subs(subs)
            basis_reps.append(np.array(rep).flatten().astype(np.float64))

        basis_tensor = torch.tensor(
            np.array(basis_reps), dtype=torch.float64
        )  # (N_basis, Dim)

        total_crystal = lat_tensor.unsqueeze(1) + basis_tensor.unsqueeze(0)

        total_crystal_flat = total_crystal.view(-1, self.dim)

        coords = total_crystal_flat @ basis_mat.T

        return coords


@dataclass(frozen=True)
class ReciprocalLattice(Lattice):
    @property
    @lru_cache
    def dual(self) -> "Lattice":  # type: ignore[override]
        basis = (1 / (2 * sy.pi)) * self.basis.inv().T
        return Lattice(basis=basis, shape=self.shape)


@dataclass(frozen=True)
class Offset(Spatial):
    rep: ImmutableDenseMatrix
    space: AffineSpace

    @property
    def dim(self) -> int:
        return self.rep.rows

    def fractional(self) -> "Offset":
        """
        Return the fractional coordinates of this Offset within its lattice space.
        """
        basis = self.space.basis
        uv = basis.LUsolve(self.rep)  # fractional coords u,v
        n = sy.Matrix([sy.floor(x) for x in uv])  # integer cell indices
        s = uv - n  # fractional coords in [0,1)
        frac = basis * s  # point inside reference cell
        return Offset(rep=sy.ImmutableDenseMatrix(frac), space=self.space)

    fractional = lru_cache(fractional)

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
class Momentum(Offset):
    def fractional(self) -> "Momentum":
        """
        Return the fractional coordinates of this Offset within its lattice space.
        """
        basis = self.space.basis
        uv = basis.LUsolve(self.rep)  # fractional coords
        n = sy.Matrix([sy.floor(x) for x in uv])  # reciprocal cell indices
        s = uv - n  # fractional coords in [0,1)
        frac = basis * s  # point inside reference cell
        return Momentum(rep=sy.ImmutableDenseMatrix(frac), space=self.space)

    fractional = lru_cache(fractional)


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


@dataclass(frozen=True)
class PointGroupBasis(Spatial):
    expr: sy.Expr
    axes: Tuple[sy.Symbol, ...]
    order: int
    rep: sy.ImmutableDenseMatrix

    @property
    def dim(self):
        return len(self.axes)

    def __str__(self):
        return f"PointGroupBasis({str(self.expr)})"

    def __repr__(self):
        return f"PointGroupBasis({repr(self.expr)})"


@dataclass(frozen=True)
class AbelianGroupOrder:
    irrep: sy.ImmutableDenseMatrix
    axes: Tuple[sy.Symbol, ...]
    basis_function_order: int

    @lru_cache
    def __full_indices(self):
        return tuple(product(*((self.axes,) * self.basis_function_order)))

    @lru_cache
    def __commute_indices(self):
        indices = self.__full_indices()
        _, select_rules = AbelianGroupOrder.__get_contract_select_rules(indices)
        sorted_rules = sorted(select_rules, key=lambda x: x[1])
        return tuple(indices[n] for n, _ in sorted_rules)

    @property
    @lru_cache
    def euclidean_basis(self) -> sy.ImmutableDenseMatrix:
        indices = self.__commute_indices()
        return sy.ImmutableDenseMatrix([sy.prod(idx) for idx in indices]).T

    @staticmethod
    @lru_cache
    def __get_contract_select_rules(indices: Tuple[Tuple[sy.Symbol, ...], ...]):
        commute_index_table: OrderedDict[Tuple[sy.Symbol, ...], int] = OrderedDict()
        contract_indices = []
        select_indices = []
        order_indices = set()
        order_idx = 0
        for n, idx in enumerate(indices):
            key = tuple(sorted(idx, key=lambda s: s.name))
            m = commute_index_table.setdefault(key, order_idx)

            contract_indices.append((n, m))
            if m not in order_indices:
                select_indices.append((n, m))
                order_indices.add(m)
                order_idx += 1

        return contract_indices, select_indices

    @property
    @lru_cache
    def full_rep(self):
        return reduce(sy.kronecker_product, (self.irrep,) * self.basis_function_order)

    @property
    @lru_cache
    def rep(self):
        indices = self.__full_indices()
        contract_indices, select_indices = self.__get_contract_select_rules(indices)

        contract_matrix = sy.zeros(len(indices), len(select_indices))
        for i, j in contract_indices:
            contract_matrix[i, j] = 1

        select_matrix = sy.zeros(len(indices), len(select_indices))
        for i, j in select_indices:
            select_matrix[i, j] = 1

        return select_matrix.T @ self.full_rep @ contract_matrix

    @property
    @lru_cache
    def basis(self) -> FrozenDict:
        transform = self.rep
        eig = transform.eigenvects()

        tbl = {}
        for v, _, vec_group in eig:
            vec = vec_group[0]
            # principle term is the first non-zero term
            principle_term = next(x for x in vec if x != 0)

            rep = vec / principle_term
            expr = sy.simplify(rep.dot(self.euclidean_basis))
            tbl[v] = PointGroupBasis(
                expr=expr, axes=self.axes, order=self.basis_function_order, rep=rep
            )

        return FrozenDict(tbl)


@dataclass(frozen=True)
class AbelianGroup(Operable):
    irrep: sy.ImmutableDenseMatrix
    axes: Tuple[sy.Symbol, ...]
    order: int

    @lru_cache
    def group_order(self, order: int):
        return AbelianGroupOrder(self.irrep, self.axes, order)

    @property
    @lru_cache
    def basis(self):
        tbl = {}
        for o in range(1, self.order):
            group_order = self.group_order(o)
            for k, v in group_order.basis.items():
                tbl.setdefault(k, v)

            if len(tbl) == self.order:
                break

        return FrozenDict(tbl)


@dispatch(AbelianGroup, PointGroupBasis)
def operator_mul(
    g: AbelianGroup, basis: PointGroupBasis
) -> Tuple[sy.Expr, PointGroupBasis]:
    if set(g.axes) != set(basis.axes):
        raise ValueError(
            f"Axes of AbelianGroup and PointGroupBasis must match: {g.axes} != {basis.axes}"
        )

    g_irrep = g.group_order(basis.order).rep
    basis_rep = basis.rep
    transformed_rep = g_irrep @ basis_rep

    phases = set()
    for n in range(transformed_rep.rows):
        if basis_rep[n] != 0:
            phases.add(sy.simplify(transformed_rep[n] / basis_rep[n]))
        else:
            if sy.simplify(transformed_rep[n]) != 0:
                raise ValueError(f"{basis} is not a basis function!")

    if not phases:
        raise ValueError(f"{basis} is a trivial basis function: zero")

    if len(phases) > 1:
        raise ValueError(f"{basis} is not a basis function!")

    return phases.pop(), basis
