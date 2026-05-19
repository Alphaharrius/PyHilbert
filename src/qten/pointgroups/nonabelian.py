"""
Non-abelian finite-group projection helpers.

This module adds a minimal non-abelian symmetry workflow alongside the existing
abelian utilities. It provides:

- `FiniteGroupRepresentation`: exact matrix representation of a finite group.
- `nonabelian_isotypic_projectors`: character-projector construction.
- `nonabelian_column_symmetrize`: projector-based column decomposition for tensors.

The implementation is intentionally conservative and does not alter abelian
APIs. It is designed as an incremental extension point for block (matrix-valued)
symmetry sectors.
"""

from dataclasses import dataclass
from typing import Dict, Hashable, Mapping, Tuple

import sympy as sy

from ..geometries.spatials import Spatial
from ..linalg.tensors import Tensor, cat
from ..symbolics import HilbertSpace, IndexSpace, U1Basis
from ..utils.collections_ext import FrozenDict


def _normalize_projector_basis(
    vec: sy.ImmutableDenseMatrix,
) -> sy.ImmutableDenseMatrix:
    first_non_zero = next((entry for entry in vec if entry != 0), None)
    if first_non_zero is None:
        return vec
    return sy.ImmutableDenseMatrix(sy.simplify(vec / first_non_zero))


@dataclass(frozen=True)
class NonAbelianSectorBasis(Spatial):
    """
    Label for a non-abelian isotypic sector.

    Attributes
    ----------
    irrep : str
        Name of the target irrep sector (for example `"A1"` or `"E"`).
    """

    irrep: str

    @property
    def dim(self) -> int:
        """Scalar symbolic label dimension."""
        return 1

    def __str__(self) -> str:
        return self.irrep

    def __repr__(self) -> str:
        return self.irrep


@dataclass(frozen=True, init=False)
class FiniteGroupRepresentation:
    """
    Exact finite-group matrix representation.

    Parameters
    ----------
    matrices : Mapping[Hashable, sy.MatrixBase]
        Mapping from group element label to representation matrix `D(g)`.
    identity : Hashable
        Label of the identity element in `matrices`.
    """

    matrices: FrozenDict[Hashable, sy.ImmutableDenseMatrix]
    identity: Hashable

    def __init__(
        self,
        matrices: Mapping[Hashable, sy.MatrixBase | sy.ImmutableDenseMatrix],
        identity: Hashable,
    ) -> None:
        if not matrices:
            raise ValueError("matrices must be non-empty.")

        normalized: Dict[Hashable, sy.ImmutableDenseMatrix] = {}
        shape: Tuple[int, int] | None = None
        for element, matrix in matrices.items():
            immutable = sy.ImmutableDenseMatrix(matrix)
            if immutable.rows != immutable.cols:
                raise ValueError(
                    f"Matrix for element {element!r} must be square, got "
                    f"{immutable.shape}."
                )
            if shape is None:
                shape = immutable.shape
            elif immutable.shape != shape:
                raise ValueError(
                    "All matrices must share the same shape, got "
                    f"{shape} and {immutable.shape}."
                )
            normalized[element] = immutable

        if identity not in normalized:
            raise ValueError(
                f"identity={identity!r} must be present in matrices keys."
            )

        object.__setattr__(self, "matrices", FrozenDict(normalized))
        object.__setattr__(self, "identity", identity)

    @property
    def dim(self) -> int:
        """Representation dimension."""
        return next(iter(self.matrices.values())).rows

    @property
    def order(self) -> int:
        """Number of represented group elements."""
        return len(self.matrices)

    @property
    def elements(self) -> Tuple[Hashable, ...]:
        """Tuple of represented element labels in insertion order."""
        return tuple(self.matrices.keys())

    def matrix(self, element: Hashable) -> sy.ImmutableDenseMatrix:
        """Return matrix `D(element)`."""
        try:
            return self.matrices[element]
        except KeyError as exc:
            raise KeyError(f"Unknown group element: {element!r}.") from exc

    def isotypic_projector(
        self,
        *,
        irrep_dim: int,
        characters: Mapping[Hashable, sy.Expr],
    ) -> sy.ImmutableDenseMatrix:
        r"""
        Build the isotypic projector from a character row.

        For a finite group representation `D(g)`, this computes

        `P = (d/|G|) * sum_g conj(chi(g)) * D(g)`,

        where `d` is the irrep dimension and `chi` is the irrep character.
        """
        if irrep_dim < 1:
            raise ValueError(f"irrep_dim must be positive, got {irrep_dim}.")

        element_set = set(self.matrices.keys())
        if set(characters.keys()) != element_set:
            raise ValueError(
                "characters keys must exactly match representation elements."
            )

        projector = sy.zeros(self.dim, self.dim)
        for element, matrix in self.matrices.items():
            projector += sy.conjugate(sy.simplify(characters[element])) * matrix

        scale = sy.Rational(irrep_dim, self.order)
        return sy.ImmutableDenseMatrix(sy.simplify(scale * projector))

    def isotypic_projectors(
        self,
        character_table: Mapping[str, tuple[int, Mapping[Hashable, sy.Expr]]],
    ) -> FrozenDict[str, sy.ImmutableDenseMatrix]:
        """
        Build all isotypic projectors from a character table.

        Parameters
        ----------
        character_table : Mapping[str, tuple[int, Mapping[Hashable, sy.Expr]]]
            Mapping `irrep_name -> (irrep_dim, characters)`.
        """
        if not character_table:
            raise ValueError("character_table must be non-empty.")

        projectors = {
            irrep: self.isotypic_projector(irrep_dim=irrep_dim, characters=characters)
            for irrep, (irrep_dim, characters) in character_table.items()
        }
        return FrozenDict(projectors)

    @staticmethod
    def projector_basis(
        projector: sy.ImmutableDenseMatrix,
    ) -> Tuple[sy.ImmutableDenseMatrix, ...]:
        """
        Return a canonical basis of the projector image.
        """
        vectors = projector.columnspace()
        canonical = []
        for vec in vectors:
            rep = sy.ImmutableDenseMatrix(vec)
            rep = _normalize_projector_basis(rep)
            if all(entry == 0 for entry in rep):
                continue
            canonical.append(rep)
        return tuple(canonical)


def nonabelian_isotypic_projectors(
    representation: FiniteGroupRepresentation,
    character_table: Mapping[str, tuple[int, Mapping[Hashable, sy.Expr]]],
) -> FrozenDict[str, sy.ImmutableDenseMatrix]:
    """
    Convenience wrapper for `FiniteGroupRepresentation.isotypic_projectors`.
    """
    return representation.isotypic_projectors(character_table)


def _attach_sector_label(seed: U1Basis | None, irrep: str) -> U1Basis:
    label = NonAbelianSectorBasis(irrep=irrep)
    if seed is None:
        return U1Basis.new(label)
    try:
        return seed.replace(label)
    except ValueError:
        return U1Basis(coef=seed.coef, base=seed.base + (label,))


def _attach_degeneracy_tag(seed: U1Basis, index: int) -> U1Basis:
    try:
        return seed.replace(index)
    except ValueError:
        return U1Basis(coef=seed.coef, base=seed.base + (index,))


def nonabelian_column_symmetrize(
    projectors: Mapping[str, Tensor],
    w: Tensor,
    full_sector: bool = False,
) -> Tensor:
    """
    Project tensor columns into non-abelian isotypic sectors.

    Parameters
    ----------
    projectors : Mapping[str, Tensor]
        Mapping `irrep_name -> projector Tensor` acting on `w.dims[0]`.
    w : Tensor
        Rank-2 tensor of columns to project.
    full_sector : bool, default False
        If `True`, return all non-zero sector components per input column.
        If `False`, keep only the largest non-zero sector component per column.
    """
    if not projectors:
        raise ValueError("projectors must be non-empty.")
    if w.rank() != 2:
        raise ValueError("w must be a rank-2 tensor of ambient-space columns.")

    row_dim = w.dims[0]
    if not isinstance(row_dim, HilbertSpace):
        raise ValueError("w.dims[0] must be a HilbertSpace.")

    input_col_dim = w.dims[1]
    seeds: list[U1Basis | None]
    if isinstance(input_col_dim, HilbertSpace):
        seeds = list(input_col_dim.elements())
    elif isinstance(input_col_dim, IndexSpace):
        seeds = [None] * input_col_dim.dim
    else:
        raise ValueError("w.dims[1] must be either an IndexSpace or a HilbertSpace.")

    first_projector = next(iter(projectors.values()))
    if first_projector.rank() != 2:
        raise ValueError("each projector must be rank-2.")
    if first_projector.dims[0] != row_dim or first_projector.dims[1] != row_dim:
        raise ValueError("projector dims must both match w.dims[0].")

    for irrep, projector in projectors.items():
        if projector.rank() != 2:
            raise ValueError(f"projector for {irrep!r} must be rank-2.")
        if projector.dims[0] != row_dim or projector.dims[1] != row_dim:
            raise ValueError(
                f"projector dims for {irrep!r} must both match w.dims[0]."
            )

    single_col = IndexSpace.linear(1)
    tol = 1e-10

    projected_cols: list[Tensor] = []
    raw_labels: list[U1Basis] = []
    for j, seed in enumerate(seeds):
        col = w[:, j : j + 1].clone().replace_dim(1, single_col)
        candidates: list[tuple[float, Tensor, U1Basis]] = []
        for irrep, projector in projectors.items():
            projected = projector @ col
            projected_norm = projected.norm()
            norm_value = float(projected_norm.item())
            if norm_value <= tol:
                continue
            candidates.append(
                (
                    norm_value,
                    projected / norm_value,
                    _attach_sector_label(seed, irrep),
                )
            )

        if full_sector:
            for _, projected, label in candidates:
                projected_cols.append(projected)
                raw_labels.append(label)
        elif candidates:
            _, projected, label = max(candidates, key=lambda item: item[0])
            projected_cols.append(projected)
            raw_labels.append(label)

    if not projected_cols:
        return Tensor(
            data=w.data.new_empty((row_dim.dim, 0), dtype=w.data.dtype),
            dims=(row_dim, IndexSpace.linear(0)),
        )

    totals: dict[U1Basis, int] = {}
    for label in raw_labels:
        totals[label] = totals.get(label, 0) + 1

    seen: dict[U1Basis, int] = {}
    labels: list[U1Basis] = []
    for label in raw_labels:
        idx = seen.get(label, 0)
        seen[label] = idx + 1
        if totals[label] > 1:
            labels.append(_attach_degeneracy_tag(label, idx))
        else:
            labels.append(label)

    out_dim = HilbertSpace.new(labels)
    return cat(projected_cols, dim=-1).replace_dim(-1, out_dim)
