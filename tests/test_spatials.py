import pytest
import sympy as sy
import torch
from sympy import ImmutableDenseMatrix
from pyhilbert.spatials import (
    Lattice,
    ReciprocalLattice,
    Offset,
    cartes,
    AffineSpace,
    AbstractLattice,
)
from pyhilbert.utils import FrozenDict


def test_lattice_creation_and_dual():
    # 2D square lattice
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice = Lattice(basis=basis, shape=(2, 2))

    assert lattice.dim == 2
    assert lattice.shape == (2, 2)
    assert isinstance(lattice.affine, AffineSpace)
    assert isinstance(lattice, AbstractLattice)
    assert isinstance(lattice.unit_cell, FrozenDict)
    assert len(lattice.unit_cell) == 1
    assert lattice.unit_cell["r"].rep == ImmutableDenseMatrix([0, 0])

    # Check dual
    reciprocal = lattice.dual
    assert isinstance(reciprocal, ReciprocalLattice)
    assert isinstance(reciprocal, AbstractLattice)
    assert reciprocal.dim == 2
    assert not hasattr(reciprocal, "unit_cell")

    # Check double dual gives back original lattice (scaled by 1/4pi^2 in this implementation)
    orig_basis = lattice.basis
    round_trip_basis = reciprocal.dual.basis

    assert round_trip_basis == orig_basis


def test_lattice_with_unit_cell():
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    unit_cell_input = {"a": (0, 0), "b": (sy.Rational(1, 2), sy.Rational(1, 2))}
    lattice = Lattice(basis=basis, shape=(2, 2), unit_cell=unit_cell_input)

    assert isinstance(lattice.unit_cell, FrozenDict)
    assert len(lattice.unit_cell) == 2
    assert isinstance(lattice.unit_cell["a"], Offset)
    assert lattice.unit_cell["a"].rep == ImmutableDenseMatrix([0, 0])
    assert isinstance(lattice.unit_cell["b"], Offset)
    assert lattice.unit_cell["b"].rep == ImmutableDenseMatrix(
        [sy.Rational(1, 2), sy.Rational(1, 2)]
    )

    # ReciprocalLattice should not accept unit_cell
    with pytest.raises(TypeError):
        ReciprocalLattice(basis=basis, shape=(2, 2), unit_cell=unit_cell_input)


def test_cartes_lattice():
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice = Lattice(basis=basis, shape=(2, 2))

    # cartes should return offsets for (0,0), (0,1), (1,0), (1,1)
    points = cartes(lattice)
    assert len(points) == 4
    assert isinstance(points[0], Offset)

    # Check content of points
    coords = set()
    for p in points:
        coords.add(tuple(p.rep))

    assert (0, 0) in coords
    assert (0, 1) in coords
    assert (1, 0) in coords
    assert (1, 1) in coords


def test_cartes_reciprocal_lattice():
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    # shape (2, 2)
    lattice = Lattice(basis=basis, shape=(2, 2))
    reciprocal = lattice.dual

    points = cartes(reciprocal)
    assert len(points) == 4

    coords = set()
    for p in points:
        # p.rep should be (n/2, m/2)
        coords.add(tuple(p.rep))

    assert (0, 0) in coords
    assert (sy.Rational(1, 2), 0) in coords
    assert (0, sy.Rational(1, 2)) in coords
    assert (sy.Rational(1, 2), sy.Rational(1, 2)) in coords


def test_coords():
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    # Default unit cell (empty -> one atom at origin)
    lattice = Lattice(basis=basis, shape=(2, 2))
    coords = lattice.coords()
    assert coords.shape == (4, 2)

    # Explicit unit cell
    unit_cell = {"a": (sy.Rational(1, 10), sy.Rational(1, 10))}
    lattice_offset = Lattice(basis=basis, shape=(2, 2), unit_cell=unit_cell)
    coords_offset = lattice_offset.coords()
    assert coords_offset.shape == (4, 2)

    # Check that (0.1, 0.1) is in the coordinates (corresponding to cell 0,0)
    expected = torch.tensor([0.1, 0.1], dtype=torch.float64)
    assert torch.any(torch.all(torch.isclose(coords_offset, expected), dim=1))
