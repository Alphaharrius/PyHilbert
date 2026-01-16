import pytest
import sympy as sy
from sympy import ImmutableDenseMatrix
from pyhilbert.spatials import Lattice, ReciprocalLattice, Offset, cartes, AffineSpace, AbstractLattice


def test_lattice_creation_and_dual():
    # 2D square lattice
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice = Lattice(basis=basis, shape=(2, 2))

    assert lattice.dim == 2
    assert lattice.shape == (2, 2)
    assert isinstance(lattice.affine, AffineSpace)
    assert isinstance(lattice, AbstractLattice)
    assert lattice.unit_cell == set()

    # Check dual
    reciprocal = lattice.dual
    assert isinstance(reciprocal, ReciprocalLattice)
    assert isinstance(reciprocal, AbstractLattice)
    assert reciprocal.dim == 2
    assert not hasattr(reciprocal, "unit_cell")

    # Check double dual gives back original lattice (scaled by 1/4pi^2 in this implementation)
    orig_basis = lattice.basis
    round_trip_basis = reciprocal.dual.basis

    assert round_trip_basis == orig_basis * (1 / (4 * sy.pi**2))


def test_lattice_with_unit_cell():
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    unit_cell = {(0,0), (0.5,0.5)}
    lattice = Lattice(basis=basis, shape=(2, 2), unit_cell=unit_cell)

    assert lattice.unit_cell == unit_cell

    # ReciprocalLattice should not accept unit_cell
    with pytest.raises(TypeError):
        ReciprocalLattice(basis=basis, shape=(2, 2), unit_cell=unit_cell)


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
