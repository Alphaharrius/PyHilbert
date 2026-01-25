import pytest
import numpy as np
import sympy as sy
from sympy import ImmutableDenseMatrix
from pyhilbert.spatials import Lattice, Offset
from pyhilbert.hilbert import hilbert, Mode, HilbertSpace
from pyhilbert.utils import FrozenDict


def test_lattice_scale():
    """
    Test Lattice.scale(M) correctly generates a supercell lattice
    and populates the unit cell with shifted atoms.
    """
    # 1. Define a simple square lattice
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    # Implicitly one atom at origin if unit_cell is empty
    lattice = Lattice(basis=basis, shape=(1, 1))

    # 2. Scale by doubling in X direction
    M = ImmutableDenseMatrix([[2, 0], [0, 1]])
    new_lattice = lattice.scale(M)

    # 3. Verify Basis: New basis should be M @ old_basis
    expected_basis = M @ basis
    assert new_lattice.basis == expected_basis

    # 4. Verify Unit Cell Population
    # Determinant is 2, so we expect 2 atoms in the new unit cell
    assert len(new_lattice.unit_cell) == 2

    # 5. Verify Atom Positions (Fractional coordinates in new basis)
    # Original atom at (0,0).
    # Shifts for M=[[2,0],[0,1]] are (0,0) and (1,0).
    # M_inv = [[0.5, 0], [0, 1]]
    # Pos 1: (0,0) @ M_inv = (0.0, 0.0)
    # Pos 2: (1,0) @ M_inv = (0.5, 0.0)

    positions = []
    for pos in new_lattice.unit_cell.values():
        positions.append(np.array(pos, dtype=float).flatten())

    # Check for presence of expected positions
    expected_positions = [np.array([0.0, 0.0]), np.array([0.5, 0.0])]

    for expected in expected_positions:
        found = any(np.allclose(p, expected) for p in positions)
        assert found, f"Expected position {expected} not found in {positions}"


def test_hilbert_scale():
    """
    Test HilbertSpace.scale(M) correctly expands the Hilbert space
    and updates mode positions.
    """
    # 1. Setup Lattice and Mode
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice = Lattice(basis=basis, shape=(1, 1))

    # Mode requires an Offset for 'r' to be scalable
    r_vec = ImmutableDenseMatrix([0, 0])
    offset = Offset(rep=r_vec, space=lattice)

    attr = FrozenDict({"orb": "s", "spin": "up", "r": offset})
    mode = Mode(count=1, attr=attr)

    hs = hilbert([mode])

    # 2. Scale by 2x2 matrix (Det = 4)
    M = ImmutableDenseMatrix([[2, 0], [0, 2]])
    new_hs = hs.scale(M)

    # 3. Verify Dimensions
    # Original dim 1 * Det 4 = 4
    assert new_hs.dim == 4
    assert len(new_hs.structure) == 4

    # 4. Verify Mode Attributes and Positions
    # Expected fractional coords in new supercell:
    # (0,0), (0, 0.5), (0.5, 0), (0.5, 0.5)
    expected_fracs = [
        (0.0, 0.0),
        (0.0, 0.5),
        (0.5, 0.0),
        (0.5, 0.5),
    ]

    found_fracs = []
    for m in new_hs.structure.keys():
        # Attributes should be preserved
        assert m.attr["orb"] == "s"
        assert m.attr["spin"] == "up"

        # Check position
        # m.r.rep contains the fractional coordinates in the new lattice basis
        rep = np.array(m['r'].rep, dtype=float).flatten()
        found_fracs.append(tuple(rep))

    # Verify all expected positions are present
    for ef in expected_fracs:
        found = False
        for ff in found_fracs:
            if np.allclose(ef, ff):
                found = True
                break
        assert found, f"Expected fraction {ef} not found in {found_fracs}"