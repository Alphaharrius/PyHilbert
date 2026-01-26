import numpy as np
import torch
from sympy import ImmutableDenseMatrix
from pyhilbert.spatials import Lattice, Offset
from pyhilbert.hilbert import hilbert, Mode, brillouin_zone
from pyhilbert.tensors import Tensor
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
        rep = np.array(m["r"].rep, dtype=float).flatten()
        found_fracs.append(tuple(rep))

    # Verify all expected positions are present
    for ef in expected_fracs:
        found = False
        for ff in found_fracs:
            if np.allclose(ef, ff):
                found = True
                break
        assert found, f"Expected fraction {ef} not found in {found_fracs}"


def test_tensor_scale():
    """
    Test Tensor.scale(M) for band folding of a 1D tight-binding chain.
    """
    # 1. Setup 1D Lattice
    a = 1.0
    basis = ImmutableDenseMatrix([[a]])
    lattice = Lattice(basis=basis, shape=(4,))  # 4 unit cells
    recip = lattice.dual

    # 2. Setup MomentumSpace (4 points)
    # cartes(recip) gives k points: 0, 1/4, 2/4, 3/4 in fractional recip coords
    k_space = brillouin_zone(recip)
    assert k_space.dim == 4

    # 3. Setup HilbertSpace (1 orbital)
    r_vec = ImmutableDenseMatrix([0.0])
    offset = Offset(rep=r_vec, space=lattice)
    mode = Mode(count=1, attr=FrozenDict({"orb": "s", "r": offset}))
    h_space = hilbert([mode])

    # 4. Create Hamiltonian Tensor
    # H(k) = -2t cos(k*a) -> E = -2, 0, 2, 0
    energies = torch.tensor([-2.0, 0.0, 2.0, 0.0], dtype=torch.float64).reshape(4, 1, 1)
    H = Tensor(data=energies, dims=(k_space, h_space, h_space))

    # 5. Scale by M=[[2]]
    M = ImmutableDenseMatrix([[2]])
    H_super = H.scale(M)

    # 6. Verify Dimensions
    # New MomentumSpace should have 2 points
    # New HilbertSpace should have 2 modes
    assert len(H_super.dims) == 3
    new_k_space = H_super.dims[0]
    new_h_space = H_super.dims[1]

    assert new_k_space.dim == 2
    assert new_h_space.dim == 2

    # 7. Verify Eigenvalues at each k-point
    # k_new = 0 (folds k_old=0 and k_old=0.5 -> E=-2, 2)
    block_0 = H_super.data[0]  # 2x2 matrix
    evals_0 = torch.linalg.eigvalsh(block_0)
    assert torch.allclose(evals_0, torch.tensor([-2.0, 2.0], dtype=torch.float64))

    # k_new = 0.5 (folds k_old=0.25 and k_old=0.75 -> E=0, 0)
    block_1 = H_super.data[1]
    evals_1 = torch.linalg.eigvalsh(block_1)
    assert torch.allclose(
        evals_1, torch.tensor([0.0, 0.0], dtype=torch.float64), atol=1e-7
    )


def test_tensor_scale_2d():
    """
    Test Tensor.scale(M) for band folding of a 2D square lattice tight-binding model.
    """
    # 1. Setup 2D Lattice
    basis = ImmutableDenseMatrix([[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(basis=basis, shape=(2, 2))
    recip = lattice.dual

    # 2. Setup MomentumSpace (4 points: 0,0; 0.5,0; 0,0.5; 0.5,0.5)
    k_space = brillouin_zone(recip)
    assert k_space.dim == 4

    # 3. Setup HilbertSpace (1 orbital)
    r_vec = ImmutableDenseMatrix([0.0, 0.0])
    offset = Offset(rep=r_vec, space=lattice)
    mode = Mode(count=1, attr=FrozenDict({"orb": "s", "r": offset}))
    h_space = hilbert([mode])

    # 4. Create Hamiltonian Tensor
    # H(k) = -2t (cos(kx*a) + cos(ky*a))
    # k points are fractional. k_cart = k_frac * b_i.
    # k.r = k_frac * 2pi.
    # E = -2 * (cos(2pi*kx) + cos(2pi*ky))

    # Extract k-points to compute energies
    k_fracs = []
    for k in k_space.elements():
        k_fracs.append(np.array(k.rep, dtype=float).flatten())
    k_fracs = np.array(k_fracs)  # (4, 2)

    kx = k_fracs[:, 0]
    ky = k_fracs[:, 1]

    # t = 1.0
    energies = -2.0 * (np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky))
    # Expected: -4, 0, 0, 4 (order depends on k_space iteration, but set is same)

    energies_tensor = torch.tensor(energies, dtype=torch.float64).reshape(4, 1, 1)
    H = Tensor(data=energies_tensor, dims=(k_space, h_space, h_space))

    # 5. Scale by M=[[2, 0], [0, 2]]
    # This folds all 4 points into Gamma (0,0) of the supercell BZ
    M = ImmutableDenseMatrix([[2, 0], [0, 2]])
    H_super = H.scale(M)

    # 6. Verify Dimensions
    # New MomentumSpace should have 1 point (Gamma)
    # New HilbertSpace should have 4 modes (1 * det(M))
    assert len(H_super.dims) == 3
    new_k_space = H_super.dims[0]
    new_h_space = H_super.dims[1]

    assert new_k_space.dim == 1
    assert new_h_space.dim == 4

    # 7. Verify Eigenvalues
    # The single block at Gamma should contain all original energies
    block = H_super.data[0]  # 4x4 matrix
    evals = torch.linalg.eigvalsh(block)

    expected_evals = torch.tensor([-4.0, 0.0, 0.0, 4.0], dtype=torch.float64)
    # Sort both to compare
    evals_sorted, _ = torch.sort(evals)
    expected_sorted, _ = torch.sort(expected_evals)

    assert torch.allclose(evals_sorted, expected_sorted, atol=1e-7)
