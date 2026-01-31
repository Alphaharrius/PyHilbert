import torch
from sympy import ImmutableDenseMatrix

from pyhilbert.spatials import Lattice, Offset
from pyhilbert.hilbert import brillouin_zone, hilbert, Mode
from pyhilbert.tensors import Tensor
from pyhilbert.transform import bandfold
from pyhilbert.utils import FrozenDict



def test_bandfold_1d():
    # 1. Setup
    # 1a. Define a 1D lattice with 4 k-points
    basis = ImmutableDenseMatrix([[1]])
    lattice = Lattice(basis=basis, shape=(4,))
    k_space = brillouin_zone(lattice.dual)
    assert k_space.dim == 4

    # 1b. Define a simple 1-dim Hilbert space
    r_offset = Offset(rep=ImmutableDenseMatrix([0]), space=lattice.affine)
    h_space = hilbert([Mode(count=1, attr=FrozenDict({"r": r_offset}))])
    assert h_space.dim == 1

    # 1c. Create an input tensor (4, 1, 1)
    # Data is just a sequence of numbers for easy tracking
    data = torch.arange(4, dtype=torch.float64).reshape(4, 1, 1)
    tensor_in = Tensor(data=data, dims=(k_space, h_space, h_space))

    # 1d. Define scaling matrix (double the unit cell)
    M = ImmutableDenseMatrix([[2]])

    # 2. Execute
    tensor_out = bandfold(M, tensor_in)

    # 3. Assert
    # 3a. Check new dimensions
    scaled_k_space = tensor_out.dims[0]
    new_h_space = tensor_out.dims[1]
    
    assert scaled_k_space.dim == 2 # 4 / det(M) = 4 / 2 = 2
    assert new_h_space.dim == 2 # 1 * det(M) = 1 * 2 = 2
    assert tensor_out.dims[2].dim == 2

    # 3b. Check the data
    # k=0 folds to k=0. k=1/2 folds to k=0.
    # k=1/4 folds to k=1/4. k=3/4 folds to k=1/4.
    # Original k-points: 0, 1/4, 1/2, 3/4
    # New k-points: 0, 1/2.
    
    # Check data for k_new=0 (index 0)
    # Maps k=0 (val 0) and k=1/2 (val 2)
    # Expected matrix: [[1, -1], [-1, 1]]
    expected_k0 = torch.tensor([[1, -1], [-1, 1]], dtype=torch.complex128)
    assert torch.allclose(tensor_out.data[0], expected_k0)

    # Check data for k_new=1/2 (index 1)
    # Maps k=1/4 (val 1) and k=3/4 (val 3)
    # Expected matrix: [[2, i], [-i, 2]]
    expected_k1 = torch.tensor([[2, 1j], [-1j, 2]], dtype=torch.complex128)
    assert torch.allclose(tensor_out.data[1], expected_k1)


def test_bandfold_2d():
    # 1. Setup
    # 1a. Define a 2D lattice with 4 k-points (2x2)
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice = Lattice(basis=basis, shape=(2, 2))
    k_space = brillouin_zone(lattice.dual)
    assert k_space.dim == 4

    # 1b. Define a simple Hilbert space
    r_offset = Offset(rep=ImmutableDenseMatrix([0, 0]), space=lattice.affine)
    h_space = hilbert([Mode(count=1, attr=FrozenDict({"orb": "s", "r": r_offset}))])
    assert h_space.dim == 1

    # 1c. Create input tensor (4, 1, 1)
    # Data: 0, 1, 2, 3
    data = torch.arange(4, dtype=torch.float64).reshape(4, 1, 1)
    tensor_in = Tensor(data=data, dims=(k_space, h_space, h_space))

    # 1d. Define scaling matrix (double in both directions)
    M = ImmutableDenseMatrix([[2, 0], [0, 2]])

    # 2. Execute
    tensor_out = bandfold(M, tensor_in)

    # 3. Assert
    # 3a. Check dimensions
    # New lattice shape: (2//2, 2//2) = (1, 1) -> 1 k-point
    # New Hilbert dim: 1 * det(M) = 4
    scaled_k_space = tensor_out.dims[0]
    new_h_space = tensor_out.dims[1]

    assert scaled_k_space.dim == 1
    assert new_h_space.dim == 4
    assert tensor_out.dims[2].dim == 4

    # 3b. Check data
    # All 4 k-points fold to the single Gamma point.
    # Expected matrix derived from folding 0, 1, 2, 3
    # Basis order: (0,0), (0,1), (1,0), (1,1)
    expected_matrix = torch.tensor([
        [ 1.5, -0.5, -1.0,  0.0],
        [-0.5,  1.5,  0.0, -1.0],
        [-1.0,  0.0,  1.5, -0.5],
        [ 0.0, -1.0, -0.5,  1.5]
    ], dtype=torch.float64)
    
    assert torch.allclose(tensor_out.data[0].real, expected_matrix)
    assert torch.allclose(tensor_out.data[0].imag, torch.zeros_like(expected_matrix))
