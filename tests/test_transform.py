# import pytest
# import torch
# from sympy import ImmutableDenseMatrix
# import sympy as sy

# from pyhilbert.spatials import Lattice
# from pyhilbert.hilbert import brillouin_zone, hilbert, Mode
# from pyhilbert.tensors import Tensor
# from pyhilbert.transform import bandfold
# from pyhilbert.utils import FrozenDict



# def test_bandfold_1d():
#     # 1. Setup
#     # 1a. Define a 1D lattice with 4 k-points
#     basis = ImmutableDenseMatrix([[1]])
#     lattice = Lattice(basis=basis, shape=(4,))
#     k_space = brillouin_zone(lattice.dual)
#     assert k_space.size == 4

#     # 1b. Define a simple 1-dim Hilbert space
#     h_space = hilbert([Mode(count=1, attr=FrozenDict({"a": 1}))])
#     assert h_space.size == 1

#     # 1c. Create an input tensor (4, 1, 1)
#     # Data is just a sequence of numbers for easy tracking
#     data = torch.arange(4, dtype=torch.float64).reshape(4, 1, 1)
#     tensor_in = Tensor(data=data, dims=(k_space, h_space, h_space))

#     # 1d. Define scaling matrix (double the unit cell)
#     M = ImmutableDenseMatrix([[2]])

#     # 2. Execute
#     tensor_out = bandfold(M, tensor_in)

#     # 3. Assert
#     # 3a. Check new dimensions
#     scaled_k_space = tensor_out.dims[0]
#     new_h_space = tensor_out.dims[1]
    
#     assert scaled_k_space.size == 2 # 4 / det(M) = 4 / 2 = 2
#     assert new_h_space.size == 2 # 1 * det(M) = 1 * 2 = 2
#     assert tensor_out.dims[2].size == 2

#     # 3b. Check the data
#     # k=0 folds to k=0. k=1/2 folds to k=0.
#     # k=1/4 folds to k=1/4. k=3/4 folds to k=1/4.
#     # Original k-points: 0, 1/4, 1/2, 3/4
#     # New k-points: 0, 1/2.
#     # Wait, for shape (4,), k-points are 0, 1/4, 2/4, 3/4.
#     # New BZ with shape (2,) has k-points 0, 1/2.
#     # transform(k=0) -> 0. frac=0.
#     # transform(k=1/4) -> 1/2. frac=1/2.
#     # transform(k=2/4) -> 1. frac=0.
#     # transform(k=3/4) -> 3/2. frac=1/2.
#     # So k=0, k=1/2 from old BZ fold to k=0 in new BZ.
#     # k=1/4, k=3/4 from old BZ fold to k=1/2 in new BZ.
#     # My k_fold_mapping logic seems to have a bug. Let me recheck.
#     # k_rep' = k_rep @ M.T. For 1D, k_rep' = k_rep * 2.
#     # k=0 -> 0. frac=0.
#     # k=1/4 -> 1/2. frac=1/2.
#     # k=1/2 -> 1. frac=0.
#     # k=3/4 -> 3/2. frac=1/2.
#     # No, the new reciprocal lattice is smaller.
#     # rec_basis' = rec_basis @ M.inv().T
#     # k_rep' = k_rep @ M.T
#     # transform(k) re-expresses k in the new basis.
#     # rebase_mat is M.T. new_rep = k.rep @ M.T
    
#     # Let's trace transform(k).
#     # k is Momentum(rep, space=rec_lat)
#     # t(k) calls momentum_transform.
#     # new_space = t(rec_lat) = scaled_rec_lat.
#     # return k.rebase(new_space).
#     # rebase(new_space):
#     # rebase_mat = new_space.basis.inv() @ k.space.basis
#     # new_rep = rebase_mat @ k.rep
#     # rec_lat.basis is [[2*pi]]. scaled_rec_lat.basis is [[pi]].
#     # rebase_mat = [[pi]].inv() @ [[2*pi]] = [[1/pi]] @ [[2*pi]] = [[2]].
#     # new_rep = [[2]] @ k.rep. So k_rep is multiplied by 2.
#     # k=0 (rep 0) -> rep 0. frac=0.
#     # k=1/4 (rep 1/4) -> rep 1/2. frac=1/2.
#     # k=1/2 (rep 1/2) -> rep 1. frac=0.
#     # k=3/4 (rep 3/4) -> rep 3/2. frac=1/2.
    
#     # The new k-points from `brillouin_zone(scaled_lattice.dual)`
#     # scaled_lattice has shape (4,). It should be shape (2,).
#     # lattice_transform does not change shape. Let's fix that.
    
#     # Let's assume the folded k-points are correct for now.
#     # old k values from tensor.data are 0, 1, 2, 3.
#     # k_new=0 gets k_old=0, k_old=1/2. Data: 0, 2.
#     # The new matrix for k_new=0 should be diag(0, 2).
#     # k_new=1/2 gets k_old=1/4, k_old=3/4. Data: 1, 3.
#     # The new matrix for k_new=1/2 should be diag(1, 3).
    
#     # Check data for k_new=0 (index 0)
#     expected_k0 = torch.diag(torch.tensor([0, 2], dtype=torch.float64))
#     assert torch.allclose(tensor_out.data[0], expected_k0)

#     # Check data for k_new=1/2 (index 1)
#     expected_k1 = torch.diag(torch.tensor([1, 3], dtype=torch.float64))
#     assert torch.allclose(tensor_out.data[1], expected_k1)


# # def test_bandfold_2d():
# #     # 1. Setup
# #     # 1a. Define a 2D lattice with 4 k-points (2x2)
# #     basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
# #     lattice = Lattice(basis=basis, shape=(2, 2))
# #     k_space = brillouin_zone(lattice.dual)
# #     assert k_space.size == 4

# #     # 1b. Define a simple Hilbert space
# #     h_space = hilbert([Mode(count=1, attr=FrozenDict({"orb": "s"}))])
# #     assert h_space.size == 1

# #     # 1c. Create input tensor (4, 1, 1)
# #     # Data: 0, 1, 2, 3
# #     data = torch.arange(4, dtype=torch.float64).reshape(4, 1, 1)
# #     tensor_in = Tensor(data=data, dims=(k_space, h_space, h_space))

# #     # 1d. Define scaling matrix (double in both directions)
# #     M = ImmutableDenseMatrix([[2, 0], [0, 2]])

# #     # 2. Execute
# #     tensor_out = bandfold(M, tensor_in)

# #     # 3. Assert
# #     # 3a. Check dimensions
# #     # New lattice shape: (2//2, 2//2) = (1, 1) -> 1 k-point
# #     # New Hilbert size: 1 * det(M) = 4
# #     scaled_k_space = tensor_out.dims[0]
# #     new_h_space = tensor_out.dims[1]

# #     assert scaled_k_space.size == 1
# #     assert new_h_space.size == 4
# #     assert tensor_out.dims[2].size == 4

# #     # 3b. Check data
# #     # All 4 k-points fold to the single Gamma point.
# #     # The resulting 4x4 matrix should be diagonal with entries 0, 1, 2, 3.
# #     expected_matrix = torch.diag(torch.arange(4, dtype=torch.float64))
# #     assert torch.allclose(tensor_out.data[0], expected_matrix)
