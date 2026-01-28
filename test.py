from pyhilbert.spatials import Lattice, Offset, Momentum
from pyhilbert.transform import BasisTransform
from sympy import ImmutableDenseMatrix
import sympy as sy
from pyhilbert.spatials import Lattice
from pyhilbert.hilbert import brillouin_zone, hilbert, Mode
from pyhilbert.tensors import Tensor
from pyhilbert.transform import bandfold
from pyhilbert.utils import FrozenDict
import torch
import matplotlib.pyplot as plt

M = ImmutableDenseMatrix([[2, 0], [0, 2]])
transform = BasisTransform(M)
basis = ImmutableDenseMatrix([[1, 0], [0, 1]])


square_lattice = Lattice(basis=basis, shape=(2, 2))
# square_lattice.plot("structure")
scaled_lattice = transform(square_lattice)
# scaled_lattice.plot("structure")
# print(scaled_lattice.unit_cell)
reciprocal_lattice = square_lattice.dual
# # print(reciprocal_lattice)

scaled_reciprocal = transform(reciprocal_lattice)
# # print(scaled_reciprocal)

r_vec = ImmutableDenseMatrix([-sy.Rational(1/4), 1])
offset = Offset(rep=r_vec, space=square_lattice)
# print(offset.fractional())
transformed_offset = transform(offset)

# # print(transformed_offset)

k_vec = ImmutableDenseMatrix([sy.Rational(1, 2), sy.Rational(1)])
momentum = Momentum(rep=k_vec, space=reciprocal_lattice)
# print(momentum.space)
# print(momentum.fractional())
transformed_momentum = transform(momentum)
# print(transformed_momentum.fractional())


# 1     # 1. Setup
# 1a. Define a 2D lattice with 4 k-points (2x2)
basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
lattice = Lattice(basis=basis, shape=(2, 2))
k_space = brillouin_zone(lattice.dual)
# 1b. Define a simple Hilbert space
h_space = hilbert([Mode(count=1, attr=FrozenDict({"orb": "s", "r" : r})) for r in lattice.unit_cell.values()])

# 1c. Create input tensor (4, 1, 1)
# Data: 0, 1, 2, 3
data = torch.arange(4, dtype=torch.float64).reshape(4, 1, 1)
tensor_in = Tensor(data=data, dims=(k_space, h_space, h_space))

M = ImmutableDenseMatrix([[2, 0], [0, 2]])

# tensor_out = bandfold(M, tensor_in)


# Test with flat band
flat_data = torch.ones(4, dtype=torch.float64).reshape(4, 1, 1)
flat_tensor_in = Tensor(data=flat_data, dims=(k_space, h_space, h_space))
flat_tensor_out = bandfold(M, flat_tensor_in)
# print("Flat band output:\n", flat_tensor_out.data)

# Test with alternating band (+a, -a)
a = 1.0
alt_data = torch.tensor([a, -a, a, -a], dtype=torch.float64).reshape(4, 1, 1)
alt_tensor_in = Tensor(data=alt_data, dims=(k_space, h_space, h_space))
alt_tensor_out = bandfold(M, alt_tensor_in)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(alt_tensor_in.data.flatten(), 'o', label='Original')
ax1.set_title("Original Band Structure")
print(f"Before folding: {torch.linalg.eigvalsh(alt_tensor_in.data).T}")
# print(torch.linalg.eigvalsh(alt_tensor_in.data))
# ax1.plot(torch.zeros_like(evals), evals, 'o', label='Original')
# print(alt_tensor_out.data)
evals = torch.linalg.eigvalsh(alt_tensor_out.data)
print(f"After folding: {evals}")
ax2.plot(torch.zeros_like(evals), evals, 'o', label='Folded')
ax2.set_title("Folded Band Structure (at Gamma)")
plt.show()