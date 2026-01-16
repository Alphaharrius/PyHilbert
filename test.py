import sympy as sy
from sympy import ImmutableDenseMatrix
from pyhilbert.spatials import Lattice, ReciprocalLattice, Offset, cartes, AffineSpace

M = ImmutableDenseMatrix([[2, 0], [0, 2]])

lattice = Lattice(basis=ImmutableDenseMatrix([[1, 0], [0, 1]]), shape=(1, 1))

nlattice, off = lattice.scale(M)
print(off)
lattice.plot("structure", backend="matplotlib", save_path="original_lattice.png")
nlattice.plot("structure", backend="matplotlib", save_path="scaled_lattice.png")