import pytest
import sympy as sy
import torch
from sympy import ImmutableDenseMatrix
from pyhilbert.spatials import Lattice


basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
unit_cell = {(0, 0)}
lattice = Lattice(basis=basis, shape=(1, 1), unit_cell=unit_cell)
M = ImmutableDenseMatrix([[2, 0], [0, 2]])
new = lattice.scale(M)
lattice.plot("structure",backend="matplotlib",save_path="lattice_plot_old.png")
new.plot("structure",backend="matplotlib",save_path="lattice_plot.png")