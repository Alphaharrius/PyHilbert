import pytest
import sympy as sy
from sympy import ImmutableDenseMatrix
from pyhilbert.spatials import PointGroupBasis, AbelianGroup
from pyhilbert.utils import FrozenDict


def test_point_group_basis_creation():
    # Define a simple basis function x
    x, y = sy.symbols("x y")
    expr = x
    axes = (x, y)
    order = 1
    rep = ImmutableDenseMatrix([1, 0])  # Vector representation of x in (x, y) basis

    pgb = PointGroupBasis(expr=expr, axes=axes, order=order, rep=rep)

    assert pgb.dim == 2
    assert str(pgb) == "PointGroupBasis(x)"
    assert repr(pgb) == "PointGroupBasis(x)"


def test_abelian_group_c2_symmetry():
    # C2 symmetry: x -> -x, y -> -y
    # Irrep matrix for C2 generator: [[-1, 0], [0, -1]]

    x, y = sy.symbols("x y")
    axes = (x, y)

    # C2 group has order 2. Generator is C2.
    irrep = ImmutableDenseMatrix([[-1, 0], [0, -1]])

    # To cover degree 2 (quadratic terms), we need order=3 in the library constructor
    g_c2 = AbelianGroup(irrep=irrep, axes=axes, order=3)
    bases = g_c2.basis

    assert isinstance(bases, FrozenDict)

    # We expect eigenvalues +1 and -1
    # Order 1 polynomials (x, y) have eigenvalue -1.
    # Order 2 polynomials (x^2, xy, y^2) have eigenvalue +1.

    has_even = False
    has_odd = False

    for eig, func in bases.items():
        if eig == 1:
            has_even = True
        elif eig == -1:
            has_odd = True

    assert has_even and has_odd


def test_operator_mul_symmetry():
    x, y = sy.symbols("x y")
    axes = (x, y)

    # C2 symmetry
    irrep = ImmutableDenseMatrix([[-1, 0], [0, -1]])
    g = AbelianGroup(irrep=irrep, axes=axes, order=2)

    # Create a manual basis function 'x' which is odd (-1) under C2
    pgb_x = PointGroupBasis(
        expr=x,
        axes=axes,
        order=1,
        rep=ImmutableDenseMatrix([1, 0]),  # x=1*x + 0*y
    )

    # Apply group operation: g * pgb_x
    phase, basis = g * pgb_x

    assert basis == pgb_x
    assert phase == -1

    # Create a manual basis function 'x^2' which is even (+1) under C2
    # Vector length 3. [1, 0, 0] corresponds to x^2 in the sorted basis used by library
    pgb_x2 = PointGroupBasis(
        expr=x**2, axes=axes, order=2, rep=ImmutableDenseMatrix([1, 0, 0])
    )

    phase, basis = g * pgb_x2
    assert basis == pgb_x2
    assert phase == 1


def test_operator_mul_invalid_basis():
    x, y = sy.symbols("x y")
    axes = (x, y)

    # C4: x->y, y->-x. Matrix [[0, -1], [1, 0]]
    irrep_c4 = ImmutableDenseMatrix([[0, -1], [1, 0]])
    g_c4 = AbelianGroup(irrep=irrep_c4, axes=axes, order=4)

    pgb_x = PointGroupBasis(
        expr=x, axes=axes, order=1, rep=ImmutableDenseMatrix([1, 0])
    )

    # g * x -> y. y != phase * x. Should raise ValueError.
    with pytest.raises(ValueError, match="is not a basis function"):
        _ = g_c4 * pgb_x
