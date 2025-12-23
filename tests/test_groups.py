import pytest
import sympy as sy
from sympy import ImmutableDenseMatrix
from pyhilbert.spatials import PointGroupBasis, AbelianGroup, operator_mul
from pyhilbert.utils import FrozenDict

def test_point_group_basis_creation():
    # Define a simple basis function x
    x, y = sy.symbols('x y')
    expr = x
    axes = (x, y)
    order = 1
    rep = ImmutableDenseMatrix([1, 0]) # Vector representation of x in (x, y) basis
    
    pgb = PointGroupBasis(expr=expr, axes=axes, order=order, rep=rep)
    
    assert pgb.dim == 2
    assert str(pgb) == "PointGroupBasis(x)"
    assert repr(pgb) == "PointGroupBasis(x)"

def test_abelian_group_c2_symmetry():
    # C2 symmetry: x -> -x, y -> -y
    # Irrep matrix for C2 generator: [[-1, 0], [0, -1]]
    
    x, y = sy.symbols('x y')
    axes = (x, y)
    
    # C2 group has order 2. Generator is C2.
    # Group elements: E, C2.
    # We define the group by its generator's representation on the fundamental axes.
    irrep = ImmutableDenseMatrix([[-1, 0], [0, -1]])
    
    # Create AbelianGroup
    # The 'order' parameter here seems to represent the order of the group (number of elements)?
    # Or the order of the tensor representation?
    # Looking at the code: AbelianGroup(..., order=int).
    # .basis iterates range(1, self.order). This implies self.order is group order?
    # Actually, let's look at AbelianGroup.basis implementation.
    # It calls group_order(o).
    # And AbelianGroupOrder uses self.basis_function_order.
    # This suggests 'order' in AbelianGroup might be something else or the iteration is over tensor powers?
    
    # Let's inspect AbelianGroup logic deeper.
    # group_order(o) creates AbelianGroupOrder with basis_function_order=o.
    # This generates basis functions of polynomial order 'o'.
    # e.g. o=1 -> x, y
    # o=2 -> x^2, xy, y^2
    
    # So 'order' in AbelianGroup seems to be the MAX polynomial order to generate bases for?
    # Let's assume we want up to quadratic terms.
    g = AbelianGroup(irrep=irrep, axes=axes, order=3) 
    
    # Test basis generation
    basis_dict = g.basis
    assert isinstance(basis_dict, FrozenDict)
    
    # For C2: x->-x, y->-y.
    # Order 1: x -> -1*x (odd), y -> -1*y (odd).
    # Order 2: x^2 -> (-x)^2 = x^2 (even), xy -> (-x)(-y) = xy (even), y^2 -> y^2 (even).
    
    # The code calculates eigenvectors of the transformation.
    # For order 1: matrix is diagonal [-1, -1]. Eigenvalues -1.
    # Basis functions should be x and y (or linear combs).
    
    # Check what we got
    # Keys are eigenvalues?
    # The code: tbl[v] = ... where v is eigenvalue.
    # Since multiple basis functions can have same eigenvalue, this might overwrite?
    # Look at code:
    # for k, v in group_order.basis.items(): tbl.setdefault(k, v)
    # This means it keeps the FIRST basis found for each eigenvalue across orders.
    # Wait, 'if len(tbl) == self.order: break'.
    # This implies 'order' is the number of distinct representations (charges) we want to find?
    # For C2, we have two irreps: A (even, +1) and B (odd, -1).
    # So if we set order=2, we expect to find one even and one odd function.
    
    # range(1, order) in library means order is exclusive limit for polynomial degree.
    # To cover degree 2, we need order=3.
    g_c2 = AbelianGroup(irrep=irrep, axes=axes, order=3)
    bases = g_c2.basis
    
    # We expect eigenvalues +1 and -1 (if they exist in the polynomial spaces checked)
    # Order 1 polynomials (x, y) have eigenvalue -1.
    # Order 2 polynomials (x^2, xy, y^2) have eigenvalue +1.
    
    # Eigenvalues in sympy might be complex or specific types, let's just check keys.
    keys = list(bases.keys())
    # Should have -1 and 1.
    
    has_even = False
    has_odd = False
    
    for eig, func in bases.items():
        if eig == 1:
            has_even = True
            # Expect quadratic
            # print(f"Even func: {func}")
        elif eig == -1:
            has_odd = True
            # Expect linear
            # print(f"Odd func: {func}")
            
    assert has_even and has_odd

def test_operator_mul_symmetry():
    x, y = sy.symbols('x y')
    axes = (x, y)
    
    # C2 symmetry
    irrep = ImmutableDenseMatrix([[-1, 0], [0, -1]])
    g = AbelianGroup(irrep=irrep, axes=axes, order=2)
    
    # Create a manual basis function 'x' which is odd (-1) under C2
    # Order 1
    pgb_x = PointGroupBasis(
        expr=x, 
        axes=axes, 
        order=1, 
        rep=ImmutableDenseMatrix([1, 0]) # x=1*x + 0*y
    )
    
    # Apply group operation: g * pgb_x
    # Should return (eigenvalue, pgb_x)
    # Eigenvalue should be -1
    phase, basis = g * pgb_x
    
    assert basis == pgb_x
    assert phase == -1
    
    # Create a manual basis function 'x^2' which is even (+1) under C2
    # Order 2
    # The library commutes indices, so x*y == y*x. 
    # For axes (x, y) order 2, the basis is sorted: x^2, xy, y^2 (or similar sorted order).
    # Full indices: (x,x), (x,y), (y,x), (y,y)
    # Commute indices: (x,x), (x,y), (y,y). Length 3.
    # x^2 corresponds to the first element if sorted by name?
    # x<y. (x,x) -> x^2. (x,y) -> xy. (y,y) -> y^2.
    # Vector length 3. [1, 0, 0] corresponds to x^2.
    
    pgb_x2 = PointGroupBasis(
        expr=x**2,
        axes=axes,
        order=2,
        rep=ImmutableDenseMatrix([1, 0, 0])
    )
    
    phase, basis = g * pgb_x2
    assert basis == pgb_x2
    assert phase == 1

def test_operator_mul_invalid_basis():
    x, y = sy.symbols('x y')
    axes = (x, y)
    irrep = ImmutableDenseMatrix([[-1, 0], [0, -1]])
    g = AbelianGroup(irrep=irrep, axes=axes, order=2)
    
    # Create a function x + x^2. This mixes orders, so can't be represented easily with current class structure 
    # as PointGroupBasis takes a single 'order'.
    # Instead, let's create a mixed symmetry state within same order if possible.
    # For C2, everything is either +1 or -1, so hard to mix in 1D.
    # But let's try a basis that is NOT an eigenvector.
    # e.g. for a rotation C4 (90 deg), x -> y.
    # Then x is not an eigenvector (x != lambda * y).
    
    # C4: x->y, y->-x. Matrix [[0, -1], [1, 0]]
    irrep_c4 = ImmutableDenseMatrix([[0, -1], [1, 0]])
    g_c4 = AbelianGroup(irrep=irrep_c4, axes=axes, order=4)
    
    pgb_x = PointGroupBasis(
        expr=x,
        axes=axes,
        order=1,
        rep=ImmutableDenseMatrix([1, 0])
    )
    
    # g * x -> y. y != phase * x. Should raise ValueError.
    with pytest.raises(ValueError, match="is not a basis function"):
        _ = g_c4 * pgb_x

