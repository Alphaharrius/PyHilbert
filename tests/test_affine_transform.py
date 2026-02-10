import sympy as sy
from sympy import ImmutableDenseMatrix

from pyhilbert.affine_transform import (
    AffineFunction,
    AffineGroupElement,
    affine_transform,
)
from pyhilbert.spatials import AffineSpace, Offset


def _space_and_offset(dim: int):
    basis = ImmutableDenseMatrix.eye(dim)
    space = AffineSpace(basis=basis)
    offset = Offset(rep=ImmutableDenseMatrix([0] * dim), space=space)
    return space, offset


def test_affine_function_dim_and_str():
    x = sy.symbols("x")
    f = AffineFunction(
        expr=x,
        axes=(x,),
        order=1,
        rep=ImmutableDenseMatrix([1]),
    )
    assert f.dim == 1
    assert "AffineFunction(x)" in str(f)
    assert "AffineFunction(x)" in repr(f)


def test_affine_group_full_rep_kronecker_power():
    x, y = sy.symbols("x y")
    _, offset = _space_and_offset(2)
    irrep = ImmutableDenseMatrix([[1, 2], [0, 1]])
    t = AffineGroupElement(
        irrep=irrep, axes=(x, y), offset=offset, basis_function_order=2
    )
    expected = sy.kronecker_product(irrep, irrep)
    assert t.full_rep == expected


def test_affine_group_rep_shape_for_order_two():
    x, y = sy.symbols("x y")
    _, offset = _space_and_offset(2)
    irrep = ImmutableDenseMatrix([[1, 0], [0, 1]])
    t = AffineGroupElement(
        irrep=irrep, axes=(x, y), offset=offset, basis_function_order=2
    )
    # Monomials: x^2, x*y, y^2 -> 3 basis terms.
    assert t.rep.shape == (3, 3)


def test_affine_group_affine_rep_identity_basis():
    x, y = sy.symbols("x y")
    space, offset = _space_and_offset(2)
    offset = Offset(rep=ImmutableDenseMatrix([1, 2]), space=space)
    irrep = ImmutableDenseMatrix([[2, 0], [0, 3]])
    t = AffineGroupElement(
        irrep=irrep, axes=(x, y), offset=offset, basis_function_order=1
    )
    expected = ImmutableDenseMatrix([[2, 0, 1], [0, 3, 2], [0, 0, 1]])
    assert t.affine_rep == expected


def test_affine_group_affine_rep_non_identity_basis():
    x, y = sy.symbols("x y")
    basis = ImmutableDenseMatrix([[2, 0], [0, 1]])
    space = AffineSpace(basis=basis)
    offset = Offset(rep=ImmutableDenseMatrix([1, 1]), space=space)
    irrep = ImmutableDenseMatrix([[1, 0], [0, 2]])
    t = AffineGroupElement(
        irrep=irrep, axes=(x, y), offset=offset, basis_function_order=1
    )
    expected = ImmutableDenseMatrix([[1, 0, 2], [0, 2, 1], [0, 0, 1]])
    assert t.affine_rep == expected


def test_affine_group_rebase_changes_space_only():
    x, y = sy.symbols("x y")
    space, offset = _space_and_offset(2)
    irrep = ImmutableDenseMatrix([[1, 0], [0, 1]])
    t = AffineGroupElement(
        irrep=irrep, axes=(x, y), offset=offset, basis_function_order=1
    )

    new_space = AffineSpace(basis=ImmutableDenseMatrix([[2, 0], [0, 2]]))
    new_t = t.rebase(new_space)

    assert new_t.irrep == t.irrep
    assert new_t.axes == t.axes
    assert new_t.basis_function_order == t.basis_function_order
    assert new_t.offset.space == new_space


def test_affine_group_basis_keys_match_eigenvalues():
    x, y = sy.symbols("x y")
    _, offset = _space_and_offset(2)
    irrep = ImmutableDenseMatrix([[1, 0], [0, -1]])
    t = AffineGroupElement(
        irrep=irrep, axes=(x, y), offset=offset, basis_function_order=1
    )

    basis = t.basis
    assert set(basis.keys()) == {1, -1}
    for val, func in basis.items():
        assert isinstance(func, AffineFunction)
        assert func.axes == (x, y)
        assert func.order == 1
        assert t.rep @ func.rep == val * func.rep
        result = affine_transform(t, func)
        assert result.phase == val


def test_affine_transform_eigenfunction_phase():
    x = sy.symbols("x")
    space, offset = _space_and_offset(1)
    irrep = ImmutableDenseMatrix([[-1]])
    t = AffineGroupElement(
        irrep=irrep, axes=(x,), offset=offset, basis_function_order=1
    )
    f = AffineFunction(expr=x, axes=(x,), order=1, rep=ImmutableDenseMatrix([1]))
    result = affine_transform(t, f)
    assert result.phase == -1


def test_affine_transform_non_eigenfunction_raises():
    x, y = sy.symbols("x y")
    _, offset = _space_and_offset(2)
    irrep = ImmutableDenseMatrix([[1, 0], [0, -1]])
    t = AffineGroupElement(
        irrep=irrep, axes=(x, y), offset=offset, basis_function_order=1
    )
    f = AffineFunction(
        expr=x + y,
        axes=(x, y),
        order=1,
        rep=ImmutableDenseMatrix([1, 1]),
    )
    try:
        affine_transform(t, f)
        assert False, "Expected ValueError for non-eigenfunction."
    except ValueError:
        pass


def test_affine_transform_axes_mismatch_raises():
    x, y = sy.symbols("x y")
    space, offset = _space_and_offset(1)
    irrep = ImmutableDenseMatrix([[1]])
    t = AffineGroupElement(
        irrep=irrep, axes=(x,), offset=offset, basis_function_order=1
    )
    f = AffineFunction(expr=y, axes=(y,), order=1, rep=ImmutableDenseMatrix([1]))
    try:
        affine_transform(t, f)
        assert False, "Expected ValueError for axes mismatch."
    except ValueError:
        pass


def test_affine_transform_order_mismatch_rebuilds():
    x = sy.symbols("x")
    space, offset = _space_and_offset(1)
    irrep = ImmutableDenseMatrix([[2]])
    t = AffineGroupElement(
        irrep=irrep, axes=(x,), offset=offset, basis_function_order=1
    )
    f = AffineFunction(expr=x**2, axes=(x,), order=2, rep=ImmutableDenseMatrix([1]))
    result = affine_transform(t, f)
    assert result.phase == 4


def test_affine_transform_zero_basis_vector_raises():
    x = sy.symbols("x")
    space, offset = _space_and_offset(1)
    irrep = ImmutableDenseMatrix([[1]])
    t = AffineGroupElement(
        irrep=irrep, axes=(x,), offset=offset, basis_function_order=1
    )
    f = AffineFunction(expr=0, axes=(x,), order=1, rep=ImmutableDenseMatrix([0]))
    try:
        affine_transform(t, f)
        assert False, "Expected ValueError for zero basis vector."
    except ValueError:
        pass
