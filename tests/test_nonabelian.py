import sympy as sy
import torch
from sympy import ImmutableDenseMatrix

from qten.linalg.tensors import Tensor
from qten.pointgroups.nonabelian import (
    FiniteGroupRepresentation,
    NonAbelianSectorBasis,
    nonabelian_column_symmetrize,
    nonabelian_isotypic_projectors,
)
from qten.symbolics.hilbert_space import HilbertSpace, U1Basis
from qten.symbolics.state_space import IndexSpace


def _perm_matrix(perm: tuple[int, int, int]) -> ImmutableDenseMatrix:
    data = [[0, 0, 0] for _ in range(3)]
    for src, dst in enumerate(perm):
        data[dst][src] = 1
    return ImmutableDenseMatrix(data)


def _s3_representation() -> FiniteGroupRepresentation:
    # Permutations act on the 3-site basis e_i -> e_{perm(i)}.
    matrices = {
        "e": _perm_matrix((0, 1, 2)),
        "(12)": _perm_matrix((1, 0, 2)),
        "(13)": _perm_matrix((2, 1, 0)),
        "(23)": _perm_matrix((0, 2, 1)),
        "(123)": _perm_matrix((1, 2, 0)),
        "(132)": _perm_matrix((2, 0, 1)),
    }
    return FiniteGroupRepresentation(matrices=matrices, identity="e")


def _s3_character_table():
    one = sy.Integer(1)
    minus_one = sy.Integer(-1)
    zero = sy.Integer(0)
    two = sy.Integer(2)
    return {
        "trivial": (
            1,
            {
                "e": one,
                "(12)": one,
                "(13)": one,
                "(23)": one,
                "(123)": one,
                "(132)": one,
            },
        ),
        "sign": (
            1,
            {
                "e": one,
                "(12)": minus_one,
                "(13)": minus_one,
                "(23)": minus_one,
                "(123)": one,
                "(132)": one,
            },
        ),
        "standard": (
            2,
            {
                "e": two,
                "(12)": zero,
                "(13)": zero,
                "(23)": zero,
                "(123)": minus_one,
                "(132)": minus_one,
            },
        ),
    }


def _sympy_to_torch(matrix: ImmutableDenseMatrix) -> torch.Tensor:
    return torch.tensor(
        [[complex(sy.N(matrix[i, j])) for j in range(matrix.cols)] for i in range(matrix.rows)],
        dtype=torch.complex128,
    )


def test_s3_isotypic_projectors_split_permutation_representation():
    rep = _s3_representation()
    projectors = nonabelian_isotypic_projectors(rep, _s3_character_table())

    p_trivial = projectors["trivial"]
    p_sign = projectors["sign"]
    p_standard = projectors["standard"]
    ident = ImmutableDenseMatrix.eye(rep.dim)

    assert sy.simplify(p_trivial.trace() - 1) == 0
    assert sy.simplify(p_sign.trace()) == 0
    assert sy.simplify(p_standard.trace() - 2) == 0

    assert sy.simplify(p_trivial @ p_trivial - p_trivial) == sy.zeros(3, 3)
    assert sy.simplify(p_standard @ p_standard - p_standard) == sy.zeros(3, 3)
    assert sy.simplify(p_sign @ p_sign - p_sign) == sy.zeros(3, 3)

    assert sy.simplify(p_trivial @ p_standard) == sy.zeros(3, 3)
    assert sy.simplify(p_trivial @ p_sign) == sy.zeros(3, 3)
    assert sy.simplify(p_standard @ p_sign) == sy.zeros(3, 3)

    assert sy.simplify(p_trivial + p_sign + p_standard - ident) == sy.zeros(3, 3)


def test_nonabelian_column_symmetrize_labels_projected_sectors():
    rep = _s3_representation()
    projectors_sym = nonabelian_isotypic_projectors(rep, _s3_character_table())

    row_space = HilbertSpace.new([U1Basis.new("s0"), U1Basis.new("s1"), U1Basis.new("s2")])
    projectors = {
        name: Tensor(
            data=_sympy_to_torch(matrix),
            dims=(row_space, row_space),
        )
        for name, matrix in projectors_sym.items()
    }

    w = Tensor(
        data=torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.complex128),
        dims=(row_space, IndexSpace.linear(1)),
    )

    full = nonabelian_column_symmetrize(projectors, w, full_sector=True)
    labels_full = list(full.dims[1].elements())
    irreps_full = {label.irrep_of(NonAbelianSectorBasis).irrep for label in labels_full}
    assert irreps_full == {"trivial", "standard"}
    assert full.data.shape == (3, 2)

    dominant = nonabelian_column_symmetrize(projectors, w, full_sector=False)
    labels_dominant = list(dominant.dims[1].elements())
    assert len(labels_dominant) == 1
    assert labels_dominant[0].irrep_of(NonAbelianSectorBasis).irrep == "standard"
