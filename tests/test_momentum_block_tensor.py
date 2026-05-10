from collections import OrderedDict

import sympy as sy
import torch
from sympy import ImmutableDenseMatrix

from qten.geometries.boundary import PeriodicBoundary
from qten.geometries.spatials import Lattice
from qten.linalg._mb_tensor import (
    MomentumBlockTensor,
)
from qten.linalg.tensors import Tensor
from qten.symbolics.hilbert_space import HilbertSpace, U1Basis
from qten.symbolics.state_space import (
    BroadcastSpace,
    MomentumBlockSpace,
    MomentumSpace,
    brillouin_zone,
)


def _band_space(name: str, size: int) -> HilbertSpace:
    return HilbertSpace.new(
        U1Basis(coef=sy.Integer(1), base=((name, i),)) for i in range(size)
    )


def _k_space() -> MomentumSpace:
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    return brillouin_zone(lattice.dual)


def _pair_space(*pairs) -> MomentumBlockSpace:
    return MomentumBlockSpace(
        structure=OrderedDict((pair, i) for i, pair in enumerate(pairs))
    )


def test_momentum_block_transpose_updates_pair_axis_when_band_dims_match():
    k_space = _k_space()
    k0, k1 = k_space.elements()
    band = _band_space("band", 2)
    pair_space = _pair_space((k0, k1), (k1, k0))

    tensor = MomentumBlockTensor(
        data=torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=torch.complex128,
        ),
        dims=(pair_space, band, band),
    )

    transposed = tensor.transpose(1, 2)

    assert isinstance(transposed, MomentumBlockTensor)
    assert tuple(transposed.dims[0].elements()) == ((k1, k0), (k0, k1))
    assert transposed.dims[1:] == (band, band)
    assert torch.allclose(transposed.data, tensor.data.transpose(1, 2))


def test_momentum_block_right_matmul_uses_pair_second_momentum():
    k_space = _k_space()
    k0, k1 = k_space.elements()
    left_band = _band_space("left", 2)
    mid_band = _band_space("mid", 2)
    right_band = _band_space("right", 2)
    pair_space = _pair_space((k0, k0), (k1, k0), (k1, k1))

    left = MomentumBlockTensor(
        data=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 0.0], [0.0, 2.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ],
            dtype=torch.complex128,
        ),
        dims=(pair_space, left_band, mid_band),
    )
    right = Tensor(
        data=torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=torch.complex128,
        ),
        dims=(k_space, mid_band, right_band),
    )

    out = left @ right
    expected = torch.stack(
        [
            left.data[0] @ right.data[0],
            left.data[1] @ right.data[0],
            left.data[2] @ right.data[1],
        ]
    )

    assert isinstance(out, MomentumBlockTensor)
    assert out.dims == (pair_space, left_band, right_band)
    assert torch.allclose(out.data, expected)


def test_momentum_block_left_matmul_uses_pair_first_momentum():
    k_space = _k_space()
    k0, k1 = k_space.elements()
    left_band = _band_space("left", 2)
    mid_band = _band_space("mid", 2)
    right_band = _band_space("right", 2)
    pair_space = _pair_space((k0, k1), (k0, k0), (k1, k0))

    left = Tensor(
        data=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 2.0]],
                [[0.0, 1.0], [3.0, 0.0]],
            ],
            dtype=torch.complex128,
        ),
        dims=(k_space, left_band, mid_band),
    )
    right = MomentumBlockTensor(
        data=torch.tensor(
            [
                [[1.0, 2.0], [0.0, 1.0]],
                [[2.0, 0.0], [1.0, 2.0]],
                [[0.0, 1.0], [4.0, 0.0]],
            ],
            dtype=torch.complex128,
        ),
        dims=(pair_space, mid_band, right_band),
    )

    out = left @ right
    expected = torch.stack(
        [
            left.data[0] @ right.data[0],
            left.data[0] @ right.data[1],
            left.data[1] @ right.data[2],
        ]
    )

    assert isinstance(out, MomentumBlockTensor)
    assert out.dims == (pair_space, left_band, right_band)
    assert torch.allclose(out.data, expected)


def test_momentum_block_matmul_accumulates_duplicate_off_diagonal_pairs():
    k_space = _k_space()
    k0, k1 = k_space.elements()
    left_band = _band_space("left", 2)
    mid_band = _band_space("mid", 2)
    right_band = _band_space("right", 2)
    left_pairs = _pair_space((k0, k0), (k0, k1))
    right_pairs = _pair_space((k0, k1), (k1, k1))

    left = MomentumBlockTensor(
        data=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 0.0], [0.0, 2.0]],
            ],
            dtype=torch.complex128,
        ),
        dims=(left_pairs, left_band, mid_band),
    )
    right = MomentumBlockTensor(
        data=torch.tensor(
            [
                [[1.0, 1.0], [0.0, 1.0]],
                [[0.0, 2.0], [3.0, 0.0]],
            ],
            dtype=torch.complex128,
        ),
        dims=(right_pairs, mid_band, right_band),
    )

    out = left @ right
    expected = left.data[0] @ right.data[0] + left.data[1] @ right.data[1]

    assert isinstance(out, MomentumBlockTensor)
    assert tuple(out.dims[0].elements()) == ((k0, k1),)
    assert torch.allclose(out.data[0], expected)


def test_momentum_block_matmul_collapses_diagonal_output_to_momentum_space():
    k_space = _k_space()
    k0, k1 = k_space.elements()
    left_band = _band_space("left", 2)
    mid_band = _band_space("mid", 2)
    right_band = _band_space("right", 2)
    left_pairs = _pair_space((k0, k0), (k0, k1))
    right_pairs = _pair_space((k0, k0), (k1, k0))

    left = MomentumBlockTensor(
        data=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ],
            dtype=torch.complex128,
        ),
        dims=(left_pairs, left_band, mid_band),
    )
    right = MomentumBlockTensor(
        data=torch.tensor(
            [
                [[2.0, 0.0], [0.0, 2.0]],
                [[1.0, 3.0], [4.0, 0.0]],
            ],
            dtype=torch.complex128,
        ),
        dims=(right_pairs, mid_band, right_band),
    )

    out = left @ right
    expected = left.data[0] @ right.data[0] + left.data[1] @ right.data[1]

    assert isinstance(out, Tensor)
    assert not isinstance(out, MomentumBlockTensor)
    assert isinstance(out.dims[0], MomentumSpace)
    assert tuple(out.dims[0].elements()) == (k0,)
    assert torch.allclose(out.data[0], expected)


def test_momentum_block_add_preserves_subtype_via_generic_tensor_add():
    k_space = _k_space()
    k0, k1 = k_space.elements()
    band = _band_space("band", 2)
    pair_space = _pair_space((k0, k1), (k1, k0))

    left = MomentumBlockTensor(
        data=torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=torch.complex128,
        ),
        dims=(pair_space, band, band),
    )
    right = MomentumBlockTensor(
        data=torch.tensor(
            [
                [[0.5, 1.0], [1.5, 2.0]],
                [[2.5, 3.0], [3.5, 4.0]],
            ],
            dtype=torch.complex128,
        ),
        dims=(pair_space, band, band),
    )

    out = left + right

    assert isinstance(out, MomentumBlockTensor)
    assert out.dims == (pair_space, band, band)
    assert torch.allclose(out.data, left.data + right.data)


def test_momentum_block_mean_downgrades_to_plain_tensor():
    k_space = _k_space()
    k0, k1 = k_space.elements()
    left_band = _band_space("left", 2)
    right_band = _band_space("right", 3)
    pair_space = _pair_space((k0, k1), (k1, k0))
    tensor = MomentumBlockTensor(
        data=torch.randn(
            pair_space.dim, left_band.dim, right_band.dim, dtype=torch.float64
        ),
        dims=(pair_space, left_band, right_band),
    )

    out = tensor.mean(dim=0)

    assert isinstance(out, Tensor)
    assert not isinstance(out, MomentumBlockTensor)
    assert out.dims == (left_band, right_band)
    assert torch.allclose(out.data, tensor.data.mean(dim=0))


def test_momentum_block_all_keepdim_downgrades_to_plain_tensor():
    k_space = _k_space()
    k0, k1 = k_space.elements()
    band = _band_space("band", 2)
    pair_space = _pair_space((k0, k1), (k1, k0))
    tensor = MomentumBlockTensor(
        data=torch.tensor(
            [
                [[True, False], [True, True]],
                [[True, True], [False, True]],
            ]
        ),
        dims=(pair_space, band, band),
    )

    out = tensor.all(dim=1, keepdim=True)

    assert isinstance(out, Tensor)
    assert not isinstance(out, MomentumBlockTensor)
    assert out.dims[0] == pair_space
    assert isinstance(out.dims[1], BroadcastSpace)
    assert out.dims[2] == band
    assert torch.equal(out.data, torch.all(tensor.data, dim=1, keepdim=True))


def test_momentum_block_unsqueeze_downgrades_to_plain_tensor():
    k_space = _k_space()
    k0, k1 = k_space.elements()
    band = _band_space("band", 2)
    pair_space = _pair_space((k0, k1), (k1, k0))
    tensor = MomentumBlockTensor(
        data=torch.randn(pair_space.dim, band.dim, band.dim, dtype=torch.float64),
        dims=(pair_space, band, band),
    )

    out = tensor.unsqueeze(0)

    assert isinstance(out, Tensor)
    assert not isinstance(out, MomentumBlockTensor)
    assert isinstance(out.dims[0], BroadcastSpace)
    assert out.dims[1:] == (pair_space, band, band)
    assert torch.allclose(out.data, tensor.data.unsqueeze(0))


def test_momentum_block_argmax_downgrades_to_plain_tensor():
    k_space = _k_space()
    k0, k1 = k_space.elements()
    left_band = _band_space("left", 2)
    right_band = _band_space("right", 3)
    pair_space = _pair_space((k0, k1), (k1, k0))
    tensor = MomentumBlockTensor(
        data=torch.randn(
            pair_space.dim, left_band.dim, right_band.dim, dtype=torch.float64
        ),
        dims=(pair_space, left_band, right_band),
    )

    out = tensor.argmax(0)

    assert isinstance(out, Tensor)
    assert not isinstance(out, MomentumBlockTensor)
    assert out.rank() == 2
    assert out.dims == (left_band, right_band)
    assert torch.equal(out.data, tensor.data.argmax(dim=0))
