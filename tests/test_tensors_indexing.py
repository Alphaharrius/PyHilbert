import pytest
import torch
from itertools import product

from pyhilbert.tensors import Tensor, where
from pyhilbert.state_space import BroadcastSpace, IndexSpace


class TestTensorAdvancedGetitem:
    def test_getitem_rejects_multiple_ellipsis_in_normal_indexing(self):
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)
        data = torch.arange(6, dtype=torch.float64).reshape(2, 3)
        tensor = Tensor(data=data, dims=(a, b))

        with pytest.raises(IndexError, match="single ellipsis"):
            _ = tensor[..., ...]

    def test_getitem_with_tensor_advanced_index_contiguous(self):
        a = IndexSpace.linear(3)
        b = IndexSpace.linear(4)
        data = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        tensor = Tensor(data=data, dims=(a, b))

        idx = Tensor(
            data=torch.tensor([2, 0], dtype=torch.long), dims=(IndexSpace.linear(2),)
        )
        out = tensor[idx, :]

        expected = data[idx.data, :]
        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (IndexSpace.linear(2), b)

    def test_getitem_with_tensor_advanced_index_separated(self):
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)
        c = IndexSpace.linear(4)
        data = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
        tensor = Tensor(data=data, dims=(a, b, c))

        i = Tensor(
            data=torch.tensor([1, 0], dtype=torch.long), dims=(IndexSpace.linear(2),)
        )
        j = Tensor(
            data=torch.tensor([3, 1], dtype=torch.long), dims=(IndexSpace.linear(2),)
        )
        out = tensor[i, :, j]

        expected = data[i.data, :, j.data]
        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (IndexSpace.linear(2), b)

    def test_getitem_with_tensor_advanced_index_and_none_contiguous(self):
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)
        c = IndexSpace.linear(4)
        d = IndexSpace.linear(5)
        data = torch.arange(120, dtype=torch.float64).reshape(2, 3, 4, 5)
        tensor = Tensor(data=data, dims=(a, b, c, d))

        i = Tensor(
            data=torch.tensor([1, 0], dtype=torch.long), dims=(IndexSpace.linear(2),)
        )
        j = Tensor(
            data=torch.tensor([3, 1], dtype=torch.long), dims=(IndexSpace.linear(2),)
        )
        out = tensor[:, i, j, None, :]

        expected = data[:, i.data, j.data, None, :]
        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (a, IndexSpace.linear(2), BroadcastSpace(), d)

    def test_getitem_with_tensor_advanced_index_and_none_separated(self):
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)
        c = IndexSpace.linear(4)
        data = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
        tensor = Tensor(data=data, dims=(a, b, c))

        i = Tensor(
            data=torch.tensor([1, 0], dtype=torch.long), dims=(IndexSpace.linear(2),)
        )
        j = Tensor(
            data=torch.tensor([3, 1], dtype=torch.long), dims=(IndexSpace.linear(2),)
        )
        out = tensor[None, i, :, j]

        expected = data[None, i.data, :, j.data]
        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (IndexSpace.linear(2), BroadcastSpace(), b)

    def test_getitem_with_tensor_advanced_index_separated_by_none(self):
        a = IndexSpace.linear(5)
        b = IndexSpace.linear(3)
        c = IndexSpace.linear(4)
        sel = IndexSpace.linear(2)
        data = torch.arange(60, dtype=torch.float64).reshape(5, 3, 4)
        tensor = Tensor(data=data, dims=(a, b, c))

        i = Tensor(data=torch.tensor([1, 0], dtype=torch.long), dims=(sel,))
        j = Tensor(data=torch.tensor([3, 1], dtype=torch.long), dims=(sel,))
        out = tensor[:, i, None, j]

        expected = data[:, i.data, None, j.data]
        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        # `None` separates advanced indices, so torch uses separated layout:
        # advanced dims are moved to the front.
        assert out.dims == (sel, a, BroadcastSpace())

    def test_getitem_with_tensor_advanced_index_rejects_invalid_mix(self):
        a = IndexSpace.linear(3)
        b = IndexSpace.linear(4)
        tensor = Tensor(
            data=torch.arange(12, dtype=torch.float64).reshape(3, 4), dims=(a, b)
        )

        state_index = IndexSpace.linear(1)
        tensor_idx = Tensor(
            data=torch.tensor([0], dtype=torch.long), dims=(IndexSpace.linear(1),)
        )
        with pytest.raises(ValueError, match="cannot be mixed"):
            _ = tensor[state_index, tensor_idx]

    def test_getitem_with_tensor_advanced_index_rejects_int_and_non_full_slice(self):
        a = IndexSpace.linear(3)
        b = IndexSpace.linear(4)
        data = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        tensor = Tensor(data=data, dims=(a, b))
        idx = Tensor(
            data=torch.tensor([1, 0], dtype=torch.long), dims=(IndexSpace.linear(2),)
        )

        with pytest.raises(TypeError, match="only supports Tensor indices"):
            _ = tensor[idx, 1]

        with pytest.raises(TypeError, match="only supports Tensor indices"):
            _ = tensor[idx, 1:3]

    def test_getitem_with_tensor_advanced_index_preserves_broadcasted_dims(self):
        row_src = IndexSpace.linear(4)
        col_src = IndexSpace.linear(5)
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)

        data = torch.arange(20, dtype=torch.float64).reshape(4, 5)
        tensor = Tensor(data=data, dims=(row_src, col_src))

        i = Tensor(
            data=torch.tensor([[0], [3]], dtype=torch.long),
            dims=(a, BroadcastSpace()),
        )
        j = Tensor(
            data=torch.tensor([[1, 4, 2]], dtype=torch.long),
            dims=(BroadcastSpace(), b),
        )

        out = tensor[i, j]
        expected = data[i.data, j.data]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (a, b)

    def test_getitem_with_tensor_advanced_index_raises_for_dim_conflict(self):
        row_src = IndexSpace.linear(4)
        col_src = IndexSpace.linear(5)
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(2)
        c = IndexSpace.linear(2)

        data = torch.arange(20, dtype=torch.float64).reshape(4, 5)
        tensor = Tensor(data=data, dims=(row_src, col_src))

        i = Tensor(
            data=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
            dims=(a, b),
        )
        j = Tensor(
            data=torch.tensor([[1, 0], [4, 2]], dtype=torch.long),
            dims=(a, c.map(lambda n: n + 10)),
        )

        with pytest.raises(ValueError, match="incompatible for broadcast"):
            _ = tensor[i, j]

    def test_getitem_with_three_advanced_indices_broadcasted(self):
        xdim = IndexSpace.linear(5)
        ydim = IndexSpace.linear(6)
        zdim = IndexSpace.linear(7)
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(4)
        c = IndexSpace.linear(3)

        data = torch.arange(5 * 6 * 7, dtype=torch.float64).reshape(5, 6, 7)
        tensor = Tensor(data=data, dims=(xdim, ydim, zdim))

        i = Tensor(
            data=torch.tensor(
                [[[0, 1, 2]], [[4, 3, 2]]],
                dtype=torch.long,
            ),
            dims=(a, BroadcastSpace(), c),
        )
        j = Tensor(
            data=torch.tensor(
                [[[0], [1], [2], [3]]],
                dtype=torch.long,
            ),
            dims=(BroadcastSpace(), b, BroadcastSpace()),
        )
        k = Tensor(
            data=torch.tensor(
                [
                    [[0, 1, 2], [3, 4, 5], [6, 0, 1], [2, 3, 4]],
                    [[5, 6, 0], [1, 2, 3], [4, 5, 6], [0, 1, 2]],
                ],
                dtype=torch.long,
            ),
            dims=(a, b, c),
        )

        out = tensor[i, j, k]
        expected = data[i.data, j.data, k.data]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (a, b, c)

    def test_getitem_with_higher_rank_advanced_indices(self):
        row_src = IndexSpace.linear(4)
        col_src = IndexSpace.linear(5)
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)
        c = IndexSpace.linear(2)

        data = torch.arange(20, dtype=torch.float64).reshape(4, 5)
        tensor = Tensor(data=data, dims=(row_src, col_src))

        i = Tensor(
            data=torch.tensor(
                [[[0], [3], [1]], [[2], [1], [0]]],
                dtype=torch.long,
            ),
            dims=(a, b, BroadcastSpace()),
        )
        j = Tensor(
            data=torch.tensor(
                [[[0, 1]], [[3, 4]]],
                dtype=torch.long,
            ),
            dims=(a, BroadcastSpace(), c),
        )

        out = tensor[i, j]
        expected = data[i.data, j.data]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (a, b, c)

    def test_getitem_with_tensor_advanced_bool_mask(self):
        src = IndexSpace.linear(5)
        keep = IndexSpace.linear(3)

        data = torch.arange(5, dtype=torch.float64)
        tensor = Tensor(data=data, dims=(src,))

        mask = Tensor(
            data=torch.tensor([True, False, True, False, True], dtype=torch.bool),
            dims=(src,),
        )
        out = tensor[mask]
        expected = data[mask.data]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (keep,)

    def test_getitem_with_tensor_advanced_ellipsis_at_end(self):
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)
        c = IndexSpace.linear(4)
        d = IndexSpace.linear(5)
        sel = IndexSpace.linear(2)

        data = torch.arange(2 * 3 * 4 * 5, dtype=torch.float64).reshape(2, 3, 4, 5)
        tensor = Tensor(data=data, dims=(a, b, c, d))

        i = Tensor(data=torch.tensor([1, 0], dtype=torch.long), dims=(sel,))
        j = Tensor(data=torch.tensor([3, 1], dtype=torch.long), dims=(sel,))

        out = tensor[..., i, j]
        expected = data[..., i.data, j.data]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (a, b, sel)

    def test_getitem_with_tensor_advanced_ellipsis_middle_separated(self):
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)
        c = IndexSpace.linear(4)
        d = IndexSpace.linear(5)
        sel = IndexSpace.linear(2)

        data = torch.arange(2 * 3 * 4 * 5, dtype=torch.float64).reshape(2, 3, 4, 5)
        tensor = Tensor(data=data, dims=(a, b, c, d))

        i = Tensor(data=torch.tensor([1, 0], dtype=torch.long), dims=(sel,))
        j = Tensor(data=torch.tensor([3, 1], dtype=torch.long), dims=(sel,))

        out = tensor[i, ..., j]
        expected = data[i.data, ..., j.data]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (sel, b, c)

    def test_getitem_with_tensor_advanced_ellipsis_none_then_index(self):
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)
        c = IndexSpace.linear(4)
        d = IndexSpace.linear(5)
        sel = IndexSpace.linear(2)

        data = torch.arange(2 * 3 * 4 * 5, dtype=torch.float64).reshape(2, 3, 4, 5)
        tensor = Tensor(data=data, dims=(a, b, c, d))

        # Value 4 is valid for axis `d` (size 5), but invalid for axis `c` (size 4).
        # If `...` is expanded too short when `None` is present, this incorrectly
        # targets axis `c` and raises IndexError.
        idx = Tensor(data=torch.tensor([4, 1], dtype=torch.long), dims=(sel,))
        out = tensor[..., None, idx]
        expected = data[..., None, idx.data]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (a, b, c, BroadcastSpace(), sel)

    def test_getitem_with_tensor_advanced_raises_for_shape_mismatch(self):
        row_src = IndexSpace.linear(4)
        col_src = IndexSpace.linear(5)
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)

        data = torch.arange(20, dtype=torch.float64).reshape(4, 5)
        tensor = Tensor(data=data, dims=(row_src, col_src))

        i = Tensor(data=torch.tensor([0, 1], dtype=torch.long), dims=(a,))
        j = Tensor(data=torch.tensor([0, 1, 2], dtype=torch.long), dims=(b,))

        with pytest.raises(ValueError, match="not broadcastable in shape"):
            _ = tensor[i, j]

    def test_getitem_with_tensor_advanced_raises_for_out_of_bounds(self):
        row_src = IndexSpace.linear(3)
        col_src = IndexSpace.linear(4)
        sel = IndexSpace.linear(2)

        data = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        tensor = Tensor(data=data, dims=(row_src, col_src))

        i = Tensor(data=torch.tensor([0, 3], dtype=torch.long), dims=(sel,))
        j = Tensor(data=torch.tensor([1, 2], dtype=torch.long), dims=(sel,))

        with pytest.raises(IndexError):
            _ = tensor[i, j]

    def test_getitem_with_tensor_advanced_empty_index(self):
        row_src = IndexSpace.linear(3)
        col_src = IndexSpace.linear(4)
        empty = IndexSpace.linear(0)

        data = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        tensor = Tensor(data=data, dims=(row_src, col_src))

        i = Tensor(data=torch.tensor([], dtype=torch.long), dims=(empty,))
        j = Tensor(data=torch.tensor([], dtype=torch.long), dims=(empty,))

        out = tensor[i, j]
        expected = data[i.data, j.data]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (empty,)

    def test_getitem_with_where_indices(self):
        row_src = IndexSpace.linear(3)
        col_src = IndexSpace.linear(4)

        data = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        tensor = Tensor(data=data, dims=(row_src, col_src))

        mask = Tensor(
            data=torch.tensor(
                [
                    [True, False, False, True],
                    [False, True, False, False],
                    [True, False, True, False],
                ],
                dtype=torch.bool,
            ),
            dims=(row_src, col_src),
        )

        row_idx, col_idx = where(mask)
        out = tensor[row_idx, col_idx]
        expected = data[mask.data]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (IndexSpace.linear(int(mask.data.sum().item())),)

    def test_getitem_with_tensor_advanced_matches_torch_ellipsis_none_patterns(self):
        # Differential test focused on risky forms involving:
        # Tensor-index, full slice (:), ellipsis (...), and None.
        rank = 4
        axis_size = 5
        dims = tuple(IndexSpace.linear(axis_size) for _ in range(rank))
        data = torch.arange(axis_size**rank, dtype=torch.float64).reshape(
            *(axis_size for _ in range(rank))
        )
        tensor = Tensor(data=data, dims=dims)
        idx = Tensor(
            data=torch.tensor([4, 1], dtype=torch.long),
            dims=(IndexSpace.linear(2),),
        )

        token_alphabet = ("A", "S", "N", "E")  # Tensor, :, None, Ellipsis
        max_key_len = rank + 2
        for key_len in range(1, max_key_len + 1):
            for token_pattern in product(token_alphabet, repeat=key_len):
                if token_pattern.count("E") > 1:
                    continue
                # Focus this sweep on ellipsis+None+advanced interactions.
                if (
                    "A" not in token_pattern
                    or "E" not in token_pattern
                    or "N" not in token_pattern
                ):
                    continue

                key = tuple(
                    idx
                    if t == "A"
                    else slice(None)
                    if t == "S"
                    else None
                    if t == "N"
                    else Ellipsis
                    for t in token_pattern
                )
                torch_key = tuple(k.data if isinstance(k, Tensor) else k for k in key)

                # Compare only valid torch indexing forms.
                try:
                    expected = data[torch_key]
                except Exception:
                    continue

                out = tensor[key]
                assert isinstance(out, Tensor)
                assert torch.equal(out.data, expected)
                assert tuple(dim.dim for dim in out.dims) == tuple(expected.shape)

    def test_getitem_rejects_multiple_ellipsis_in_advanced_indexing(self):
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)
        data = torch.arange(6, dtype=torch.float64).reshape(2, 3)
        tensor = Tensor(data=data, dims=(a, b))
        idx = Tensor(
            data=torch.tensor([1, 0], dtype=torch.long), dims=(IndexSpace.linear(2),)
        )

        with pytest.raises(IndexError, match="single ellipsis"):
            _ = tensor[..., idx, ...]
