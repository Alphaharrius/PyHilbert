r"""
Momentum-block tensor types and specialized multiplication rules.

This module defines a rank-3 [`Tensor`][qten.linalg.tensors.Tensor] subtype for
momentum-resolved block matrices whose first axis stores ordered momentum pairs
`(k1, k2)`. Each entry on that leading axis labels one matrix block acting
between two Hilbert-space sectors, so a
[`MomentumBlockTensor`][qten.MomentumBlockTensor]
has dims `(MomentumBlockSpace, HilbertSpace, HilbertSpace)`.

Mathematical convention
-----------------------
Write the leading momentum-pair space as
\(K_{\mathrm{block}} = \{(k_1, k_2)\}\). A momentum block tensor represents a
family of matrices
\(A_{(k_1, k_2)}\), stored as a rank-3 tensor with

Axis `0`: momentum-pair labels `(k1, k2)`.
Axis `1`: left Hilbert-space basis labels.
Axis `2`: right Hilbert-space basis labels.

The specialized `@` rules in this module interpret the pair axis as block
metadata rather than as a generic batch axis:

`MomentumBlockTensor @ Tensor`: uses the second momentum `k2` to select the
right-hand momentum block before multiplying.
`Tensor @ MomentumBlockTensor`: uses the first momentum `k1` to select the
left-hand momentum block.
`MomentumBlockTensor @ MomentumBlockTensor`: composes compatible block pairs
`(k1, k2)` and `(k2, k3)` into `(k1, k3)` and accumulates duplicates.

Repository usage
----------------
This type sits between the generic symbolic tensor layer in
[`qten.linalg.tensors`][qten.linalg.tensors] and momentum-resolved operator
logic. It is intentionally more structured than a plain
[`Tensor`][qten.linalg.tensors.Tensor]: the rank is fixed to `3`, the first dim
must be a [`MomentumBlockSpace`][qten.symbolics.state_space.MomentumBlockSpace],
and the last two dims must be
[`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] axes.

Axis-order restrictions
-----------------------
Because axis `0` is not an ordinary batch axis but a
[`MomentumBlockSpace`][qten.symbolics.state_space.MomentumBlockSpace] carrying
ordered pairs `(k1, k2)`, only matrix-leg swaps are meaningfully supported.

Supported: `transpose(1, 2)` and `permute(0, 2, 1)`. They swap the matrix
legs and simultaneously transpose the momentum-pair labels
`(k1, k2) -> (k2, k1)`.
Supported: `h(1, 2)` and the common shorthand `h(-2, -1)`. They conjugate the
matrix data and then swap the last two axes.
Unsupported: any transpose or permutation involving axis `0`, such as
`transpose(0, 1)`, `transpose(0, 2)`, `permute(1, 0, 2)`, or
`permute(2, 1, 0)`. These operations move the momentum-block axis out of
position `0`, break the structural invariant of the type, and are expected to
fail validation.
"""

from dataclasses import dataclass
from dataclasses import replace
from collections import OrderedDict
from typing import Tuple, cast

import torch

from .tensors import Tensor, permute, strict_dims, transpose
from ..abstracts import Operable
from ..geometries import Momentum
from ..symbolics import HilbertSpace, MomentumSpace, MomentumBlockSpace
from ..validations import need_validation


def _promote_matmul_operands(
    left: torch.Tensor, right: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast two tensors to a shared dtype suitable for `torch.matmul`."""
    common_dtype = torch.promote_types(left.dtype, right.dtype)
    if left.dtype != common_dtype:
        left = left.to(common_dtype)
    if right.dtype != common_dtype:
        right = right.to(common_dtype)
    return left, right


def _contract_gather_indices(
    contract_space: MomentumBlockSpace,
    momentum_space: MomentumSpace,
    component: int,
    *,
    device: torch.device,
) -> torch.LongTensor:
    """
    Build gather indices from one component of a momentum-pair axis.

    Parameters
    ----------
    contract_space : MomentumBlockSpace
        Pair-labelled axis whose entries are queried component-wise.
    momentum_space : MomentumSpace
        Single-momentum axis providing the destination index mapping.
    component : int
        Pair component to extract: `0` for `k1`, `1` for `k2`.
    device : torch.device
        Torch device on which the index tensor should be allocated.

    Returns
    -------
    torch.LongTensor
        One index per entry of `contract_space`, ordered like that space.

    Raises
    ------
    ValueError
        If any requested momentum is missing from `momentum_space`.
    """
    try:
        return cast(
            torch.LongTensor,
            torch.tensor(
                [
                    momentum_space.structure[pair[component]]
                    for pair in contract_space.structure.keys()
                ],
                dtype=torch.long,
                device=device,
            ),
        )
    except KeyError as exc:
        raise ValueError(
            "The momentum-resolved operand is missing a momentum required by the contract space."
        ) from exc


def _validate_momentum_block_tensor(tensor: Tensor) -> None:
    """Validate the fixed symbolic layout of a momentum block tensor."""
    if tensor.rank() != 3:
        raise ValueError("MomentumBlockTensor must have exactly 3 dimensions.")
    K, B1, B2 = tensor.dims
    if not isinstance(K, MomentumBlockSpace):
        raise ValueError(
            "The first dimension of MomentumBlockTensor must be a MomentumBlockSpace."
        )
    if not isinstance(B1, HilbertSpace) or not isinstance(B2, HilbertSpace):
        raise ValueError(
            "The second and third dimensions of MomentumBlockTensor must be HilbertSpaces."
        )


def _pair_space(pairs: tuple[tuple[Momentum, Momentum], ...]) -> MomentumBlockSpace:
    """Build a contiguous [`MomentumBlockSpace`][qten.symbolics.state_space.MomentumBlockSpace] from ordered pairs."""
    return MomentumBlockSpace(
        structure=OrderedDict((pair, i) for i, pair in enumerate(pairs))
    )


def _momentum_space(momentums: tuple[Momentum, ...]) -> MomentumSpace:
    """Build a contiguous [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] from ordered momenta."""
    return MomentumSpace(structure=OrderedDict((k, i) for i, k in enumerate(momentums)))


@need_validation(_validate_momentum_block_tensor)
@strict_dims
@dataclass(frozen=True)
class MomentumBlockTensor(Tensor):
    """
    Rank-3 tensor subtype for momentum-labelled matrix blocks.

    A [`MomentumBlockTensor`][qten.MomentumBlockTensor]
    stores matrix blocks on its last two axes and a
    [`MomentumBlockSpace`][qten.symbolics.state_space.MomentumBlockSpace]
    on axis `0`. The common layout is
    `(MomentumBlockSpace, HilbertSpace, HilbertSpace)`.

    Operations such as `transpose(1, 2)` and `h(1, 2)` are specialized so the
    momentum-pair metadata is updated together with the matrix-leg swap.

    Notes
    -----
    This subtype does not support arbitrary axis-reordering operations.

    Valid reorderings: `transpose(1, 2)`, `permute(0, 2, 1)`, `h(1, 2)`, and
    `h(-2, -1)`.
    Invalid reorderings: `transpose(0, 1)`, `transpose(0, 2)`, and any
    `permute(...)` that moves axis `0` away from the front.

    The reason is semantic, not just shape-related: axis `0` stores
    momentum-pair labels whose orientation must track the left/right matrix
    legs. Only swapping the two matrix axes has a well-defined update rule for
    those labels.
    """

    def __post_init__(self):
        """
        Finalize construction.

        The base [`Tensor`][qten.linalg.tensors.Tensor] post-init is executed
        so any forced-device construction logic still applies.
        """
        super().__post_init__()

    def __repr__(self) -> str:
        """
        Return a compact developer-facing representation of the tensor.

        This mirrors [`Tensor.__repr__`][qten.linalg.tensors.Tensor.__repr__]
        while preserving the concrete subtype name.
        """
        device_type = self.data.device.type
        device = "GPU" if device_type in {"cuda", "mps"} else "CPU"
        if self.dims:
            shape = ", ".join(f"{type(dim).__name__}:{dim.dim}" for dim in self.dims)
            shape_repr = f"({shape})"
        else:
            shape_repr = "()"
        return (
            f"<{device} {type(self).__name__} "
            f"grad={self.data.requires_grad} shape={shape_repr}>"
        )

    __str__ = __repr__


# mypy does not model multimethod's dynamic .register API.
@permute.register  # type: ignore[attr-defined]
def _(
    tensor: MomentumBlockTensor, *order: int | tuple[int, ...] | list[int]
) -> MomentumBlockTensor:
    """
    Permute a momentum block tensor while preserving block-axis semantics.

    The special case `order == (0, 2, 1)` swaps the matrix legs and therefore
    also reverses each momentum pair `(k1, k2) -> (k2, k1)` on the leading
    [`MomentumBlockSpace`][qten.symbolics.state_space.MomentumBlockSpace].
    Other permutations reuse the generic `Tensor` metadata permutation and then
    rely on the subtype validator to reject unsupported layouts.
    """
    normalized: Tuple[int, ...]
    if len(order) == 1 and isinstance(order[0], (tuple, list)):
        normalized = tuple(order[0])
    else:
        normalized = cast(Tuple[int, ...], tuple(order))

    if len(normalized) != tensor.rank():
        raise ValueError(
            f"Permutation order length {len(normalized)} does not match tensor dimensions {tensor.rank()}!"
        )

    result = cast(
        MomentumBlockTensor,
        replace(
            tensor,
            data=tensor.data.permute(normalized),
            dims=tuple(tensor.dims[i] for i in normalized),
        ),
    )

    if normalized == (0, 2, 1):
        return cast(
            MomentumBlockTensor,
            replace(
                result,
                dims=(
                    cast(MomentumBlockSpace, tensor.dims[0]).transposed(),
                    result.dims[1],
                    result.dims[2],
                ),
            ),
        )
    return result


@transpose.register  # type: ignore[attr-defined]
def _(tensor: MomentumBlockTensor, dim0: int, dim1: int) -> MomentumBlockTensor:
    """
    Transpose a momentum block tensor via its specialized permutation rule.

    Only matrix-leg swaps are supported in practice. `transpose(1, 2)` is the
    meaningful case and updates the leading momentum-pair axis from `(k1, k2)`
    to `(k2, k1)`. Transposes involving axis `0` are expected to fail because
    they violate the fixed `(MomentumBlockSpace, HilbertSpace, HilbertSpace)`
    layout.
    """
    order = list(range(tensor.rank()))
    order[dim0], order[dim1] = order[dim1], order[dim0]
    return permute(tensor, tuple(order))


@Operable.__matmul__.register  # type: ignore[attr-defined]
def _(self: MomentumBlockTensor, other: Tensor) -> MomentumBlockTensor:
    """
    Contract a band-pair tensor with a momentum-resolved operator on the right.

    `self` stores matrix blocks labelled by momentum pairs `(k1, k2)` on its
    first axis. This multiplication applies a right operand defined on single
    momenta `k2` by selecting, for each pair in `self`, the block from `other`
    whose momentum matches that pair's second component. The selected blocks are
    then multiplied with one batched `torch.matmul`.

    Parameters
    ----------
    self : MomentumBlockTensor
        Left operand with dims `(MomentumBlockSpace, HilbertSpace,
        HilbertSpace)`.
    other : Tensor
        Right operand with dims `(MomentumSpace, HilbertSpace, HilbertSpace)`.

    Returns
    -------
    MomentumBlockTensor
        Tensor with dims `(self.dims[0], self.dims[1], other.dims[2])`.

    Raises
    ------
    ValueError
        If `other` is not rank-3, if its first dim is not a
        [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace], or if any
        momentum needed from `self` is missing from `other`.
    """
    if other.rank() != 3:
        raise ValueError(
            "MomentumBlockTensor @ Tensor requires the right operand to have exactly 3 dimensions."
        )

    other_k, _, other_out = other.dims
    if not isinstance(other_k, MomentumSpace):
        raise ValueError(
            "The first dimension of the right operand must be a MomentumSpace."
        )
    other_out = cast(HilbertSpace, other_out)

    other = other.align(1, self.dims[2])

    contract_space = cast(MomentumBlockSpace, self.dims[0])
    left_in = cast(HilbertSpace, self.dims[1])
    gather_indices = _contract_gather_indices(
        contract_space, other_k, 1, device=other.data.device
    )
    gathered = other.data.index_select(0, gather_indices)
    left_data, right_data = _promote_matmul_operands(self.data, gathered)
    data = torch.matmul(left_data, right_data)
    return MomentumBlockTensor(
        data=data,
        dims=(contract_space, left_in, other_out),
    )


@Operable.__matmul__.register  # type: ignore[attr-defined]
def _(self: Tensor, other: MomentumBlockTensor) -> MomentumBlockTensor:
    """
    Contract a momentum-resolved operator with a band-pair tensor on the right.

    `other` stores matrix blocks labelled by momentum pairs `(k1, k2)` on its
    first axis. This multiplication applies a left operand defined on single
    momenta `k1` by selecting, for each pair in `other`, the block from `self`
    whose momentum matches that pair's first component. The selected blocks are
    then multiplied with one batched `torch.matmul`.

    Parameters
    ----------
    self : Tensor
        Left operand with dims `(MomentumSpace, HilbertSpace, HilbertSpace)`.
    other : MomentumBlockTensor
        Right operand with dims `(MomentumBlockSpace, HilbertSpace,
        HilbertSpace)`.

    Returns
    -------
    MomentumBlockTensor
        Tensor with dims `(other.dims[0], self.dims[1], other.dims[2])`.

    Raises
    ------
    ValueError
        If `self` is not rank-3, if its first dim is not a
        [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace], or if any
        momentum needed from `other` is missing from `self`.
    """
    if self.rank() != 3:
        raise ValueError(
            "Tensor @ MomentumBlockTensor requires the left operand to have exactly 3 dimensions."
        )

    self_k, self_out, _ = self.dims
    if not isinstance(self_k, MomentumSpace):
        raise ValueError(
            "The first dimension of the left operand must be a MomentumSpace."
        )
    self_out = cast(HilbertSpace, self_out)

    self = self.align(2, other.dims[1])

    contract_space = cast(MomentumBlockSpace, other.dims[0])
    right_out = cast(HilbertSpace, other.dims[2])
    gather_indices = _contract_gather_indices(
        contract_space, self_k, 0, device=self.data.device
    )
    gathered = self.data.index_select(0, gather_indices)
    left_data, right_data = _promote_matmul_operands(gathered, other.data)
    data = torch.matmul(left_data, right_data)
    return MomentumBlockTensor(
        data=data,
        dims=(contract_space, self_out, right_out),
    )


@Operable.__matmul__.register  # type: ignore[attr-defined]
def _(self: MomentumBlockTensor, other: MomentumBlockTensor) -> Tensor:
    """
    Compose two band-pair tensors through their shared middle momentum.

    For each left block labelled `(k1, k2)` and right block labelled `(k2, k3)`,
    this computes the matrix product of those blocks and accumulates the result
    into output pair `(k1, k3)`. Contributions landing on the same output pair
    are summed. If every surviving output pair is diagonal `(k, k)`, the pair
    axis is collapsed to a standard `MomentumSpace`.

    Parameters
    ----------
    self : MomentumBlockTensor
        Left operand with dims `(MomentumBlockSpace, HilbertSpace,
        HilbertSpace)`.
    other : MomentumBlockTensor
        Right operand with dims `(MomentumBlockSpace, HilbertSpace,
        HilbertSpace)`.

    Returns
    -------
    Tensor
        Either a `MomentumBlockTensor` with dims
        `(MomentumBlockSpace, self.dims[1], other.dims[2])`, or a plain
        `Tensor` with dims `(MomentumSpace, self.dims[1], other.dims[2])` when
        every output pair is diagonal.
    """
    other = other.align(1, self.dims[2])

    left_space = cast(MomentumBlockSpace, self.dims[0])
    left_out = cast(HilbertSpace, self.dims[1])
    right_space = cast(MomentumBlockSpace, other.dims[0])
    right_out = cast(HilbertSpace, other.dims[2])
    left_pairs = tuple(left_space.structure.keys())
    right_pairs = tuple(right_space.structure.keys())

    right_by_start: dict[Momentum, list[int]] = {}
    for right_idx, (k2, _) in enumerate(right_pairs):
        right_by_start.setdefault(k2, []).append(right_idx)

    left_indices: list[int] = []
    right_indices: list[int] = []
    output_indices: list[int] = []
    output_pairs: list[tuple[Momentum, Momentum]] = []
    output_lookup: dict[tuple[Momentum, Momentum], int] = {}

    for left_idx, (k1, k2) in enumerate(left_pairs):
        for right_idx in right_by_start.get(k2, ()):
            out_pair = (k1, right_pairs[right_idx][1])
            out_idx = output_lookup.get(out_pair)
            if out_idx is None:
                out_idx = len(output_pairs)
                output_lookup[out_pair] = out_idx
                output_pairs.append(out_pair)
            left_indices.append(left_idx)
            right_indices.append(right_idx)
            output_indices.append(out_idx)

    if not output_pairs:
        empty_space = _pair_space(tuple())
        return MomentumBlockTensor(
            data=self.data.new_zeros((0, left_out.dim, right_out.dim)),
            dims=(empty_space, left_out, right_out),
        )

    left_gather = torch.tensor(left_indices, dtype=torch.long, device=self.data.device)
    right_gather = torch.tensor(
        right_indices, dtype=torch.long, device=other.data.device
    )
    out_index = torch.tensor(output_indices, dtype=torch.long, device=self.data.device)

    left_data = self.data.index_select(0, left_gather)
    right_data = other.data.index_select(0, right_gather)
    left_data, right_data = _promote_matmul_operands(left_data, right_data)
    contributions = torch.matmul(left_data, right_data)

    data = contributions.new_zeros((len(output_pairs), left_out.dim, right_out.dim))
    data.index_add_(0, out_index, contributions)

    output_pairs_tuple = tuple(output_pairs)
    if all(k1 == k2 for k1, k2 in output_pairs_tuple):
        momentum_space = _momentum_space(tuple(k1 for k1, _ in output_pairs_tuple))
        return Tensor(data=data, dims=(momentum_space, left_out, right_out))

    contract_space = _pair_space(output_pairs_tuple)
    return MomentumBlockTensor(
        data=data,
        dims=(contract_space, left_out, right_out),
    )
