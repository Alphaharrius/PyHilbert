from typing import Tuple, Union, Sequence, cast, Dict
from numbers import Number
from dataclasses import dataclass
from multipledispatch import dispatch  # type: ignore[import-untyped]
import torch

from .abstracts import Operable, Plottable
from .hilbert import (
    StateSpace,
    HilbertSpace,
    BroadcastSpace,
    embedding_order,
    same_span,
    flat_permutation_order,
    Mode,
)

@dataclass(frozen=True)
class Tensor(Operable, Plottable):
    data: torch.Tensor
    dims: Tuple[StateSpace, ...]

    def conj(self) -> "Tensor":
        """
        Compute the complex conjugate of the given tensor.

        Returns
        -------
        `Tensor`
            The complex conjugate of the tensor.
        """
        return conj(self)

    def permute(self, *order: Union[int, Sequence[int]]) -> "Tensor":
        """
        Permute the dimensions according to the specified order.

        Parameters
        ----------
        order : `Union[int, Sequence[int]]`
            The desired order of dimensions.

        Returns
        -------
        `Tensor`
            The permuted tensor.
        """
        return permute(self, *order)

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        """
        Transpose the specified dimensions.

        Parameters
        ----------
        dim0 : `int`
            The first dimension to transpose.
        dim1 : `int`
            The second dimension to transpose.

        Returns
        -------
        `Tensor`
            The transposed tensor.
        """
        return transpose(self, dim0, dim1)

    def h(self, dim0: int, dim1: int) -> "Tensor":
        """
        Hermitian transpose (conjugate transpose) of the specified dimensions.

        Parameters
        ----------
        dim0 : `int`
            The first dimension to transpose.
        dim1 : `int`
            The second dimension to transpose.

        Returns
        -------
        `Tensor`
            The Hermitian transposed tensor.
        """
        return self.conj().transpose(dim0, dim1)

    def align(self, dim: int, target_dim: StateSpace) -> "Tensor":
        """
        Align the specified dimension to the target StateSpace.

        Parameters
        ----------
        dim : `int`
            The dimension index to align.
        target_dim : `StateSpace`
            The target StateSpace to align to.

        Returns
        -------
        `Tensor`
            The aligned tensor.
        """
        return align(self, dim, target_dim)

    def unsqueeze(self, dim: int) -> "Tensor":
        """
        Unsqueeze the specified dimension.

        Parameters
        ----------
        dim : `int`
            The dimension to unsqueeze.

        Returns
        -------
        `Tensor`
            The unsqueezed tensor.
        """
        return unsqueeze(self, dim)

    def squeeze(self, dim: int) -> "Tensor":
        """
        Squeeze the specified dimension.

        Parameters
        ----------
        dim : `int`
            The dimension to squeeze.

        Returns
        -------
        `Tensor`
            The squeezed tensor.
        """
        return squeeze(self, dim)

    def rank(self) -> int:
        """
        Get the rank (number of dimensions) of the tensor.

        Returns
        -------
        `int`
            The rank of the tensor.
        """
        return rank(self)

    def expand_to_union(self, union_dims: list[StateSpace]) -> "Tensor":
        """
        Expand the tensor to the union of the specified dimensions.

        Parameters
        ----------
        union_dims : `list[StateSpace]`
            The dimensions to expand to the union of.

        Returns
        -------
        `Tensor`
            The expanded tensor.
        """
        return expand_to_union(self, union_dims)

    def item(self) -> Union[Number, int, float]:
        """
        Return the value of a 0-dimensional tensor as a standard Python number.

        Returns
        -------
        `number`
            The value of the tensor.

        Raises
        ------
        ValueError
            If the tensor is not 0-dimensional.
        """
        return self.data.item()

    def cpu(self) -> "Tensor":
        """
        Copy the tensor data to CPU memory and create a new `Tensor` instance.

        Returns
        -------
        `Tensor`
            The new `Tensor` instance with copied data on CPU.
        """
        return Tensor(data=self.data.cpu(), dims=self.dims)

    def gpu(self) -> "Tensor":
        """
        Copy the tensor data to GPU memory and create a new `Tensor` instance.

        Returns
        -------
        `Tensor`
            The new `Tensor` instance with copied data on GPU.

        Raises
        ------
        RuntimeError
            If GPU is not available on this system.
        """
        if torch.cuda.is_available():
            return Tensor(data=self.data.cuda(), dims=self.dims)
        elif torch.backends.mps.is_available():
            return Tensor(data=self.data.to("mps"), dims=self.dims)
        else:
            raise RuntimeError(
                "Only CUDA and MPS devices are supported for GPU operations!"
            )


    @property
    def requires_grad(self) -> bool:
        """
        Check if the tensor data requires gradient tracking.

        Returns
        -------
        `bool`
            True if the tensor data requires gradient tracking, False otherwise.
        """
        return self.data.requires_grad

    def attach(self) -> "Tensor":
        """
        Enable gradient tracking for the tensor data and return the attached `Tensor` instance.

        Behavior
        --------
        - If `requires_grad` is already `True`, this returns `self` unchanged.
        - Otherwise, this detaches the underlying data from any existing autograd graph,
          clones it to ensure a fresh leaf tensor, and sets `requires_grad` to `True`.
        - The returned tensor preserves the original `dims`, device, and dtype.

        Returns
        -------
        `Tensor`
            The new `Tensor` instance with gradient tracking enabled.
        """
        if self.data.requires_grad:
            return self
        return Tensor(
            data=self.data.detach().clone().requires_grad_(True), dims=self.dims
        )

    def detach(self) -> "Tensor":
        """
        Disable gradient tracking for the tensor data and create a new `Tensor` instance.

        Behavior
        --------
        - Always returns a new `Tensor` whose data is a detached view of the
          original tensor (no clone), so it shares storage with the original.
        - The returned tensor preserves the original `dims`, device, and dtype.

        Returns
        -------
        `Tensor`
            The new `Tensor` instance with gradient tracking disabled.
        """
        return Tensor(data=self.data.detach(), dims=self.dims)

    def clone(self) -> "Tensor":
        """
        Create a deep copy of the tensor.

        Returns
        -------
        `Tensor`
            The cloned tensor.
        """
        return Tensor(data=self.data.clone(), dims=self.dims)
    
    def replace_dim(self, dim: int, new_dim: StateSpace)-> "Tensor":
        """
        Replace the StateSpace at the specified dimension with a new StateSpace.

        Parameters
        ----------
        tensor : `Tensor`
            The tensor to modify.
        dim : `int`
            The index of the dimension to replace.
        new_dim : `StateSpace`
            The new StateSpace to assign to the dimension.

        Returns
        -------
        `Tensor`
            A new Tensor with the updated dimension.
        """
        return replace_dim(self, dim, new_dim)



    def __repr__(self) -> str:
        device_type = self.data.device.type
        device = "GPU" if device_type in {"cuda", "mps"} else "CPU"
        if self.dims:
            shape = ", ".join(f"{type(dim).__name__}:{dim.size}" for dim in self.dims)
            shape_repr = f"({shape})"
        else:
            shape_repr = "()"
        return f"<{device} Tensor grad={self.data.requires_grad} shape={shape_repr}>"

    __str__ = __repr__  # Override str to use the same representation


def _match_dims_for_matmul(left: Tensor, right: Tensor) -> Tuple[Tensor, Tensor]:
    if left.rank() == 1:
        left = left.unsqueeze(0)
    if right.rank() == 1:
        right = right.unsqueeze(-1)

    if left.rank() > right.rank():
        # Unsqueeze right tensor
        for _ in range(left.rank() - right.rank()):
            right = right.unsqueeze(0)
    elif right.rank() > left.rank():
        # Unsqueeze left tensor
        for _ in range(right.rank() - left.rank()):
            left = left.unsqueeze(0)
    return left, right


def _align_dims_for_matmul(left: Tensor, right: Tensor) -> Tuple[Tensor, Tensor]:
    ignores = set()
    for n, ld in enumerate(left.dims[:-2]):
        if not isinstance(ld, BroadcastSpace):
            continue
        rd = right.dims[n]
        if isinstance(rd, BroadcastSpace):
            continue
        left = left.align(n, rd)
        ignores.add(n)

    for n, ld in enumerate(left.dims[:-2]):
        if n in ignores:
            continue
        right = right.align(n, ld)

    return left, right


def matmul(left: Tensor, right: Tensor) -> Tensor:
    """
    Perform matrix multiplication between two Tensors with StateSpace-aware
    alignment and torch-style rank handling.

    Both operands must be at least 1D. If either operand is 1D, this follows
    `torch.matmul` behavior by temporarily unsqueezing it to 2D, performing the
    matmul, then squeezing out the added dimension(s).

    The function first makes the tensors have the same number of dimensions by
    unsqueezing leading dimensions with `BroadcastSpace`. It then aligns any
    leading (batch) dimensions so that `BroadcastSpace` can expand to concrete
    StateSpaces and any non-broadcast StateSpaces are reordered to match. Finally,
    the right tensor's second-to-last dimension is aligned to the left tensor's
    last dimension, and `torch.matmul` is applied.

    The contraction always happens between `left.dims[-1]` and `right.dims[-2]`.
    Leading dimensions behave like batch dimensions and follow the broadcast and
    alignment rules described above. The output keeps all aligned leading
    dimensions (including any `BroadcastSpace` that remain), drops the contracted
    dimension, and appends the right-most dimension from `right`.

    Parameters
    ----------
    left : `Tensor`
        The left tensor operand.
    right : `Tensor`
        The right tensor operand.

    Returns
    -------
    `Tensor`
        A tensor with data `torch.matmul(left.data, right.data)` and dimensions
        `left.dims[:-1] + right.dims[-1:]`, after the alignment and any
        1D squeeze handling.

    Raises
    ------
    ValueError
        If either operand is 0D or any StateSpace alignment fails during the
        broadcast or contraction alignment steps.
    """
    left_rank = left.rank()
    right_rank = right.rank()

    if left_rank < 1:
        raise ValueError("Left tensor must have rank at least 1 for matmul!")
    if right_rank < 1:
        raise ValueError("Right tensor must have rank at least 1 for matmul!")

    left, right = _match_dims_for_matmul(left, right)
    left, right = _align_dims_for_matmul(left, right)

    right = right.align(-2, left.dims[-1])
    data = torch.matmul(left.data.to(dtype=torch.complex128), right.data.to(dtype=torch.complex128))
    new_dims = left.dims[:-1] + right.dims[-1:]

    prod = Tensor(data=data, dims=new_dims)

    if left_rank == 1 and right_rank == 1:
        prod = prod.squeeze(0).squeeze(-1)
    elif right_rank == 1:
        prod = prod.squeeze(-1)
    elif left_rank == 1:
        prod = prod.squeeze(-2)

    return prod


@dispatch(Tensor, Tensor)
def operator_matmul(left: Tensor, right: Tensor) -> Tensor:
    """Perform matrix multiplication (contraction) between two `Tensor`."""
    return matmul(left, right)


def _match_dims_for_tensoradd(left: Tensor, right: Tensor) -> Tuple[Tensor, Tensor]:
    if left.rank() > right.rank():
        # Unsqueeze right tensor
        for _ in range(left.rank() - right.rank()):
            right = right.unsqueeze(0)
    elif right.rank() > left.rank():
        # Unsqueeze left tensor
        for _ in range(right.rank() - left.rank()):
            left = left.unsqueeze(0)
    return left, right


@dispatch(Tensor, Tensor)
def operator_add(left: Tensor, right: Tensor) -> Tensor:
    """
    Add two tensors with the same order of dimensions.
    If the intra-ordering within the `StateSpace`s differ,
    the `right` tensor is permuted to match the ordering
    of the `left` tensor before addition.

    Parameters
    ----------
    left : `Tensor`
        The left tensor to add.
    right : `Tensor`
        The right tensor to add.

    Returns
    -------
    `Tensor`
        The resulting tensor on the union of StateSpaces.
    """
    left, right = _match_dims_for_tensoradd(left, right)

    # calculate the union of the StateSpaces
    union_dims = []
    for l_dim, r_dim in zip(left.dims, right.dims):
        union_dims.append(l_dim + r_dim)

    # Expand BroadcastSpace to the union StateSpace to ensure data expansion
    left = left.expand_to_union(union_dims)
    right = right.expand_to_union(union_dims)

    # calculate the new shape
    new_shape = tuple(u.size for u in union_dims)
    new_data = torch.zeros(new_shape, dtype=left.data.dtype, device=left.data.device)
    # fill the left tensor into the new data
    left_slices = tuple(slice(0, d.size) for d in left.dims)
    new_data[left_slices] = left.data
    # fill the right tensor into the new data
    right_embedding_order = (
        torch.tensor(embedding_order(r, u), dtype=torch.long, device=left.data.device)
        for r, u in zip(right.dims, union_dims)
    )
    new_data.index_put_(
        torch.meshgrid(*right_embedding_order, indexing="ij"),
        right.data,
        accumulate=True,
    )

    return Tensor(data=new_data, dims=tuple(union_dims))


@dispatch(Tensor)
def operator_neg(tensor: Tensor) -> Tensor:
    """
    Perform negation on the given tensor.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to negate.

    Returns
    -------
    `Tensor`
        The negated tensor.
    """
    return Tensor(data=-tensor.data, dims=tensor.dims)


@dispatch(Tensor, Tensor)
def operator_sub(left: Tensor, right: Tensor) -> Tensor:
    """
    Subtract the right tensor from the left tensor with the same order of dimensions.
    If the intra-ordering within the `StateSpace`s differ, the `right` tensor is
    permuted to match the ordering of the `left` tensor before addition.

    Parameters
    ----------
    left : `Tensor`
        The tensor from which to subtract.
    right : `Tensor`
        The tensor to subtract.

    Returns
    -------
    `Tensor`
        The resulting tensor after subtraction.
    """
    return left + (-right)


@dispatch(Number, Tensor)
def operator_mul(left: Number, right: Tensor) -> Tensor:
    """
    Perform element-wise multiplication of a number and a tensor.

    Parameters
    ----------
    left : `Number`
        The scalar value.
    right : `Tensor`
        The tensor.
    Returns
    -------
    `Tensor`
        A new tensor with each element multiplied by the scalar.
    """
    return Tensor(data=left * right.data, dims=right.dims)


@dispatch(Tensor, Number)  # type: ignore[no-redef]
def operator_mul(left: Tensor, right: Number) -> Tensor:
    """
    Perform element-wise multiplication of a tensor and a number.

    Parameters
    ----------
    left : `Tensor`
        The tensor.
    right : `Number`
        The scalar value.
    Returns
    -------
    `Tensor`
        A new tensor with each element multiplied by the scalar.
    """
    return Tensor(data=left.data * right, dims=left.dims)


@dispatch(Number, Tensor)  # type: ignore[no-redef]
def operator_add(left: Number, right: Tensor) -> Tensor:
    """
    Add a number to the diagonal of the tensor (broadcasting over batch dimensions).

    This treats the tensor as a batch of matrices (defined by the last two dimensions).
    The scalar is added to the diagonal elements of these matrices.
    For rank-2 tensors, this is equivalent to M + c*I.
    Parameters
    ----------
    left : `Number`
        The scalar value to add to the diagonal.
    right : `Tensor`
        The target tensor (must be at least rank 2).
    Returns
    -------
    `Tensor`
        The result of adding the scalar to the diagonal.
    """
    eye = identity(right.dims)
    return left * eye + right


@dispatch(Tensor, Number)  # type: ignore[no-redef]
def operator_add(left: Tensor, right: Number) -> Tensor:
    """
    Add a number to the diagonal of the tensor (broadcasting over batch dimensions).

    This treats the tensor as a batch of matrices (defined by the last two dimensions).
    The scalar is added to the diagonal elements of these matrices.
    For rank-2 tensors, this is equivalent to M + c*I.
    Parameters
    ----------
    left : `Tensor`
        The target tensor (must be at least rank 2).
    right : `Number`
        The scalar value to add to the diagonal.
    Returns
    -------
    `Tensor`
        The result of adding the scalar to the diagonal.
    """
    eye = identity(left.dims)
    return left + right * eye


@dispatch(Number, Tensor)  # type: ignore[no-redef]
def operator_sub(left: Number, right: Tensor) -> Tensor:
    """
    Subtract a tensor from a number (broadcasted on diagonal).

    This treats the tensor as a batch of matrices (defined by the last two dimensions).
    The operation is performed as (c*I - T), where I is the identity matrix broadcasted
    over the batch dimensions.
    Parameters
    ----------
    left : `Number`
        The scalar value.
    right : `Tensor`
        The tensor to subtract.
    Returns
    -------
    `Tensor`
        The result of the subtraction.
    """
    eye = identity(right.dims)
    return left * eye + (-right)


@dispatch(Tensor, Number)  # type: ignore[no-redef]
def operator_sub(left: Tensor, right: Number) -> Tensor:
    """
    Subtract a number from a tensor (broadcasted on diagonal).

    This treats the tensor as a batch of matrices (defined by the last two dimensions).
    The operation is performed as (T - c*I), where I is the identity matrix broadcasted
    over the batch dimensions.
    Parameters
    ----------
    left : `Tensor`
        The tensor.
    right : `Number`
        The scalar value to subtract from the diagonal.
    Returns
    -------
    `Tensor`
        The result of the subtraction.
    """
    eye = identity(left.dims)
    return left + (-right) * eye


@dispatch(Tensor, Number)
def operator_truediv(left: Tensor, right: Number) -> Tensor:
    """
    Perform element-wise division of a tensor by a number.
    Parameters
    ----------
    left : `Tensor`
        The tensor.
    right : `Number`
        The scalar divisor.
    Returns
    -------
    `Tensor`
        A new tensor with each element divided by the scalar.
    """
    return left * (1.0 / right)  # type: ignore[operator]


def permute(tensor: Tensor, *order: Union[int, Sequence[int]]) -> Tensor:
    """
    Permute the dimensions of the tensor according to the specified order.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to permute.
    order : `Union[int, Sequence[int]]`
        The desired order of dimensions.

    Returns
    -------
    `Tensor`
        The permuted tensor.
    """
    _order: Tuple[int, ...]
    if len(order) == 1 and isinstance(order[0], (tuple, list)):
        _order = tuple(order[0])
    else:
        # We assume that if it's not a single list/tuple, it's a sequence of ints
        _order = cast(Tuple[int, ...], tuple(order))

    if len(_order) != tensor.rank():
        raise ValueError(
            f"Permutation order length {len(_order)} does not match tensor dimensions {tensor.rank()}!"
        )

    new_data = tensor.data.permute(_order)
    new_dims = tuple(tensor.dims[i] for i in _order)

    return Tensor(data=new_data, dims=new_dims)


def transpose(tensor: Tensor, dim0: int, dim1: int) -> Tensor:
    """
    Transpose the specified dimensions of the tensor.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to transpose.
    dim0 : `int`
        The first dimension to transpose.
    dim1 : `int`
        The second dimension to transpose.

    Returns
    -------
    `Tensor`
        The transposed tensor.
    """
    new_data = tensor.data.transpose(dim0, dim1)

    # Convert tuple to list to modify
    new_dims_list = list(tensor.dims)
    # Swap elements
    new_dims_list[dim0], new_dims_list[dim1] = new_dims_list[dim1], new_dims_list[dim0]

    return Tensor(data=new_data, dims=tuple(new_dims_list))


def conj(tensor: Tensor) -> Tensor:
    """
    Compute the complex conjugate of the given tensor.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to conjugate.

    Returns
    -------
    `Tensor`
        The complex conjugate of the tensor.
    """
    return Tensor(data=tensor.data.conj(), dims=tensor.dims)


def unsqueeze(tensor: Tensor, dim: int) -> Tensor:
    """
    Unsqueeze the specified dimension of the tensor.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to unsqueeze.
    dim : `int`
        The dimension to unsqueeze.

    Returns
    -------
    `Tensor`
        The unsqueezed tensor.
    """
    if dim < 0:
        dim = dim + len(tensor.dims) + 1
    new_data = tensor.data.unsqueeze(dim)
    new_dims = tensor.dims[:dim] + (BroadcastSpace(),) + tensor.dims[dim:]

    return Tensor(data=new_data, dims=new_dims)


def squeeze(tensor: Tensor, dim: int) -> Tensor:
    """
    Squeeze the specified dimension of the tensor.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to squeeze.
    dim : `int`
        The dimension to squeeze.

    Returns
    -------
    `Tensor`
        The squeezed tensor.
    """
    if dim < 0:
        dim = dim + len(tensor.dims)
    if not isinstance(tensor.dims[dim], BroadcastSpace):
        return tensor  # No squeezing needed if not BroadcastSpace

    new_data = tensor.data.squeeze(dim)
    new_dims = tensor.dims[:dim] + tensor.dims[dim + 1 :]

    return Tensor(data=new_data, dims=new_dims)


def align(tensor: Tensor, dim: int, target_dim: StateSpace) -> Tensor:
    """
    Align the specified dimension of the tensor to the target StateSpace.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to align.
    dim : `int`
        The dimension index to align.
    target : `StateSpace`
        The target StateSpace to align to.

    Returns
    -------
    `Tensor`
        The aligned tensor.
    """
    current_dim = tensor.dims[dim]
    if isinstance(target_dim, BroadcastSpace):
        return tensor  # No alignment needed for BroadcastSpace

    if isinstance(current_dim, BroadcastSpace):
        # Expand broadcast dimension to match the target StateSpace size.
        expanded_shape = list(tensor.data.shape)
        expanded_shape[dim] = target_dim.size
        aligned_data = tensor.data.expand(*expanded_shape)
        return Tensor(
            data=aligned_data,
            dims=tensor.dims[:dim] + (target_dim,) + tensor.dims[dim + 1 :],
        )

    if type(current_dim) is not type(target_dim):
        raise ValueError(
            f"Cannot align dimensions with different StateSpace types: "
            f"current dim={type(current_dim)} vs target dim={type(target_dim)}!"
        )
    if not same_span(current_dim, target_dim):
        raise ValueError(f"StateSpace at {dim} cannot be aligned to target StateSpace!")

    target_order = flat_permutation_order(current_dim, target_dim)
    aligned_data = torch.index_select(
        tensor.data,
        dim,
        torch.tensor(target_order, dtype=torch.long, device=tensor.data.device),
    )

    aligned_tensor = Tensor(
        data=aligned_data,
        dims=tensor.dims[:dim] + (target_dim,) + tensor.dims[dim + 1 :],
    )

    return aligned_tensor


def rank(tensor: Tensor) -> int:
    """
    Get the rank (number of dimensions) of the tensor.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor whose rank is to be determined.

    Returns
    -------
    `int`
        The rank of the tensor.
    """
    return len(tensor.dims)


def expand_to_union(tensor: Tensor, union_dims: list[StateSpace]) -> Tensor:
    """
    Expand BroadcastSpace dimensions in the tensor to match union_dims sizes.
    Performs expansion in a single pass to avoid intermediate Tensor creation.
    """

    if not any(isinstance(d, BroadcastSpace) for d in tensor.dims):
        return tensor
    target_shape = []
    new_dims = []
    needs_expansion = False

    for dim, u_dim, size in zip(tensor.dims, union_dims, tensor.data.shape):
        if isinstance(dim, BroadcastSpace) and not isinstance(u_dim, BroadcastSpace):
            target_shape.append(u_dim.size)
            new_dims.append(u_dim)
            needs_expansion = True
        else:
            target_shape.append(size)
            new_dims.append(dim)

    if not needs_expansion:
        return tensor

    return Tensor(data=tensor.data.expand(target_shape), dims=tuple(new_dims))


def mapping_matrix(
    from_space: HilbertSpace, to_space: HilbertSpace, mapping: Dict[Mode, Mode]
) -> Tensor:
    # TODO: Use globally defined complex dtype
    mat = torch.zeros((from_space.size, to_space.size), dtype=torch.complex64)
    for fm, tm in mapping.items():
        fslice = from_space.get_slice(fm)
        tslice = to_space.get_slice(tm)

        flen = fslice.stop - fslice.start
        tlen = tslice.stop - tslice.start
        if flen != tlen:
            raise ValueError(
                f"Cannot create mapping matrix between modes of different sizes: {flen} != {tlen}"
            )

        mat[fslice, tslice] = torch.eye(flen, dtype=mat.dtype, device=mat.device)

    return Tensor(data=mat, dims=(from_space, to_space))


def identity(dims: Tuple[StateSpace, ...]) -> Tensor:
    """
    Create an identity tensor based on the last two dimensions.
    Returns a rank-2 Tensor corresponding to the identity of the matrix part.
    """
    if len(dims) < 2:
        raise ValueError(
            f"Identity tensor creation requires at least rank 2, got rank {len(dims)}!"
        )
    matrix_dims = dims[-2:]
    rows = matrix_dims[0].size
    cols = matrix_dims[1].size
    return Tensor(data=torch.eye(rows, cols), dims=matrix_dims)

def replace_dim(tensor: Tensor, dim: int, new_dim: StateSpace) -> Tensor:
    """
    Replace the StateSpace at the specified dimension with a new StateSpace.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to modify.
    dim : `int`
        The index of the dimension to replace.
    new_dim : `StateSpace`
        The new StateSpace to assign to the dimension.

    Returns
    -------
    `Tensor`
        A new Tensor with the updated dimension.
    """
    if dim < 0:
        dim += len(tensor.dims)

    if dim < 0 or dim >= len(tensor.dims):
        raise IndexError(
            f"Dimension index {dim} out of range for tensor of rank {len(tensor.dims)}"
        )

    current_size = tensor.data.shape[dim]

    # Check size compatibility
    # If new_dim is BroadcastSpace with no structure (size 0), it matches data dimension of size 1.
    if isinstance(new_dim, BroadcastSpace) and new_dim.dim == 0:
        if current_size != 1:
            raise ValueError(
                f"Cannot replace dimension of size {current_size} with empty BroadcastSpace (expects size 1)."
            )
    elif new_dim.size != current_size:
        raise ValueError(
            f"New StateSpace size {new_dim.size} does not match tensor data size {current_size} at dimension {dim}!"
        )

    new_dims = list(tensor.dims)
    new_dims[dim] = new_dim
    return Tensor(data=tensor.data, dims=tuple(new_dims))
