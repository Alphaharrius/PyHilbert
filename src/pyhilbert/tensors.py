from typing import Tuple
from numbers import Number
from dataclasses import dataclass
from multipledispatch import dispatch

import torch

from . import hilbert
from .abstracts import Operable
from .hilbert import StateSpace


@dataclass(frozen=True)
class Tensor(Operable):
    data: torch.Tensor
    dims: Tuple[StateSpace, ...]

    def __post_init__(self) -> None:
        # Ensure that data is detached from any computation graph.
        # In the future if we need to do autograd we will use nn.Module instead.
        object.__setattr__(self, "data", self.data.detach())

    def conj(self) -> 'Tensor':
        """
        Compute the complex conjugate of the given tensor.
        
        Returns
        -------
        `Tensor`
            The complex conjugate of the tensor.
        """
        return conj(self)

    def permute(self, *order: Tuple[int, ...]) -> 'Tensor':
        """
        Permute the dimensions according to the specified order.
        
        Parameters
        ----------
        order : `Tuple[int, ...]`
            The desired order of dimensions.

        Returns
        -------
        `Tensor`
            The permuted tensor.
        """
        return permute(self, *order)
    
    def transpose(self, dim0: int, dim1: int) -> 'Tensor':
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
    
    def align(self, dim: int, target_dim: StateSpace) -> 'Tensor':
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
    
    def unsqueeze(self, dim: int) -> 'Tensor':
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
    
    def squeeze(self, dim: int) -> 'Tensor':
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
    
    def item(self) -> Number:
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
    
    def cpu(self) -> 'Tensor':
        """
        Copy the tensor data to CPU memory and create a new `Tensor` instance.
        
        Returns
        -------
        `Tensor`
            The new `Tensor` instance with copied data on CPU.
        """
        return Tensor(data=self.data.cpu(), dims=self.dims)
    
    def gpu(self) -> 'Tensor':
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
            return Tensor(data=self.data.to('mps'), dims=self.dims)
        else:
            raise RuntimeError("Only CUDA and MPS devices are supported for GPU operations!")

    def __repr__(self) -> str:
        device_type = self.data.device.type
        device = "GPU" if device_type in {"cuda", "mps"} else "CPU"
        if self.dims:
            shape = ", ".join(f"{type(dim).__name__}:{dim.size}" for dim in self.dims)
            shape_repr = f"({shape})"
        else:
            shape_repr = "()"
        return f"<Tensor {device} grad={self.data.requires_grad} shape={shape_repr}>"

    __str__ = __repr__ # Override str to use the same representation


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
    ignores = []
    for n, ld in enumerate(left.dims[:-2]):
        if not isinstance(ld, hilbert.BroadcastSpace):
            continue
        rd = right.dims[n]
        if isinstance(rd, hilbert.BroadcastSpace):
            continue
        left = left.align(n, rd)
        ignores.append(n)
    
    ignores = set(ignores)
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
    data = torch.matmul(left.data, right.data)
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
    """ Perform matrix multiplication (contraction) between two `Tensor`. """
    return matmul(left, right)


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
    if left.rank() != right.rank():
        raise ValueError("Tensors must have the same number of dimensions to be added.")
    if left.dims == right.dims:
        return Tensor(data=left.data + right.data, dims=left.dims)
    # calculate the union of the StateSpaces
    union_dims = []
    for l_dim, r_dim in zip(left.dims, right.dims):
        union_dims.append(l_dim + r_dim)
    # calculate the new shape
    new_shape = tuple(u.size for u in union_dims)
    new_data = torch.zeros(new_shape, dtype=left.data.dtype, device=left.data.device)
    # fill the left tensor into the new data
    left_slices = tuple(slice(0, d.size) for d in left.dims)
    new_data[left_slices] = left.data
    # fill the right tensor into the new data
    right_embedding_order = (
        torch.tensor(hilbert.embedding_order(r, u), dtype=torch.long, device=left.data.device) 
        for r, u in zip(right.dims, union_dims)
    )
    new_data.index_put_(torch.meshgrid(*right_embedding_order, indexing='ij'), right.data, accumulate=True)

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


def permute(tensor: Tensor, *order: Tuple[int, ...]) -> Tensor:
    """
    Permute the dimensions of the tensor according to the specified order.
    
    Parameters
    ----------
    tensor : `Tensor`
        The tensor to permute.
    order : `Tuple[int, ...]`
        The desired order of dimensions.

    Returns
    -------
    `Tensor`
        The permuted tensor.
    """
    if len(order) == 1 and isinstance(order[0], (tuple, list)):
        order = tuple(order[0])
    else:
        order = tuple(order)
    if len(order) != tensor.rank():
        raise ValueError(
            f"Permutation order length {len(order)} does not match tensor dimensions {tensor.rank()}!"
        )
    
    new_data = tensor.data.permute(order)
    new_dims = tuple(tensor.dims[i] for i in order)
    
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
    new_dims = tensor.dims[:dim] + (hilbert.BroadcastSpace(),) + tensor.dims[dim:]
    
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
    if not isinstance(tensor.dims[dim], hilbert.BroadcastSpace):
        return tensor  # No squeezing needed if not BroadcastSpace
    
    new_data = tensor.data.squeeze(dim)
    new_dims = tensor.dims[:dim] + tensor.dims[dim+1:]
    
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
    if isinstance(target_dim, hilbert.BroadcastSpace):
        return tensor  # No alignment needed for BroadcastSpace
    
    if isinstance(current_dim, hilbert.BroadcastSpace):
        # Expand broadcast dimension to match the target StateSpace size.
        expanded_shape = list(tensor.data.shape)
        expanded_shape[dim] = target_dim.size
        aligned_data = tensor.data.expand(*expanded_shape)
        return Tensor(
            data=aligned_data, dims=tensor.dims[:dim] + (target_dim,) + tensor.dims[dim+1:])

    if type(current_dim) is not type(target_dim):
        raise ValueError(
            f"Cannot align dimensions with different StateSpace types: "
            f"current dim={type(current_dim)} vs target dim={type(target_dim)}!"
        )
    if not hilbert.same_span(current_dim, target_dim):
        raise ValueError(f"StateSpace at {dim} cannot be aligned to target StateSpace!")

    target_order = hilbert.flat_permutation_order(current_dim, target_dim)
    aligned_data = torch.index_select(
        tensor.data, 
        dim, 
        torch.tensor(target_order, dtype=torch.long, device=tensor.data.device))

    aligned_tensor = Tensor(
        data=aligned_data, dims=tensor.dims[:dim] + (target_dim,) + tensor.dims[dim+1:])

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
