from typing import Tuple
from dataclasses import dataclass
from multipledispatch import dispatch

import torch

from .abstracts import Operable
from .hilbert import StateSpace


@dataclass(frozen=True)
class Tensor(Operable):
    data: torch.Tensor
    dims: Tuple[StateSpace, ...]

    def __post_init__(self) -> None:
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
    
    # TODO: Add informative print-outs


@dispatch(Tensor, Tensor)
def operator_matmul(left: Tensor, right: Tensor) -> Tensor:
    """
    Perform matrix multiplication (contraction) between two Tensors with
    the same order of dimensions. If the intra-ordering within the `StateSpace`s differ, 
    the `right` tensor is permuted to match the ordering of the `left` tensor before multiplication.
    
    Parameters
    ----------
    left : `Tensor`
        The left tensor to multiply.
    right : `Tensor`
        The right tensor to multiply.

    Returns
    -------
    `Tensor`
        The resulting tensor after multiplication.
    """
    left_dim = left.dims[-1]
    right_dim = right.dims[-2]

    if type(left_dim) is not type(right_dim):
        raise ValueError(
            f"Cannot contract Tensors with different types of StateSpaces: "
            f"{type(left_dim)} and {type(right_dim)}!"
        )
    
    if not left_dim.has_same_span(right_dim):
        raise ValueError(f"Cannot contract Tensors with different StateSpaces!")
    
    right_order = right_dim.permute_order(left_dim)
    right_data = torch.index_select(right.data, -2, torch.tensor(right_order, dtype=torch.long))

    new_data = torch.matmul(left.data, right_data)
    new_dims = left.dims[:-1] + right.dims[:-2] + (right.dims[-1],)

    return Tensor(data=new_data, dims=new_dims)


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
    if len(left.dims) != len(right.dims):
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
    grid_indices = (torch.tensor(StateSpace.embedding_indices(r, u), dtype=torch.long, device=left.data.device) 
                    for r, u in zip(right.dims, union_dims))
    new_data.index_put_(torch.meshgrid(*grid_indices, indexing='ij'), right.data, accumulate=True)

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
    if len(order) != len(tensor.dims):
        raise ValueError(
            f"Permutation order length {len(order)} does not match tensor dimensions {len(tensor.dims)}!"
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
