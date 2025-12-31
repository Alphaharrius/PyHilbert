from typing import Tuple
from dataclasses import dataclass
from multipledispatch import dispatch

import torch

from .abstracts import Operable, Plottable
from .hilbert import StateSpace


@dataclass(frozen=True)
class Tensor(Operable, Plottable):
    data: torch.Tensor
    dims: Tuple[StateSpace, ...]

    def conj(self) -> 'Tensor':
        return Tensor(data=self.data.conj(), dims=self.dims)
    
    # TODO: Use *args and **kwargs to allow for more flexible operations?
    def permute(self, order) -> 'Tensor':
        return permute(self, order)
    
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
        The resulting tensor after addition.
    """
    # TODO: Implement addition
    raise NotImplementedError()


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
    # TODO: Implement negation
    raise NotImplementedError()


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


# TODO: Use *args and **kwargs to allow for more flexible operations?
def permute(tensor: Tensor, order: Tuple[int, ...]) -> Tensor:
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
    # TODO: Implement permutation
    raise NotImplementedError()


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
    raise NotImplementedError()
