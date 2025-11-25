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

    def conj(self) -> 'Tensor':
        return Tensor(data=self.data.conj(), dims=self.dims)


@dispatch(Tensor, Tensor)
def operator_matmul(left: Tensor, right: Tensor) -> Tensor:
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
