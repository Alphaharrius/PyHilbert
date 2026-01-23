from typing import Dict, Tuple
from typing import cast

from multipledispatch import dispatch

import numpy as np
import torch

from .spatials import Momentum, Offset
from .hilbert import MomentumSpace, HilbertSpace, Mode
from .hilbert import mode_mapping
from .tensors import Tensor
from .tensors import mapping_matrix


@dispatch(tuple, tuple)
def fourier_transform(K: Tuple[Momentum, ...], R: Tuple[Offset, ...]) -> torch.Tensor:
    """
    Compute Fourier phase factors between momentum and real-space offsets.

    This returns the kernel `exp(-2π i k·r)` evaluated for all pairs of
    momentum points in `K` and offsets in `R`.

    Parameters
    ----------
    `K` : `Tuple[Momentum, ...]`
        Momentum points.
    `R` : `Tuple[Offset, ...]`
        Real-space offsets.

    Returns
    -------
    `torch.Tensor`
        Complex tensor of shape `(len(K), len(R))` with elements
        `exp(-2π i k·r)`.
    """
    ten_K = torch.from_numpy(  # (K, d)
        np.stack([np.array(k.rep, dtype=np.float64).reshape(-1) for k in K], axis=0)
    )
    ten_R = torch.from_numpy(  # (d, R)
        np.stack([np.array(r.rep, dtype=np.float64).reshape(-1) for r in R], axis=1)
    )
    exponent = -2j * np.pi * torch.matmul(ten_K, ten_R)  # (K, R)
    return torch.exp(exponent)  # (K, R)


@dispatch(MomentumSpace, HilbertSpace, HilbertSpace)  # type: ignore[no-redef]
def fourier_transform(
    k_space: MomentumSpace,
    bloch_space: HilbertSpace,
    region_space: HilbertSpace,
    *,
    r_name: str = "r",
) -> torch.Tensor:
    """
    Build the Fourier transform tensor between `k_space` and `region_space`.

    This computes phase factors for `k_space` against the offsets collected
    from `region_space`, then maps region modes into `bloch_space` using
    the coordinate named by `r_name`.

    Parameters
    ----------
    `k_space` : `MomentumSpace`
        Momentum space defining the k points.
    `bloch_space` : `HilbertSpace`
        Bloch space to map region modes into.
    `region_space` : `HilbertSpace`
        Real-space region defining offsets.
    `r_name` : `str`, default `"r"`
        Name of the spatial coordinate in region modes used for the mapping.

    Returns
    -------
    `Tensor`
        Tensor with data shape `(K, B, R)` and dims
        `(k_space, bloch_space, region_space)`.
    """
    K: Tuple[Momentum] = k_space.elements()
    R: Tuple[Offset] = region_space.collect(r_name)
    f = fourier_transform(K, R)  # (K, R)

    region_to_bloch: Dict[Mode, Mode] = mode_mapping(
        region_space, bloch_space, lambda m: cast(Offset, m[r_name]).fractional()
    )

    map = mapping_matrix(region_space, bloch_space, region_to_bloch).transpose(
        0, 1
    )  # (B, R)
    # (K, 1, R) * (1, B, R)
    f = f.unsqueeze(1) * map.data.unsqueeze(0)
    return Tensor(data=f, dims=(k_space, bloch_space, region_space))  # (K, B, R)
