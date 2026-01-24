from typing import List
from collections import namedtuple

import torch

from .hilbert import FactorSpace
from .hilbert import same_span
from .tensors import Tensor


def _band_counts(eigenvalues: torch.Tensor, eps: float) -> List[int]:
    n = eigenvalues.shape[-1]
    if eps < 0.0:
        eps = 0.0
    if n == 0:
        raise ValueError("Eigenvalues tensor has zero size along the last dimension.")
    reference = eigenvalues.reshape(-1, n)[0]
    diffs = (reference[1:] - reference[:-1]).abs()
    split_idxs = torch.nonzero(diffs > eps, as_tuple=False).flatten()
    if split_idxs.numel() == 0:
        return [n]
    splits = (split_idxs + 1).tolist()
    band_counts: List[int] = []
    prev = 0
    for s in splits:
        band_counts.append(s - prev)
        prev = s
    band_counts.append(n - prev)
    return band_counts


EigH = namedtuple("EigH", ["eigenvalues", "eigenvectors"])


def _assert_eig_dims(tensor: Tensor) -> None:
    if tensor.rank() < 2:
        raise ValueError(
            "Input tensor must have at least two dimensions for matrix decomposition."
        )

    dim0, dim1 = tensor.dims[-2], tensor.dims[-1]
    if not same_span(dim0, dim1):
        raise ValueError(
            "The last two dimensions of the tensor must span the same Hilbert space."
        )


def eigh(tensor: Tensor, group_band_eps: float = 1e-8) -> EigH:
    """
    Perform eigen-value decomposition on a `Tensor` with Hermitian matrix at the last two indices.

    Parameters
    ----------
    `tensor` : `Tensor`
        Input tensor with Hermitian matrices at the last two indices.
    `group_band_eps` : `float`, default `1e-8`
        Tolerance for grouping eigenvalues into bands. Eigenvalues within this
        tolerance are considered part of the same band.

    Returns
    -------
    `EigH`
        A namedtuple `(eigenvalues, eigenvectors)` where:
        - `eigenvalues` is a `Tensor` containing the eigenvalues.
        - `eigenvectors` is a `Tensor` containing the corresponding eigenvectors.
        - `eigenvalues` dtype matches the real dtype of the input (complex inputs
          yield real eigenvalues of the corresponding real dtype).
        - `eigenvectors` dtype matches the input dtype.
        - `eigenvalues.dims` keeps all leading dimensions and replaces the last
          two matrix dimensions with a single `FactorSpace` dimension.
        - `eigenvectors.dims` keeps the leading dimensions, then uses the second
          last dimension (the row space) followed by the `FactorSpace` dimension.

    Notes
    -----
    `torch.linalg.eigh` is differentiable for Hermitian inputs, but the gradients
    can be ill-defined or unstable when eigenvalues are degenerate or nearly
    degenerate. If you use this in autograd, consider stabilizing the spectrum
    (e.g., with a small perturbation) or avoiding backpropagation through
    eigenvectors when bands are expected to merge.
    """
    _assert_eig_dims(tensor)

    dim0 = tensor.dims[-2]
    target = tensor.align(-1, dim0)  # Align column space to match the row space
    eigenvalues, eigenvectors = torch.linalg.eigh(target.data)

    band_counts = _band_counts(eigenvalues, float(group_band_eps))
    spectrum = FactorSpace.from_band_counts(band_counts)

    print(target.dims)

    eigvals = Tensor(
        data=eigenvalues,
        dims=target.dims[:-2] + (spectrum,),
    )
    eigvecs = Tensor(
        data=eigenvectors,
        dims=target.dims[:-2] + (dim0, spectrum),
    )

    return EigH(eigvals, eigvecs)


def eigvalsh(tensor: Tensor, group_band_eps: float = 1e-8) -> Tensor:
    """
    Compute eigenvalues of a `Tensor` with Hermitian matrix at the last two indices.

    Parameters
    ----------
    `tensor` : `Tensor`
        Input tensor with Hermitian matrices at the last two indices.
    `group_band_eps` : `float`, default `1e-8`
        Tolerance for grouping eigenvalues into bands. Eigenvalues within this
        tolerance are considered part of the same band.

    Returns
    -------
    `Tensor`
        A `Tensor` containing the eigenvalues with:
        - dtype matching the real dtype of the input (complex inputs
          yield real eigenvalues of the corresponding real dtype).
        - dims keeping all leading dimensions and replacing the last
          two matrix dimensions with a single `FactorSpace` dimension.
    """
    _assert_eig_dims(tensor)

    dim0 = tensor.dims[-2]
    target = tensor.align(-1, dim0)  # Align column space to match the row space
    eigenvalues = torch.linalg.eigvalsh(target.data)

    band_counts = _band_counts(eigenvalues, float(group_band_eps))
    spectrum = FactorSpace.from_band_counts(band_counts)

    vals = Tensor(
        data=eigenvalues,
        dims=target.dims[:-2] + (spectrum,),
    )

    return vals


def eig(tensor: Tensor, group_band_eps: float = 1e-8) -> EigH:
    """
    Perform eigen-value decomposition on a `Tensor` with general square matrices
    at the last two indices.

    Parameters
    ----------
    `tensor` : `Tensor`
        Input tensor with square matrices at the last two indices.
    `group_band_eps` : `float`, default `1e-8`
        Tolerance for grouping eigenvalues into bands. Eigenvalues within this
        tolerance are considered part of the same band based on the order
        returned by `torch.linalg.eig`.

    Returns
    -------
    `EigH`
        A namedtuple `(eigenvalues, eigenvectors)` where:
        - `eigenvalues` is a `Tensor` containing the eigenvalues.
        - `eigenvectors` is a `Tensor` containing the corresponding eigenvectors.
        - `eigenvalues` dtype is the complex dtype of the input (real inputs
          yield complex eigenvalues).
        - `eigenvectors` dtype matches the complex dtype of the input.
        - `eigenvalues.dims` keeps all leading dimensions and replaces the last
          two matrix dimensions with a single `FactorSpace` dimension.
        - `eigenvectors.dims` keeps the leading dimensions, then uses the second
          last dimension (the row space) followed by the `FactorSpace` dimension.

    Notes
    -----
    `torch.linalg.eig` does not guarantee any ordering of the eigenvalues. Band
    grouping is therefore based on the returned order and may not match
    expectations for non-Hermitian spectra.
    """
    _assert_eig_dims(tensor)

    dim0 = tensor.dims[-2]
    target = tensor.align(-1, dim0)  # Align column space to match the row space
    eigenvalues, eigenvectors = torch.linalg.eig(target.data)

    band_counts = _band_counts(eigenvalues, float(group_band_eps))
    spectrum = FactorSpace.from_band_counts(band_counts)

    eigvals = Tensor(
        data=eigenvalues,
        dims=target.dims[:-2] + (spectrum,),
    )
    eigvecs = Tensor(
        data=eigenvectors,
        dims=target.dims[:-2] + (dim0, spectrum),
    )

    return EigH(eigvals, eigvecs)


def eigvals(tensor: Tensor, group_band_eps: float = 1e-8) -> Tensor:
    """
    Compute eigenvalues of a `Tensor` with general square matrices at the last
    two indices.

    Parameters
    ----------
    `tensor` : `Tensor`
        Input tensor with square matrices at the last two indices.
    `group_band_eps` : `float`, default `1e-8`
        Tolerance for grouping eigenvalues into bands. Eigenvalues within this
        tolerance are considered part of the same band based on the order
        returned by `torch.linalg.eigvals`.

    Returns
    -------
    `Tensor`
        A `Tensor` containing the eigenvalues with:
        - dtype matching the complex dtype of the input (real inputs
          yield complex eigenvalues).
        - dims keeping all leading dimensions and replacing the last
          two matrix dimensions with a single `FactorSpace` dimension.

    Notes
    -----
    `torch.linalg.eigvals` does not guarantee any ordering of the eigenvalues.
    Band grouping is therefore based on the returned order and may not match
    expectations for non-Hermitian spectra.
    """
    _assert_eig_dims(tensor)

    dim0 = tensor.dims[-2]
    target = tensor.align(-1, dim0)  # Align column space to match the row space
    eigenvalues = torch.linalg.eigvals(target.data)

    band_counts = _band_counts(eigenvalues, float(group_band_eps))
    spectrum = FactorSpace.from_band_counts(band_counts)

    vals = Tensor(
        data=eigenvalues,
        dims=target.dims[:-2] + (spectrum,),
    )

    return vals


QR = namedtuple("QR", ["Q", "R"])


def qr(tensor: Tensor) -> QR:
    """
    Perform QR decomposition on a `Tensor` with matrices at the last two indices.

    Returns
    -------
    `QR`
        A namedtuple `(Q, R)` where:
        - `Q` is a `Tensor` with orthonormal columns (reduced QR).
        - `R` is an upper-triangular `Tensor`.
        - Output dims preserve leading dimensions and map the last two dims to
          `(row_dim, spectral_dim)` for `Q` and `(spectral_dim, col_dim)` for
          `R`, where `spectral_dim` is a `FactorSpace` describing the
          reduced QR bond dimension.
    """
    if tensor.rank() < 2:
        raise ValueError(
            "Input tensor must have at least two dimensions for matrix decomposition."
        )

    row_dim = tensor.dims[-2]
    col_dim = tensor.dims[-1]

    q_data, r_data = torch.linalg.qr(tensor.data, mode="reduced")
    spectral_dim = FactorSpace.from_band_counts([q_data.shape[-1]])

    q = Tensor(
        data=q_data,
        dims=tensor.dims[:-2] + (row_dim, spectral_dim),
    )
    r = Tensor(
        data=r_data,
        dims=tensor.dims[:-2] + (spectral_dim, col_dim),
    )

    return QR(q, r)
