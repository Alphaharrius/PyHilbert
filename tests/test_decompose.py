import torch

from pyhilbert.decompose import eig, eigh
from pyhilbert.hilbert import Mode, Spectrum, hilbert
from pyhilbert.tensors import Tensor
from pyhilbert.utils import FrozenDict


def test_eigh_reconstructs_hermitian_matrix():
    torch.manual_seed(0)

    mode = Mode(count=3, attr=FrozenDict({"name": "m"}))
    space = hilbert([mode])

    data = torch.randn(3, 3, dtype=torch.complex64)
    hermitian = data + data.conj().transpose(-2, -1)
    tensor = Tensor(data=hermitian, dims=(space, space))

    eigvals, eigvecs = eigh(tensor)

    assert isinstance(eigvals.dims[-1], Spectrum)
    assert eigvecs.dims[-2] == space
    assert eigvecs.dims[-1] is eigvals.dims[-1]

    diag = torch.diag(eigvals.data).to(eigvecs.data.dtype)
    recon = eigvecs.data @ diag @ eigvecs.data.conj().transpose(-2, -1)
    assert torch.allclose(recon, hermitian, atol=1e-5, rtol=1e-5)


def test_eig_reconstructs_general_matrix():
    torch.manual_seed(0)

    mode = Mode(count=3, attr=FrozenDict({"name": "m"}))
    space = hilbert([mode])

    data = torch.randn(3, 3, dtype=torch.complex64)
    tensor = Tensor(data=data, dims=(space, space))

    eigvals, eigvecs = eig(tensor)

    assert isinstance(eigvals.dims[-1], Spectrum)
    assert eigvecs.dims[-2] == space
    assert eigvecs.dims[-1] is eigvals.dims[-1]

    diag = torch.diag(eigvals.data)
    recon = eigvecs.data @ diag @ torch.linalg.inv(eigvecs.data)
    assert torch.allclose(recon, data, atol=1e-5, rtol=1e-5)
