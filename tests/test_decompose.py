import torch

from pyhilbert.decompose import eig, eigh, qr
from pyhilbert.hilbert import Mode, FactorSpace, hilbert
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

    assert isinstance(eigvals.dims[-1], FactorSpace)
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

    assert isinstance(eigvals.dims[-1], FactorSpace)
    assert eigvecs.dims[-2] == space
    assert eigvecs.dims[-1] is eigvals.dims[-1]

    diag = torch.diag(eigvals.data)
    recon = eigvecs.data @ diag @ torch.linalg.inv(eigvecs.data)
    assert torch.allclose(recon, data, atol=1e-5, rtol=1e-5)


def test_qr_reconstructs_tall_matrix():
    torch.manual_seed(0)

    row_mode = Mode(count=4, attr=FrozenDict({"name": "row"}))
    col_mode = Mode(count=3, attr=FrozenDict({"name": "col"}))
    row_space = hilbert([row_mode])
    col_space = hilbert([col_mode])

    data = torch.randn(4, 3, dtype=torch.float64)
    tensor = Tensor(data=data, dims=(row_space, col_space))

    q, r = qr(tensor)

    assert q.dims[-2] == row_space
    assert isinstance(q.dims[-1], FactorSpace)
    assert r.dims[-1] == col_space
    assert r.dims[-2] is q.dims[-1]

    recon = q.data @ r.data
    assert torch.allclose(recon, data, atol=1e-6, rtol=1e-6)


def test_qr_reconstructs_wide_matrix():
    torch.manual_seed(0)

    row_mode = Mode(count=3, attr=FrozenDict({"name": "row"}))
    col_mode = Mode(count=5, attr=FrozenDict({"name": "col"}))
    row_space = hilbert([row_mode])
    col_space = hilbert([col_mode])

    data = torch.randn(3, 5, dtype=torch.float64)
    tensor = Tensor(data=data, dims=(row_space, col_space))

    q, r = qr(tensor)

    assert q.dims[-2] == row_space
    assert isinstance(q.dims[-1], FactorSpace)
    assert r.dims[-1] == col_space
    assert r.dims[-2] is q.dims[-1]

    recon = q.data @ r.data
    assert torch.allclose(recon, data, atol=1e-6, rtol=1e-6)
