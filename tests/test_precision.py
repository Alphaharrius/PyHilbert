import pytest
import torch
import pyhilbert


@pytest.mark.filterwarnings("ignore:ComplexHalf support is experimental")
def test_set_precision():
    # 1. Test Half Precision ('16')
    pyhilbert.set_precision("16")
    assert torch.get_default_dtype() == torch.float16

    real_tensor = torch.tensor([1.0])
    # This line triggers the warning, but pytest will now ignore it
    complex_tensor = torch.tensor([1 + 1j])

    assert real_tensor.dtype == torch.float16
    assert complex_tensor.dtype == torch.complex32  # 16 + 16 bits

    # 2. Test Single Precision ('32')
    pyhilbert.set_precision("32")
    assert torch.get_default_dtype() == torch.float32
    assert torch.tensor([1 + 1j]).dtype == torch.complex64

    # 3. Test Double Precision ('64')
    pyhilbert.set_precision("64")
    assert torch.get_default_dtype() == torch.float64
    assert torch.tensor([1 + 1j]).dtype == torch.complex128
