import pytest
from pyhilbert.utils import FrozenDict
import pyhilbert
import torch


def test_frozendict_creation_and_access():
    d = FrozenDict({"a": 1, "b": 2})
    assert len(d) == 2
    assert d["a"] == 1
    assert d["b"] == 2

    with pytest.raises(KeyError):
        _ = d["c"]


def test_frozendict_immutability():
    d = FrozenDict({"a": 1})

    # It's a Mapping, so it doesn't have __setitem__ or __delitem__ exposed by default if not implemented.
    # But we should check it doesn't allow modification.
    with pytest.raises(TypeError):
        d["a"] = 2


def test_frozendict_hash():
    d1 = FrozenDict({"a": 1, "b": 2})
    d2 = FrozenDict({"b": 2, "a": 1})  # Order shouldn't matter

    assert hash(d1) == hash(d2)
    assert d1 == d2

    # Can be used as dictionary key
    mapping = {d1: "value"}
    assert mapping[d2] == "value"


def test_frozendict_eq():
    d1 = FrozenDict({"a": 1})
    d2 = {"a": 1}
    assert d1 == d2  # Should compare equal to dict

    d3 = FrozenDict({"a": 2})
    assert d1 != d3


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
