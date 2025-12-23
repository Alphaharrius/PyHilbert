import pytest
from pyhilbert.hilbert import Mode, hilbert, HilbertSpace
from pyhilbert.utils import FrozenDict

def test_mode_creation():
    attr = FrozenDict({'a': 1})
    m = Mode(count=2, attr=attr)
    assert m.count == 2
    assert m.dim == 2
    assert m['a'] == 1

def test_hilbert_space_creation():
    attr1 = FrozenDict({'id': 1})
    m1 = Mode(count=2, attr=attr1)
    
    attr2 = FrozenDict({'id': 2})
    m2 = Mode(count=3, attr=attr2)
    
    hs = hilbert([m1, m2])
    assert isinstance(hs, HilbertSpace)
    # The dimension of the StateSpace is the number of elements in the structure (keys),
    # which are the modes. So it should be 2.
    assert hs.dim == 2 
    assert len(hs.structure) == 2

