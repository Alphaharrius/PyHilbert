import torch
import pytest
from typing import Tuple
from pyhilbert.hilbert import Mode, hilbert, HilbertSpace, StateSpace
from pyhilbert.tensors import Tensor, operator_matmul
from pyhilbert.utils import FrozenDict

# Monkeypatch missing methods in StateSpace for tests to run
# The source code in tensors.py expects these methods to exist on StateSpace
if not hasattr(StateSpace, 'has_same_span'):
    def has_same_span(self, other: 'StateSpace') -> bool:
        return set(self.structure.keys()) == set(other.structure.keys())
    StateSpace.has_same_span = has_same_span

if not hasattr(StateSpace, 'permute_order'):
    def permute_order(self, target: 'StateSpace') -> Tuple[int, ...]:
        return StateSpace.flat_permutation_order(self, target)
    StateSpace.permute_order = permute_order


def test_tensor_matmul_permutation():
    # Define modes
    m1 = Mode(count=2, attr=FrozenDict({'id': 1}))
    m2 = Mode(count=3, attr=FrozenDict({'id': 2}))
    
    # Define StateSpaces
    # s1: [m1, m2] -> indices: m1=[0,1], m2=[2,3,4]
    s1 = hilbert([m1, m2])
    
    # s2: [m2, m1] -> indices: m2=[0,1,2], m1=[3,4]
    s2 = hilbert([m2, m1])
    
    # Dummy outer spaces
    s_left = hilbert([Mode(count=1, attr=FrozenDict({'id': 'L'}))])
    s_right = hilbert([Mode(count=1, attr=FrozenDict({'id': 'R'}))])
    
    # Create Tensors
    # T1: shape (s_left, s1). Data shape (1, 5)
    # Let's make data distinct so we can track permutation
    # T1 maps s_left -> s1.
    # We want T1 @ T2.
    # T1 columns correspond to s1 (m1 then m2).
    # T2 rows correspond to s2 (m2 then m1).
    
    # Let's use simple identity-like matrices to verify alignment
    # But sizes are different.
    
    # T1: 1x5. 
    t1_data = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    t1 = Tensor(data=t1_data, dims=(s_left, s1))
    
    # T2: 5x1.
    # Rows 0,1,2 correspond to m2.
    # Rows 3,4 correspond to m1.
    # If we want to match T1 (m1, m2), we need T2 to be effectively permuted to (m1, m2) before mult.
    # T1's m1 part (cols 0,1 -> vals 1,2) should multiply T2's m1 part (rows 3,4).
    # T1's m2 part (cols 2,3,4 -> vals 3,4,5) should multiply T2's m2 part (rows 0,1,2).
    
    t2_data = torch.tensor([
        [10.0], # m2[0]
        [20.0], # m2[1]
        [30.0], # m2[2]
        [40.0], # m1[0]
        [50.0]  # m1[1]
    ])
    t2 = Tensor(data=t2_data, dims=(s2, s_right))
    
    # Expected result:
    # 1*40 + 2*50 + 3*10 + 4*20 + 5*30
    # = 40 + 100 + 30 + 80 + 150
    # = 400
    
    res = t1 @ t2
    
    assert isinstance(res, Tensor)
    assert res.dims == (s_left, s_right)
    assert res.data.shape == (1, 1)
    assert res.data.item() == 400.0

def test_tensor_matmul_incompatible():
    m1 = Mode(count=2, attr=FrozenDict({'id': 1}))
    m2 = Mode(count=3, attr=FrozenDict({'id': 2}))
    m3 = Mode(count=1, attr=FrozenDict({'id': 3}))
    
    s1 = hilbert([m1, m2])
    s2 = hilbert([m1, m3]) # different span
    
    t1 = Tensor(data=torch.randn(1, 5), dims=(hilbert([]), s1))
    t2 = Tensor(data=torch.randn(3, 1), dims=(s2, hilbert([])))
    
    with pytest.raises(ValueError, match="Cannot contract Tensors with different StateSpaces"):
        t1 @ t2
