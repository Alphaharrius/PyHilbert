import unittest
import torch
from collections import OrderedDict
from pyhilbert.tensors import Tensor, matmul
from pyhilbert.hilbert import HilbertSpace, StateSpace, Mode, BroadcastSpace
from pyhilbert.utils import FrozenDict

class TestMode(Mode):
    pass

class TestMatmul(unittest.TestCase):
    def setUp(self):
        # Define some dummy modes
        self.mode_a = TestMode(count=2, attr=FrozenDict({'name': 'a'}))
        self.mode_b = TestMode(count=3, attr=FrozenDict({'name': 'b'}))
        self.mode_c = TestMode(count=4, attr=FrozenDict({'name': 'c'}))
        
        # Create StateSpaces
        # Space 1: A + B (dim 5)
        structure1 = OrderedDict()
        structure1[self.mode_a] = slice(0, 2)
        structure1[self.mode_b] = slice(2, 5)
        self.space1 = HilbertSpace(structure=structure1)
        
        # Space 2: C (dim 4)
        structure2 = OrderedDict()
        structure2[self.mode_c] = slice(0, 4)
        self.space2 = HilbertSpace(structure=structure2)
        
        # Space 3: B + A (dim 5) - Same span as Space 1 but different order
        structure3 = OrderedDict()
        structure3[self.mode_b] = slice(0, 3)
        structure3[self.mode_a] = slice(3, 5)
        self.space3 = HilbertSpace(structure=structure3)

    def test_basic_matmul(self):
        # A (2, 5) x B (5, 4) -> C (2, 4)
        # Dimensions: (space2, space1) x (space1, space2)
        # Wait, matmul contracts left[-1] and right[-2]
        # left: (M, K) -> (space2, space1)
        # right: (K, N) -> (space1, space2)
        # result: (M, N) -> (space2, space2)
        
        data_left = torch.randn(self.space2.size, self.space1.size)
        tensor_left = Tensor(data=data_left, dims=(self.space2, self.space1))
        
        data_right = torch.randn(self.space1.size, self.space2.size)
        tensor_right = Tensor(data=data_right, dims=(self.space1, self.space2))
        
        result = matmul(tensor_left, tensor_right)
        
        expected_data = torch.matmul(data_left, data_right)
        self.assertTrue(torch.allclose(result.data, expected_data))
        self.assertEqual(result.dims, (self.space2, self.space2))

    def test_matmul_with_alignment(self):
        # Test where contraction dimensions have different internal order (space1 vs space3)
        # left: (space2, space1)
        # right: (space3, space2)
        # space1 and space3 cover {A, B} but in different order.
        
        data_left = torch.randn(self.space2.size, self.space1.size)
        tensor_left = Tensor(data=data_left, dims=(self.space2, self.space1))
        
        # Create data for right tensor corresponding to space3 ordering
        # space3: B (0-3), A (3-5)
        # space1: A (0-2), B (2-5)
        # logical vector v in space3 order [b0, b1, b2, a0, a1]
        # logical vector v in space1 order [a0, a1, b0, b1, b2]
        
        data_right_s3 = torch.randn(self.space3.size, self.space2.size)
        tensor_right = Tensor(data=data_right_s3, dims=(self.space3, self.space2))
        
        # When matmul(left, right) happens:
        # right is aligned to left.dims[-1] (space1).
        # alignment permutes right from space3 to space1.
        
        result = matmul(tensor_left, tensor_right)
        
        # Manually align right data to check result
        # Permutation from space3 to space1
        # space3 indices: 0,1,2 (B), 3,4 (A)
        # space1 indices: A first (so take 3,4 from space3), B second (take 0,1,2 from space3)
        # align indices: [3, 4, 0, 1, 2]
        indices = torch.tensor([3, 4, 0, 1, 2], dtype=torch.long)
        aligned_data_right = torch.index_select(data_right_s3, 0, indices)
        
        expected_data = torch.matmul(data_left, aligned_data_right)
        self.assertTrue(torch.allclose(result.data, expected_data), "Data mismatch after alignment")
        self.assertEqual(result.dims, (self.space2, self.space2))

    def test_broadcasting_missing_dims(self):
        # left: (space1) -> interpreted as (space1)
        # right: (space2, space1, space2)
        # unsqueezed left -> (Broadcast, Broadcast, space1)
        # This might fail because matmul expects left to be at least 1D? 
        # Actually matmul aligns left[-1] with right[-2].
        # If left is 1D: dims=(space1,), left[-1] is space1.
        # If right is 3D: dims=(D1, space1, D2). right[-2] is space1.
        # Broadcast match:
        # left unsqueezed to match rank 3: (Broadcast, Broadcast, space1)
        # batch dims: 
        # 0: left(Broadcast) vs right(D1) -> match
        # 1: left(Broadcast) vs right(space1) -> This is the contraction dim for right? No, right[-2] is contraction.
        # Wait, if right is (D1, space1, D2), rank is 3.
        # left is (space1), rank 1.
        # _match_dims makes left rank 3: (Broadcast, Broadcast, space1).
        # _align_dims:
        # n=0: left(Broadcast), right(D1) -> ok.
        # n=1: left(Broadcast), right(space1) -> ok.
        # Contraction: left[-1] (space1) vs right[-2] (space1). Match!
        
        # Let's try:
        # Left: (space1, )
        # Right: (space2, space1, space2)
        
        data_left = torch.randn(self.space1.size)
        tensor_left = Tensor(data=data_left, dims=(self.space1,))
        
        data_right = torch.randn(self.space2.size, self.space1.size, self.space2.size)
        tensor_right = Tensor(data=data_right, dims=(self.space2, self.space1, self.space2))
        
        result = matmul(tensor_left, tensor_right)
        
        # Expected:
        # left broadcasted to (1, 1, space1.size)
        # right is (space2.size, space1.size, space2.size)
        # but the broadcasting logic in matmul implementation:
        # unsqueeze adds BroadcastSpace.
        # align expands BroadcastSpace if the other is not BroadcastSpace.
        # align(tensor, dim, target):
        # if current is Broadcast, expand data.
        
        # left becomes (space2, space1_broadcast?? No wait)
        # _align_dims_for_matopt iterates up to -2.
        # left dims: (Broadcast, Broadcast, space1)
        # right dims: (space2, space1, space2)
        # loop n=0: left[0] is Broadcast, right[0] is space2. left aligned to space2.
        # data_left expanded at dim 0.
        
        # loop n=1: left[1] is Broadcast, right[1] is space1. 
        # But wait, contraction is left[-1] vs right[-2].
        # left[-1] is space1. right[-2] is space1.
        # So left is (space2, Broadcast, space1).
        # right is (space2, space1, space2).
        # This seems overlapping.
        # Contraction logic:
        # "The contraction always happens between left.dims[-1] and right.dims[-2]."
        
        # So for left=(space1), right=(space2, space1, space2)
        # left -> (Broadcast, Broadcast, space1)
        # align n=0: left aligned to space2 -> (space2, Broadcast, space1)
        # align n=1: left aligned to ? right[1] is space1.
        # left aligned to space1 -> (space2, space1, space1) (Broadcast expands to space1)
        # result dims: left[:-1] + right[-1:]
        # left[:-1] is (space2, space1). right[-1:] is (space2).
        # result: (space2, space1, space2).
        
        # But wait, torch.matmul logic:
        # left (B, N, K) x right (B, K, M)
        # here left is (space2, space1, space1). right is (space2, space1, space2).
        # torch.matmul(left, right) -> left shape (S2, S1, S1), right shape (S2, S1, S2).
        # batch dim S2 matches.
        # matrix mul: (S1, S1) x (S1, S2) -> (S1, S2).
        # final shape (S2, S1, S2).
        
        # Let's verify this behavior is what we expect.
        # Effectively batched vector-matrix mult?
        # A (K) x B (M, K, N)
        # -> A broadcasted to (M, K) -> (M, 1, K) or something?
        
        # Actually usually A(K) @ B(..., K, N) works in torch.
        # In torch, (K) @ (M, K, N) -> (M, N).
        # But here we force ranks to match.
        # left becomes (1, 1, K) -> expand to (M, 1, K)? No, align expands to target size.
        # left dim 0 aligned to right dim 0 (space2). size M.
        # left dim 1 aligned to right dim 1 (space1). size K.
        # so left becomes (M, K, K).
        # right is (M, K, N).
        # result (M, K, N).
        
        # Is this intended?
        # If I have a vector v and a batch of matrices M_i, I might want v @ M_i for all i.
        # v: (K). M: (Batch, K, N).
        # v -> (1, 1, K).
        # align dim 0: (Batch, 1, K). (Broadcast to Batch).
        # align dim 1: (Batch, K, K) ?? Why align dim 1 to K?
        # right dim 1 is K (from K, N).
        # So left becomes (Batch, K, K).
        # data is replicated K times along dim 1? That seems wrong for standard matmul.
        
        # Standard broadcast: (K) vs (B, K, N)
        # (K) treated as (1, K) if we follow numpy/torch broadcasting rules?
        # But here `_match_dims` unsqueezes at 0.
        # (K) -> (1, 1, K).
        # right is (B, K, N).
        # dim 0: 1 vs B -> expand to B. -> (B, 1, K).
        # dim 1: 1 vs K -> expand to K. -> (B, K, K).
        
        # This seems to imply PyHilbert's matmul might behave differently than torch.matmul if broadcasting aligns batch dims aggressively.
        # Or maybe I should not align dim 1 if it is the contraction dim for right?
        # `_align_dims_for_matopt` loops `left.dims[:-2]`.
        # So for left (B, B, K) it only aligns dim 0.
        # dim 1 is left.dims[-2]. It is skipped by loop.
        # So left remains (B, Broadcast, K).
        # right is (B, K, N).
        # torch.matmul((B, 1, K), (B, K, N)) -> (B, N).
        # Result dims: left[:-1] (B, Broadcast) + right[-1:] (N) -> (B, Broadcast, N).
        # Wait, BroadcastSpace is a valid dimension in output?
        # If so, size is 1.
        
        expected_data_torch = torch.matmul(data_left, data_right)
        # torch result for (K) @ (M, K, N) is (M, N).
        
        result = matmul(tensor_left, tensor_right)
        
        # If the code works as analyzed:
        # left unsqueezed: (Broadcast, Broadcast, space1)
        # align loop over [:-2]: index 0 only.
        # index 0: left[0] (Broadcast) aligns to right[0] (space2).
        # left -> (space2, Broadcast, space1). Data shape (S2, 1, S1).
        # torch.matmul((S2, 1, S1), (S2, S1, S2))
        # -> (S2, 1, S2).
        # Result dims: left[:-1] + right[-1:] = (space2, Broadcast) + (space2) = (space2, Broadcast, space2).
        # Shape (S2, 1, S2).
        # This is essentially (S2, S2) with an unsqueezed middle dim.
        
        # Check if result matches this expectation
        self.assertEqual(len(result.dims), 3)
        self.assertEqual(result.dims[0], self.space2)
        self.assertIsInstance(result.dims[1], BroadcastSpace)
        self.assertEqual(result.dims[2], self.space2)
        
        # data check
        # torch.matmul(v, M) -> (M, N) usually?
        # A = torch.randn(5)
        # B = torch.randn(4, 5, 4)
        # torch.matmul(A, B).shape -> (4, 4).
        # My result is (4, 1, 4). Squeeze -> (4, 4).
        self.assertTrue(torch.allclose(result.data.squeeze(1), expected_data_torch))

    def test_broadcast_space_alignment(self):
        # Explicit broadcasting test using BroadcastSpace
        # left: (Broadcast, space1) -> effectively (1, 5)
        # right: (space2, space1, space2) -> (4, 5, 4)
        
        # Create a tensor with explicit BroadcastSpace
        # BroadcastSpace has size 1 implicitly for data generation usually? 
        # But Tensor.data shape must match dims. BroadcastSpace doesn't have a fixed size property that returns 1? 
        # StateSpace.size relies on structure. BroadcastSpace structure is empty, size is 0?
        # Let's check StateSpace.size implementation.
        # size returns structure[last].stop. If empty, 0.
        
        # If BroadcastSpace size is 0, we can't create a tensor with dim size 0 and expect it to broadcast to N?
        # Usually broadcasting dim has size 1.
        # But BroadcastSpace in PyHilbert seems to handle "unsqueezed" dims.
        # unsqueeze() creates data with dim size 1.
        
        # Let's use unsqueeze to create the tensor with BroadcastSpace
        data_orig = torch.randn(self.space1.size)
        tensor_orig = Tensor(data=data_orig, dims=(self.space1,))
        from pyhilbert.tensors import unsqueeze
        tensor_left = unsqueeze(tensor_orig, 0) # (Broadcast, space1)
        
        # right tensor
        data_right = torch.randn(self.space2.size, self.space1.size, self.space2.size)
        tensor_right = Tensor(data=data_right, dims=(self.space2, self.space1, self.space2))
        
        # This is effectively the same as missing dims test but we manually created the BroadcastSpace tensor
        result = matmul(tensor_left, tensor_right)
        
        self.assertEqual(len(result.dims), 3)
        self.assertEqual(result.dims[0], self.space2)
        # The result logic in matmul: left[:-1] + right[-1:]
        # left: (Broadcast, space1). left[:-1] -> (Broadcast,)
        # right: (space2, space1, space2). right[-1:] -> (space2,)
        # Wait, but during alignment:
        # left (Broadcast, space1), right (space2, space1, space2)
        # match dims -> left unsqueezed -> (Broadcast, Broadcast, space1)
        # So tensor_left which is (Broadcast, space1) is unsqueezed to (Broadcast, Broadcast, space1).
        # And proceeds as before.
        
        # What if we have (space2, Broadcast) x (space2, space1)?
        # left: (space2, Broadcast) (e.g. 4, 1)
        # right: (space2, space1) (e.g. 4, 5)
        # contraction: Broadcast vs space2.
        # Broadcast vs space2 -> This should fail or expand?
        # contraction requires alignment.
        # align(left, -1, space2). left[-1] is Broadcast.
        # align: if current is Broadcast, expand.
        # So left becomes (space2, space2). Data expanded.
        # Then matmul (S2, S2) x (S2, S1) -> (S2, S1).
        # This is outer product if S2 was 1, but here S2=4.
        # It's (4, 4) x (4, 5) -> (4, 5).
        # where (4, 4) is formed by repeating the column vector 4 times?
        
        # Test case where right tensor has BroadcastSpace at contraction dimension
        # left: (space2, space2) -> (4, 4)
        # right: (Broadcast, space1) -> (1, 5)
        # contraction: space2 (4) vs Broadcast (1).
        # matmul aligns right[-2] (Broadcast) to left[-1] (space2).
        # expected: right expands to (4, 5). matmul((4,4), (4,5)) -> (4, 5).
        
        data_left = torch.randn(self.space2.size, self.space2.size)
        tensor_left = Tensor(data=data_left, dims=(self.space2, self.space2))
        
        data_right = torch.randn(1, self.space1.size)
        tensor_right = Tensor(data=data_right, dims=(BroadcastSpace(structure=OrderedDict()), self.space1))
        
        result = matmul(tensor_left, tensor_right)
        
        # Verify
        # right expanded
        expanded_right = data_right.expand(self.space2.size, self.space1.size)
        expected = torch.matmul(data_left, expanded_right)
        
        self.assertEqual(result.dims, (self.space2, self.space1))
        self.assertTrue(torch.allclose(result.data, expected))
        
    def test_incompatible_shapes(self):
        # left: (space2, space2) -> (4, 4)
        # right: (space1, space2) -> (5, 4)
        # contraction: space2 (4) vs space1 (5) -> Error
        
        t1 = Tensor(torch.randn(4, 4), (self.space2, self.space2))
        t2 = Tensor(torch.randn(5, 4), (self.space1, self.space2))
        
        with self.assertRaises(ValueError):
            matmul(t1, t2)

if __name__ == '__main__':
    unittest.main()

