# Numerical Precision Analysis in PyHilbert

This document analyzes the current state of numerical precision handling in the PyHilbert library. The goal is to identify inconsistencies and areas where precision is hardcoded, to inform the implementation of a global precision setting API.

## Findings

Based on a codebase search, here are the locations where numerical precision is explicitly mentioned or hardcoded:

### `pyhilbert/tensors.py`

- In `Field.identity`, the identity matrix is initialized with `torch.complex128`:
  ```python
  mat = torch.zeros((from_space.dim, to_space.dim), dtype=torch.complex128)
  ```
- In `zeros_like` and `eye_like`, the `dtype` of the input tensor is preserved.
- `torch.long` is used for indexing, which is appropriate and should not be affected by the floating point precision settings.

### `pyhilbert/utils.py`

- The `set_dtype` function allows setting the default `torch` floating point and complex precision. It supports `float32`, `float64`, and `float128`.
- In `k_vecs_to_distances`, `torch.float64` is hardcoded when creating tensors.
  ```python
  k_vecs = torch.tensor(np.array(k_vecs_list), dtype=torch.float64)
  ...
  [torch.tensor([0.0], dtype=torch.float64), torch.cumsum(diffs, dim=0)]
  ...
  k_dist = torch.tensor([], dtype=torch.float64)
  ```

### `pyhilbert/spatials.py`

- When creating tensors from numpy arrays, `np.float64` and `torch.float64` are hardcoded.
  ```python
  np.array(basis_eval).astype(np.float64), dtype=torch.float64
  ...
  lat_reps.append(np.array(off.rep).flatten().astype(np.float64))
  ...
  np.array(lat_reps), dtype=torch.float64
  ...
  basis_reps.append(np.zeros(self.dim, dtype=np.float64))
  ...
  basis_reps.append(np.array(site_vec).flatten().astype(np.float64))
  ...
  np.array(basis_reps), dtype=torch.float64
  ```

### `pyhilbert/fourier.py`

- When creating tensors from numpy arrays, `np.float64` is hardcoded.
  ```python
  np.stack([np.array(k.rep, dtype=np.float64).reshape(-1) for k in K], axis=0)
  ...
  np.stack([np.array(r.rep, dtype=np.float64).reshape(-1) for r in R], axis=1)
  ```

### `pyhilbert/decompose.py`

- In `decompose`, the `dtype` of the input tensor is preserved.

## Summary

The `pyhilbert.utils.set_dtype` function is intended to provide a mechanism for setting the numerical precision. However, this setting is not respected in several places in the code, where `torch.float64`, `np.float64`, or `torch.complex128` are hardcoded.

To implement a uniform precision setting, the following changes are required:

1.  All hardcoded `dtype` values should be replaced with a global `dtype` variable that can be set by the user.
2.  The `pyhilbert.utils.set_dtype` function should be used to set this global `dtype` variable.
3.  Care must be taken to handle both floating point and complex data types correctly based on the global setting.
4.  The places where `numpy` arrays are converted to `torch` tensors should be updated to respect the global `dtype`.

This concludes the analysis of the numerical precision in the PyHilbert API.