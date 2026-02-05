from typing import Literal
import torch


def set_precision(precision: Literal["16", "32", "64", "128"]) -> None:
    """
    Set the default floating point precision for PyTorch tensors.

    This implicitly controls the precision of complex number operations:
    - '32':  Sets float32 (Single Precision).
             Complex tensors will default to complex64 (two 32-bit components).
    - '64':  Sets float64 (Double Precision).
             Complex tensors will default to complex128 (two 64-bit components).
    - '128': Sets float128 (Quad Precision), if supported.

    Parameters
    ----------
    precision : Literal['16', '32', '64', '128']
        The desired bit-width for the real floating point components.
    """
    dtype_map = {
        "16": torch.float16,
        "32": torch.float32,
        "64": torch.float64,
    }
    if precision == "128":
        if hasattr(torch, "float128"):
            dtype = torch.float128
        else:
            raise ValueError("float128 and complex256 is not supported on this system.")
    elif precision in dtype_map:
        dtype = dtype_map[precision]
    else:
        raise ValueError(f"Precision must be one of {list(dtype_map.keys()) + ['128']}")

    # Apply the default dtype
    torch.set_default_dtype(dtype)
