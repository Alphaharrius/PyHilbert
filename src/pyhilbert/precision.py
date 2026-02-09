from typing import Literal, Tuple, Union
import numpy as np
import torch

global_float_dtype = torch.float64
global_complex_dtype = torch.complex128
global_np_float_dtype = np.dtype(np.float64)
global_np_complex_dtype = np.dtype(np.complex128)

PrecisionInput = Union[Literal["32", "64"], int, torch.dtype, np.dtype]


def _normalize_precision(
    precision: PrecisionInput,
) -> Tuple[torch.dtype, torch.dtype, np.dtype, np.dtype]:
    """
    Normalize user precision input to (torch_float, torch_complex, np_float, np_complex).
    """
    if precision in ("32", 32):
        return (
            torch.float32,
            torch.complex64,
            np.dtype(np.float32),
            np.dtype(np.complex64),
        )
    if precision in ("64", 64):
        return (
            torch.float64,
            torch.complex128,
            np.dtype(np.float64),
            np.dtype(np.complex128),
        )

    if isinstance(precision, np.dtype) or (
        isinstance(precision, type) and issubclass(precision, np.generic)
    ):
        np_dt = np.dtype(precision)
        if np_dt in (np.dtype(np.float32), np.dtype(np.complex64)):
            return (
                torch.float32,
                torch.complex64,
                np.dtype(np.float32),
                np.dtype(np.complex64),
            )
        if np_dt in (np.dtype(np.float64), np.dtype(np.complex128)):
            return (
                torch.float64,
                torch.complex128,
                np.dtype(np.float64),
                np.dtype(np.complex128),
            )

    if isinstance(precision, torch.dtype):
        if precision in (torch.float32, torch.complex64):
            return (
                torch.float32,
                torch.complex64,
                np.dtype(np.float32),
                np.dtype(np.complex64),
            )
        if precision in (torch.float64, torch.complex128):
            return (
                torch.float64,
                torch.complex128,
                np.dtype(np.float64),
                np.dtype(np.complex128),
            )

    raise ValueError("Precision must be 32/64, torch.dtype, or np.dtype")


def set_precision(
    precision: PrecisionInput,
    set_torch_default: bool = True,
) -> None:
    """
    Sets default precision and updates global dtype references.
    """
    global global_float_dtype, global_complex_dtype
    global global_np_float_dtype, global_np_complex_dtype

    float_dt, complex_dt, np_float_dt, np_complex_dt = _normalize_precision(precision)

    if set_torch_default:
        torch.set_default_dtype(float_dt)

    global_float_dtype = float_dt
    global_complex_dtype = complex_dt
    global_np_float_dtype = np_float_dt
    global_np_complex_dtype = np_complex_dt


def get_precision_config() -> Tuple[torch.dtype, torch.dtype, np.dtype, np.dtype]:
    """
    Returns current global precision config (torch_float, torch_complex, np_float, np_complex).
    """
    return (
        global_float_dtype,
        global_complex_dtype,
        global_np_float_dtype,
        global_np_complex_dtype,
    )
