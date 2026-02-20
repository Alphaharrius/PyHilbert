from collections.abc import Mapping
from abc import ABCMeta
from typing import (
    Iterator,
    Any,
    List,
    Optional,
    Tuple,
    Dict,
    Iterable,
    Callable,
    Type,
    cast,
)
from sympy import ImmutableDenseMatrix, Rational, Float
import torch


class FrozenDict(Mapping):
    __slots__ = ("__items", "__hash")

    def __init__(self, *args, **kwargs):
        data = dict(*args, **kwargs)
        try:
            fitems = frozenset(data.items())  # ensures all keys/vals are hashable
        except TypeError as e:
            raise TypeError(
                "All keys and values must be hashable. "
                "Use deep_freeze() for nested mutables."
            ) from e
        object.__setattr__(
            self, "_FrozenDict__items", tuple(fitems)
        )  # hidden, immutable
        object.__setattr__(self, "_FrozenDict__hash", hash(fitems))

    # internal accessor that bypasses the guard
    def _items(self):
        return object.__getattribute__(self, "_FrozenDict__items")

    # --- Mapping interface ---
    def __len__(self) -> int:
        return len(self._items())

    def __iter__(self) -> Iterator:
        for k, _ in self._items():
            yield k

    def __getitem__(self, key: Any) -> Any:
        for k, v in self._items():
            if k == key:
                return v
        raise KeyError(key)

    # convenient immutable snapshots
    def keys(self):
        return tuple(k for k, _ in self._items())

    def items(self):
        return tuple(self._items())

    def values(self):
        return tuple(v for _, v in self._items())

    # --- equality & hash ---
    def __hash__(self) -> int:
        return object.__getattribute__(self, "_FrozenDict__hash")

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, Mapping):
            try:
                return frozenset(self._items()) == frozenset(other.items())
            except TypeError:
                return False
        return NotImplemented

    def __str__(self) -> str:
        return str(dict(self._items()))

    def __repr__(self) -> str:
        return repr(dict(self._items()))

    def __getattribute__(self, name: str):
        if name in {"_FrozenDict__items"}:
            raise AttributeError("Private storage is hidden")
        return super().__getattribute__(name)


# --- Plotting Helpers ---


def compute_bonds(
    coords: torch.Tensor, dim: int
) -> Tuple[
    List[Optional[float]], List[Optional[float]], Optional[List[Optional[float]]]
]:
    """
    Generate bond lines connecting nearest neighbors using PyTorch.
    Returns (x_lines, y_lines, z_lines) where lists contain coordinates separated by None.
    z_lines is None if dim != 3.
    """
    if coords.size(0) < 2:
        return [], [], None if dim != 3 else None

    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    dists = torch.norm(diff, dim=-1)

    dists.fill_diagonal_(float("inf"))

    min_dist = torch.min(dists)
    if torch.isinf(min_dist):
        return [], [], None if dim != 3 else None

    tol = 1e-4
    pairs = torch.nonzero(dists <= min_dist + tol)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]

    if pairs.size(0) == 0:
        return [], [], None if dim != 3 else None

    p1 = coords[pairs[:, 0]]
    p2 = coords[pairs[:, 1]]

    p1_np = p1.numpy()
    p2_np = p2.numpy()

    x_lines: List[Optional[float]] = []
    y_lines: List[Optional[float]] = []
    z_lines: Optional[List[Optional[float]]] = [] if dim == 3 else None
    nan = None

    for i in range(len(p1_np)):
        x_lines.extend([p1_np[i, 0], p2_np[i, 0], nan])
        y_lines.extend([p1_np[i, 1], p2_np[i, 1], nan])
        if dim == 3 and z_lines is not None:
            z_lines.extend([p1_np[i, 2], p2_np[i, 2], nan])

    return x_lines, y_lines, z_lines


def matchby(
    source: Iterable[Any], dest: Iterable[Any], base_func: Callable[[Any], Any]
) -> Dict[Any, Any]:
    """
    Map elements from source to destination using a provided mapping function.
    Parameters
    ----------
    `source` : `Iterable[Any]`
        The source elements to be mapped.
    `dest` : `Iterable[Any]`
        The destination elements to map to.
    `base_func` : `Callable[[Any], Any]`
        A function that defines the comparison baseline.

    Returns
    -------
    `Dict[Any, Any]`
        A dictionary mapping each source element to its corresponding destination element `source -> dest`.
    """
    mapping: Dict[Any, Any] = {}

    source_base: Dict[Any, Any] = {m: base_func(m) for m in source}
    dest_base: Dict[Any, Any] = {base_func(m): m for m in dest}

    if len(dest_base) != len(tuple(dest)):
        raise ValueError("Destination elements have non-unique base values!")

    for sm, sb in source_base.items():
        if sb not in dest_base:
            raise ValueError(
                f"Source element {sm} with base {sb} has no match in destination!"
            )
        mapping[sm] = dest_base[sb]

    return mapping


def subtypes(cls: Type) -> Tuple[ABCMeta, ...]:
    """
    Return all transitive subclasses of a class.

    Parameters
    ----------
    `cls` : `Type`
        The class to inspect.

    Returns
    -------
    `Tuple[ABCMeta, ...]`
        A tuple containing all direct and indirect subclasses of `cls`.
    """
    out = set()
    stack = list(cls.__subclasses__())
    while stack:
        sub = stack.pop()
        if sub not in out:
            out.add(sub)
            stack.extend(sub.__subclasses__())
    return cast(Tuple[ABCMeta, ...], tuple(out))

def validate_matrix(mat: ImmutableDenseMatrix, name: str = "matrix") -> None:
    """
    Validates that all entries in the matrix are exact SymPy expressions (no Floats).
    Raises TypeError with index information if an invalid entry is found.
    """
    for i in range(mat.rows):
        for j in range(mat.cols):
            val = mat[i, j]
            if getattr(val, 'has', lambda x: False)(Float):
                raise TypeError(
                    f"Invalid entry in {name} at ({i}, {j}): {val} (type {type(val)}). "
                    "Entries must be exact (no floating-point numbers allowed)."
                )
