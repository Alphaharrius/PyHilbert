from collections.abc import Mapping
from abc import ABCMeta
from typing import (
    Iterator,
    Any,
    Generic,
    List,
    Optional,
    Tuple,
    Dict,
    TypeVar,
    Union,
    Iterable,
    Callable,
    Type,
    cast,
)
import torch
import numpy as np

from .precision import get_precision_config


_K = TypeVar("_K")
_V = TypeVar("_V")


class FrozenDict(Mapping[_K, _V], Generic[_K, _V]):
    __slots__ = ("__items", "__hash")

    def __init__(self, *args: Any, **kwargs: Any):
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
    def _items(self) -> Tuple[Tuple[_K, _V], ...]:
        return cast(
            Tuple[Tuple[_K, _V], ...],
            object.__getattribute__(self, "_FrozenDict__items"),
        )

    # --- Mapping interface ---
    def __len__(self) -> int:
        return len(self._items())

    def __iter__(self) -> Iterator[_K]:
        for k, _ in self._items():
            yield k

    def __getitem__(self, key: _K) -> _V:
        for k, v in self._items():
            if k == key:
                return v
        raise KeyError(key)

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


def generate_k_path(
    points: Dict[str, Union[List, np.ndarray, torch.Tensor]],
    path_labels: List[str],
    resolution: int = 30,
) -> tuple:
    """
    Generates a k-path through high-symmetry points.

    Args:
        points: Dictionary mapping labels to coordinates (e.g. {'G': [0,0], 'M': [0.5, 0.5]})
        path_labels: List of labels defining the path (e.g. ['G', 'M', 'K', 'G'])
        resolution: Number of points per segment.

    Returns:
        (k_vecs, k_dist, node_indices)
        k_vecs: Tensor of k-vectors (N, D)
        k_dist: Tensor of cumulative distances (N,)
        node_indices: List of indices for the high-symmetry points.
    """
    precision = get_precision_config()
    k_vecs_list = []
    node_indices = [0]

    # Convert points to numpy for easier math
    pts_np = {}
    for k, v in points.items():
        if isinstance(v, torch.Tensor):
            pts_np[k] = v.detach().cpu().numpy().astype(precision.np_float)
        else:
            pts_np[k] = np.array(v, dtype=precision.np_float)

    for i in range(len(path_labels) - 1):
        start_label = path_labels[i]
        end_label = path_labels[i + 1]

        start_vec = pts_np[start_label]
        end_vec = pts_np[end_label]

        # Determine number of points
        # If it's the last segment, include the end point
        is_last = i == len(path_labels) - 2
        num = resolution + 1 if is_last else resolution

        t = np.linspace(0, 1, num, endpoint=is_last)

        for ti in t:
            vec = (1 - ti) * start_vec + ti * end_vec
            k_vecs_list.append(vec)

        # The next node is at the current total length of list
        # Note: k_vecs_list length grows by 'resolution' each time (except last)
        next_idx = len(k_vecs_list) - 1
        node_indices.append(next_idx)

    k_vecs = torch.tensor(np.array(k_vecs_list), dtype=precision.torch_float)

    # Recalculate distances precisely from the vectors
    if len(k_vecs) > 0:
        diffs = torch.norm(k_vecs[1:] - k_vecs[:-1], dim=1)
        k_dist = torch.cat(
            [
                torch.tensor([0.0], dtype=precision.torch_float),
                torch.cumsum(diffs, dim=0),
            ]
        )
    else:
        k_dist = torch.tensor([], dtype=precision.torch_float)

    return k_vecs, k_dist, node_indices


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
