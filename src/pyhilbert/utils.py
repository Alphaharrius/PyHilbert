from collections.abc import Mapping
from typing import Iterator, Any
import pandas as pd


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
        items = pd.DataFrame(self._items(), columns=["Key", "Value"])
        return items.to_string(index=False)

    def __repr__(self) -> str:
        return str(self)

    def __getattribute__(self, name: str):
        if name in {"_FrozenDict__items"}:
            raise AttributeError("Private storage is hidden")
        return super().__getattribute__(name)
