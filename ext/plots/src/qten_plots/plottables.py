from dataclasses import dataclass, field
from typing import FrozenSet, Iterable

from qten.geometries.spatials import Offset
from qten.plottings import Plottable
from ._utils import normalize_pointcloud_marker


@dataclass(frozen=True)
class PointCloud(Plottable):
    """
    Plottable collection of spatial offsets with optional display styling.
    """

    offsets: FrozenSet[Offset] = field(default_factory=frozenset)
    name: str | None = None
    color: str | None = None
    marker: str | None = None
    opacity: float | None = None
    size: float | None = None
    border_color: str | None = None
    border_width: float | None = None

    def __post_init__(self):
        from qten.geometries.spatials import Offset

        normalized = frozenset(self.offsets)
        if not all(isinstance(offset, Offset) for offset in normalized):
            raise TypeError("PointCloud offsets must all be Offset instances.")
        if self.opacity is not None and not (0.0 <= self.opacity <= 1.0):
            raise ValueError(
                f"PointCloud opacity must lie in [0, 1], got {self.opacity}."
            )
        normalize_pointcloud_marker(self.marker)
        if self.size is not None and self.size <= 0:
            raise ValueError(f"PointCloud size must be positive, got {self.size}.")
        if self.border_width is not None and self.border_width < 0:
            raise ValueError(
                f"PointCloud border_width must be non-negative, got {self.border_width}."
            )
        object.__setattr__(self, "offsets", normalized)

    @classmethod
    def of(
        cls,
        offsets: Iterable[Offset],
        name: str | None = None,
        color: str | None = None,
        marker: str | None = None,
        opacity: float | None = None,
        size: float | None = None,
        border_color: str | None = None,
        border_width: float | None = None,
    ) -> "PointCloud":
        """Construct a [`PointCloud`][qten_plots.plottables.PointCloud] from any iterable of offsets."""
        return cls(
            offsets=frozenset(offsets),
            name=name,
            color=color,
            marker=marker,
            opacity=opacity,
            size=size,
            border_color=border_color,
            border_width=border_width,
        )
