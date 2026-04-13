from __future__ import annotations

from collections import OrderedDict
import math
from itertools import product
from typing import Dict, Optional, Sequence, Union

import numpy as np
import sympy as sy
from sympy import ImmutableDenseMatrix

from . import AffineSpace, Lattice, Offset
from .spatials import Momentum, ReciprocalLattice


def _cutoff_from_sites(
    sites_with_distances: list[tuple[float, Offset[Lattice]]], n_nearest: int
) -> float | None:
    shell_count = 0
    previous_distance: float | None = None
    for distance, _ in sites_with_distances:
        if previous_distance is None or not math.isclose(
            distance, previous_distance, rel_tol=1e-9, abs_tol=1e-9
        ):
            shell_count += 1
            previous_distance = distance
            if shell_count == n_nearest:
                return distance
    return None


def nearest_sites(
    lattice: Lattice, center: Offset[AffineSpace] | Offset[Lattice], n_nearest: int
) -> tuple[Offset[Lattice], ...]:
    """
    Return lattice sites through the `n_nearest`-th distinct distance shell.

    Sites are ordered by increasing distance from `center`, with lattice-site
    ordering used to break ties deterministically. `n_nearest=1` returns the
    nearest-distance shell, `n_nearest=2` returns the first two distinct
    distance shells, and so on.

    Parameters
    ----------
    `lattice` : `Lattice`
        Finite lattice whose sites define the candidate region.
    `center` : `Offset[AffineSpace] | Offset[Lattice]`
        Center used to rank lattice sites by distance. The center may be an
        arbitrary offset in the lattice affine space and does not need to lie
        on a lattice site.
    `n_nearest` : `int`
        Number of distinct distance shells to include. `0` returns an empty
        region. If `n_nearest` exceeds the number of distinct distance shells
        in the finite lattice, all sites are returned.

    Returns
    -------
    `tuple[Offset[Lattice], ...]`
        Tuple of lattice sites whose distances from `center` lie in the first
        `n_nearest` distinct distance shells, ordered by increasing distance
        and then by the lattice-site ordering.

    Raises
    ------
    `ValueError`
        If `n_nearest` is negative or if `center.dim` does not match
        `lattice.dim`.
    """
    if n_nearest < 0:
        raise ValueError(f"n_nearest must be non-negative, got {n_nearest}.")
    if center.dim != lattice.dim:
        raise ValueError(
            f"center must have dimension {lattice.dim} to match the lattice, got {center.dim}."
        )
    if n_nearest == 0:
        return ()

    unit_cell_sites = tuple(lattice.unit_cell.values())
    total_sites = math.prod(lattice.shape) * len(unit_cell_sites)
    center_rep = center.rebase(lattice).rep
    origin_cell = tuple(
        int(math.floor(float(center_rep[i, 0]))) for i in range(lattice.dim)
    )

    discovered: dict[Offset[Lattice], float] = {}
    included_count = -1
    stable_after_cutoff = False
    max_local_radius = max(shape // 2 for shape in lattice.shape)

    for radius in range(max_local_radius + 1):
        ranges = [range(cell - radius, cell + radius + 1) for cell in origin_cell]
        for cell_offset in product(*ranges):
            if radius and all(
                abs(cell_offset[i] - origin_cell[i]) < radius
                for i in range(lattice.dim)
            ):
                continue
            cell_rep = ImmutableDenseMatrix(cell_offset)
            for site in unit_cell_sites:
                candidate = Offset(
                    rep=ImmutableDenseMatrix(cell_rep + site.rep), space=lattice
                )
                if candidate in discovered:
                    continue
                discovered[candidate] = center.distance(candidate)

        if len(discovered) == total_sites:
            break

        if len(discovered) < n_nearest:
            continue

        local_sites = sorted(
            ((distance, site) for site, distance in discovered.items()),
            key=lambda item: (item[0], item[1]),
        )
        cutoff_distance = _cutoff_from_sites(local_sites, n_nearest)
        if cutoff_distance is None:
            continue

        new_included_count = sum(
            1
            for distance, _ in local_sites
            if distance < cutoff_distance
            or math.isclose(distance, cutoff_distance, rel_tol=1e-9, abs_tol=1e-9)
        )
        if new_included_count == included_count:
            stable_after_cutoff = True
            break
        included_count = new_included_count

    if stable_after_cutoff or len(discovered) == total_sites:
        sites_with_distances = sorted(
            ((distance, site) for site, distance in discovered.items()),
            key=lambda item: (item[0], item[1]),
        )
    else:
        sites_with_distances = sorted(
            (
                (center.distance(candidate), candidate)
                for cell in lattice.boundaries.representatives()
                for site in unit_cell_sites
                for candidate in (
                    Offset(rep=ImmutableDenseMatrix(cell + site.rep), space=lattice),
                )
            ),
            key=lambda item: (item[0], item[1]),
        )

    cutoff_distance = _cutoff_from_sites(sites_with_distances, n_nearest)

    if cutoff_distance is None:
        return tuple(site for _, site in sites_with_distances)

    return tuple(
        site
        for distance, site in sites_with_distances
        if distance < cutoff_distance
        or math.isclose(distance, cutoff_distance, rel_tol=1e-9, abs_tol=1e-9)
    )


def interpolate_reciprocal_path(
    recip: ReciprocalLattice,
    waypoints: Sequence[Union[tuple[float, ...], str]],
    n_points: int = 100,
    labels: Optional[Sequence[str]] = None,
    points: Optional[Dict[str, tuple[float, ...]]] = None,
) -> "BzPath":
    """Build a dense reciprocal-space sample along a piecewise-linear path."""
    if len(waypoints) < 2:
        raise ValueError("At least two waypoints are required to define a path.")

    _points: Dict[str, tuple[float, ...]] = points or {}

    resolved_wp: list[tuple[float, ...]] = []
    auto_labels: list[str] = []
    for i, wp in enumerate(waypoints):
        if isinstance(wp, str):
            if wp not in _points:
                raise ValueError(
                    f"Waypoint {i} is the name '{wp}' but it was not found in "
                    f"the points dictionary. Available names: "
                    f"{sorted(_points.keys()) if _points else '(empty)'}."
                )
            resolved_wp.append(_points[wp])
            auto_labels.append(wp)
        else:
            resolved_wp.append(tuple(wp))
            auto_labels.append(str(tuple(wp)))

    dim = recip.dim
    for i, wp in enumerate(resolved_wp):
        if len(wp) != dim:
            raise ValueError(f"Waypoint {i} has {len(wp)} components, expected {dim}.")
    if n_points < len(resolved_wp):
        raise ValueError(
            f"n_points ({n_points}) must be >= number of waypoints ({len(resolved_wp)})."
        )

    basis_mat = np.array(recip.basis.evalf(), dtype=float)
    wp_frac = np.array(resolved_wp, dtype=float)
    wp_cart = wp_frac @ basis_mat.T

    seg_lengths = np.array(
        [np.linalg.norm(wp_cart[i + 1] - wp_cart[i]) for i in range(len(resolved_wp) - 1)]
    )
    total_length = seg_lengths.sum()
    n_segments = len(resolved_wp) - 1

    if total_length < 1e-15:
        raise ValueError("All waypoints are identical; path has zero length.")

    remaining = n_points - n_segments - 1
    interior_per_seg = np.zeros(n_segments, dtype=int)
    if remaining > 0:
        ideal = (seg_lengths / total_length) * remaining
        interior_per_seg = np.floor(ideal).astype(int)
        deficit = remaining - interior_per_seg.sum()
        fracs = ideal - interior_per_seg
        for idx in np.argsort(-fracs)[:deficit]:
            interior_per_seg[idx] += 1

    all_fracs: list[np.ndarray] = []
    waypoint_indices: list[int] = []

    for seg in range(n_segments):
        n_interior = int(interior_per_seg[seg])
        n_seg_points = n_interior + 1
        t_vals = np.linspace(0.0, 1.0, n_seg_points, endpoint=False)
        start = wp_frac[seg]
        end = wp_frac[seg + 1]
        waypoint_indices.append(len(all_fracs))
        for t in t_vals:
            all_fracs.append(start + t * (end - start))

    waypoint_indices.append(len(all_fracs))
    all_fracs.append(wp_frac[-1])

    seen: dict[Momentum, int] = {}
    unique_momenta: list[Momentum] = []
    path_order: list[int] = []

    for frac in all_fracs:
        rep = ImmutableDenseMatrix([sy.Rational(f).limit_denominator(10**9) for f in frac])
        k = Momentum(rep=rep, space=recip)
        if k not in seen:
            seen[k] = len(unique_momenta)
            unique_momenta.append(k)
        path_order.append(seen[k])

    from ..symbolics.state_space import BzPath, MomentumSpace

    structure: OrderedDict[Momentum, int] = OrderedDict(
        (k, i) for i, k in enumerate(unique_momenta)
    )
    k_space = MomentumSpace(structure=structure)

    all_cart = np.stack(all_fracs) @ basis_mat.T
    diffs = np.diff(all_cart, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    positions = np.concatenate(([0.0], np.cumsum(dists)))

    if labels is None:
        labels = tuple(auto_labels)
    else:
        if len(labels) != len(resolved_wp):
            raise ValueError(
                f"Number of labels ({len(labels)}) must match number of waypoints ({len(resolved_wp)})."
            )
        labels = tuple(labels)

    return BzPath(
        k_space=k_space,
        labels=labels,
        waypoint_indices=tuple(waypoint_indices),
        path_order=tuple(path_order),
        path_positions=tuple(float(p) for p in positions),
    )


# Backward-compatible alias. Prefer interpolate_reciprocal_path in new code.
interpolate_path = interpolate_reciprocal_path
