from __future__ import annotations

import math

from sympy import ImmutableDenseMatrix

from .spatials import AffineSpace, Lattice, Offset


def nearest_sites(
    lattice: Lattice, center: Offset[AffineSpace] | Offset[Lattice], n_nearest: int
) -> tuple[Offset[Lattice], ...]:
    """
    Return lattice sites through the `n_nearest`-th distinct distance shell.

    Sites are ordered by increasing distance from `center`, with lattice-site
    ordering used to break ties deterministically. `n_nearest=1` returns the
    nearest-distance shell, `n_nearest=2` returns the first two distinct
    distance shells, and so on. If `n_nearest` exceeds the number of distinct
    distance shells in the finite lattice, all sites are returned.
    """
    if n_nearest < 0:
        raise ValueError(f"n_nearest must be non-negative, got {n_nearest}.")
    if center.dim != lattice.dim:
        raise ValueError(
            f"center must have dimension {lattice.dim} to match the lattice, got {center.dim}."
        )
    if n_nearest == 0:
        return ()

    sites_with_distances = sorted(
        (
            (center.distance(candidate), candidate)
            for cell in lattice.boundaries.representatives()
            for site in lattice.unit_cell.values()
            for candidate in (
                Offset(rep=ImmutableDenseMatrix(cell + site.rep), space=lattice),
            )
        ),
        key=lambda item: (item[0], item[1]),
    )
    shell_count = 0
    cutoff_distance: float | None = None
    previous_distance: float | None = None
    for distance, _ in sites_with_distances:
        if previous_distance is None or not math.isclose(
            distance, previous_distance, rel_tol=1e-9, abs_tol=1e-9
        ):
            shell_count += 1
            previous_distance = distance
            if shell_count == n_nearest:
                cutoff_distance = distance

    if cutoff_distance is None:
        return tuple(site for _, site in sites_with_distances)

    return tuple(
        site
        for distance, site in sites_with_distances
        if distance < cutoff_distance
        or math.isclose(distance, cutoff_distance, rel_tol=1e-9, abs_tol=1e-9)
    )
