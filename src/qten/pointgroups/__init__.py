"""
Point-group and abelian symmetry helpers.

This package provides compact constructors and symbolic representations for
finite abelian point operations, especially Cartesian rotations, mirrors, and
their induced actions on polynomial bases.

Core exports
------------
[`pointgroup`][qten.pointgroups] parses a compact query string into a symmetry
object. [`AbelianGroup`][qten.pointgroups.abelian.AbelianGroup] represents a
linear abelian symmetry acting on coordinate functions.
[`AbelianOpr`][qten.pointgroups.abelian.AbelianOpr] is the affine extension of
an abelian group with translation. [`AbelianBasis`][qten.pointgroups.abelian.AbelianBasis]
is the eigen-basis function object produced from symmetry representations.

Joint-basis helper
------------------
[`joint_abelian_basis`][qten.pointgroups.ops.joint_abelian_basis] constructs a
common eigen-basis for compatible commuting operators.

Non-abelian helper
------------------
[`FiniteGroupRepresentation`][qten.pointgroups.nonabelian.FiniteGroupRepresentation]
and its projector helpers provide a first-step workflow for finite non-abelian
isotypic decomposition.
"""

from ._pointgroups import pointgroup as pointgroup

from .abelian import (
    AbelianBasis as AbelianBasis,
    AbelianGroup as AbelianGroup,
    AbelianOpr as AbelianOpr,
)
from .nonabelian import (
    FiniteGroupRepresentation as FiniteGroupRepresentation,
    NonAbelianSectorBasis as NonAbelianSectorBasis,
    nonabelian_column_symmetrize as nonabelian_column_symmetrize,
    nonabelian_isotypic_projectors as nonabelian_isotypic_projectors,
)
from .ops import (
    joint_abelian_basis as joint_abelian_basis,
)
