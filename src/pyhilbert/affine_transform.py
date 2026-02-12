from dataclasses import dataclass
from typing import Dict, Literal, Tuple, cast
from collections import OrderedDict
from itertools import product
from functools import lru_cache, partial, reduce

import sympy as sy

from .abstracts import HasBase, Functional, Gaugable, GaugeBasis, Gauged, GaugeInvariant
from .spatials import AffineSpace, Spatial, Offset, Momentum
from .hilbert import Mode, MomentumSpace, HilbertSpace, hilbert
from .tensors import Tensor, mapping_matrix
from .fourier import fourier_transform
from .utils import FrozenDict


@dataclass(frozen=True)
class AffineFunction(Spatial, Gaugable, GaugeBasis):
    """
    Symbolic affine function expressed in a polynomial basis over given axes.

    Attributes
    ----------
    `expr`: `sy.Expr`
        Symbolic expression in `axes` representing the affine function.
    `axes`: `Tuple[sy.Symbol, ...]`
        Ordered tuple of symbols defining the coordinate axes.
    `order`: `int`
        Polynomial order used to build the basis representation.
    `rep`: `sy.ImmutableDenseMatrix`
        Coefficient vector in the Euclidean monomial basis (column matrix).
    """

    expr: sy.Expr
    axes: Tuple[sy.Symbol, ...]
    order: int
    rep: sy.ImmutableDenseMatrix

    @property
    def dim(self):
        """Number of axes (spatial dimension) for this affine function."""
        return len(self.axes)

    def __str__(self):
        return f"AffineFunction({str(self.expr)})"

    def __repr__(self):
        return f"AffineFunction({repr(self.expr)})"


@dataclass(frozen=True)
class AffineGroupElement(Functional, HasBase[AffineSpace]):
    """
    Affine group element acting on polynomial coordinate functions.

    This class combines a linear representation (`irrep`) with a translation
    (`offset`) and exposes multiple representations:
    - `full_rep`: Kronecker power of `irrep` for the full tensor product basis.
    - `rep`: Symmetrized representation on the commuting Euclidean monomial basis.
    - `affine_rep`: Homogeneous affine matrix in the physical basis of `offset.space`.

    Parameters
    ----------
    irrep : sy.ImmutableDenseMatrix
        Linear representation matrix acting on the coordinate axes.
    axes : Tuple[sy.Symbol, ...]
        Ordered symbols defining the coordinate axes (used to build monomials).
    offset : Offset
        Translation component with its associated `AffineSpace`.
    basis_function_order : int
        Polynomial order used to build the monomial basis (degree).
    """

    irrep: sy.ImmutableDenseMatrix
    axes: Tuple[sy.Symbol, ...]
    offset: Offset
    basis_function_order: int

    @lru_cache
    def __full_indices(self):
        return tuple(product(*((self.axes,) * self.basis_function_order)))

    @lru_cache
    def __commute_indices(self):
        indices = self.__full_indices()
        _, select_rules = AffineGroupElement.__get_contract_select_rules(indices)
        sorted_rules = sorted(select_rules, key=lambda x: x[1])
        return tuple(indices[n] for n, _ in sorted_rules)

    @property
    @lru_cache
    def euclidean_basis(self) -> sy.ImmutableDenseMatrix:
        indices = self.__commute_indices()
        return sy.ImmutableDenseMatrix([sy.prod(idx) for idx in indices]).T

    @staticmethod
    @lru_cache
    def __get_contract_select_rules(indices: Tuple[Tuple[sy.Symbol, ...], ...]):
        commute_index_table: OrderedDict[Tuple[sy.Symbol, ...], int] = OrderedDict()
        contract_indices = []
        select_indices = []
        order_indices = set()
        order_idx = 0
        for n, idx in enumerate(indices):
            key = tuple(sorted(idx, key=lambda s: s.name))
            m = commute_index_table.setdefault(key, order_idx)

            contract_indices.append((n, m))
            if m not in order_indices:
                select_indices.append((n, m))
                order_indices.add(m)
                order_idx += 1

        return contract_indices, select_indices

    @property
    @lru_cache
    def full_rep(self):
        return reduce(sy.kronecker_product, (self.irrep,) * self.basis_function_order)

    @property
    @lru_cache
    def rep(self) -> sy.ImmutableDenseMatrix:
        indices = self.__full_indices()
        contract_indices, select_indices = self.__get_contract_select_rules(indices)

        contract_matrix = sy.zeros(len(indices), len(select_indices))
        for i, j in contract_indices:
            contract_matrix[i, j] = 1

        select_matrix = sy.zeros(len(indices), len(select_indices))
        for i, j in select_indices:
            select_matrix[i, j] = 1

        return select_matrix.T @ self.full_rep @ contract_matrix

    @property
    @lru_cache
    def affine_rep(self) -> sy.ImmutableDenseMatrix:
        """
        Use the `AffineSpace` of `offset` to build the affine transform matrix in
        physical (space-basis) coordinates.
        It will take the form of:
        ```
        [ R | t ]
        [ 0 | 1 ]
        ```
        where R and t are mapped into the `offset.space` basis via:
        `R = B * irrep * B^-1` and `t = B * offset.rep`, with `B = offset.space.basis`.
        """
        space = self.offset.space
        B = space.basis
        if not isinstance(B, sy.ImmutableDenseMatrix):
            B = sy.ImmutableDenseMatrix(B)
        B_inv = B.inv()

        R = self.irrep
        if not isinstance(R, sy.ImmutableDenseMatrix):
            R = sy.ImmutableDenseMatrix(R)
        R = B @ R @ B_inv

        t = self.offset.rep
        if not isinstance(t, sy.ImmutableDenseMatrix):
            t = sy.ImmutableDenseMatrix(t)
        t = B @ t

        top = R.row_join(t)
        bottom = sy.zeros(1, R.cols).row_join(sy.ones(1, 1))
        return sy.ImmutableDenseMatrix(top.col_join(bottom))

    @property
    @lru_cache
    def basis(self) -> FrozenDict:
        transform = self.rep
        eig = transform.eigenvects()

        tbl = {}
        for v, _, vec_group in eig:
            vec = vec_group[0]
            # principle term is the first non-zero term
            principle_term = next(x for x in vec if x != 0)

            rep = vec / principle_term
            expr = sy.simplify(rep.dot(self.euclidean_basis))
            tbl[v] = AffineFunction(
                expr=expr, axes=self.axes, order=self.basis_function_order, rep=rep
            )

        return FrozenDict(tbl)

    def base(self) -> AffineSpace:
        """Get the acting space of this affine group element."""
        return self.offset.space

    def rebase(self, new_base: AffineSpace) -> "AffineGroupElement":
        """
        Change the acting space of this affine group element to a new `AffineSpace`.
        """
        return AffineGroupElement(
            irrep=self.irrep,
            axes=self.axes,
            offset=self.offset.rebase(new_base),
            basis_function_order=self.basis_function_order,
        )

    def with_origin(self, origin: Offset) -> "AffineGroupElement":
        """
        Return an equivalent affine group element expressed relative to a new origin.

        Given the affine action in coordinate form:
            `x -> R x + t`
        shifting the origin by `o` (so x = x' + o) yields:
            `x' -> R x' + t'`
        with:
            `t' = t + (R - I) o`

        Parameters
        ----------
        `origin` : `Offset`
            The new origin expressed in an affine space. If it differs from this
            element's space, the element is rebased to `origin.space` first.

        Returns
        -------
        `AffineGroupElement`
            A new affine group element with the same linear part and adjusted
            translation so the action is expressed about `origin`.
        """
        if origin.space != self.offset.space:
            t = self.rebase(origin.space)
        else:
            t = self

        irrep = t.irrep
        if not isinstance(irrep, sy.ImmutableDenseMatrix):
            irrep = sy.ImmutableDenseMatrix(irrep)

        o_rep = origin.rep
        if not isinstance(o_rep, sy.ImmutableDenseMatrix):
            o_rep = sy.ImmutableDenseMatrix(o_rep)

        t_rep = t.offset.rep
        if not isinstance(t_rep, sy.ImmutableDenseMatrix):
            t_rep = sy.ImmutableDenseMatrix(t_rep)

        ident = sy.eye(irrep.rows)
        if not isinstance(ident, sy.ImmutableDenseMatrix):
            ident = sy.ImmutableDenseMatrix(ident)

        new_rep = t_rep + (irrep - ident) @ o_rep
        new_offset = Offset(rep=sy.ImmutableDenseMatrix(new_rep), space=origin.space)

        return AffineGroupElement(
            irrep=irrep,
            axes=t.axes,
            offset=new_offset,
            basis_function_order=t.basis_function_order,
        )

    def group_elements(self, max_order: int = 128) -> Tuple["AffineGroupElement", ...]:
        """
        Generate the cyclic group elements produced by this irrep.

        Starts from the identity and repeatedly multiplies by this element,
        stopping once an element repeats or `max_order` is reached.
        """
        if max_order <= 0:
            return tuple()

        irrep = self.irrep
        axes = self.axes
        basis_order = self.basis_function_order

        current = sy.eye(irrep.rows)
        if not isinstance(current, sy.ImmutableDenseMatrix):
            current = sy.ImmutableDenseMatrix(current)

        elements = []
        seen = set()
        for _ in range(max_order):
            if current in seen:
                break
            seen.add(current)
            elements.append(
                AffineGroupElement(
                    irrep=current,
                    axes=axes,
                    offset=self.offset,
                    basis_function_order=basis_order,
                )
            )
            next_mat = current @ irrep
            if not isinstance(next_mat, sy.ImmutableDenseMatrix):
                next_mat = sy.ImmutableDenseMatrix(next_mat)
            current = next_mat

        return tuple(elements)


@AffineGroupElement.register(GaugeInvariant)
def _affine_transform_gauge_invariant(
    t: AffineGroupElement, v: GaugeInvariant
) -> Gauged[GaugeInvariant, sy.Expr]:
    """Apply an affine group element to a gauge-invariant object."""
    return Gauged(gaugable=v, gauge=sy.Integer(1))


@AffineGroupElement.register(AffineFunction)
def _affine_transform_affine_function(
    t: AffineGroupElement, f: AffineFunction
) -> Gauged[AffineFunction, sy.Expr]:
    """
    Apply an affine group element to a basis function and extract its phase factor.

    This treats the affine group element as a linear operator on the monomial basis
    of order `f.order`. The result of applying the representation to `f.rep`
    (the coefficient vector of `f` in that basis) must be a scalar multiple of the
    original vector for `f` to be an eigenfunction of the transform. When that holds,
    the scalar is the phase factor and we return it together with the original
    function `f` (the basis function itself does not change, only its phase).

    The procedure is as follows:
    - Compute `transformed_rep = t.rep @ f.rep`.
    - If `transformed_rep == phase * f.rep` for a single scalar `phase`, then
      `f` is a basis function for this transform and `phase` is returned.
    - If no such scalar exists (or the basis vector is zero), raise a ValueError.

    Parameters
    ----------
    `t` : `AffineGroupElement`
        The affine group element (transform) to apply.
    `f` : `AffineFunction`
        The basis function to be transformed.

    Returns
    -------
    `Gauged[AffineFunction, sy.Expr]`
        A named tuple containing:
        - `gauge`: The symbolic phase factor (sy.Expr) such that
          `t.rep @ f.rep == phase * f.rep`.
        - `gaugable`: The original `AffineFunction` (unchanged).

    Raises
    ------
    `ValueError`
        If the axes of `t` and `f` do not match, or if `f` is not an eigenfunction
        of the transform represented by `t`.
    """
    if set(t.axes) != set(f.axes):
        raise ValueError(
            f"Axes of AbelianGroup and PointGroupBasis must match: {t.axes} != {f.axes}"
        )

    if t.basis_function_order != f.order:
        t = AffineGroupElement(
            irrep=t.irrep,
            axes=t.axes,
            offset=t.offset,
            basis_function_order=f.order,
        )

    g_irrep = t.rep
    basis_rep = f.rep
    transformed_rep = g_irrep @ basis_rep

    phase = None
    for n in range(transformed_rep.rows):
        basis_term = basis_rep[n]
        transformed_term = transformed_rep[n]
        if basis_term != 0:
            if phase is None:
                phase = sy.simplify(transformed_term / basis_term)
            else:
                if sy.simplify(transformed_term - phase * basis_term) != 0:
                    raise ValueError(f"{f} is not a basis function!")
        else:
            if sy.simplify(transformed_term) != 0:
                raise ValueError(f"{f} is not a basis function!")

    if phase is None:
        raise ValueError(f"{f} is a trivial basis function: zero")

    return Gauged(gauge=phase, gaugable=f)


@AffineGroupElement.register(Offset)
def _affine_transform_offset(t: AffineGroupElement, offset: Offset) -> Offset:
    """
    Apply an affine group element to a spatial Offset using homogeneous coordinates.

    This implementation:
    - Ensures the transform acts in the same AffineSpace as the input Offset by
      rebasing the transform if necessary.
    - Uses the affine (homogeneous) representation of the transform to combine
      rotation/shear and translation in a single matrix multiply.
    - Preserves the input Offset's space in the returned result.

    Parameters
    ----------
    `t` : `AffineGroupElement`
        The affine group element to apply. If its internal `offset.space` does
        not match `offset.space`, the transform is rebased to the Offset's space.
    `offset` : `Offset`
        The spatial offset (column vector) to transform.

    Returns
    -------
    `Offset`
        A new `Offset` expressed in the same `AffineSpace` as the input `offset`.

    Notes
    -----
    The method constructs a homogeneous coordinate vector:
    `[offset.rep; 1]`, multiplies by `t.affine_rep`, then discards the trailing
    homogeneous component. The result remains a column vector of shape `(dim, 1)`.
    """
    if offset.space != t.offset.space:
        t = t.rebase(offset.space)

    affine_rep = t.affine_rep
    rep = offset.rep
    if not isinstance(rep, sy.ImmutableDenseMatrix):
        rep = sy.ImmutableDenseMatrix(rep)

    hom = rep.col_join(sy.ones(1, 1))
    new_hom = affine_rep @ hom
    new_rep = new_hom[:-1, :]
    return Offset(rep=sy.ImmutableDenseMatrix(new_rep), space=offset.space)


@AffineGroupElement.register(Momentum)
def _affine_transform_momentum(t: AffineGroupElement, k: Momentum) -> Momentum:
    """
    Apply an affine group element to a Momentum in fractional reciprocal coordinates.

    This implementation assumes:
    - `k.rep` stores fractional coordinates in the reciprocal lattice basis
      (values typically in [0, 1) per component).
    - The affine group's linear part is expressed in the *physical* real-space
      basis of `t.base()` via `t.affine_rep`.
    - Translations do not act on momenta, so only the linear part is used.

    Let:
    - `R_phys` be the real-space linear map in physical coordinates,
      i.e. the top-left block of `t.affine_rep`.
    - `G` be the reciprocal lattice basis (columns), `G = k.base().basis`.
      In this codebase `G = 2π * B^{-T}` for real-space basis `B`.
    - `k_frac` be the fractional reciprocal coordinates (`k.rep`).

    Then physical momentum is `k_phys = G * k_frac`, and it transforms as
    `k_phys' = (R_phys^{-1})^T * k_phys` (contravariant rule).
    Mapping back to fractional coordinates gives:

        k_frac' = G^{-1} * (R_phys^{-1})^T * G * k_frac

    The `2π` factor in `G` cancels with `G^{-1}`, so it does not appear explicitly.
    The output is wrapped with `.fractional()` to keep components in the first
    Brillouin zone.

    Parameters
    ----------
    `t` : `AffineGroupElement`
        The affine group element to apply. If its base affine space does not
        match the real-space dual of `k`, it is rebased accordingly.
    `k` : `Momentum`
        The momentum expressed in fractional reciprocal coordinates of its
        reciprocal lattice basis.

    Returns
    -------
    `Momentum`
        The transformed momentum in the same reciprocal lattice space as `k`,
        wrapped into the first Brillouin zone via `.fractional()`.
    """
    real_space = k.base().dual.affine
    if t.base() != real_space:
        t = t.rebase(real_space)

    linear_rep = t.affine_rep[:-1, :-1]
    if not isinstance(linear_rep, sy.ImmutableDenseMatrix):
        linear_rep = sy.ImmutableDenseMatrix(linear_rep)

    # Transform fractional reciprocal coordinates using
    # k_frac' = G^{-1} * (R_phys^{-1})^T * G * k_frac
    recip_basis = k.base().basis
    if not isinstance(recip_basis, sy.ImmutableDenseMatrix):
        recip_basis = sy.ImmutableDenseMatrix(recip_basis)
    recip_basis_inv = recip_basis.inv()
    reciprocal_rep = recip_basis_inv @ linear_rep.inv().T @ recip_basis

    rep = k.rep
    if not isinstance(rep, sy.ImmutableDenseMatrix):
        rep = sy.ImmutableDenseMatrix(rep)
    new_rep = reciprocal_rep @ rep
    return Momentum(rep=sy.ImmutableDenseMatrix(new_rep), space=k.base()).fractional()


@AffineGroupElement.register(Gaugable, order="back")
def _affine_transform_gaugable(
    t: AffineGroupElement, v: Gaugable
) -> Gauged[Gaugable, sy.Expr]:
    """Transform a gaugable object by updating its gauge phase.

    Parameters
    ----------
    `t` : `AffineGroupElement`
        Affine symmetry operation applied to the gauge representation of `v`.
        The operation is evaluated through `t.apply(...)`, and only its
        gauge phase contribution is used by this method.
    `v` : `Gaugable`
        Object that provides a gauge representation via `.gauge_repr()`. The
        original value is preserved and wrapped in a `Gauged` container.

    Returns
    -------
    `Gauged[Gaugable, sy.Expr]`
        A gauged wrapper whose `gaugable` field is the original input `v` and
        whose `gauge` field is the phase returned by applying `t` to `v`'s gauge
        representation.
    """
    basis = v.gauge_repr()
    phase, _ = t.apply(basis)
    return Gauged(gaugable=v, gauge=phase)


def _optional_transform_mode_attr(t: AffineGroupElement, v: Mode):
    """Transform a Mode attribute if the transform allows it."""
    if not t.allows(v):
        return v
    return t.apply(v)


@AffineGroupElement.register(Mode, order="front")
def _affine_transform_mode(t: AffineGroupElement, m: Mode) -> Mode:
    """Apply an affine transformation to each transformable attribute of a mode.

    Parameters
    ----------
    `t` : `AffineGroupElement`
        Affine transformation to apply. For each attribute in `m`, this
        function checks `t.allows(attr)` and applies `t.transform(attr)` only
        when that attribute is supported by the transform.
    `m` : `Mode`
        Input mode whose named attributes are visited and conditionally
        transformed.

    Returns
    -------
    `Mode`
        A mode instance with the same attribute structure as `m`, where each
        attribute is transformed by `t` if allowed, and left unchanged
        otherwise.
    """
    attr_names = m.attr_names()
    func = partial(_optional_transform_mode_attr, t)
    apply = {name: func for name in attr_names}
    return m.update(**apply)


@AffineGroupElement.register(HilbertSpace)
def _affine_transform_hilbert(t: AffineGroupElement, h: HilbertSpace) -> Tensor:
    mode_to_gauged: Dict[Mode, Gauged[Mode, sy.Expr]] = {
        m: cast(Gauged[Mode, sy.Expr], t(m)) for m in h
    }
    gauge_table: Dict[Tuple[Mode, Mode], sy.Expr] = {
        (m, cast(Mode, gauged.gaugable)): gauged.gauge
        for m, gauged in mode_to_gauged.items()
    }
    mode_mapping: Dict[Mode, Mode] = {m: gm for m, gm in gauge_table.keys()}
    gh = hilbert(m for m in mode_mapping.values())

    tmat = mapping_matrix(h, gh, mode_mapping)  # (h, gh)
    return tmat


def bandtransform(
    t: AffineGroupElement,
    tensor: Tensor,
    opt: Literal["left", "right", "both"] = "both",
) -> Tensor:
    """
    Apply an affine symmetry action to a momentum-resolved operator tensor.

    The expected tensor shape is `(K, B_left, B_right)` where `K` is a
    `MomentumSpace` and `B_left`, `B_right` are `HilbertSpace`s. Depending on
    `opt`, this function applies the symmetry-induced basis transform on the
    left side, right side, or both sides of the band tensor.

    For each transformed side, a k-dependent matrix is built from:
    - the affine action on the Hilbert space basis (`t(space)`), and
    - Fourier transforms that connect Bloch and real-space sectors.

    Momentum handling:
    - The k action is treated as a relabeling/permutation of sectors.
    - We align the k-axis of the transform tensors to the canonical `kspace`
      ordering before multiplication.
    - The input tensor itself is not pre-remapped in k; remapping is used only
      to align transform blocks with each momentum sector.

    Parameters
    ----------
    `t` : `AffineGroupElement`
        Affine transformation to apply.
    `tensor` : `Tensor`
        Momentum-space tensor with dims
        `(MomentumSpace, HilbertSpace, HilbertSpace)`.
    `opt` : `Literal["left", "right", "both"]`, default `"both"`
        Which side(s) to transform.

    Returns
    -------
    `Tensor`
        The transformed tensor with the same dimension types.

    Raises
    ------
    `ValueError`
        If `opt` is invalid, if `tensor` is not rank-3 with dims
        `(MomentumSpace, HilbertSpace, HilbertSpace)`, or if a Hilbert space
        side is not symmetry-compatible with `t`.
    """
    if opt not in ("both", "left", "right"):
        raise ValueError(f"Invalid option {opt} for bandtransform!")
    if not len(tensor.dims) == 3:
        raise ValueError("Input tensor must have exactly 3 dimensions.")
    if not isinstance(tensor.dims[0], MomentumSpace):
        raise ValueError("First dimension of tensor must be a MomentumSpace.")
    if not isinstance(tensor.dims[1], HilbertSpace):
        raise ValueError("Second dimension of tensor must be a HilbertSpace.")
    if not isinstance(tensor.dims[2], HilbertSpace):
        raise ValueError("Third dimension of tensor must be a HilbertSpace.")

    kspace: MomentumSpace = cast(MomentumSpace, tensor.dims[0])

    def build_transform(space: HilbertSpace) -> Tensor:
        bloch_transform: Tensor = cast(Tensor, t(space)).h(-2, -1)  # (B', B)
        # The transformation will distort the unit-cell of the Hilbert space,
        # we will use fractional to return it to the original unit-cell.
        bloch_transform = bloch_transform.replace_dim(
            0, cast(HilbertSpace, bloch_transform.dims[0]).update(r=Offset.fractional)
        )  # (B'=B, B)
        gspace = cast(HilbertSpace, bloch_transform.dims[0])
        if not space.same_span(gspace):
            raise ValueError(
                f"Hilbert space {space} is not symmetric under the transform {t}!"
            )
        left_fourier = fourier_transform(kspace, space, gspace)  # (K, B, B'=B)
        right_fourier = fourier_transform(kspace, space, space)  # (K, B, B)
        # (K, B, B'=B) @ (B'=B, B) @ (B, B)
        transform = (
            left_fourier @ bloch_transform @ right_fourier.h(-2, -1)
        )  # (K, B, B)
        return transform

    mapped_kspace = kspace.map(lambda k: cast(Momentum, t(k)).fractional())

    if opt in ("both", "left"):
        left_fourier = build_transform(cast(HilbertSpace, tensor.dims[1]))  # (K, B, B)
        left_fourier = left_fourier.replace_dim(0, mapped_kspace).align(
            0, kspace
        )  # (K, B, B)
        tensor = cast(Tensor, (left_fourier @ tensor))  # (K, B, B)

    if opt in ("both", "right"):
        right_fourier = build_transform(cast(HilbertSpace, tensor.dims[2]))  # (K, B, B)
        right_fourier = right_fourier.replace_dim(0, mapped_kspace).align(
            0, kspace
        )  # (K, B, B)
        tensor = cast(Tensor, (tensor @ right_fourier.h(-2, -1)))  # (K, B, B)

    return tensor
