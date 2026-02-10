from dataclasses import dataclass
from typing import Any, NamedTuple, Tuple
from collections import OrderedDict
from itertools import product
from functools import lru_cache, reduce

from multipledispatch import dispatch
import sympy as sy

from .abstracts import HasBase, Transform
from .spatials import AffineSpace, Spatial, Offset
from .utils import FrozenDict


@dataclass(frozen=True)
class AffineFunction(Spatial):
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
class AffineGroupElement(Transform, HasBase[AffineSpace]):
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


class FunctionTransformResult(NamedTuple):
    func: AffineFunction
    phase: sy.Expr


@dispatch(AffineGroupElement, AffineFunction)
def affine_transform(
    t: AffineGroupElement, f: AffineFunction
) -> FunctionTransformResult:
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
    `FunctionTransformResult`
        A named tuple containing:
        - `func`: The original `AffineFunction` (unchanged).
        - `phase`: The symbolic phase factor (sy.Expr) such that
          `t.rep @ f.rep == phase * f.rep`.

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

    return FunctionTransformResult(func=f, phase=phase)


AffineGroupElement.register_transform_method(AffineFunction)(affine_transform)


@dispatch(AffineGroupElement, Offset)  # type: ignore[no-redef]
def affine_transform(t: AffineGroupElement, offset: Offset) -> Offset:
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
    t : AffineGroupElement
        The affine group element to apply. If its internal `offset.space` does
        not match `offset.space`, the transform is rebased to the Offset's space.
    offset : Offset
        The spatial offset (column vector) to transform.

    Returns
    -------
    Offset
        A new Offset expressed in the same AffineSpace as the input `offset`.

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


AffineGroupElement.register_transform_method(Offset)(affine_transform)


@dispatch(AffineGroupElement, Any)  # type: ignore[no-redef]
def affine_transform(t: AffineGroupElement, v: Any) -> Any:
    """
    Apply a Affine transform to a transformable object.

    Affine transform is described by the transformation of elements based on
    their base `AffineSpace`.
    """
    raise NotImplementedError(
        f"Transforming object of type {type(v)} by {type(t)} is not supported."
    )
