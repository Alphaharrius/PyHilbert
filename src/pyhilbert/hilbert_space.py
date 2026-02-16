from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Self, Tuple, Type, TypeVar, Generic
from typing import cast
from typing_extensions import override
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from functools import lru_cache

import numpy as np
import torch
import sympy as sy
from multipledispatch import dispatch  # type: ignore[import-untyped]

from .abstracts import Operable, Functional
from .spatials import Spatial
from .state_space import StateSpace, restructure
from .tensors import Tensor
from .precision import get_precision_config


class HilbertElement(Spatial, ABC):
    """Defines an element in the `HilbertSpace`."""

    @abstractmethod
    def unit(self) -> "HilbertElement":
        """Return the unit basis of this element."""
        pass


@dispatch(HilbertElement, HilbertElement)
def same_span(a: HilbertElement, b: HilbertElement) -> bool:
    return a.unit() == b.unit()


_ObservableType = TypeVar("_ObservableType")
_OperatedType = TypeVar("_OperatedType", bound=HilbertElement)


class Operator(Generic[_ObservableType], Functional, Operable, ABC):
    """
    A composable operator that acts on observable-compatible objects.

    `Operator` combines two core behaviors:

    1. `Functional` dispatch and chaining
       Implementations are registered via `Functional.register` for pairs of
       `(input_type, operator_subclass)`. At runtime, :meth:`apply` resolves the
       function chain for the concrete input object and executes each function in
       order.

    2. `Operable` matrix-application syntax
       Because `Operator` is `Operable`, it participates in the overloaded `@`
       operator. This module defines `operator_matmul(Operator, HilbertElement)`,
       so `op @ value` applies the operator and returns only the transformed
       value component.

    Conceptually, an operator application returns two outputs:

    - an observation/measurement-like payload (`_ObservableType`)
    - the transformed object (same runtime type as the input)

    The observation payload allows operator execution to report auxiliary
    information while still producing an updated value.

    Type Parameters
    ---------------
    `_ObservableType`
        The type of the first element returned by an operator application
        (e.g. expectation value, coefficient, metadata, or any domain-specific
        observable artifact).

    Implementation Guidelines
    -------------------------
    - The registered callable chain for an operator must produce a
      2-tuple `(observable, transformed_value)`.
    - The transformed value must have the same runtime type as the input object.
      This invariant is validated by :meth:`apply` using assertions.
    - Closure validation in :meth:`apply` has two branches:
      - for `HilbertElement` inputs, closure is span-based
        (`same_span(input, transformed_value)`).
      - for non-`HilbertElement` inputs, closure is value-based
        (`input == transformed_value`).
      In either branch, if closure fails, the observable must be `None`.
    - If no registration exists for `(type(input), type(operator))`,
      :class:`NotImplementedError` is raised by `Functional.apply`.

    Usage Pattern
    -------------
    1. Define an `Operator` subclass.
    2. Register behavior with `@YourOperatorSubclass.register(InputType)`.
    3. Apply by either:
       - `obs, out = op(input_obj)` to receive both outputs, or
       - `out = op @ input_obj` to receive only the transformed value.
    """

    @override
    def apply(  # type: ignore[override]
        self, v: _OperatedType, **kwargs
    ) -> Tuple[_ObservableType, _OperatedType]:
        result = super().apply(v, **kwargs)
        assert isinstance(result, tuple), (
            f"Operator {type(self)} acting on {type(v).__name__} should yield a Tuple[Any, Any]!"
        )
        o, ov = result
        assert isinstance(ov, type(v)), (
            f"Operator {type(self)} acting on {type(v).__name__} should yield a Tuple[Any, {type(v).__name__}]!"
        )
        if isinstance(v, HilbertElement) and isinstance(ov, HilbertElement):
            needs_none = not same_span(v, ov)
        else:
            needs_none = v != ov
        if needs_none:
            assert o is None, (
                f"Un-closed operator action should have undefined irrep (None), got {o}!"
            )

        return o, ov

    def eigen_opr(self) -> Self:
        """
        Return an eigenstate-validating copy of this operator.

        This helper creates a shallow copy of the operator and wraps its
        :meth:`apply` method with a post-condition check: the transformed output
        must be equal to the input value. If the value changes, the wrapped
        operator raises :class:`ValueError`.

        Use this when downstream logic assumes that the input is already an
        eigenstate of the operator and should therefore remain unchanged by the
        operator action.

        Returns
        -------
        Self
            A copy of the current operator whose application enforces
            the same closure condition as :meth:`apply`.

        Raises
        ------
        ValueError
            Raised at call time if the wrapped operator is applied to a value
            that is not closure-preserving under :meth:`apply` semantics.

        Examples
        --------
        A common use case is validating basis assumptions in algorithms that
        require eigen-aligned states:

        - Build `checked = op.eigen_opr()`.
        - Apply `checked(state)` before a specialized computation.
        - Fail fast with `ValueError` if `state` is not an eigenstate.
        """
        op = copy(self)
        apply = op.apply

        def eigen_apply(
            v: _OperatedType, **kwargs
        ) -> Tuple[_ObservableType, _OperatedType]:
            """
            Apply the copied operator and enforce closure-preserving output.

            This wrapper delegates to the copied operator's original
            :meth:`apply` implementation, then validates that the transformed
            output remains closed with the input under the same semantics used by
            :meth:`Operator.apply`:

            - If both input and output are `HilbertElement`, closure is checked
              by span membership with `same_span(v, ov)`.
            - Otherwise, closure is checked by strict value equality `ov == v`.

            Parameters
            ----------
            `v` : `_OperatedType`
                Input object to transform.
            `**kwargs`
                Keyword arguments forwarded to the wrapped :meth:`apply` call.

            Returns
            -------
            `Tuple[_ObservableType, _OperatedType]`
                The observable payload and transformed output from the wrapped
                operator call, when closure is preserved.

            Raises
            ------
            `ValueError`
                If the transformed output is not closure-preserving with respect
                to `v` under the branch-specific rule above.
            """
            o, ov = apply(v, **kwargs)
            if isinstance(v, HilbertElement) and isinstance(ov, HilbertElement):
                is_closed = same_span(v, ov)
            else:
                is_closed = ov == v
            if not is_closed:
                raise ValueError(
                    f"{type(op).__name__} expected a closure-preserving state, but output "
                    "{ov!r} is not closed with input {v!r}."
                )
            return o, ov

        object.__setattr__(op, "apply", eigen_apply)
        return op


@dispatch(Operator, Operable)
def operator_matmul(o: Operator, v: Operable):
    _, v = o(v)
    return v


class HilbertSpan(HilbertElement, ABC):
    """
    Abstract base type for finite collections of Hilbert-space elements.

    `HilbertSpan` defines the shared interface for container-like Hilbert
    objects (for example, symbolic sums of basis states). Concrete subclasses
    choose storage and ordering, while exposing their members through
    `elements()`.

    Notes
    -----
    This class is purely an interface and does not prescribe linear-algebra
    simplification, normalization, or coefficient handling.
    """

    @abstractmethod
    def elements(self) -> Tuple[HilbertElement, ...]:
        """
        Return the elements contained in this span.

        Returns
        -------
        `Tuple[HilbertElement, ...]`
            Immutable tuple of Hilbert elements represented by this span.
        """
        pass

    @abstractmethod
    def gram(self, another: Self) -> Any:
        """
        Return the gram matrix of this span to `another` with the current span
        at the row space and `another` as the col space.

        Parameters
        ----------
        `another` : `HilbertSpan`
            The another span.

        Returns
        -------
        `Any`
            Matrix like object representing the gram matrix between two spans.
        """
        pass


@dataclass(frozen=True)
class HilbertSpace(StateSpace[HilbertElement], HilbertSpan):
    __hash__ = StateSpace.__hash__

    def lookup(self, query: Dict[Type[Any], Any]) -> HilbertElement:
        """
        Return the unique element that exactly matches all typed-irrep query entries.

        Parameters
        ----------
        `query` : `Dict[Type[Any], Any]`
            Mapping from irrep runtime type to expected irrep value.
            A candidate element matches only if, for every `(T, value)` pair,
            it contains an irrep of exact type `T` and `irrep == value`.

        Returns
        -------
        `HilbertElement`
            The unique matching element.

        Raises
        ------
        `ValueError`
            If no element matches, if multiple elements match, or if `query` is empty.
        """
        if not query:
            raise ValueError("lookup query cannot be empty.")

        matches: list[HilbertElement] = []
        for el in self.elements():
            if not isinstance(el, U1State):
                continue

            is_match = True
            for T, expected in query.items():
                try:
                    actual = el.irrep_of(T)
                except ValueError:
                    is_match = False
                    break
                if actual != expected:
                    is_match = False
                    break
            if is_match:
                matches.append(el)

        if not matches:
            raise ValueError(f"No state found for query={query}.")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple states found for query={query}; expected a unique match."
            )
        return matches[0]

    @lru_cache
    def flatten(self) -> "HilbertSpace":
        """
        Return a new instance of `HilbertSpace` with all `HilbertSpan` flattened to its elements
        respecting the original ordering.

        Returns
        -------
        `HilbertSpace`
            A new instance of `HilbertSpace` with all `HilbertSpan` flattened.
        """
        flattened_elements: OrderedDict[HilbertElement, slice] = OrderedDict()
        for el in self.structure.keys():
            if issubclass(type(el), HilbertSpan):
                for m in cast(HilbertSpan, el).elements():
                    flattened_elements[m] = slice(0, m.dim)
                continue
            flattened_elements[el] = slice(0, el.dim)
        return HilbertSpace(restructure(flattened_elements))

    def elements(self) -> Tuple[HilbertElement, ...]:
        """Get the flattened elements of this `HilbertSpace`."""
        return cast(Tuple[HilbertElement, ...], tuple(self.flatten().structure.keys()))

    @override
    def unit(self) -> "HilbertSpace":
        return hilbert(el.unit() for el in self)

    @override
    def gram(self, another: "HilbertSpace") -> Tensor:
        span = U1Span(cast(Tuple[U1State, ...], self.elements()))
        new_span = U1Span(cast(Tuple[U1State, ...], another.elements()))
        irrep = span.gram(new_span)
        precision = get_precision_config()
        data = torch.from_numpy(
            np.asarray(irrep.tolist(), dtype=precision.np_complex)
        ).to(dtype=precision.torch_complex)
        return Tensor(data=data, dims=(self, another))


def hilbert(itr: Iterable[HilbertElement]) -> HilbertSpace:
    structure: OrderedDict[HilbertElement, slice] = OrderedDict()
    base = 0
    for el in itr:
        structure[el] = slice(base, base + el.dim)
        base += el.dim
    return HilbertSpace(structure=structure)


@dispatch(HilbertSpace, HilbertSpace)  # type: ignore[no-redef]
def same_span(a: HilbertSpace, b: HilbertSpace) -> bool:
    return set(m.unit() for m in a.structure.keys()) == set(
        m.unit() for m in b.structure.keys()
    )


_IrrepType = TypeVar("_IrrepType")
"""Defines a irreducible representation type."""


@dataclass(frozen=True)
class FuncOpr(Generic[_IrrepType], Operator[sy.Integer]):
    T: Type[_IrrepType]
    func: Callable


class State(Generic[_IrrepType], HilbertElement, ABC):
    @abstractmethod
    def ket(self, ket: Self) -> _IrrepType:
        pass


@dataclass(frozen=True)  # eq=False, Skipping Operable.__eq__
class Ket(Generic[_IrrepType], Operable):
    """
    A single basis label in the Hilbert construction.

    A `Ket` wraps one irreducible-representation object (`irrep`). It does not
    store amplitudes; it is only the symbolic building block used to form larger
    tensor-product states with `@`.
    """

    irrep: _IrrepType


@dataclass(frozen=True)
class U1State(State[sy.Expr]):
    """
    Immutable single-particle basis state built from typed irreps.

    `U1State` is a symbolic tensor-product state with U(1) irreducible representation
    presented as an ordered tuple of `Ket` objects. Each ket contributes one
    irreducible representation label (`ket.irrep`) to the state. This object is
    intentionally symbolic: it stores basis labels only and does not store amplitudes
    or coefficients.

    A key invariant is enforced at construction time: for each concrete irrep
    type, the state must contain exactly one ket of that type. In other words,
    irrep multiplicities must be unity. This guarantees that type-based updates
    via `replace` are unambiguous and fast.

    Parameters
    ----------
    `irrep`: `sy.Expr`
        The irrep of this state under an recent operation.
    `kets` : `Tuple[Ket[Any], ...]`
        Ordered tuple of kets that defines the state.

    Attributes
    ----------
    `irrep`: `sy.Expr`
        The irrep of this state under an recent operation.
    `kets` : `Tuple[Ket[Any], ...]`
        Immutable tuple of basis labels in tensor-product order.

    Notes
    -----
    - `dim` is always `1`; this type represents one basis vector.
    - `replace(irrep)` substitutes the unique ket whose irrep has the same
      concrete runtime type as `irrep`.
    - `@` dispatch overloads combine kets/states into a new `U1State`.
    - `+` dispatch overloads build an `U1Span` of distinct `U1State` values.

    Raises
    ------
    `ValueError`
        Raised in `__post_init__` when any irrep type appears with multiplicity
        different from `1`.
    """

    irrep: sy.Expr
    kets: Tuple[Ket[Any], ...]

    def __post_init__(self) -> None:
        counts: Dict[Type, int] = {}
        for ket in self.kets:
            irrep_type = type(ket.irrep)
            counts[irrep_type] = counts.get(irrep_type, 0) + 1
        non_singletons = {t: c for t, c in counts.items() if c != 1}
        if non_singletons:
            detail = ", ".join(f"{t.__name__}:{c}" for t, c in non_singletons.items())
            raise ValueError(
                "U1State allows only irrep with unity multiplicity; "
                f"got multiple non-singleton types ({detail})."
            )

    @property
    def dim(self) -> int:
        """The dimension of a single particle state is always `1`."""
        return 1

    def replace(self, irrep: Any) -> "U1State":
        """
        Return a new state where the irrep of the same concrete type is replaced.

        The method searches this state's ket tuple for the unique ket whose irrep has
        the same *exact* runtime type as `irrep` (using `type(x) is type(y)`), then
        returns a new `U1State` with that ket substituted. The original instance is
        not modified.

        Parameters
        ----------
        `irrep` : `Any`
            Replacement irrep instance. Its concrete type must already exist in this
            state exactly once (enforced by `U1State.__post_init__`).

        Returns
        -------
        `U1State`
            A new `U1State` with one ket replaced.

        Raises
        ------
        `ValueError`
            If this state does not contain any ket whose irrep has the same concrete
            type as `irrep`.
        """
        target_type = type(irrep)
        kets = self.kets
        for i, ket in enumerate(kets):
            if type(ket.irrep) is target_type:
                return replace(self, kets=kets[:i] + (Ket(irrep),) + kets[i + 1 :])
        raise ValueError(
            f"U1State has no irrep of type {target_type.__name__} to replace."
        )

    def irrep_of(self, T: Type[_IrrepType]) -> _IrrepType:
        """
        Return the unique irrep in this state whose concrete type is `T`.

        This method performs a direct scan over `self.kets` and returns the
        first irrep satisfying `type(ket.irrep) is T`. Because
        `U1State.__post_init__` enforces unity multiplicity for each irrep
        type, the match is unique whenever it exists.

        Parameters
        ----------
        `T` : `Type[_IrrepType]`
            Concrete irrep type to retrieve.

        Returns
        -------
        `_IrrepType`
            The irrep instance of type `T` contained in this state.

        Raises
        ------
        `ValueError`
            If no irrep with concrete type `T` exists in this state.

        Notes
        -----
        - Matching uses exact runtime type identity (`type(x) is T`), not
          subclass checks.
        - Runtime is linear in the number of kets (`O(n)`), with no temporary
          mapping allocations.
        """
        for ket in self.kets:
            irrep = ket.irrep
            if type(irrep) is T:
                return cast(_IrrepType, irrep)
        raise ValueError(f"U1State {self} has no irrep of type {T.__name__}.")

    @override
    def ket(self, psi: "U1State") -> sy.Expr:
        if not self.kets == psi.kets:
            return sy.Integer(0)
        return cast(sy.Expr, (sy.conjugate(self.irrep) * psi.irrep).simplify())

    def __str__(self) -> str:
        ket_repr = "⊗".join(
            f"|{irrep_repr}⟩"
            if len(irrep_repr := repr(ket.irrep)) <= 32
            else f"|{type(ket.irrep).__name__}⟩"
            for ket in self.kets
        )
        if self.irrep != sy.Integer(1):
            ket_repr = f"({self.irrep}) * " + ket_repr
        return ket_repr

    def __repr__(self) -> str:
        return self.__str__()

    @override
    def unit(self) -> "U1State":
        """Get a new copy from this `U1State` with the U(1) irrep being `1`."""
        return replace(self, irrep=sy.Integer(1))


@dispatch(U1State, U1State)  # type: ignore[no-redef]
def same_span(a: U1State, b: U1State) -> bool:
    """Check if the unit basis of two `U1State` are the same."""
    return a.unit() == b.unit()


@FuncOpr.register(U1State)
def _func_opr_u1_state(f: FuncOpr, psi: U1State) -> Tuple[sy.Integer | None, U1State]:
    irrep = psi.irrep_of(f.T)
    new_irrep = f.func(irrep)
    func_irrep = sy.Integer(1) if irrep == new_irrep else None
    new_psi = psi.replace(new_irrep)
    return func_irrep, new_psi


@dataclass(frozen=True)
class U1Span(HilbertSpan):
    """
    Finite span of distinct single-particle basis states.

    `U1Span` is the additive container used by `U1State`'s `*` operator. It
    stores an ordered tuple of basis states and represents the symbolic span
    generated by those states. The object is immutable (`frozen=True`) and
    preserves insertion order.

    This type is intentionally lightweight: it does not store amplitudes,
    coefficients, or perform linear-algebra simplification. Duplicate handling
    is implemented in `operator_add` overloads, which keep only one copy of an
    existing state when building a span.

    Parameters
    ----------
    `span` : `Tuple[U1State, ...]`
        Ordered tuple of `U1State` elements contained in this span.

    Attributes
    ----------
    `span` : `Tuple[U1State, ...]`
        The underlying immutable sequence of basis states.

    Notes
    -----
    The `dim` property returns `len(span)`, i.e., the number of basis states
    currently tracked by this symbolic span.
    """

    span: Tuple[U1State, ...]

    @property
    def dim(self) -> int:
        """Get the length of this single particle state span."""
        return len(self.span)

    def __iter__(self) -> Iterator[U1State]:
        """Iterate over states in this span preserving insertion order."""
        return iter(self.span)

    @override
    def elements(self) -> Tuple[U1State, ...]:
        return self.span

    @override
    def unit(self) -> "U1Span":
        """Return the actual span without any basis scaling by a irrep."""
        return U1Span(tuple(m.unit() for m in self.span))

    @override
    def gram(self, ket: "U1Span") -> sy.ImmutableDenseMatrix:
        tbl: Dict[U1State, Tuple[int, U1State]] = {
            psi.unit(): (n, psi) for n, psi in enumerate(ket.span)
        }
        out = sy.zeros(self.dim, ket.dim)
        for n, psi in enumerate(self.span):
            unit = psi.unit()
            if unit not in tbl:
                continue
            m, kpsi = tbl[unit]
            out[n, m] = psi.ket(kpsi)
        return sy.ImmutableDenseMatrix(out)


@FuncOpr.register(U1Span)
def _func_opr_u1_span(f: FuncOpr, s: U1Span) -> Tuple[sy.Integer | None, U1Span]:
    new_s: U1Span = replace(s, span=tuple(f @ psi for psi in s.span))
    func_irrep = sy.Integer(1) if same_span(s, new_s) else None
    return func_irrep, new_s


@FuncOpr.register(HilbertSpace)
def _func_opr_hilbert(
    f: FuncOpr, h: HilbertSpace
) -> Tuple[sy.Integer | None, HilbertSpace]:
    new_h = hilbert(f @ el for el in h)
    func_irrep = sy.Integer(1) if same_span(h, new_h) else None
    return func_irrep, new_h


@dispatch(U1Span, U1Span)  # type: ignore[no-redef]
def same_span(a: U1Span, b: U1Span) -> bool:
    return set(a.unit().span) == set(b.unit().span)


@dispatch(Ket, Ket)  # type: ignore[no-redef]
def operator_matmul(a: Ket, b: Ket) -> U1State:
    return U1State(sy.Integer(1), (a, b))


@dispatch(U1State, Ket)  # type: ignore[no-redef]
def operator_matmul(psi: U1State, ket: Ket) -> U1State:
    return U1State(psi.irrep, psi.kets + (ket,))


@dispatch(Ket, U1State)  # type: ignore[no-redef]
def operator_matmul(ket: Ket, psi: U1State) -> U1State:
    return U1State(psi.irrep, (ket,) + psi.kets)


@dispatch(U1State, U1State)  # type: ignore[no-redef]
def operator_matmul(a: U1State, b: U1State) -> U1State:
    if not a.kets:
        return b
    if not b.kets:
        return a
    return U1State((a.irrep * b.irrep).simplify(), a.kets + b.kets)


@dispatch(U1State, U1State)
def operator_add(a: U1State, b: U1State) -> U1Span:
    if a == b:
        return U1Span((a,))
    return U1Span((a, b))


@dispatch(U1Span, U1State)  # type: ignore[no-redef]
def operator_add(span: U1Span, state: U1State) -> U1Span:
    if state in span.span:
        return span
    return U1Span(span.span + (state,))


@dispatch(U1State, U1Span)  # type: ignore[no-redef]
def operator_add(state: U1State, span: U1Span) -> U1Span:
    if state in span.span:
        return span
    return U1Span((state,) + span.span)
