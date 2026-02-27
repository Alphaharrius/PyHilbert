from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from typing import (
    Any,
    Callable,
    Dict,
    ClassVar,
    Literal,
    Set,
    Self,
    Tuple,
    Generic,
    Type,
    TypeVar,
)

from multipledispatch import dispatch

from .utils import subtypes


@dataclass(frozen=True)
class Operable(ABC):
    def __add__(self, other):
        return operator_add(self, other)

    def __neg__(self):
        return operator_neg(self)

    def __sub__(self, other):
        return operator_sub(self, other)

    def __mul__(self, other):
        return operator_mul(self, other)

    def __matmul__(self, other):
        return operator_matmul(self, other)

    def __truediv__(self, other):
        return operator_truediv(self, other)

    def __floordiv__(self, other):
        return operator_floordiv(self, other)

    def __pow__(self, other):
        return operator_pow(self, other)

    def __eq__(self, value):
        return operator_eq(self, value)

    def __lt__(self, other):
        return operator_lt(self, other)

    def __le__(self, other):
        return operator_le(self, other)

    def __gt__(self, other):
        return operator_gt(self, other)

    def __ge__(self, other):
        return operator_ge(self, other)

    def __and__(self, other):
        return operator_and(self, other)

    def __or__(self, other):
        return operator_or(self, other)

    def __radd__(self, other):
        return operator_add(other, self)

    def __rsub__(self, other):
        return operator_sub(other, self)

    def __rmul__(self, other):
        return operator_mul(other, self)

    def __rtruediv__(self, other):
        return operator_truediv(other, self)


@dispatch(Operable, Operable)
def operator_add(a, b):
    raise NotImplementedError(f"Addition of {type(a)} and {type(b)} is not supported!")


@dispatch(Operable)
def operator_neg(a):
    raise NotImplementedError(f"Negation of {type(a)} is not supported!")


@dispatch(Operable, Operable)
def operator_sub(a, b):
    return a + (-b)


@dispatch(Operable, Operable)
def operator_mul(a, b):
    raise NotImplementedError(
        f"Multiplication of {type(a)} and {type(b)} is not supported!"
    )


@dispatch(Operable, Operable)
def operator_matmul(a, b):
    raise NotImplementedError(
        f"Matrix multiplication of {type(a)} and {type(b)} is not supported!"
    )


@dispatch(Operable, Operable)
def operator_truediv(a, b):
    raise NotImplementedError(f"Division of {type(a)} and {type(b)} is not supported!")


@dispatch(Operable, Operable)
def operator_floordiv(a, b):
    raise NotImplementedError(
        f"Floor division of {type(a)} and {type(b)} is not supported!"
    )


@dispatch(Operable, Operable)
def operator_pow(a, b):
    raise NotImplementedError(
        f"Exponentiation of {type(a)} and {type(b)} is not supported!"
    )


@dispatch(Operable, Operable)
def operator_eq(a, b):
    raise NotImplementedError(
        f"Equality comparison of {type(a)} and {type(b)} is not supported!"
    )


@dispatch(Operable, Operable)
def operator_lt(a, b):
    raise NotImplementedError(
        f"Less-than comparison of {type(a)} and {type(b)} is not supported!"
    )


@dispatch(Operable, Operable)
def operator_le(a, b):
    raise NotImplementedError(
        f"Less-than-or-equal comparison of {type(a)} and {type(b)} is not supported!"
    )


@dispatch(Operable, Operable)
def operator_gt(a, b):
    raise NotImplementedError(
        f"Greater-than comparison of {type(a)} and {type(b)} is not supported!"
    )


@dispatch(Operable, Operable)
def operator_ge(a, b):
    raise NotImplementedError(
        f"Greater-than-or-equal comparison of {type(a)} and {type(b)} is not supported!"
    )


@dispatch(Operable, Operable)
def operator_and(a, b):
    raise NotImplementedError(
        f"Logical AND of {type(a)} and {type(b)} is not supported!"
    )


@dispatch(Operable, Operable)
def operator_or(a, b):
    raise NotImplementedError(
        f"Logical OR of {type(a)} and {type(b)} is not supported!"
    )


UpdatableType = TypeVar("UpdatableType", bound="Updatable")


class Updatable(ABC, Generic[UpdatableType]):
    """
    An object that can be updated to a new state.
    """

    def update(self, **kwargs) -> UpdatableType:
        """
        Return an updated instance of this object.

        This method delegates the construction of the updated object to
        :meth:`_updated`, then enforces common safety and dataclass consistency
        rules:
        1. ``_updated`` must return a new object (not ``self``).
        2. If both ``self`` and the returned object are dataclasses of the same
           runtime type, fields with ``init=False`` are copied from ``self`` to
           the returned object.

        Parameters
        ----------
        `**kwargs` : `Any`
            Keyword arguments forwarded to :meth:`_updated` to define the new
            state.

        Returns
        -------
        `UpdatableType`
            A new instance representing the updated state.

        Raises
        ------
        `RuntimeError`
            If :meth:`_updated` returns ``self`` instead of a new instance.
        """
        out = self._updated(**kwargs)
        if out is self:
            raise RuntimeError(
                f"{type(self).__name__}._updated() must not return self; return a new object."
            )
        if type(out) is type(self) and is_dataclass(self) and is_dataclass(out):
            for f in fields(self):
                if not f.init:
                    object.__setattr__(out, f.name, getattr(self, f.name))
        return out

    @abstractmethod
    def _updated(self, **kwargs) -> UpdatableType:
        pass


class Plottable(ABC):
    """
    An object that can be plottable.
    """

    _plot_methods: ClassVar[Dict[Tuple[str, str], Callable]] = {}

    @classmethod
    def register_plot_method(cls, name: str, backend: str = "plotly"):
        def decorator(func: Callable):
            cls._plot_methods[(name, backend)] = func
            return func

        return decorator

    def plot(self, method: str, backend: str = "plotly", *args, **kwargs):
        """
        Dispatch the plot method to the registered function.
        """
        if (method, backend) not in self._plot_methods:
            raise ValueError(
                f"Plot method {method} not found. Available methods: {list(self._plot_methods.keys())}"
            )
        return self._plot_methods[(method, backend)](self, *args, **kwargs)


class HasDual(ABC):
    """
    An object that has a dual.
    """

    @property
    @abstractmethod
    def dual(self):
        raise NotImplementedError()


BaseType = TypeVar("BaseType")


class HasBase(Generic[BaseType], ABC):
    """
    An object that is expressed in a specific base (basis/coordinate system).

    "Base" here means the same thing you would mean for a vector: a basis (or
    basis-like structure) that defines how the object's representation is
    written. Examples include a vector's basis, a lattice/affine space's basis,
    a function expanded in a basis of functions, or an operator expressed in a
    particular coordinate frame.

    The key idea is that the *mathematical object* is the same, but its
    *representation* depends on the base. Implementations should therefore
    provide `rebase(...)` to return a new equivalent object expressed in a new
    base, without mutating the original.
    """

    @abstractmethod
    def base(self) -> BaseType:
        """
        Return the base (basis/coordinate system) this object is currently expressed in.

        This should be a lightweight, stable descriptor of the representation context
        (e.g., a basis matrix, lattice, coordinate frame, or function basis). The
        returned base is used by `rebase(...)` to construct an equivalent object in a
        new base, so implementations should not mutate internal state and should
        prefer returning an immutable or effectively immutable object.
        """
        raise NotImplementedError()

    @abstractmethod
    def rebase(self, new_base: BaseType) -> "HasBase[BaseType]":
        """
        Return an equivalent object expressed in ``new_base``.

        Implementations must preserve the underlying mathematical object while
        changing only its representation. This method should be pure: do not
        mutate ``self`` or ``new_base``. Prefer returning a new instance, even if
        the base is unchanged; if you choose to return ``self`` for identical
        bases, document that behavior and ensure immutability.

        ``new_base`` is expected to be compatible with the object. If it is not,
        raise a clear error (typically ``ValueError``). Do not silently coerce
        incompatible bases.
        """
        raise NotImplementedError()


_InnerProductType = TypeVar("_InnerProductType")


class AbstractKet(Generic[_InnerProductType], ABC):
    """
    The base class for all ket-like objects that the inner product is defined via `<bra|ket>` syntax.

    The `_InnerProductType` is the type of the inner product mapping between this ket and its dual bra.
    """

    @abstractmethod
    def ket(self, another: Self) -> _InnerProductType:
        """Return the inner product mapping between this ket and `another` ket."""
        raise NotImplementedError()


@dataclass(frozen=True)
class Functional(ABC):
    _registered_methods: ClassVar[Dict[Tuple[type, type], Tuple[Callable, ...]]] = {}
    _overwrite_locked_by_subclass: ClassVar[Set[Tuple[type, type]]] = set()

    @classmethod
    def register(  # TODO: Rename to register
        cls, obj_type: type, order: Literal["overwrite", "front", "back"] = "overwrite"
    ):
        """
        Register a function defining the action of the `Functional` on a specific object type.
        Registration is applied to `obj_type` and all currently defined subclasses
        of `obj_type`. For each affected type:
        - `'overwrite'` replaces any existing function.
        - `'front'` prepends the new function to the existing function chain.
        - `'back'` appends the new function to the existing function chain.

        At call time, chained functions are executed in front-to-end order, where each
        function receives the output of the previous function as its `obj` input.

        Parameters
        ----------
        `obj_type` : `type`
            The type of object the function applies to.
        `order` : `Literal["overwrite", "front", "back"]`, optional
            The order in which to register the function relative to existing functions.
            By default, 'overwrite' which replaces any existing function.

        Returns
        -------
        `Callable`
            A decorator that registers the function for the specified object type.
        """
        if order not in ("overwrite", "front", "back"):
            raise ValueError(
                f"Invalid registration order: {order}. "
                "Expected one of 'overwrite', 'front', 'back'."
            )

        def decorator(func: Callable):
            chain: Tuple[Callable, ...] = tuple()
            for target_type in (obj_type, *subtypes(obj_type)):
                key = (target_type, cls)
                is_superclass_propagation = target_type is not obj_type
                if (
                    is_superclass_propagation
                    and key in cls._overwrite_locked_by_subclass
                ):
                    if order == "overwrite":
                        raise ValueError(
                            f"Cannot overwrite function for {target_type.__name__} via "
                            f"superclass registration on {obj_type.__name__} because "
                            "the subclass was previously registered with "
                            "order='overwrite'."
                        )
                    continue

                existing = cls._registered_methods.get(key)
                if existing is None:
                    existing_chain: Tuple[Callable, ...] = ()
                elif isinstance(existing, tuple):
                    existing_chain = existing
                else:
                    existing_chain = (existing,)

                if order == "overwrite" or not existing_chain:
                    chain = (func,)
                elif order == "front":
                    chain = (func,) + existing_chain
                else:  # order == "back"
                    chain = existing_chain + (func,)

                cls._registered_methods[key] = chain
                if not is_superclass_propagation and order == "overwrite":
                    cls._overwrite_locked_by_subclass.add(key)
            return func

        return decorator

    @staticmethod
    def get_applicable_types(cls) -> Tuple[Type, ...]:
        """
        Get all object types that can be applied by this `Functional`.

        Returns
        -------
        Tuple[Type, ...]
            A tuple of all registered object types that this `Functional` can handle.
        """
        types = set()
        for obj_type, functional_type in cls._registered_methods.keys():
            if functional_type is cls:
                types.add(obj_type)
        return tuple(types)

    def allows(self, obj: Any) -> bool:
        """
        Check if this `Functional` can be applied on the given object.

        Parameters
        ----------
        obj : Any
            The object to check for applicability.

        Returns
        -------
        bool
            True if this `Functional` can be applied on the object, False otherwise.
        """
        functional_class = type(self)
        obj_class = type(obj)
        key = (obj_class, functional_class)
        return key in self._registered_methods

    def apply(self, obj: Any, **kwargs) -> Any:
        functional_class = type(self)
        obj_class = type(obj)
        key = (obj_class, functional_class)

        chain = self._registered_methods.get(key)

        if chain is None or not chain:
            raise NotImplementedError(
                f"No function registered for {obj_class.__name__} "
                f"with {functional_class.__name__}"
            )

        out = obj
        for func in chain:
            out = func(self, out, **kwargs)
        return out

    def __call__(self, obj: Any, **kwargs) -> Any:
        return self.apply(obj, **kwargs)


_ElementType = TypeVar("_ElementType")
_MappingType = TypeVar("_MappingType")


class Span(ABC, Generic[_ElementType, _MappingType]):
    """
    An object representing the span of a set of elements.

    The specific meaning of "span" depends on the context. For example, in a
    vector space, the span of a set of vectors is the set of all linear
    combinations of those vectors. In a topological space, the span of a set
    of points might be the smallest closed set containing those points.

    Implementations should define what it means to be an element and how to
    determine if an object is contained within the span.

    The `_ElementType` type variable represents the type of elements that define the span,
    while the `_MappingType` is the type of the mapping between spans of this type.
    """

    @abstractmethod
    def elements(self) -> Tuple[_ElementType, ...]:
        """
        Return the elements contained in this span.

        Returns
        -------
        `Tuple[_ElementType, ...]`
            Immutable tuple of elements represented by this span.
        """
        pass

    @abstractmethod
    def gram(self, another: Self) -> _MappingType:
        """
        Return the gram matrix of this span to `another` with the current span
        at the row space and `another` as the col space.

        Parameters
        ----------
        `another` : `Self`
            The another span.

        Returns
        -------
        `_MappingType`
            The mapping between this span and `another`, typically represented as a matrix.
        """
        pass


class HasUnit(ABC):
    """
    An object that has a unit representation.
    """

    @abstractmethod
    def unit(self) -> Self:
        """Return the unit representation of this object."""
        raise NotImplementedError()
