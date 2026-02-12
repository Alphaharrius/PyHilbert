from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from multipledispatch import dispatch
from typing import (
    Any,
    Callable,
    Dict,
    ClassVar,
    Literal,
    NamedTuple,
    Set,
    Tuple,
    Generic,
    Type,
    TypeVar,
    Union,
)

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
        out = self._updated(**kwargs)
        if out is self:
            raise RuntimeError(
                f"{type(self).__name__}._updated() must not return self; return a new object."
            )
        return out

    @abstractmethod
    def _updated(self, **kwargs) -> UpdatableType:
        pass


class Plottable:
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


@dataclass(frozen=True)
class Transform(ABC):
    _register_transform_method: ClassVar[
        Dict[Tuple[type, type], Tuple[Callable, ...]]
    ] = {}
    _overwrite_locked_by_subclass: ClassVar[Set[Tuple[type, type]]] = set()

    @classmethod
    # def register_transform_method(cls, obj_type: type, chain: bool = False):
    def register_transform_method(
        cls, obj_type: type, order: Literal["overwrite", "front", "back"] = "overwrite"
    ):
        """
        Register a function defining the action of the `Transform` on a specific object type.

        Registration is applied to `obj_type` and all currently defined subclasses
        of `obj_type`. For each affected type:
        - `'overwrite'` replaces any existing transform function.
        - `'front'` prepends the new function to the existing function chain.
        - `'back'` appends the new function to the existing function chain.

        At call time, chained functions are executed in front-to-end order, where each
        function receives the output of the previous function as its `obj` input.

        Parameters
        ----------
        `obj_type` : `type`
            The type of object the transform function applies to.
        `order` : `Literal["overwrite", "front", "back"]`, optional
            The order in which to register the transform function relative to existing functions.
            By default, 'overwrite' which replaces any existing function.

        Returns
        -------
        `Callable`
            A decorator that registers the transform function for the specified object type.
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
                            f"Cannot overwrite transform for {target_type.__name__} via "
                            f"superclass registration on {obj_type.__name__} because "
                            "the subclass was previously registered with "
                            "order='overwrite'."
                        )
                    continue

                existing = cls._register_transform_method.get(key)
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

                cls._register_transform_method[key] = chain
                if not is_superclass_propagation and order == "overwrite":
                    cls._overwrite_locked_by_subclass.add(key)
            return func

        return decorator

    @staticmethod
    def get_transformable_types(cls) -> Tuple[Type, ...]:
        """
        Get all object types that can be transformed by this Transform.

        Returns
        -------
        Tuple[Type, ...]
            A tuple of all registered object types that this Transform can handle.
        """
        types = set()
        for obj_type, transform_type in cls._register_transform_method.keys():
            if transform_type is cls:
                types.add(obj_type)
        return tuple(types)

    def allows(self, obj: Any) -> bool:
        """
        Check if this Transform can transform the given object.

        Parameters
        ----------
        obj : Any
            The object to check for transformability.

        Returns
        -------
        bool
            True if this Transform can transform the object, False otherwise.
        """
        transform_class = type(self)
        obj_class = type(obj)
        key = (obj_class, transform_class)
        return key in self._register_transform_method

    def transform(self, obj: Any, **kwargs) -> Any:
        transform_class = type(self)
        obj_class = type(obj)
        key = (obj_class, transform_class)

        chain = self._register_transform_method.get(key)

        if chain is None or not chain:
            raise NotImplementedError(
                f"No transform registered for {obj_class.__name__} "
                f"with {transform_class.__name__}"
            )

        out = obj
        for func in chain:
            out = func(self, out, **kwargs)
        return out

    def __call__(self, obj: Any, **kwargs) -> Any:
        return self.transform(obj, **kwargs)


@dataclass(frozen=True)
class GaugeBasis(ABC):
    """
    A marker class for gaugable objects that define a specific transform type.
    """

    pass


@dataclass(frozen=True)
class GaugeInvariant(GaugeBasis):
    """
    Marker gaugable whose transform is always the base `Transform`.

    This represents gauge-invariant values that do not require a specialized
    transform implementation.
    """

    pass


@dataclass(frozen=True)
class Gaugable(ABC):
    """
    Container base class for objects that own a `GaugeBasis` instance.

    Implementations compose a `gauge_basis` value so downstream code can resolve
    transform compatibility through a consistent attribute.
    """

    _gauge_basis: GaugeBasis = field(default_factory=GaugeInvariant, init=False)

    def gauge_repr(self) -> GaugeBasis:
        """Get the gauge basis representation of this gaugable object."""
        return self._gauge_basis


_GaugableType = TypeVar("_GaugableType", bound=Union[Gaugable, GaugeInvariant])
_GaugeType = TypeVar("_GaugeType")


class Gauged(Generic[_GaugableType, _GaugeType], NamedTuple):
    """
    A simple named tuple to hold a gaugable object and its associated gauge after a transform.

    This is primarily a convenience wrapper to group together a `Gaugable` object
    and the resulting gauge after applying a `Transform`. It allows for clear
    and type-safe handling of gaugable objects along with their gauges in
    various operations.

    For example, if the gauge is `U(1)` then the gauge is a complex number.
    """

    gauge: _GaugeType
    """ The gauge associated with the gaugable object. """
    gaugable: _GaugableType
    """ The gaugable object being transformed. """
