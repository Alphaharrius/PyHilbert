from abc import ABC, abstractmethod
from dataclasses import dataclass
from multipledispatch import dispatch
from typing import Any, Callable, Dict, ClassVar, Tuple, Generic, TypeVar


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


class Updatable(ABC):
    """
    An object that can be updated to a new state.
    """

    def update(self, **kwargs) -> "Updatable":
        out = self._updated(**kwargs)
        if out is self:
            raise RuntimeError(
                f"{type(self).__name__}._updated() must not return self; return a new object."
            )
        return out

    @abstractmethod
    def _updated(self, **kwargs) -> "Updatable":
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
    _register_transform_method: ClassVar[Dict[Tuple[type, type], Callable]] = {}

    @classmethod
    def register_transform_method(cls, obj_type: type):
        """Register a transform method for a specific object type."""

        def decorator(func: Callable):
            key = (obj_type, cls)
            cls._register_transform_method[key] = func
            return func

        return decorator

    def transform(self, obj: Any, **kwargs) -> Any:
        transform_class = type(self)
        obj_class = type(obj)
        key = (obj_class, transform_class)

        # Use the correct attribute name
        callable = self._register_transform_method.get(key)

        if callable is None:
            raise NotImplementedError(
                f"No transform registered for {obj_class.__name__} "
                f"with {transform_class.__name__}"
            )

        return callable(self, obj, **kwargs)

    def __call__(self, obj: Any, **kwargs) -> Any:
        return self.transform(obj, **kwargs)
