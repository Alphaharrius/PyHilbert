from abc import ABC, abstractmethod
from dataclasses import dataclass
from multipledispatch import dispatch
from typing import Callable, Dict, ClassVar, Tuple, Type


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


class Plottable(ABC):
    """
    An object that supports dynamic plotting backends.
    """

    _registry: ClassVar[Dict[Tuple[Type, str, str], Callable]] = {}

    @classmethod
    def register_plot_method(cls, name: str, backend: str = "plotly"):
        """
        Decorator to register a plotting function for a specific class.
        Usage: @MyClass.register_plot_method("scatter")
        """

        def decorator(func: Callable):
            # We register against 'cls' - the class this method was called on.
            Plottable._registry[(cls, name, backend)] = func
            return func

        return decorator

    def plot(self, method: str, backend: str = "plotly", *args, **kwargs):
        """
        Dispatch the plot method to the registered function via MRO.
        """
        # Iterate over the MRO (Method Resolution Order) of the instance
        for class_in_hierarchy in type(self).__mro__:
            key = (class_in_hierarchy, method, backend)

            # Check the central registry
            if key in Plottable._registry:
                plot_func = Plottable._registry[key]
                return plot_func(self, *args, **kwargs)

        # If we reach here, no method was found. Provide a helpful error.
        self._raise_method_not_found(method, backend)

    def _raise_method_not_found(self, method: str, backend: str):
        # Filter available methods to only those relevant to this object (subclasses of valid types)
        available = []
        for reg_cls, reg_name, reg_backend in Plottable._registry:
            if isinstance(self, reg_cls):
                available.append(f"{reg_name} ({reg_backend})")

        msg = (
            f"No plot method '{method}' with backend '{backend}' found for {type(self).__name__}.\n"
            f"Available methods for this object: {', '.join(available) or 'None'}"
        )
        raise ValueError(msg)


class HasDual(ABC):
    """
    An object that has a dual.
    """

    @property
    @abstractmethod
    def dual(self):
        raise NotImplementedError()
