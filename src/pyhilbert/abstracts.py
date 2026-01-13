from abc import ABC, abstractmethod
from dataclasses import dataclass
from multipledispatch import dispatch
from typing import Callable, Dict, ClassVar, Tuple


@dataclass(frozen=True)
class Operable(ABC):
    def __add__(self, other: "Operable"):
        return operator_add(self, other)

    def __neg__(self):
        return operator_neg(self)

    def __sub__(self, other: "Operable"):
        return operator_sub(self, other)

    def __mul__(self, other: "Operable"):
        return operator_mul(self, other)

    def __matmul__(self, other: "Operable"):
        return operator_matmul(self, other)

    def __truediv__(self, other: "Operable"):
        return operator_truediv(self, other)

    def __floordiv__(self, other: "Operable"):
        return operator_floordiv(self, other)

    def __pow__(self, other: "Operable"):
        return operator_pow(self, other)

    def __eq__(self, value):
        return operator_eq(self, value)

    def __lt__(self, other: "Operable"):
        return operator_lt(self, other)

    def __le__(self, other: "Operable"):
        return operator_le(self, other)

    def __gt__(self, other: "Operable"):
        return operator_gt(self, other)

    def __ge__(self, other: "Operable"):
        return operator_ge(self, other)

    def __and__(self, other: "Operable"):
        return operator_and(self, other)

    def __or__(self, other: "Operable"):
        return operator_or(self, other)


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
    _plot_methods: ClassVar[Dict[Tuple[str,str], Callable]] = {}
    @classmethod
    def register_plot_method(cls, name: str, backend: str='plotly'):
        def decorator(func: Callable):
            cls._plot_methods[(name, backend)] = func
            return func

        return decorator
    
    def plot(self, method: str, backend: str='plotly', *args, **kwargs):
        """
        Dispatch the plot method to the registered function.
        """
        if (method, backend) not in self._plot_methods:
            raise ValueError(f"Plot method {method} not found. Available methods: {list(self._plot_methods.keys())}")
        return self._plot_methods[(method, backend)](self, *args, **kwargs)


class HasDual(ABC):
    """
    An object that has a dual.
    """

    @property
    @abstractmethod
    def dual(self):
        raise NotImplementedError()
