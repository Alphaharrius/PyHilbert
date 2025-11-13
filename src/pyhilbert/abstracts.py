from dataclasses import dataclass
from multipledispatch import dispatch


@dataclass(frozen=True)
class Operable:
    def __add__(self, other: 'Operable'):
        return operator_add(self, other)

    def __neg__(self):
        return operator_neg(self)

    def __sub__(self, other: 'Operable'):
        return operator_sub(self, other)

    def __mul__(self, other: 'Operable'):
        return operator_mul(self, other)

    def __matmul__(self, other: 'Operable'):
        return operator_matmul(self, other)

    def __truediv__(self, other: 'Operable'):
        return operator_truediv(self, other)

    def __floordiv__(self, other: 'Operable'):
        return operator_floordiv(self, other)

    def __pow__(self, other: 'Operable'):
        return operator_pow(self, other)
    
    def __lt__(self, other: 'Operable'):
        return operator_lt(self, other)
    
    def __le__(self, other: 'Operable'):
        return operator_le(self, other)
    
    def __gt__(self, other: 'Operable'):
        return operator_gt(self, other)
    
    def __ge__(self, other: 'Operable'):
        return operator_ge(self, other)
    
    def __and__(self, other: 'Operable'):
        return operator_and(self, other)
    
    def __or__(self, other: 'Operable'):
        return operator_or(self, other)


@dispatch(Operable, Operable)
def operator_add(a, b):
    return NotImplementedError(f'Addition of {type(a)} and {type(b)} is not supported!')


@dispatch(Operable, Operable)
def operator_neg(a):
    return NotImplementedError(f'Negation of {type(a)} is not supported!')


@dispatch(Operable, Operable)
def operator_sub(a, b):
    return a + (-b)


@dispatch(Operable, Operable)
def operator_mul(a, b):
    return NotImplementedError(f'Multiplication of {type(a)} and {type(b)} is not supported!')


@dispatch(Operable, Operable)
def operator_matmul(a, b):
    return NotImplementedError(f'Matrix multiplication of {type(a)} and {type(b)} is not supported!')


@dispatch(Operable, Operable)
def operator_truediv(a, b):
    return NotImplementedError(f'Division of {type(a)} and {type(b)} is not supported!')


@dispatch(Operable, Operable)
def operator_floordiv(a, b):
    return NotImplementedError(f'Floor division of {type(a)} and {type(b)} is not supported!')


@dispatch(Operable, Operable)
def operator_pow(a, b):
    return NotImplementedError(f'Exponentiation of {type(a)} and {type(b)} is not supported!')


@dispatch(Operable, Operable)
def operator_lt(a, b):
    return NotImplementedError(f'Less-than comparison of {type(a)} and {type(b)} is not supported!')


@dispatch(Operable, Operable)
def operator_le(a, b):
    return NotImplementedError(f'Less-than-or-equal comparison of {type(a)} and {type(b)} is not supported!')


@dispatch(Operable, Operable)
def operator_gt(a, b):
    return NotImplementedError(f'Greater-than comparison of {type(a)} and {type(b)} is not supported!')


@dispatch(Operable, Operable)
def operator_ge(a, b):
    return NotImplementedError(f'Greater-than-or-equal comparison of {type(a)} and {type(b)} is not supported!')


@dispatch(Operable, Operable)
def operator_and(a, b):
    return NotImplementedError(f'Logical AND of {type(a)} and {type(b)} is not supported!')


@dispatch(Operable, Operable)
def operator_or(a, b):
    return NotImplementedError(f'Logical OR of {type(a)} and {type(b)} is not supported!')
