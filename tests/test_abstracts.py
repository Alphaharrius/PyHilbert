import pytest
from dataclasses import dataclass
from pyhilbert.abstracts import Operable, Updatable


@dataclass(frozen=True)
class MockOperable(Operable):
    pass


@dataclass(frozen=True)
class MockUpdatable(Updatable):
    val: int = 0

    def _updated(self, **kwargs) -> "Updatable":
        new_val = kwargs.get("val", self.val)
        return MockUpdatable(val=new_val)


@dataclass(frozen=True)
class BadUpdatable(Updatable):
    def _updated(self, **kwargs):
        return self


def test_operable_unimplemented():
    a = MockOperable()
    b = MockOperable()

    # Test all default implementations return NotImplementedError

    # Arithmetics
    # Note: The current implementation returns the Error object rather than raising it?
    # Let's check the code: return NotImplementedError(...)
    # It returns an instance of the exception, it doesn't raise it.

    res = a + b
    assert isinstance(res, NotImplementedError)

    res = -a
    assert isinstance(res, NotImplementedError)

    # a - b calls a + (-b).
    # -b returns NotImplementedError.
    # a + NotImplementedError -> ??
    # The dispatch is for (Operable, Operable).
    # NotImplementedError is not Operable.
    # So this might raise a Dispatch error or standard TypeError depending on multipledispatch behavior.

    # But let's check explicit calls to operators if they return the error object.

    assert isinstance(a * b, NotImplementedError)
    assert isinstance(a @ b, NotImplementedError)
    assert isinstance(a / b, NotImplementedError)
    assert isinstance(a // b, NotImplementedError)
    assert isinstance(a**b, NotImplementedError)

    # Comparisons
    assert isinstance(a < b, NotImplementedError)
    assert isinstance(a <= b, NotImplementedError)
    assert isinstance(a > b, NotImplementedError)
    assert isinstance(a >= b, NotImplementedError)

    # Logical
    assert isinstance(a & b, NotImplementedError)
    assert isinstance(a | b, NotImplementedError)


def test_updatable_correct():
    u = MockUpdatable(val=1)
    u2 = u.update(val=2)
    assert u2.val == 2
    assert u2 is not u


def test_updatable_bad_implementation():
    b = BadUpdatable()
    with pytest.raises(RuntimeError, match="must not return self"):
        b.update(foo="bar")
