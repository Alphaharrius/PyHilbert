import pytest
from dataclasses import dataclass
from pyhilbert.validations import Validator, Validatable, no_validate, validate_by


def test_validatable_runs_validators_after_regular_init():
    class PositiveValue(Validator["RegularValidated"]):
        def validate(self, value: "RegularValidated") -> None:
            if value.number <= 0:
                raise ValueError("number must be positive")

    @validate_by(PositiveValue())
    class RegularValidated(Validatable):
        def __init__(self, number: int):
            self.number = number

    instance = RegularValidated(3)
    assert instance.number == 3

    with pytest.raises(ValueError, match="number must be positive"):
        RegularValidated(0)


def test_validatable_runs_validators_after_dataclass_init():
    class PositiveValue(Validator["DataclassValidated"]):
        def validate(self, value: "DataclassValidated") -> None:
            if value.number <= 0:
                raise ValueError("number must be positive")

    @validate_by(PositiveValue())
    @dataclass(frozen=True)
    class DataclassValidated(Validatable):
        number: int

    instance = DataclassValidated(3)
    assert instance.number == 3

    with pytest.raises(ValueError, match="number must be positive"):
        DataclassValidated(0)


def test_validatable_inherits_validators_in_base_to_derived_order():
    calls: list[str] = []

    class RecordValidator(Validator["Validatable"]):
        def __init__(self, name: str):
            self.name = name

        def validate(self, value: Validatable) -> None:
            calls.append(self.name)

    base_validator = RecordValidator("base")
    middle_validator = RecordValidator("middle")
    leaf_validator = RecordValidator("leaf")

    @validate_by(base_validator)
    class Base(Validatable):
        def __init__(self):
            pass

    @validate_by(middle_validator)
    class Middle(Base):
        def __init__(self):
            super().__init__()

    @validate_by(leaf_validator)
    class Leaf(Middle):
        def __init__(self):
            super().__init__()

    assert Base.__validators__ == [base_validator]
    assert Middle.__validators__ == [middle_validator]
    assert Leaf.__validators__ == [leaf_validator]
    assert Base.__validators__ is not Middle.__validators__
    assert Middle.__validators__ is not Leaf.__validators__

    Leaf().validate()

    assert calls == ["base", "middle", "leaf", "base", "middle", "leaf"]


def test_validatable_runs_local_validators_at_each_init_layer():
    calls: list[int] = []

    class CountCalls(Validator["Validatable"]):
        def __init__(self, expected: int):
            self.expected = expected

        def validate(self, value: Validatable) -> None:
            calls.append(self.expected)

    @validate_by(CountCalls(1))
    class Base(Validatable):
        def __init__(self):
            self.value = 1

    @validate_by(CountCalls(2))
    class Middle(Base):
        def __init__(self):
            super().__init__()

    @validate_by(CountCalls(3))
    class Leaf(Middle):
        def __init__(self):
            super().__init__()

    Leaf()

    assert calls == [1, 2, 3]


def test_validatable_runs_local_validators_at_each_dataclass_post_init_layer():
    calls: list[int] = []

    class CountCalls(Validator["Validatable"]):
        def __init__(self, expected: int):
            self.expected = expected

        def validate(self, value: Validatable) -> None:
            calls.append(self.expected)

    @validate_by(CountCalls(1))
    @dataclass(frozen=True)
    class Base(Validatable):
        value: int

    @validate_by(CountCalls(2))
    @dataclass(frozen=True)
    class Middle(Base):
        def __post_init__(self) -> None:
            super().__post_init__()

    @validate_by(CountCalls(3))
    @dataclass(frozen=True)
    class Leaf(Middle):
        def __post_init__(self) -> None:
            super().__post_init__()

    Leaf(1)

    assert calls == [1, 2, 3]


def test_validatable_dataclass_validate_runs_full_chain_in_base_to_derived_order():
    calls: list[str] = []

    class RecordValidator(Validator["Validatable"]):
        def __init__(self, name: str):
            self.name = name

        def validate(self, value: Validatable) -> None:
            calls.append(self.name)

    base_validator = RecordValidator("base")
    middle_validator = RecordValidator("middle")
    leaf_validator = RecordValidator("leaf")

    @validate_by(base_validator)
    @dataclass(frozen=True)
    class Base(Validatable):
        value: int

    @validate_by(middle_validator)
    @dataclass(frozen=True)
    class Middle(Base):
        def __post_init__(self) -> None:
            super().__post_init__()

    @validate_by(leaf_validator)
    @dataclass(frozen=True)
    class Leaf(Middle):
        def __post_init__(self) -> None:
            super().__post_init__()

    assert Base.__validators__ == [base_validator]
    assert Middle.__validators__ == [middle_validator]
    assert Leaf.__validators__ == [leaf_validator]
    assert Base.__validators__ is not Middle.__validators__
    assert Middle.__validators__ is not Leaf.__validators__

    Leaf(1).validate()

    assert calls == ["base", "middle", "leaf", "base", "middle", "leaf"]


def test_no_validate_disables_construction_and_explicit_validation():
    calls: list[int] = []

    class PositiveValue(Validator["LocalValidated"]):
        def validate(self, value: "LocalValidated") -> None:
            calls.append(value.number)
            if value.number <= 0:
                raise ValueError("number must be positive")

    @validate_by(PositiveValue())
    class LocalValidated(Validatable):
        def __init__(self, number: int):
            self.number = number

    with no_validate():
        instance = LocalValidated(0)
        assert instance.number == 0
        assert instance.validate() is instance

    assert calls == []

    with pytest.raises(ValueError, match="number must be positive"):
        instance.validate()

    assert calls == [0]
