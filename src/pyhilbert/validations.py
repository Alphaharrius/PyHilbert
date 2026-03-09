from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from threading import local
from typing import Any, Callable, Generic, Iterator, List, Self, Type, TypeVar, cast


_VALIDATION_STATE = local()

V = TypeVar("V", bound="Validatable", contravariant=True)
"""
Type variable for :class:`Validatable` objects.

This bound keeps validator signatures type-safe: a ``Validator[V]`` can only be
attached to classes that inherit from :class:`Validatable`, and its
``validate()`` method will receive the concrete validatable instance type.
"""


def _validation_disabled_depth() -> int:
    return cast(int, getattr(_VALIDATION_STATE, "disabled_depth", 0))


@contextmanager
def no_validate() -> Iterator[None]:
    """
    Temporarily disable validation in the current thread.

    While this context manager is active, construction-time validators and
    explicit :meth:`Validatable.validate` calls become no-ops. Nesting is
    supported.
    """
    previous_depth = _validation_disabled_depth()
    _VALIDATION_STATE.disabled_depth = previous_depth + 1
    try:
        yield
    finally:
        if previous_depth == 0:
            delattr(_VALIDATION_STATE, "disabled_depth")
        else:
            _VALIDATION_STATE.disabled_depth = previous_depth


def validate_by(
    *validators: "Validator[V]",
) -> Callable[[Type[V]], Type[V]]:
    """
    Attach validators to a :class:`Validatable` subclass.

    The decorator appends ``validators`` to the class-local validator list
    prepared during :meth:`Validatable.__init_subclass__`. Validators are
    stored on the class that declares them; they are not copied onto
    subclasses.

    During construction, each wrapped ``__init__`` or ``__post_init__`` runs
    the validators declared on that class after that specific layer completes.
    Explicit calls to :meth:`Validatable.validate` walk the class hierarchy in
    base-to-derived order and run the full validator chain once.

    Parameters
    ----------
    `validators` : `Validator[V]`
        Validator instances to append to the decorated class.

    Returns
    -------
    `Callable[[Type[V]], Type[V]]`
        A class decorator that augments the target class and returns it.
    """

    def decorator(cls: Type[V]) -> Type[V]:
        cls.__validators__ = [*cls.__validators__, *validators]
        return cls

    return decorator


class Validator(Generic[V], ABC):
    """
    Abstract interface for object-level validation.

    A validator encapsulates one validation rule for instances of a particular
    :class:`Validatable` subtype. Validators are typically attached to classes
    with :func:`validate_by` and then executed automatically by
    :meth:`Validatable.validate` after object construction.
    """

    @abstractmethod
    def validate(self, value: V) -> None:
        """
        Validate a fully constructed object instance.

        Parameters
        ----------
        `value` : `V`
            The instance being validated.

        Raises
        ------
        `ValueError`
            Raised when the instance violates the validator's constraint.
        """
        pass


class Validatable(ABC):
    """
    Mixin that provides inherited, post-construction validation.

    Subclasses own only their locally declared validators. Validators attached
    through :func:`validate_by` are stored on the class that declares them and
    are not copied onto descendants.

    Validation is designed to run after initialization:
    - For regular classes, a class-defined ``__init__`` is wrapped so
      that class's validators run after that constructor returns.
    - For dataclasses without a class-defined ``__init__``, ``__post_init__``
      is used as the hook so that class's validators run after the
      dataclass-generated ``__init__`` has assigned all fields.

    Notes
    -----
    ``__validators__`` is not shared mutably across the hierarchy. Every
    subclass gets its own local list prepared during :meth:`__init_subclass__`.
    """

    __validators__: List[Validator[Self]] = []
    """Validators declared directly on the concrete class."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Prepare subclass-local validator state and validation hooks.

        When a new :class:`Validatable` subclass is created, this method:
        1. Initializes a fresh class-local validator list for the subclass.
        2. Wraps any subclass-defined ``__init__`` so that class's validators
           run after its constructor returns.
        3. If the subclass does not define ``__init__``, installs or wraps
           ``__post_init__`` so dataclass-generated construction can still run
           that class's validators.

        Parameters
        ----------
        `kwargs` : `Any`
            Standard subclass initialization keyword arguments forwarded to the
            parent implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.__validators__ = []

        init = cls.__dict__.get("__init__")
        if init is not None:
            setattr(
                cls,
                "__init__",
                Validatable._wrap_init(cls, cast(Callable[..., None], init)),
            )
            return

        post_init = cls.__dict__.get("__post_init__")
        if post_init is None:
            setattr(cls, "__post_init__", Validatable._make_post_init(cls))
        else:
            setattr(
                cls,
                "__post_init__",
                Validatable._wrap_post_init(cls, cast(Callable[..., None], post_init)),
            )

    @staticmethod
    def _run_class_validators(
        owner: Type["Validatable"], instance: "Validatable"
    ) -> None:
        if _validation_disabled_depth() > 0:
            return
        for validator in owner.__validators__:
            validator.validate(instance)

    @staticmethod
    def _wrap_init(
        owner: Type["Validatable"], init: Callable[..., None]
    ) -> Callable[..., None]:
        """
        Wrap an ``__init__`` implementation to validate after construction.

        Parameters
        ----------
        `owner` : `Type[Validatable]`
            The class that owns the wrapped constructor.
        `init` : `Callable[..., None]`
            The original constructor defined on the subclass.

        Returns
        -------
        `Callable[..., None]`
            A constructor wrapper that preserves the original initialization
            logic and then runs the validators declared on `owner`.
        """

        def wrapped(self: Self, *args: Any, **kwargs: Any) -> None:
            init(self, *args, **kwargs)
            Validatable._run_class_validators(owner, self)

        return wrapped

    @staticmethod
    def _make_post_init(owner: Type["Validatable"]) -> Callable[..., None]:
        """
        Create a default ``__post_init__`` implementation for dataclasses.

        Parameters
        ----------
        `owner` : `Type[Validatable]`
            The class that owns the generated post-init hook.

        Returns
        -------
        `Callable[..., None]`
            A ``__post_init__`` method that runs the validators declared on
            `owner`.
        """

        def wrapped(self: Self, *args: Any, **kwargs: Any) -> None:
            Validatable._run_class_validators(owner, self)

        return wrapped

    @staticmethod
    def _wrap_post_init(
        owner: Type["Validatable"],
        post_init: Callable[..., None],
    ) -> Callable[..., None]:
        """
        Wrap an ``__post_init__`` implementation to validate afterward.

        This is primarily used for dataclasses, whose generated ``__init__``
        invokes ``__post_init__`` once field assignment is complete.

        Parameters
        ----------
        `owner` : `Type[Validatable]`
            The class that owns the wrapped post-init hook.
        `post_init` : `Callable[..., None]`
            The original ``__post_init__`` method defined on the subclass.

        Returns
        -------
        `Callable[..., None]`
            A wrapper that preserves the original post-initialization logic and
            then runs the validators declared on `owner`.
        """

        def wrapped(self: Self, *args: Any, **kwargs: Any) -> None:
            post_init(self, *args, **kwargs)
            Validatable._run_class_validators(owner, self)

        return wrapped

    def validate(self) -> Self:
        """
        Execute the full validator chain for this instance.

        Validators run in base-to-derived order across the validatable classes
        in the instance's method resolution order. The method returns ``self``
        so callers may validate inline when useful.

        Returns
        -------
        `Self`
            The validated instance.

        Raises
        ------
        `ValueError`
            Propagated from any validator that rejects the instance.
        """
        if _validation_disabled_depth() > 0:
            return self
        seen: List[Validator[Any]] = []
        for cls in reversed(type(self).__mro__):
            if not issubclass(cls, Validatable):
                continue
            for validator in getattr(cls, "__validators__", ()):
                if validator not in seen:
                    seen.append(validator)
                    validator.validate(self)
        return self
