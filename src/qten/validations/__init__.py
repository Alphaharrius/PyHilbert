"""
Validation decorators and symbolic validation helpers.

This package centralizes the runtime validation layer used across QTen
dataclasses and symbolic objects.

Decorator exports
-----------------
- [`need_validation`][qten.validations]
  Attach validators to a class or function.
- [`validate`][qten.validations]
  Execute registered validators explicitly.
- [`no_validate`][qten.validations]
  Temporarily suppress validation.

Related module
--------------
- [`qten.validations.symbolics`][qten.validations.symbolics]
  Symbolics-specific validation rules used by geometry and Hilbert-space
  objects.
"""

from ._validations import (
    need_validation as need_validation,
    no_validate as no_validate,
    validate as validate,
)
