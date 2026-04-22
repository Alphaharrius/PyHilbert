"""
Lightweight physics-facing data structures built on QTen primitives.

This package currently exposes small domain objects that represent physical
relationships and observables while integrating with the symbolic and plotting
infrastructure elsewhere in the library.

Exports
-------
- [`Bond`][qten.phys]
  Connection object describing a relation between sites or offsets.
- [`FFObservable`][qten.phys]
  Form-factor style observable wrapper.
"""

from ._bonds import Bond as Bond
from ._ff_observables import FFObservable as FFObservable
