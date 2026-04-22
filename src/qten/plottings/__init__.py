"""
Core plotting dispatch layer for QTen objects.

This package defines the lightweight plotting interface used by QTen and its
plotting extensions. Backend-specific implementations may live elsewhere, but
objects become plottable by inheriting from the exported base class and
registering plotting methods.

Exports
-------
- [`Plottable`][qten.plottings]
  Base class providing backend-aware plot-method registration and dispatch.

Notes
-----
The concrete plot object types exposed to users are primarily documented in
[`qten_plots`][qten_plots], while this package focuses on the shared dispatch
mechanism.
"""

from ._plottings import Plottable as Plottable
