"""Compatibility helpers for third-party numeric libraries."""

from __future__ import annotations

import numpy as np


def ensure_numpy_legacy_float_aliases() -> None:
    """Restore deprecated NumPy aliases expected by older dependencies."""
    if not hasattr(np, "float_"):
        np.float_ = np.float64  # type: ignore[attr-defined]


def ensure_numpy_legacy_complex_aliases() -> None:
    """Restore deprecated NumPy complex aliases expected by older dependencies."""
    if not hasattr(np, "complex_"):
        np.complex_ = np.complex128  # type: ignore[attr-defined]


def ensure_numpy_legacy_aliases() -> None:
    """Restore the NumPy scalar aliases that older Pyomo versions expect."""
    ensure_numpy_legacy_float_aliases()
    ensure_numpy_legacy_complex_aliases()
