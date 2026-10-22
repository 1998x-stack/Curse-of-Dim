"""Analytic reference results and geometry helpers for experiment assertions.

Assumptions: independent uniform coordinates on [0,1], Euclidean distance
except the periodic L-infinity nearest-neighbor radius.
"""
from __future__ import annotations

import math


def grid_cell_count(dimension: int, bins_per_axis: int = 10) -> int:
    """Cells in an equally discretized d-dimensional unit cube (exact integer)."""
    if dimension < 1 or bins_per_axis < 1:
        raise ValueError("dimension and bins_per_axis must be positive")
    return bins_per_axis**dimension


def central_volume_fraction(dimension: int, inner_width: float = 0.8) -> float:
    """Share with every coordinate in the central interval of width inner_width."""
    if dimension < 1 or not 0.0 <= inner_width <= 1.0:
        raise ValueError("dimension >= 1 and width in [0, 1] required")
    return inner_width**dimension


def uniform_squared_distance_moments(dimension: int) -> tuple[float, float]:
    """E[||X-Y||²] and Var[||X-Y||²] for i.i.d. U(0,1) vectors."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    return dimension / 6.0, 7.0 * dimension / 180.0


def euclidean_distance_cv_delta_approx(dimension: int) -> float:
    """Leading-order delta-method approximation of CV(||X-Y||)."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    return math.sqrt(7.0 / (20.0 * dimension))


def median_periodic_linf_nearest_radius(n_samples: int, dimension: int) -> float:
    """Median radius from a fixed query to n uniform samples on a periodic cube.

    For 0 <= r <= 1/2, P(R_min > r) = (1-(2r)^d)^n.
    Excludes the query itself from the sample set.
    """
    if n_samples < 1 or dimension < 1:
        raise ValueError("n_samples and dimension must be positive")
    p = -math.expm1(math.log(0.5) / n_samples)
    return 0.5 * math.exp(math.log(p) / dimension)
