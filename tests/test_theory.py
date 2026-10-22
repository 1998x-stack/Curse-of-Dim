import pytest

from curse_of_dim.theory import (
    central_volume_fraction,
    euclidean_distance_cv_delta_approx,
    grid_cell_count,
    median_periodic_linf_nearest_radius,
    uniform_squared_distance_moments,
)


def test_grid_growth_exact_integer():
    assert grid_cell_count(100) == 10**100
    assert grid_cell_count(4, bins_per_axis=2) == 16


def test_central_fraction():
    assert central_volume_fraction(3) == pytest.approx(0.8**3)
    assert central_volume_fraction(100) < central_volume_fraction(2)


def test_squared_distance_moments():
    mean, var = uniform_squared_distance_moments(10)
    assert mean == pytest.approx(10 / 6)
    assert var == pytest.approx(70 / 180)


def test_delta_method_cv_shrinks_as_inverse_sqrt_d():
    a = euclidean_distance_cv_delta_approx(10)
    b = euclidean_distance_cv_delta_approx(40)
    assert b / a == pytest.approx(0.5)


def test_periodic_linf_median_matches_distribution():
    n, d = 1000, 5
    r = median_periodic_linf_nearest_radius(n, d)
    assert 0.0 < r < 0.5
    assert (1 - (2 * r) ** d) ** n == pytest.approx(0.5)


@pytest.mark.parametrize("func,args", [
    (grid_cell_count, (0,)),
    (central_volume_fraction, (0,)),
    (uniform_squared_distance_moments, (-1,)),
    (euclidean_distance_cv_delta_approx, (0,)),
    (median_periodic_linf_nearest_radius, (0, 2)),
])
def test_reject_bad_arguments(func, args):
    with pytest.raises(ValueError):
        func(*args)


def test_noise_free_orthonormal_map_preserves_distances():
    import numpy as np
    rng = np.random.default_rng(111)
    z = rng.normal(size=(50, 2))
    q, _ = np.linalg.qr(rng.normal(size=(128, 2)))
    x = z @ q.T
    assert np.allclose(np.linalg.norm(z[1:] - z[:-1], axis=1),
                       np.linalg.norm(x[1:] - x[:-1], axis=1), atol=1e-12)
