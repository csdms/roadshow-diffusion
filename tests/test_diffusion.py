"""Test the diffusion module."""
import numpy as np
import pytest

from roadshow_diffusion import calculate_time_step
from roadshow_diffusion import make_grid
from roadshow_diffusion import set_initial_profile
from roadshow_diffusion import solve1d


def test_time_step_is_float():
    time_step = calculate_time_step(1, 1)
    assert isinstance(time_step, float)


def test_time_step_with_zero_spacing():
    assert calculate_time_step(0.0, 1) == pytest.approx(0.0)


def test_time_step_increases_with_spacing():
    time_steps = [
        calculate_time_step(spacing, 1.0)
        for spacing in [1.0, 10.0, 100.0, 1000.0]
    ]
    assert np.all(np.diff(time_steps) > 0.0)


def test_time_step_decreases_with_diffusivity():
    time_steps = [
        calculate_time_step(1.0, diffusivity)
        for diffusivity in [1.0, 10.0, 100.0, 1000.0]
    ]
    assert np.all(np.diff(time_steps) < 0.0)


def test_initial_profile_is_array_of_float():
    """Check profile is a numpy array of floats."""
    z = set_initial_profile()
    return isinstance(z, np.ndarray) and np.issubdtype(z.dtype, np.floating)


def test_initial_profile_length():
    """Check the array is of the correct length."""
    z = set_initial_profile(grid_size=100)
    assert len(z) == 100


def test_initial_profile_min_max():
    """Check values are in range."""
    z = set_initial_profile(boundary_left=500, boundary_right=0)
    assert np.all(z >= 0.0) and np.all(z <= 500.0)
    assert z.min() == pytest.approx(0.0)
    assert z.max() == pytest.approx(500.0)


def test_make_grid_length():
    """Check the length of the grid."""
    x, size = make_grid(100.0, 1.0)
    assert len(x) == 100
    assert size == len(x)


def test_make_grid_spacing():
    """Check the grid spacing."""
    x, size = make_grid(50.0, 10.0)

    assert size == 5
    assert np.all(np.diff(x) == pytest.approx(10.0))


def test_make_grid_end_points():
    """Check the grid end points."""
    width, spacing = 500.0, 0.5
    x, size = make_grid(width, spacing)

    assert size == 1000
    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(width - spacing)


def test_solve1d_does_something():
    """Check that things actually change."""
    z = np.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    z_initial = z.copy()

    solve1d(z, time_step=0.5)

    assert np.abs(z - z_initial).max() > 0


def test_solve1d_fixed_boundaries():
    """Check that boundary values don't change."""
    time_step = 0.5 * calculate_time_step(1.0, 1.0)
    z = np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])

    solve1d(z, time_step=time_step)
    assert z[0] == pytest.approx(0.0)
    assert z[-1] == pytest.approx(0.0)

    z[0] = 10.0
    z[-1] = 100.0
    for _ in range(1000):
        solve1d(z, time_step=1.0)

    assert z[0] == pytest.approx(10.0)
    assert z[-1] == pytest.approx(100.0)


def test_solve1d_in_bounds():
    """Check that values remain with max/min."""
    time_step = 0.5 * calculate_time_step(1.0, 1.0)
    z = np.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    solve1d(z, time_step=time_step)

    assert np.all(z <= 1.0)
    assert np.all(z >= 0.0)


def test_solve1d_mass_balance():
    """Check that mass is concerved."""
    time_step = 0.5 * calculate_time_step(1.0, 1.0)
    z = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    expected = z.sum()
    solve1d(z, time_step=time_step)
    actual = z.sum()

    assert actual == pytest.approx(expected)


def test_solve1d_time_step():
    """Check there's less diffusion with small time step."""
    z_initial = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    time_step = 0.5 * calculate_time_step(1.0, 1.0)
    z = z_initial.copy()

    solve1d(z, time_step=time_step)
    dz_large_dt = z_initial - z

    time_step *= 0.5
    z = z_initial.copy()
    solve1d(z, time_step=time_step)
    dz_small_dt = z_initial - z

    assert np.abs(dz_large_dt - dz_small_dt).max() > 0
    assert np.all(np.abs(dz_large_dt) >= np.abs(dz_small_dt))
