"""Test the diffusion module."""
import math

import numpy as np
import pytest

from roadshow_diffusion.diffusion import calculate_time_step, set_initial_profile, make_grid, solve1d

DOMAIN_SIZE = 100
TOLERANCE = 0.01
ZMAX = 500.0


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


def test_initial_profile_defaults():
    z = set_initial_profile()
    assert type(z) is np.ndarray
    assert len(z) == DOMAIN_SIZE
    assert math.isclose(z.max(), ZMAX, rel_tol=TOLERANCE)


def test_make_grid():
    pass


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
