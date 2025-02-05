"""Test the diffusion module."""
import numpy as np
import pytest

from roadshow_diffusion import calculate_stable_time_step
from roadshow_diffusion import diffuse_until
from roadshow_diffusion import step_like


def test_time_step_is_float():
    time_step = calculate_stable_time_step(1, 1)
    assert isinstance(time_step, float)


def test_time_step_with_zero_spacing():
    assert calculate_stable_time_step(0.0, 1) == pytest.approx(0.0)


def test_time_step_increases_with_spacing():
    time_steps = [
        calculate_stable_time_step(spacing, 1.0)
        for spacing in [1.0, 10.0, 100.0, 1000.0]
    ]
    assert np.all(np.diff(time_steps) > 0.0)


def test_time_step_decreases_with_diffusivity():
    time_steps = [
        calculate_stable_time_step(1.0, diffusivity)
        for diffusivity in [1.0, 10.0, 100.0, 1000.0]
    ]
    assert np.all(np.diff(time_steps) < 0.0)


def test_step_like_is_array_of_float():
    """Check profile is a numpy array of floats."""
    z = step_like([1.0, 2.0, 3.0])
    return isinstance(z, np.ndarray) and np.issubdtype(z.dtype, np.floating)

    z = step_like([1, 2, 3])
    return isinstance(z, np.ndarray) and np.issubdtype(z.dtype, np.floating)


def test_step_like_length():
    """Check the array is of the correct length."""
    z = step_like(np.arange(100))
    assert len(z) == 100


def test_step_like_min_max():
    """Check values are in range."""
    x = np.arange(100.0)
    z = step_like(np.arange(100.0), step_at=len(x) // 2)
    assert np.all(z >= 0.0) and np.all(z <= 1.0)
    assert z.min() == pytest.approx(0.0)
    assert z.max() == pytest.approx(1.0)


def test_diffuse_until_does_something():
    """Check that things actually change."""
    z_initial = np.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    z = diffuse_until(z_initial, 1.0)

    assert np.abs(z - z_initial).max() > 0


def test_diffuse_until_fixed_boundaries():
    """Check that boundary values don't change."""
    z = diffuse_until(np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]), 10.0)
    assert z[0] == pytest.approx(0.0)
    assert z[-1] == pytest.approx(0.0)

    z[0] = 10.0
    z[-1] = 100.0
    z = diffuse_until(z, 10.0)

    assert z[0] == pytest.approx(10.0)
    assert z[-1] == pytest.approx(100.0)


def test_diffuse_until_in_bounds():
    """Check that values remain with max/min."""
    z = diffuse_until(np.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), 1.0)

    assert np.all(z <= 1.0)
    assert np.all(z >= 0.0)


def test_diffuse_until_mass_balance():
    """Check that mass is concerved."""
    time_step = 0.5 * calculate_stable_time_step(1.0, 1.0)

    z_initial = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    expected = z_initial.sum()
    z = diffuse_until(z_initial, time_step)
    actual = z.sum()

    assert actual == pytest.approx(expected)


def test_diffuse_until_time_step():
    """Check there's less diffusion with small time step."""
    z_initial = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    time_step = 0.5 * calculate_stable_time_step(1.0, 1.0)

    dz_large_dt = z_initial - diffuse_until(z_initial, time_step)
    dz_small_dt = z_initial - diffuse_until(z_initial, time_step / 2.0)

    assert np.abs(dz_large_dt - dz_small_dt).max() > 0
    assert np.all(np.abs(dz_large_dt) >= np.abs(dz_small_dt))
