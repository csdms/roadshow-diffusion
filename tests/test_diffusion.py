"""Test the diffusion module."""
import math

import numpy as np
import pytest

from roadshow_diffusion.diffusion import calculate_time_step, set_initial_profile, make_grid, solve1d

DOMAIN_SIZE = 100
TIME_STEP = 0.5
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


def test_solve1d():
    z = np.zeros(DOMAIN_SIZE)
    z[DOMAIN_SIZE // 2] = ZMAX
    solve1d(z, time_step=TIME_STEP)
    assert type(z) is np.ndarray
    assert len(z) == DOMAIN_SIZE
    assert z.max() < ZMAX
    assert z.sum() == ZMAX
