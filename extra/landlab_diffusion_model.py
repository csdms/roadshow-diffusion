import os
import sys
import tomllib

import numpy as np
import matplotlib.pyplot as plt
from landlab import RasterModelGrid


def calculate_stable_time_step(dx, diffusivity):
    return 0.25 * dx**2 / diffusivity


def new_profile(grid, step_at=1.0):
    z = grid.zeros(at="node")
    z[grid.x_of_node >= step_at] = 1.0
    return z


def plot_profile(grid, concentration, color="r"):
    grid.imshow(concentration)


def calculate_elevation_change(grid, z, diffusivity):
    dzdl = grid.calc_grad_at_link(z)
    qs_at_link = -diffusivity * dzdl
    dzdt = -grid.calc_flux_div_at_node(qs_at_link)
    return dzdt


def diffuse_until(grid, z_initial, stop_time, diffusivity=1.0):
    stable_dt = 0.9 * calculate_stable_time_step(np.min(grid.length_of_link), diffusivity)
    z = z_initial.copy()
    
    time = 0
    while time < stop_time:
        dt = min(stable_dt, stop_time - time)
        z += dt* calculate_elevation_change(grid, z, diffusivity)
        time += dt

    return z


def run_diffusion_model():
    shape = (100, 200)
    stop_time = 5.0
    diffusivity = 10.0
    grid = RasterModelGrid(shape, xy_spacing=(1.0, 1.0))
    z_initial = new_profile(grid, step_at=100)
    z = diffuse_until(grid, z_initial, stop_time, diffusivity=diffusivity)
    plot_profile(grid, z)
