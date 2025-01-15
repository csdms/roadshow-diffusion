import os
import tomllib

import numpy as np
import matplotlib.pyplot as plt


def calculate_stable_time_step(dx, diffusivity):
    return 0.5 * dx**2 / diffusivity


def step_like(x, step_at=0):
    y = np.empty_like(x, dtype=float)

    y[:step_at] = 1.0
    y[step_at] = 0.5
    y[step_at+1:] = 0.0

    return y


def plot_profile(x, concentration, color="r"):
    plt.figure()
    plt.plot(x, concentration, color)
    plt.xlabel("x")
    plt.ylabel("C")
    plt.title("Concentration profile")


def calculate_second_derivative(y, dx=1.0):
    d2y_dx2 = np.empty_like(y)

    d2y_dx2[1:-1] = (y[:-2] - 2 * y[1:-1] + y[2:]) / dx**2
    d2y_dx2[0] = 0.0
    d2y_dx2[-1] = 0.0

    return d2y_dx2


def diffuse_until(y_initial, stop_time, dx=1.0, diffusivity=1.0):
    stable_dt = 0.9 * calculate_stable_time_step(dx, diffusivity)

    y = y_initial.copy()

    time_remaining = stop_time
    while time_remaining > 0.0:
        dt = min(time_remaining, stable_dt)

        y += diffusivity * dt * calculate_second_derivative(y, dx=dx)

        time_remaining -= dt

    return y


def run_diffusion_model(diffusivity=100.0, width=100.0, stop_time=1.0, n_points=81):
    x, dx = np.linspace(0, width, num=n_points, retstep=True)
    initial_concentration = step_like(x, step_at=len(x) // 2)

    concentration = diffuse_until(
        initial_concentration, stop_time, dx=dx, diffusivity=diffusivity
    )

    plot_profile(x, initial_concentration, "g")
    plot_profile(x, concentration, "r")
    plt.show()

    return concentration


def load_params_from_path(filepath):
    with open(filepath, "rb") as stream:
        params = tomllib.load(stream)
    return params


if __name__ == "__main__":
    print("Diffusion model")
    filepath = "diffusion.toml"

    if os.path.isfile(filepath):
        params = load_params_from_path(filepath)
    else:
        params = {}
    concentration = run_diffusion_model(**params)

    print(concentration)
