import numpy as np
import matplotlib.pyplot as plt


def calculate_time_step(grid_spacing, diffusivity):
    return 0.5 * grid_spacing**2 / diffusivity


def set_initial_profile(grid_size=100, boundary_left=500, boundary_right=0):
    profile = np.empty(grid_size)
    profile[: grid_size // 2] = boundary_left
    profile[grid_size // 2 :] = boundary_right
    return profile


def plot_profile(concentration, grid, color="r"):
    plt.figure()
    plt.plot(grid, concentration, color)
    plt.xlabel("x")
    plt.ylabel("C")
    plt.title("Concentration profile")


def make_grid(domain_size, grid_spacing):
    grid = np.arange(start=0, stop=domain_size, step=grid_spacing)
    return (grid, len(grid))


def solve1d(concentration, grid_spacing=1.0, time_step=1.0, diffusivity=1.0):
    """Solve the one-dimensional diffusion equation with fixed boundary conditions.

    Parameters
    ----------
    concentration : ndarray
        The quantity being diffused.
    grid_spacing : float (optional)
        Distance between grid nodes.
    time_step : float (optional)
        Time step of model.
    diffusivity : float (optional)
        Diffusivity.

    Returns
    -------
    result : ndarray
        The concentration after a time step.

    Examples
    --------
    >>> import numpy as np
    >>> from roadshow_diffusion.solver import solve1d
    >>> z = np.zeros(5)
    >>> z[2] = 5
    >>> z
    array([0.0, 0.0, 5.0, 0.0, 0.0])
    >>> solve1d(z, diffusivity=0.25)
    >>> z
    array([0.0, 1.2, 2.5, 1.2, 0.0])
    """
    centered_diff = np.roll(concentration, -1) - 2*concentration + np.roll(concentration, 1)
    concentration[1:-1] += diffusivity * time_step / grid_spacing**2 * centered_diff[1:-1]


def diffusion_model():
    D = 100
    Lx = 7
    dx = 0.5

    x, nx = make_grid(Lx, dx)
    dt = calculate_time_step(dx, D)
    C = set_initial_profile(nx, boundary_left=500, boundary_right=0)

    print("Time = 0\n", C)
    for t in range(0, 5):
        solve1d(C, dx, dt, D)
        print(f"Time = {t*dt:.4f}\n", C)


if __name__ == "__main__":
    print("Diffusion model")
    diffusion_model()