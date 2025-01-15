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


def plot_profile(concentration, grid, color="r"):
    plt.figure()
    plt.plot(grid, concentration, color)
    plt.xlabel("x")
    plt.ylabel("C")
    plt.title("Concentration profile")


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
    n_points = 81

    x, dx = np.linspace(0, Lx, num=n_points, retstep=True)
    dt = calculate_stable_time_step(dx, D)
    C = step_like(x, step_at=len(x) // 2)

    print("Time = 0\n", C)
    for t in range(0, 5):
        solve1d(C, dx, dt, D)
        print(f"Time = {t*dt:.4f}\n", C)


if __name__ == "__main__":
    print("Diffusion model")
    diffusion_model()
