# # A 1D diffusion model
# Here we develop a one-dimensional model of diffusion.
# It assumes a constant diffusivity.
# It uses a regular grid.
# It has fixed boundary conditions.
# The diffusion equation:
# 
# $$ \frac{\partial C}{\partial t} = D\frac{\partial^2 C}{\partial x^2} $$
# 
# The discretized version of the diffusion equation we'll solve with our model:
# 
# $$ C^{t+1}_x = C^t_x + {D \Delta t \over \Delta x^2} (C^t_{x+1} - 2C^t_x + C^t_{x-1}) $$
# 
# This is the explicit FTCS scheme as described in Slingerland and Kump (2011).

# We'll use two libraries, [NumPy](https://numpy.org/) (for arrays) and [Matplotlib](https://matplotlib.org/) (for plotting), that aren't a part of the core Python library.
import numpy as np
import matplotlib.pyplot as plt


# Write a function to calculate the time step of the model.
def calculate_time_step(grid_spacing, diffusivity):
    return 0.5 * grid_spacing**2 / diffusivity


# Make a function that creates the initial concentration profile.
def set_initial_profile(grid_size=100, boundary_left=500, boundary_right=0):
    profile = np.empty(grid_size)
    profile[: grid_size // 2] = boundary_left
    profile[grid_size // 2 :] = boundary_right
    return profile


# Make a function to plot the concentration profile.
def plot_profile(concentration, grid, color="r"):
    plt.figure()
    plt.plot(grid, concentration, color)
    plt.xlabel("x")
    plt.ylabel("C")
    plt.title("Concentration profile")


# Write a function to set up the model grid.
def make_grid(domain_size, grid_spacing):
    grid = np.arange(start=0, stop=domain_size, step=grid_spacing)
    return (grid, len(grid))


# Finally, write a function to solve the diffusion equation, advancing the model by one time step.
def solve1d(concentration, grid_spacing=1.0, time_step=1.0, diffusivity=1.0):
    centered_diff = np.roll(concentration, -1) - 2*concentration + np.roll(concentration, 1)
    concentration[1:-1] += diffusivity * time_step / grid_spacing**2 * centered_diff[1:-1]


# Run the model in an example.
def diffusion_example():
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
    print("Diffusion model example")
    diffusion_example()
