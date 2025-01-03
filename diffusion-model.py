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


# ## Run the model

# Start by setting two fixed model parameters--the diffusivity and the size of the model domain.
D = 100
Lx = 300

# Next, set up the model grid using a NumPy array.
dx = 0.5
x, nx = make_grid(Lx, dx)

# Set the initial conditions for the model.
# The concentration `C` is a step function with a high value on the left, a low value on the right, and the step at the center of the domain.
C_left = 500
C_right = 0
C = set_initial_profile(nx, boundary_left=C_left, boundary_right=C_right)

# Plot the initial concentration profile.
plot_profile(C, x)

# Set the start time of the model and the number of time steps. Then calculate a stable time step for the model using the Von Neumann stability criterion.
time = 0
nt = 5000
dt = calculate_time_step(dx, D)

# Loop over the time steps of the model,
# solving the diffusion equation using the FTCS scheme described above.
# The boundary conditions are clamped, so reset them after each time step.
for t in range(0, nt):
    solve1d(C, dx, dt, D)

# Plot the result.
plot_profile(C, x, color="b")

