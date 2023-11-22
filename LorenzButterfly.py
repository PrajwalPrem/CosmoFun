import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Lorenz system equations
def lorenz(t, xyz, sigma, rho, beta):
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Function to simulate the Lorenz system
def simulate_lorenz(sigma, rho, beta, t_span, initial_conditions):
    sol = solve_ivp(
        lorenz,
        t_span,
        initial_conditions,
        args=(sigma, rho, beta),
        dense_output=True,
    )
    return sol

# Visualizing Lorenz butterfly with multiple trajectories in x-z plane
def visualize_multiple_trajectories(sigma, rho, beta, t_span, num_trajectories=100):
    plt.figure(figsize=(15, 8))

    # Plot multiple trajectories
    for _ in range(num_trajectories):
        initial_conditions = np.random.uniform(low=[-10, -10, 0], high=[10, 10, 40])
        sol = simulate_lorenz(sigma, rho, beta, t_span, initial_conditions)
        plt.plot(sol.y[0], sol.y[2], color='blue', alpha=0.1)

    plt.title('Lorenz Butterfly: Multiple Trajectories in x-z Plane')
    plt.xlabel('X-axis')
    plt.ylabel('Z-axis')
    plt.show()

# Visualizing a single trajectory in the Lorenz butterfly in x-z plane
def visualize_single_trajectory(sigma, rho, beta, t_span, initial_conditions):
    plt.figure(figsize=(15, 8))

    # Plot a single trajectory with extended time span
    sol = simulate_lorenz(sigma, rho, beta, t_span, initial_conditions)
    plt.plot(sol.y[0], sol.y[2], color='red', linewidth=0.2, label='Single Trajectory')

    plt.title('Lorenz Butterfly: Single Trajectory in x-z Plane')
    plt.xlabel('X-axis')
    plt.ylabel('Z-axis')
    plt.legend()
    plt.show()

# Time span for multiple trajectories
t_span_multiple = [0, 5]

# Time span for a single trajectory
t_span_single = [0, 800]

# Number of trajectories for the multiple trajectories plot
num_trajectories = 100

# Visualizing multiple trajectories in the Lorenz butterfly in x-z plane
visualize_multiple_trajectories(sigma=10, rho=28, beta=8/3, t_span=t_span_multiple, num_trajectories=num_trajectories)

# Visualizing a single trajectory in the Lorenz butterfly in x-z plane
# Using the same initial conditions for both plots for illustration purposes
initial_conditions = np.random.uniform(low=[0, -5, 0], high=[10, 10, 40])
visualize_single_trajectory(sigma=10, rho=28, beta=8/3, t_span=t_span_single, initial_conditions=initial_conditions)


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Lorenz system equations
def lorenz(t, xyz, sigma, rho, beta):
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Function to simulate the Lorenz system
def simulate_lorenz(sigma, rho, beta, t_span, initial_conditions):
    sol = solve_ivp(
        lorenz,
        t_span,
        initial_conditions,
        args=(sigma, rho, beta),
        dense_output=True,
    )
    return sol

# Set parameters
sigma = 10
rho = 28
beta = 8/3

# Set time span
t_span = [0, 50]

# Set initial conditions
initial_conditions = [1.0, 0.0, 0.0]

# Simulate the Lorenz system
sol = simulate_lorenz(sigma, rho, beta, t_span, initial_conditions)

# Plot x, y, z oscillations with respect to time
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='x')
#plt.plot(sol.t, sol.y[1], label='y')
plt.plot(sol.t, sol.y[2], label='z')
plt.title('Lorenz System: Oscillations of x, z with Time')
plt.xlabel('Time')
plt.ylabel('Oscillations')
plt.legend()
plt.show()