"""
Part (a) Physical Example of a Lorenz System:
The equations relate the properties of a two-dimensional fluid layer uniformly
warmed from below and cooled from above. In particular, the equations describe 
the rate of change of three quantities with respect to time: x is proportional 
to the rate of convection, y to the horizontal temperature variation, and z to 
the vertical temperature variation.[3] The constants σ, ρ, and β are system 
parameters proportional to the Prandtl number, Rayleigh number, and certain 
physical dimensions of the layer itself.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import matplotlib.animation as animation

# Define the Lorenz system
def lorenz(t, state, sigma, r, b):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (r - z) - y
    dzdt = x * y - b * z
    return [dxdt, dydt, dzdt]

# Parameters for the Lorenz system
sigma = 10  # ω
r = 48      # ε
b = 3       # ϑ

# Time span and evaluation points
t_span = (0, 12)
t_eval = np.linspace(0, 12, 10000)

# Initial condition (you can choose different values to see varying trajectories)
initial_state = [1.0, 1.0, 1.0]

# Solve the Lorenz system
sol = solve_ivp(lorenz, t_span, initial_state, args=(sigma, r, b), t_eval=t_eval)

# Extract the solution components
x = sol.y[0]
y = sol.y[1]
z = sol.y[2]

# Plot the time series for x, y, and z
plt.figure(figsize=(10, 6))
plt.plot(sol.t, x, label='x(t)')
plt.plot(sol.t, y, label='y(t)')
plt.plot(sol.t, z, label='z(t)')
plt.xlabel('Time')
plt.ylabel('State Variables')
plt.title('Time Evolution of the Lorenz System')
plt.legend()
plt.show()
plt.savefig("lorenz_time_series.png")

# Additionally, plot the 3D Lorenz attractor
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.set_title('Lorenz Attractor')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
plt.savefig("lorenz_attractor.png")


# Set up the figure and 3D axis for the animation
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim((np.min(x), np.max(x)))
ax.set_ylim((np.min(y), np.max(y)))
ax.set_zlim((np.min(z), np.max(z)))
ax.set_title("Lorenz Attractor Trajectory")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# Initialize the line (trajectory) with alpha value and a dot for the current point
line, = ax.plot([], [], [], lw=2, alpha=0.8)
current_point, = ax.plot([], [], [], 'o', markersize=8, color='red')

# Initialization function for the animation
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    current_point.set_data([], [])
    current_point.set_3d_properties([])
    return line, current_point

# Update function to incrementally draw the trajectory and update the current point
def update(num):
    line.set_data(x[:num], y[:num])
    line.set_3d_properties(z[:num])
    # Update the current point to the latest position
    current_point.set_data([x[num]], [y[num]])
    current_point.set_3d_properties([z[num]])
    print("Frame", num)
    return line, current_point


# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(t_eval),
                              init_func=init, interval=1.2 , blit=True)

# Save the animation as an MP4 video file (ensure 'ffmpeg' is installed)
ani.save("lorenz_attractor.mp4", writer="ffmpeg")