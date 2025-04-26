import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Parameters
L = 1.0        # Length of the domain
T = 1.0        # Total time 
nx = 400       # Number of spatial points
nt = 2000      # Number of time steps
D = 0.1        # Diffusion coefficient

# Discretization
dx = L / (nx - 1)
dt = T / nt

# Stability condition auto-adjust
if D * dt / dx**2 > 0.5:
    print("Adjusting dt and nt to satisfy stability condition...")
    dt = 0.4 * dx**2 / D
    nt = int(T / dt)
    dt = T / nt  # Recalculate dt exactly

nt = int(nt / 100)  # Artificial slowdown for animation

print(f"Using dt = {dt:.4e}, nt = {nt}")

# Initialize x and u
x = np.linspace(0, L, nx)

# Initial condition: smooth narrow Gaussian
sigma = 0.01
u = np.exp(-(x - L/2)**2 / (2 * sigma**2))

# Normalize initial condition
u /= np.sum(u) * dx

# Setup figure
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

fig, ax = plt.subplots()
line, = ax.plot(x, u)

# Compute peak height for setting y-axis
peak_height = 1 / (np.sqrt(2 * np.pi) * sigma)
ax.set_ylim(0, peak_height * 1.05)

ax.set_xlabel(r'$x$', fontsize=20)
ax.set_ylabel(r'$p(x,t)$', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=12)

# Tiny time counter text
time_text = ax.text(0.85, 0.05, '', transform=ax.transAxes, fontsize=10, verticalalignment='bottom')


# Function to update the plot
def update(frame):
    global u
    unew = np.copy(u)
    unew[1:-1] = u[1:-1] + D * dt / dx**2 * (u[2:] - 2*u[1:-1] + u[:-2])
    u = unew

    # Normalize at every step
    u /= np.sum(u) * dx

    line.set_ydata(u)
    current_time = frame * dt
    time_text.set_text(r'$t=%.4f$' % current_time)
    return line, time_text

ani = animation.FuncAnimation(fig, update, frames=nt, interval=100, blit=True)


# Save the animation as a movie
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani.save("../Movies/diffusion_evolution.mp4", writer=writer)


plt.show()

