
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Parameters
L = 1.0        # Length of the domain
T = 1.0      # Total time 
nx = 400      # Number of spatial points doubled
nt = 2000       # Number of time steps
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

nt=int(nt/20)

print(f"Using dt = {dt:.4e}, nt = {nt}")

# Initialize x and u
x = np.linspace(0, L, nx)

# Initial condition: smooth narrow Gaussian
sigma = 0.01
u = np.exp(-(x - L/2)**2 / (2 * sigma**2))

# Setup figure
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

fig, ax = plt.subplots()
line, = ax.plot(x, u)
ax.set_ylim(0, 1.0)
ax.set_xlabel(r'$x$',fontsize=20)
ax.set_ylabel(r'$p(x,t)$',fontsize=20)

# Tiny time counter text
time_text = ax.text(0.85, 0.05, '', transform=ax.transAxes, fontsize=10, verticalalignment='bottom')

# Diffusion update function
def diffuse(u):
    un = u.copy()
    u[1:-1] = un[1:-1] + D * dt / dx**2 * (un[2:] - 2 * un[1:-1] + un[:-2])
    return u

# Animation update function
def update(frame):
    global u
    u = diffuse(u)
    line.set_ydata(u)
    current_time = frame * dt
    time_text.set_text(r'$t=%.4f$' % current_time)
    return line, time_text

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=nt, blit=True, interval=100)

# Make sure output directory exists
output_dir = os.path.abspath(os.path.join('..', 'Movies'))
os.makedirs(output_dir, exist_ok=True)

# Save the animation
output_path = os.path.join(output_dir, 'diffusion_evolution.mp4')
ani.save(output_path, writer='ffmpeg')

print(f"Movie saved to {output_path}")

plt.show()
