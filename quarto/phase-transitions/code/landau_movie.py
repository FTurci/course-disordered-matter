import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the Landau free energy function
def landau_free_energy(m, a2, a4=1.0):
    return 0.5 * a2 * m**2 + 0.25 * a4 * m**4

# Magnetization range
m = np.linspace(-2.2, 2.2, 400)

# Setup figure
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], lw=2)
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2, 4)
ax.set_xlabel('Magnetization (m)')
ax.set_ylabel('Free Energy $F(m)$')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.grid(True)

def animate(a2):
    ax.clear()
    F = landau_free_energy(m, a2)
    ax.plot(m, F, lw=2, color='blue')
    ax.set_title(f'Landau Free Energy for $a_2$ = {a2:.2f}',fontsize=16)
    ax.set_xlabel('Magnetization (m)',fontsize=20)
    ax.set_ylabel('Free Energy $F(m)$',fontsize=20)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2, 4)
    ax.grid(True)

# Create animation
a2_values = np.linspace(2, -2, 100)
ani = animation.FuncAnimation(fig, animate, frames=a2_values, interval=100)

# Save animation as movie
ani.save('../Movies/landau_free_energy_evolution.mp4', writer='ffmpeg', fps=10)

# Show animation
plt.show()
