import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

a2_values = np.linspace(2, -4.25, 200)

# Magnetization range
m = np.linspace(-3, 3, 500)

# Define the Landau free energy function
def landau_free_energy(m, a2, a4=1.0):
    return 0.5 * a2 * m**2 + 0.25 * a4 * m**4

# Setup figure
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], lw=2)
left_minima_line, = ax.plot([], [], lw=2, color='blue', label='Left Minimum')
right_minima_line, = ax.plot([], [], lw=2, color='red', label='Right Minimum')

ax.set_xlim(-3, 3)
ax.set_ylim(-5, 5)
ax.legend()

# Text for displaying 'a' value
a2_text = ax.text(0.45, 0.95, '', transform=ax.transAxes, va='top', fontsize=16)

# Store full history of minima
history_left_x = []
history_left_y = []
history_right_x = []
history_right_y = []

# Precompute all minima positions
for a2 in a2_values:
    F = landau_free_energy(m, a2)
    minima = []
    for j in range(1, len(m)-1):
        if F[j-1] > F[j] and F[j] < F[j+1]:
            minima.append((m[j], F[j]))

    if len(minima) >= 2:
        (x1, y1), (x2, y2) = minima[0], minima[-1]
        history_left_x.append(x1)
        history_left_y.append(y1)
        history_right_x.append(x2)
        history_right_y.append(y2)
    elif len(minima) == 1:
        (xm, ym) = minima[0]
        if xm < 0:
            history_left_x.append(xm)
            history_left_y.append(ym)
            history_right_x.append(np.nan)
            history_right_y.append(np.nan)
        else:
            history_right_x.append(xm)
            history_right_y.append(ym)
            history_left_x.append(np.nan)
            history_left_y.append(np.nan)
    else:
        history_left_x.append(np.nan)
        history_left_y.append(np.nan)
        history_right_x.append(np.nan)
        history_right_y.append(np.nan)

def init():
    line.set_data([], [])
    left_minima_line.set_data([], [])
    right_minima_line.set_data([], [])
    a2_text.set_text('')
    return line, left_minima_line, right_minima_line, a2_text

ax.set_xlabel('Magnetization (m)',fontsize=20)
ax.set_ylabel('Free Energy $F(m)$',fontsize=20)


ax.set_xticks([0])
ax.set_yticks([0])
ax.set_xticklabels(['0'], fontsize=14)  # Set font size for x=0
ax.set_yticklabels(['0'], fontsize=14)  # Set font size for y=0
ax.axhline(0, color='black', linestyle='--', linewidth=1)

def animate(i):
    a2 = a2_values[i]
    F = landau_free_energy(m, a2)
    line.set_data(m, F)

    left_minima_line.set_data(history_left_x[:i+1], history_left_y[:i+1])
    right_minima_line.set_data(history_right_x[:i+1], history_right_y[:i+1])

    a2_text.set_text(f'$a_2$ = {a2:.2f}')

    return line, left_minima_line, right_minima_line, a2_text

ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=len(a2_values), interval=50, blit=True)


# Save animation as movie
ani.save('../Movies/landau_free_energy_evolution.mp4', writer='ffmpeg', fps=10)

# Show animation
plt.show()
