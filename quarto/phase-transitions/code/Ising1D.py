import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# Number of spins
N = 20  # updated from 15 to 20

# Initialize spins (+1 or -1)
spins = np.random.choice([-1, 1], size=N)

# Initial temperature
T = 2.0

# Set up figure and axis for the spins
fig, ax = plt.subplots(figsize=(10, 2))
plt.subplots_adjust(bottom=0.25)  # make room for slider
ax.set_xlim(-0.5, N - 0.5)
ax.set_ylim(-1, 1)
ax.axis('off')

# Create text objects for each spin
texts = []
for i in range(N):
    arrow = '↑' if spins[i] == 1 else '↓'
    t = ax.text(i, 0, arrow, fontsize=24, ha='center', va='center')
    texts.append(t)

def update(frame):
    """Perform Metropolis updates over all spins, then refresh display."""
    global spins, T
    for _ in range(N):
        i = np.random.randint(N)
        left = spins[(i - 1) % N]
        right = spins[(i + 1) % N]
        deltaE = 2 * spins[i] * (left + right)
        # Metropolis criterion
        if deltaE < 0 or np.random.rand() < np.exp(-deltaE / T):
            spins[i] *= -1
    # Update arrows on screen
    for idx, t in enumerate(texts):
        t.set_text('↑' if spins[idx] == 1 else '↓')
    return texts

# Create the animation with caching disabled and blit turned off
ani = FuncAnimation(
    fig,
    update,
    interval=200,
    blit=False,
    cache_frame_data=False
)

# Add a temperature slider
ax_T = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgray')
slider_T = Slider(ax_T, 'Temperature T', 0.1, 5.0, valinit=T)

def on_T_change(val):
    """Callback to update T when the slider changes."""
    global T
    T = val

slider_T.on_changed(on_T_change)

# Show the plot (ani is kept in scope so it won't be deleted)
plt.show()


