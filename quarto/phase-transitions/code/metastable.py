import numpy as np
import matplotlib.pyplot as plt

# Parameters for free energy landscape
a = -1.0   # quadratic coefficient
b = 1.0    # quartic coefficient
H = -0.2   # external field (tilts the potential)

# Magnetization range
m = np.linspace(-2, 2, 400)

# Free energy density function
f = a*m**2 + b*m**4 - H*m

# Derivative for extrema (minima/maxima)
df = 2*a*m + 4*b*m**3 - H
extrema_indices = np.where(np.diff(np.sign(df)))[0]

# Plot free energy landscape
plt.figure(figsize=(6,4))
plt.plot(m, f, lw=2, color="blue")  # blue curve
for idx in extrema_indices:
    plt.scatter(m[idx], f[idx], color="red", s=60, zorder=5)

# Axis formatting
plt.ylim(-0.5, 1.0)
plt.xlabel("Order parameter $m$ (magnetization)")
plt.ylabel("Free energy density $f(m)$")
plt.gca().set_yticklabels([])  # remove y tick labels
plt.grid(True, alpha=0.3)

# Adjust x-axis tick labels to show half the actual value
ax = plt.gca()
ticks = ax.get_xticks()
ax.set_xticklabels([f"{tick/2:.1f}" for tick in ticks])

# Save figure
plt.savefig("free_energy_landscape_blue_halfxticks.png", dpi=300, bbox_inches="tight")
plt.show()
