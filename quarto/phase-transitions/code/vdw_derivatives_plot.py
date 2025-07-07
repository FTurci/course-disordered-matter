
import matplotlib.pyplot as plt
import numpy as np

# Define the first and second derivatives
dp_dv = lambda v: (-8 / 3) / (v - 1/3)**2 + 6 / v**3
d2p_dv2 = lambda v: (16 / 3) / (v - 1/3)**3 - 18 / v**4

# Volume ranges
v1 = np.linspace(0.7, 1.4, 500)
v2 = np.linspace(0.9, 1.4, 500)

# Create side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# First derivative plot with label (a)
ax1.plot(v1, dp_dv(v1))
ax1.set_xlim(0.7, 1.4)
ax1.set_ylim(-2.5, 0)
ax1.set_xlabel(r"$v$",fontsize=20)
ax1.set_ylabel(r"$\frac{\partial p}{\partial v}$",fontsize=20)
ax1.axhline(0, color='black', lw=0.8)
ax1.text(0.72, -0.2, "(a)", fontsize=16, fontweight='bold')
ax1.grid(True)

# Second derivative plot with label (b)
ax2.plot(v2, d2p_dv2(v2))
ax2.set_xlim(0.9, 1.4)
ax2.set_ylim(-0.5, 2)
ax2.spines['bottom'].set_position(('data', 0))
ax2.spines['left'].set_position(('data', 0.9))
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(axis='x', direction='in', pad=5)
ax2.tick_params(axis='y', direction='in', pad=5)
ax2.set_xlabel(r"$v$",fontsize=20)
ax2.set_ylabel(r"$\frac{\partial^2 p}{\partial v^2}$",fontsize=20)
ax2.text(0.92, 1.8, "(b)", fontsize=16, fontweight='bold')
ax2.grid(True)

ax1.tick_params(axis='both', labelsize=16)
ax2.tick_params(axis='both', labelsize=16)
plt.tight_layout()

# Save the figure
plt.savefig("../Figs/vdw_derivatives.png", dpi=300)

# Optionally show the plot (can be commented out in headless environments)
plt.show()
