
import numpy as np
import matplotlib.pyplot as plt

# Critical values
T_c = 1.0
rho_c = 0.5

# Temperature array
T = np.linspace(0.5, T_c, 300)

# Common critical exponent for smoothness
beta = 0.325

# Asymmetric amplitudes: B > A
A = 0.25  # gas side amplitude
B = 0.45  # liquid side amplitude (larger)

# Coexistence curves
delta = (T_c - T)**beta
rho_gas = rho_c - A * delta
rho_liquid = rho_c + B * delta

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the two branches
ax.plot(rho_liquid, T, color='black')
ax.plot(rho_gas, T, color='black')

# Mark the critical point
ax.plot(rho_c, T_c, 'o', color='black')

# Labels
ax.text(0.25, 0.75, r'$\rho_{gas}$', fontsize=20, ha='center', va='center')
ax.text(0.9, 0.85, r'$\rho_{liquid}$', fontsize=20, ha='center', va='center')

# Set labels and limits
ax.set_xlabel(r'Density $\rho$', fontsize=20)
ax.set_ylabel(r'Temperature $T$', fontsize=20)
ax.set_xlim(0.2, 1)
ax.set_ylim(0.5, 1.2)

# Customize ticks
ax.set_xticks([])  # remove x-axis tick labels
ax.set_yticks([T_c])
ax.set_yticklabels([r'$T_c$'], fontsize=20)
ax.tick_params(axis='x', direction='out', length=6, width=1)
ax.tick_params(axis='y', direction='out', length=6, width=1, labelsize=16)

# Draw horizontal line at Tc
ax.axhline(y=T_c, color='black', linestyle='--', linewidth=1)

# Hide top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Hide vertical gridlines
ax.grid(False)

# Save figure
plt.savefig('../Figs/readl_fluid_pd_schem.png', dpi=300, bbox_inches='tight')

plt.show()
