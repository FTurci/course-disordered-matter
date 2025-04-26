
import numpy as np
import matplotlib.pyplot as plt

# Critical values
T_c = 1.0  # normalized critical temperature
rho_c = 0.5  # critical density (symmetric Ising)

# Temperature array
T = np.linspace(0.0, T_c, 300)

# Coexistence curve (mean-field square-root behavior)
delta_rho = (T_c - T)**0.5
rho_upper = rho_c + delta_rho / 2
rho_lower = rho_c - delta_rho / 2

# Set up plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the two branches without shading
ax.plot(rho_upper, T, color='black')
ax.plot(rho_lower, T, color='black')

# Mark the critical point
ax.plot(rho_c, T_c, 'o', color='black')

# Label the gas and liquid branches
ax.text(0.05, 0.4, r'$\rho_{gas}$', fontsize=20, ha='center', va='center')
ax.text(0.96, 0.4, r'$\rho_{liquid}$', fontsize=20, ha='center', va='center')

# Set labels and limits
ax.set_xlabel(r'Density $\rho$', fontsize=20)
ax.set_ylabel(r'Temperature $T$', fontsize=20)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.2)

# Customize y-ticks: only at Tc
ax.set_yticks([T_c])
ax.set_yticklabels([r'$T_c$'], fontsize=20)

# Customize x-ticks
ax.tick_params(axis='x', direction='out', length=6, width=1, labelsize=16)
ax.tick_params(axis='y', direction='out', length=6, width=1, labelsize=16)

# Draw horizontal grid line at Tc
ax.axhline(y=T_c, color='black', linestyle='--', linewidth=1)

# Hide top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Hide vertical gridlines
ax.grid(False)

# Save figure as PNG
plt.savefig('../Figs/lattice_gas_phase_diagram.png', dpi=300, bbox_inches='tight')

plt.show()
