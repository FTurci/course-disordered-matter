
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
T_c = 1.0  # Critical temperature
T1 = np.linspace(0.5, 0.98, 250)  # Below T_c
T2 = np.linspace(1.02, 1.5, 250)  # Above T_c
gamma = 1.0  # Mean-field critical exponent for susceptibility

# Define magnetic susceptibility
chi1 = 1.17 * (T_c - T1)**(-gamma)       # 10% reduction from 1.3
chi2 = 1.5 * (T2 - T_c)**(-gamma)        # 50% increase

# Remove points above chi=9.7
T1 = T1[chi1 <= 9.7]
chi1 = chi1[chi1 <= 9.7]
T2 = T2[chi2 <= 9.7]
chi2 = chi2[chi2 <= 9.7]

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(T1, chi1, linewidth=2)
plt.plot(T2, chi2, linewidth=2)

# Full vertical dashed line at T_c
plt.axvline(x=T_c, ymin=0, ymax=1, color='gray', linestyle='--')

# T_c label
plt.text(T_c + 0.04, 2.2, r'$T_c$', ha='center', va='bottom', fontsize=14)

# Curve labels
plt.text(0.625, 7, r'$\chi \propto (-t)^{-1}$', fontsize=16)
plt.text(1.25, 7, r'$\chi \propto t^{-1}$', fontsize=16)

# Axis labels
plt.xlabel('Temperature $T$', fontsize=20)
plt.ylabel('Susceptibility $\chi$', fontsize=20)

# Hide x-axis ticks
plt.xticks([T_c], [])
plt.tick_params(axis='x', which='both', bottom=False, top=False)

# Hide y-axis scale
plt.yticks([])
plt.tick_params(axis='y', which='both', left=False, right=False)

# No gridlines
plt.grid(axis='y', which='both', linestyle='', linewidth=0)

# Save figure
plt.savefig('mf_chi_final_adjusted.png', dpi=300, bbox_inches='tight')

# Show plot
plt.show()
