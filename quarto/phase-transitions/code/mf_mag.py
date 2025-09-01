
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
T_c = 1.0  # Critical temperature
T = np.linspace(0, 1.2, 500)  # Temperature range
beta = 0.5  # Mean-field critical exponent

# Define magnetisation
M = np.where(T < T_c, (T_c - T)**beta, 0)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(T, M, linewidth=2)

# Draw shorter vertical line at T_c
plt.axvline(x=T_c, ymin=0, ymax=0.5, color='gray', linestyle='--')

# Mark T_c slightly lower
plt.text(T_c, -0.07, r'$T_c$', ha='center', va='top', fontsize=14)

# Label next to the curve
plt.text(0.55, 0.7, r'$|m| \propto (-t)^{1/2}$', fontsize=16)

# Axis labels with British spelling and larger font size
plt.xlabel('Temperature $T$', fontsize=20)
plt.ylabel('Magnetisation $|m|$', fontsize=20)

# Customising x-axis ticks
plt.xticks([T_c], [])
plt.tick_params(axis='x', which='both', bottom=False, top=False)

# Customising y-axis ticks: only show 0
plt.yticks([0], ['0'], fontsize=14)
plt.tick_params(axis='y', which='both', left=False, right=False)

# Remove horizontal gridlines
plt.grid(axis='y', which='both', linestyle='', linewidth=0)

# Save figure as PNG
plt.savefig('mean_field_magnetisation.png', dpi=300, bbox_inches='tight')

# Show plot
plt.show()
