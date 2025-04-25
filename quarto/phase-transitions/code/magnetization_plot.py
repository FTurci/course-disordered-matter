
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# Define critical temperature
T_c = 1.0
H_half = 0.005  # Small positive field H/2

# Create temperature range
T = np.linspace(0.6, 1.3, 500)

# Define mean field magnetization without field
m = np.zeros_like(T)
for i, temp in enumerate(T):
    if temp < T_c:
        m[i] = (T_c - temp)**0.5
    else:
        m[i] = 0

# Define reflected magnetization
m_reflected = -m

# Mean-field equation in presence of small field
def mean_field_equation(m_val, temp, T_c, H):
    return m_val**3 - (T_c - temp) * m_val - H

# Compute magnetization with small positive field
m_smooth_half = np.zeros_like(T)
for i, temp in enumerate(T):
    if temp < T_c:
        guess = (T_c - temp)**0.5
    else:
        guess = H_half / (temp - T_c + 0.05)
    sol = root_scalar(mean_field_equation, args=(temp, T_c, H_half), bracket=[0, 2], method='bisect')
    m_smooth_half[i] = sol.root if sol.converged else 0

# Plotting
plt.figure(figsize=(8, 6))

# Plot H=0 curve and its reflection
line1, = plt.plot(m, T)
plt.plot(-m, T, linestyle=line1.get_linestyle(), color=line1.get_color())

# Plot small positive H/2 curve
plt.plot(m_smooth_half, T, linestyle='--')

# Label for H=0 on negative m side
idx_label = 100
plt.text(-m[idx_label]-0.026, T[idx_label], 'H=0', fontsize=10, ha='right', va='center')

# Label for small positive H (moved +0.003 along x axis)
idx_Tc = np.abs(T - T_c).argmin()
plt.text(m_smooth_half[idx_Tc]+0.037, T[idx_Tc]+0.08, 'Small positive H', fontsize=10, ha='center', va='bottom')

# Get x-axis limits dynamically
x_min, x_max = plt.xlim()

# Extend the dashed line at T=1.0 by -0.05 to the left
plt.hlines(y=T_c, xmin=x_min, xmax=x_max, color='gray', linestyle=':')

# Change label for T_c to T=T_c
plt.text(x_min + 0.2, T_c+0.02, r'$T=T_c$', fontsize=12, verticalalignment='center', horizontalalignment='right')

# Remove all automatic ticks and tick marks except m=0
plt.yticks([])
plt.tick_params(axis='y', which='both', left=False, right=False)
plt.xticks([0])
plt.tick_params(axis='x', which='both', top=False, bottom=True)

# Label axes
plt.xlabel('Magnetization m')
plt.ylabel('Temperature')

# No automatic grid
plt.grid(False)

# Save the plot as a PNG file
plt.savefig('isingmag_new.png', dpi=300)

plt.show()
