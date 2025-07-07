
import matplotlib.pyplot as plt
import numpy as np

# Define the reduced van der Waals equation of state
def p_vdw(v, t):
    return (8 * t / 3) / (v - 1/3) - 3 / v**2

# Volume and temperature ranges
v = np.linspace(0.4, 3.0, 800)
t_values = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1]

# Create the plot
plt.figure(figsize=(10, 6))
for t in t_values:
    p = p_vdw(v, t)
    plt.plot(v, p, label=f't = {t:.2f}')

# Add lines and formatting
plt.axhline(0, color='black', lw=0.5)
plt.axvline(1, color='grey', linestyle='--', lw=0.5)
plt.xlim(0, 3)
plt.ylim(0, 1.5)
plt.xlabel(r'$v = V/Vc$',fontsize=20)
plt.ylabel(r'$p = P/Pc$',fontsize=20)
#plt.title('Reduced Van der Waals Isotherms (v = 0 to 3, t = 0.85 to 1.1)')
plt.legend()
plt.tick_params(axis='both', labelsize=16)
plt.grid(True)
plt.savefig("../Figs/vdw_isotherms.png", format='png', bbox_inches='tight')
plt.show()