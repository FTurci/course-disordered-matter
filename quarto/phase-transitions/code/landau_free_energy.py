import numpy as np
import matplotlib.pyplot as plt

# Define the Landau free energy function
def landau_free_energy(m, a2, a4=1.0):
    return 0.5 * a2 * m**2 + 0.25 * a4 * m**4

# Magnetization range
m = np.linspace(-2.2, 2.2, 400)

# Different a2 values: positive, zero, negative
a2_values = [2, 1, 0, -1, -2]

plt.figure(figsize=(10, 6))

# Plot for each a2 value
for a2 in a2_values:
    F = landau_free_energy(m, a2)
    plt.plot(m, F, label=f'$a_2 = {a2}$')

# Labeling
#plt.title('Landau Free Energy for Different $a_2$ Values')
plt.xlabel('Magnetization (m)')
plt.ylabel('Free Energy $F(m)$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure to a file
plt.savefig('../Figs/landau_free_energy.png', dpi=300)
# Display plot
plt.show()
