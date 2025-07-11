import numpy as np
import matplotlib.pyplot as plt

# Critical temperature for 2D Ising model (J/k_B units)
Tc = 2 / np.log(1 + np.sqrt(2))

# Temperature range for T < Tc
T = np.linspace(0.1, Tc, 300)
beta_mf = 0.5
M_mf = (1 - T / Tc)**beta_mf

# Reduced-amplitude spinodal for each T
f = 0.7
M_spin = f * np.sqrt(1 - T / Tc)

# Midpoint in the region for arrow targets
T_mid = Tc / 2
M_mid = (f * np.sqrt(1 - T_mid / Tc) + (1 - T_mid / Tc)**beta_mf) / 2

# Label positions
label_x_nuc, label_y_nuc = 0.95, 1.95
spin_label_y = 0.7

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Binodal (mean-field branches)
ax.plot(M_mf, T, color='orange')
ax.plot(-M_mf, T, color='orangered')

# Spinodal boundary
ax.plot(M_spin, T, '--', color='magenta')
ax.plot(-M_spin, T, '--', color='magenta')

# Critical temperature line
ax.axhline(Tc, linestyle=':', color='gold')

# Shade regions
ax.fill_betweenx(T, M_spin, M_mf, color='grey', alpha=0.2)
ax.fill_betweenx(T, -M_mf, -M_spin, color='grey', alpha=0.2)
ax.fill_betweenx(T, -M_spin, M_spin, color='lightblue', alpha=0.2)

# Labels
ax.text(0.0, spin_label_y, "Spinodal decomposition", fontsize=18, ha='center', va='center')
ax.text(label_x_nuc, label_y_nuc, "Nucleation and growth", fontsize=18, ha='center', va='center')
ax.text(-1.03, Tc + 0.05, r'$T_c$', fontsize=18, va='bottom')

# Arrows (unchanged)
ax.annotate('', xy=(M_mid, T_mid), xytext=(0.95, 1.85),
            arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(-M_mid, T_mid), xytext=(0.95, 1.85),
            arrowprops=dict(arrowstyle='->', lw=2))

# Axes labels and styling
ax.set_xlabel('Magnetisation', fontsize=18)
ax.set_ylabel('Temperature', fontsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.set_xticks([])
ax.set_yticks([])

# Limits
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(0, Tc + 0.2)

# Save to PNG
plt.savefig('../Figs/spinodal.png', dpi=300, bbox_inches='tight')

plt.show()
