
import matplotlib.pyplot as plt
import numpy as np

# Parameters
mean = 0.475
std_dev = 0.02
x_end = 1.1 * 0.8 * 0.75
tick_x_pos = x_end - 0.02 - 0.03 - 0.015
tick_height = 0.045

# Generate data
x = np.linspace(-0.5, 1.1, 500)
y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
x_limited = x[x <= x_end]
y_limited = y[:len(x_limited)]

# Plot
plt.figure(figsize=(6, 4))
plt.plot(x_limited, y_limited, color='gray', alpha=0.7)

# Hide spines and ticks
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# Arrows
arrowprops = dict(arrowstyle='->,head_width=0.32,head_length=0.64', linewidth=1.5, color='black')
plt.annotate('', xy=(x_end, 0), xytext=(-0.5, 0), arrowprops=arrowprops)
plt.annotate('', xy=(0, max(y)*1.2), xytext=(0, 0), arrowprops=arrowprops)

# Labels
plt.text(0.2, -max(y)*0.07, 'Block variable', ha='center', va='top', fontsize=12)
plt.text(-0.05, max(y)*1.15, 'Probability', ha='right', va='center', rotation='vertical', fontsize=12)

# Tick marks and labels
plt.plot([tick_x_pos, tick_x_pos], [-tick_height, tick_height], color='black', linewidth=1)
plt.text(tick_x_pos, -max(y)*0.05, '1', ha='center', va='top')

q_tick_x = 0.475
plt.plot([q_tick_x, q_tick_x], [-tick_height, tick_height], color='black', linewidth=1)
plt.text(q_tick_x, -max(y)*0.05, 'Q', ha='center', va='top')

# Zero label
plt.text(0, -max(y)*0.05, '0', ha='center')

# Limits
plt.xlim(-0.5, x_end)
plt.ylim(0, max(y)*1.3)

plt.tight_layout()
plt.savefig("../Figs/schem_op_ordered.png", dpi=300)
plt.show()
