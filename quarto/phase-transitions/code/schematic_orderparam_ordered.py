
import matplotlib.pyplot as plt
import numpy as np

# Parameters
mean = 0.475 * 0.8 * 0.8
std_dev = 0.02 / 2  # halved width
x_end = 1.1 * 0.8 * 0.75 * 0.8
tick_x_pos = (1.1 * 0.8 * 0.75 - 0.02 - 0.03 - 0.015) * 0.8 * 0.8
q_tick_x = 0.475 * 0.8 * 0.8
tick_height = 0.045

# New x start after 30% additional reduction
new_x_start = -0.5 + 0.3 * 0.5  # first reduction
new_x_start = new_x_start + 0.3 * (0 - new_x_start)  # additional 30% reduction

# Generate data
x = np.linspace(new_x_start, 1.1, 500)
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
plt.annotate('', xy=(x_end, 0), xytext=(new_x_start, 0), arrowprops=arrowprops)
plt.annotate('', xy=(0, max(y)*1.2), xytext=(0, 0), arrowprops=arrowprops)

# Labels
plt.text(0.2 * 0.8 * 0.8, -max(y)*0.07, 'Block variable', ha='center', va='top', fontsize=12)
plt.text(-0.025, max(y)*1.15, 'Probability', ha='right', va='center', rotation='vertical', fontsize=12)

# Tick marks
plt.plot([tick_x_pos, tick_x_pos], [-tick_height, tick_height], color='black', linewidth=1)
plt.text(tick_x_pos, -max(y)*0.05, '1', ha='center', va='top')

plt.plot([q_tick_x, q_tick_x], [-tick_height, tick_height], color='black', linewidth=1)
plt.text(q_tick_x, -max(y)*0.05, 'Q', ha='center', va='top')

# Zero label
plt.text(0, -max(y)*0.05, '0', ha='center')

# Limits
plt.xlim(new_x_start, x_end)
plt.ylim(0, max(y)*1.3)

plt.tight_layout()
plt.savefig("final_scaled_gaussian.png", dpi=300)
plt.show()
