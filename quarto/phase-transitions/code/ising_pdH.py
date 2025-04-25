
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

fig, ax = plt.subplots(figsize=(8, 6))

Tc = 1.0

T_first_order = np.linspace(0, Tc, 100)
H_first_order = np.zeros_like(T_first_order)
ax.plot(T_first_order, H_first_order, 'k-')

ax.plot(Tc, 0, 'ro')
ax.text(Tc - 0.2, -0.1, 'Critical Point', color='r', fontsize=10, ha='left', va='bottom')

ax.set_xlabel('Temperature (T)')
ax.set_ylabel('Magnetic Field (H)')
ax.grid(False)

ax.text(1.2, 0.0, 'Paramagnetic', fontsize=12)
ax.text(0.3, 0.5, 'Positive\nMagnetisation', fontsize=12, ha='center')
ax.text(0.3, -0.65, 'Negative\nMagnetisation', fontsize=12, ha='center')

for spine in ax.spines.values():
    spine.set_visible(True)

ax.set_xlim(0, 2)
ax.set_ylim(-1, 1)
ax.set_yticks([])
ax.set_xticks([Tc])
ax.set_xticklabels([r'$T_c$'], fontsize=12)

ax.annotate('', xy=(0.5, 0.2), xytext=(0.5, -0.2),
            arrowprops=dict(arrowstyle='<->', linewidth=1.5, color='red'))
ax.text(0.55, 0.125, 'First-order\nPhase Transition', va='bottom', fontsize=10)

center_x, center_y = 0.3, 0.28
width_x, height_y = 0.12 * 1.5 * 0.8, 0.12 * 1.5
x_vals = np.linspace(center_x - width_x/2, center_x + width_x/2, 4)
y_vals = np.linspace(center_y - height_y/2, center_y + height_y/2, 4)

arrow_length = 0.045
arrow_width = 0.002

for x in x_vals:
    for y in y_vals:
        arrow = FancyArrow(x, y, 0, arrow_length, width=arrow_width,
                           head_width=0.015, head_length=0.015,
                           length_includes_head=True, color='k')
        ax.add_patch(arrow)

for x in x_vals:
    for y in y_vals:
        arrow = FancyArrow(x, -y, 0, -arrow_length, width=arrow_width,
                           head_width=0.015, head_length=0.015,
                           length_includes_head=True, color='k')
        ax.add_patch(arrow)

vertical_shift = 0.025
dilation_factor = 1.05 ** 2

box_half_width = (width_x / 2 + 0.01) * dilation_factor
box_left = center_x - box_half_width
box_right = center_x + box_half_width
box_bottom = center_y - height_y/2 - 0.01 - 0.025 + vertical_shift
box_top = center_y + height_y/2 + 0.01 + 0.025 + vertical_shift

ax.plot([box_left, box_right], [box_bottom, box_bottom], 'k-')
ax.plot([box_left, box_right], [box_top, box_top], 'k-')
ax.plot([box_left, box_left], [box_bottom, box_top], 'k-')
ax.plot([box_right, box_right], [box_bottom, box_top], 'k-')

ax.plot([box_left, box_right], [-box_bottom, -box_bottom], 'k-')
ax.plot([box_left, box_right], [-box_top, -box_top], 'k-')
ax.plot([box_left, box_left], [-box_bottom, -box_top], 'k-')
ax.plot([box_right, box_right], [-box_bottom, -box_top], 'k-')

# Save the figure
plt.savefig('ising_phase_diagram_output.png', dpi=300, bbox_inches='tight')
plt.show()
