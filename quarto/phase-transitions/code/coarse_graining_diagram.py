
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(figsize=(7, 8))
ax.set_xlim(-2, 14)
ax.set_ylim(0, 16)
ax.axis('off')

def draw_colored_box(ax, x, y, text, color):
    ax.add_patch(patches.Rectangle((x, y), 6, 2, edgecolor='black', facecolor=color, linewidth=1.5))
    ax.text(x + 3, y + 1, text, ha='center', va='center', fontsize=13, weight='bold', color='white')

draw_colored_box(ax, 2, 13, 'Microscopic', '#4C72B0')
draw_colored_box(ax, 2, 8, 'Coarse graining', '#55A868')
draw_colored_box(ax, 2, 3, 'Universal', '#C44E52')

def draw_further_shortened_arrow(ax, x, y_top, y_bottom, shift_fraction=0.2):
    full_length = y_top - y_bottom
    shorten = 2 * shift_fraction * full_length
    y1 = y_top - shorten / 2
    y2 = y_bottom + shorten / 2
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='-|>', lw=6, color='black'))
    return (y1 + y2) / 2

mid1 = draw_further_shortened_arrow(ax, 5, 12.8, 10.2)
mid2 = draw_further_shortened_arrow(ax, 5, 7.8, 5.2)

ax.text(5, mid1 - 0.3, 'Large ξ\nSystem specific details', ha='center', va='center', fontsize=12)
ax.text(5, mid2 - 0.3, 'Large L', ha='center', va='center', fontsize=12)

spin_color = '#348ABD'
particle_color = '#E24A33'

spin_box_x = -1.0
spin_box_y = 13.0
spin_box_height = 1.8
ax.add_patch(patches.Rectangle((spin_box_x, spin_box_y), 2, spin_box_height, fill=False, edgecolor='black', linewidth=1.2))

for i in range(3):
    for j in range(4):
        ax.arrow(spin_box_x + 0.3 + i*0.5, spin_box_y + 0.3 + j*0.4, 0, 0.3,
                 head_width=0.1, head_length=0.1, fc=spin_color, ec=spin_color)

particle_box_x = 9.2
particle_box_y = 13.2
box_size = 2.0
ax.add_patch(patches.Rectangle((particle_box_x, particle_box_y), box_size, box_size, fill=False, edgecolor='black', linewidth=1.2))

np.random.seed(123)
n_particles = 20
for _ in range(n_particles):
    x = particle_box_x + np.random.rand() * box_size
    y = particle_box_y + np.random.rand() * box_size
    ax.add_patch(patches.Circle((x, y), 0.1, color=particle_color))

plt.tight_layout()
plt.show()
