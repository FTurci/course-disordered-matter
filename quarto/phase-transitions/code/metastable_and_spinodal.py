# Metastable vs. Spinodal sketches for a Landau free energy
# F(m) = (a/2) m^2 + (b/4) m^4 - h m, with a<0, b>0.
# Styling per request:
# - y-axis maximum = 1.0
# - no tick marks
# - no tick labels

import numpy as np
import matplotlib.pyplot as plt

# --- Model parameters ---
a = -1.0
b = 1.0

# Fields: one in the metastable region, one at the spinodal
m_s = np.sqrt(-a / (3 * b))           # |m| at the spinodal
h_s_neg = (2 * a / 3.0) * (m_s)       # negative for a<0; take |.| for plotting h>0 case
h_meta = 0.20
h_spin = abs(h_s_neg)

def F(m, a, b, h):
    return 0.5 * a * m**2 + 0.25 * b * m**4 - h * m

m = np.linspace(-2.0, 2.0, 1000)

def style_axes():
    # Set y max; remove ticks and tick labels on both axes
    plt.xlim(-2, 2)
    plt.ylim(None, 1.0)
    plt.tick_params(axis='both', which='both', length=0, width=0,
                    labelbottom=False, labelleft=False, labeltop=False, labelright=False)
    plt.xticks([])
    plt.yticks([])

# --- Plot 1: Metastable region ---
plt.figure()
plt.plot(m, F(m, a, b, h_meta), label=f"Metastable (h = {h_meta:.2f})")
plt.axhline(0, linewidth=0.5)
plt.axvline(0, linewidth=0.5)
plt.xlabel("Order parameter m")
plt.ylabel("Free energy F(m)")
plt.title("Metastable region")
plt.legend()
style_axes()
# plt.savefig("metastable.png", dpi=200, bbox_inches="tight")
plt.show()

# --- Plot 2: Spinodal ---
plt.figure()
plt.plot(m, F(m, a, b, h_spin), label=f"Spinodal (h ≈ {h_spin:.3f})")
plt.axhline(0, linewidth=0.5)
plt.axvline(0, linewidth=0.5)
plt.xlabel("Order parameter m")
plt.ylabel("Free energy F(m)")
plt.title("Spinodal")
plt.legend()
style_axes()
# plt.savefig("spinodal.png", dpi=200, bbox_inches="tight")
plt.show()
