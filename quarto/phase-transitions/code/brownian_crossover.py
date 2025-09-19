import matplotlib.pyplot as plt
import numpy as np

# Time range
t = np.logspace(-3, 2, 500)

# Parameters
v_th = 1.0   # thermal velocity (arbitrary units)
tau = 1.0    # characteristic relaxation time
D = v_th**2 * tau / 2  # diffusion coefficient

# Mean squared displacement (MSD)
msd_ballistic = (v_th**2) * t**2
msd_diffusive = 2 * D * t
msd_crossover = 2 * D * (t - tau*(1 - np.exp(-t/tau)))

# Save the plot as PNG
output_path = "brownian_crossover.png"

plt.figure(figsize=(7,5))
plt.loglog(t, msd_ballistic, '--', label="Ballistic ($\\propto t^2$)")
plt.loglog(t, msd_diffusive, '--', label="Diffusive ($\\propto t$)")
plt.loglog(t, msd_crossover, '-', color="black", label="Crossover (Brownian)")

plt.xlabel("Time (t)")
plt.ylabel("Mean Squared Displacement")
plt.legend()
plt.grid(True, which="both", axis="y", ls=":")

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()
