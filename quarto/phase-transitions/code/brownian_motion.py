
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit

# Parameters
n_fluid = 300
box_size = 20.0
n_steps = 10000
sigma_f = 1.0
sigma_c = 10.0
epsilon = 0.05
mass_f = 1.0
mass_c = 2.0
dt = 1e-5
dt_e = 1e-5

# Echo the parameter values
print("Simulation Parameters:")
print(f"n_fluid = {n_fluid}")
print(f"box_size = {box_size}")
print(f"n_steps = {n_steps}")
print(f"sigma_f = {sigma_f}")
print(f"sigma_c = {sigma_c}")
print(f"epsilon = {epsilon}")
print(f"mass_f = {mass_f}")
print(f"mass_c = {mass_c}")
print(f"dt = {dt}")
 
 # Derived quantities

sigma_f6 = sigma_f ** 6
sigma_f12 = sigma_f **12
sigma_cf = 0.5 * (sigma_f + sigma_c)
sigma_cf6 = sigma_cf ** 6
sigma_cf12 = sigma_cf ** 12

np.random.seed(42)

# Safe initialization to avoid overlaps with colloid and other fluid particles
def initialize_fluid_positions(n_fluid, box_size, sigma_f, sigma_c, colloid_pos, min_dist_factor=0.85):
    min_dist_ff = min_dist_factor * sigma_f
    min_dist_cf = min_dist_factor * 0.5 * (sigma_f + sigma_c)
    positions = []
    max_attempts = 20000

    for _ in range(n_fluid):
        for attempt in range(max_attempts):
            trial = np.random.rand(2) * box_size
            too_close = False

            # Check distance to colloid center
            if np.linalg.norm(trial - colloid_pos[0]) < min_dist_cf:
                too_close = True

            # Check distances to already placed fluid particles
            for existing in positions:
                if np.linalg.norm(trial - existing) < min_dist_ff:
                    too_close = True
                    break

            if not too_close:
                positions.append(trial)
                break
        else:
            raise RuntimeError("Failed to place a fluid particle without overlap after many attempts.")
    
    return np.array(positions)

colloid_pos = np.array([[box_size / 2, box_size / 2]])
fluid_pos = initialize_fluid_positions(n_fluid, box_size, sigma_f, sigma_c, colloid_pos)
fluid_vel = (np.random.rand(n_fluid, 2) - 0.5)
colloid_vel = np.zeros((1, 2))

@njit
def compute_forces_numba(fluid_pos, colloid_pos, sigma_cf6, sigma_cf12, sigma_f6, sigma_f12, epsilon, box_size, n_fluid):
    forces_f = np.zeros_like(fluid_pos)
    force_c = np.zeros_like(colloid_pos)

    for i in range(n_fluid):
        # Fluid-colloid interaction
        rij = fluid_pos[i] - colloid_pos[0]
        rij -= box_size * np.round(rij / box_size)
        dist2 = np.dot(rij, rij)
        if dist2 < (2.5 ** 2) * ((sigma_cf6 ** (1/6)) ** 2) and dist2 > 1e-10:
            r2 = dist2
            r6 = r2 ** 3
            r12 = r6 ** 2
            fmag = 48 * epsilon * ((sigma_cf12 / r12) - 0.5 * (sigma_cf6 / r6)) / r2
            fvec = fmag * rij
            forces_f[i] += fvec
            force_c[0] -= fvec

        for j in range(i + 1, n_fluid):
            rij = fluid_pos[i] - fluid_pos[j]
            rij -= box_size * np.round(rij / box_size)
            dist2 = np.dot(rij, rij)
            if dist2 < (2.5 ** 2) * ((sigma_f6 ** (1/6)) ** 2) and dist2 > 1e-10:
                r2 = dist2
                r6 = r2 ** 3
                r12 = r6 ** 2
                fmag = 48 * epsilon * ((sigma_f12 / r12) - 0.5 * (sigma_f6 / r6)) / r2
                fvec = fmag * rij
                forces_f[i] += fvec
                forces_f[j] -= fvec

    return forces_f, force_c

fluid_history = []
colloid_history = []
forces_f, force_c = compute_forces_numba(fluid_pos, colloid_pos, sigma_cf6, sigma_cf12, sigma_f6, sigma_f12, epsilon, box_size, n_fluid)

n_equilibration = 2000  # Number of steps to equilibrate before tracking

# Equilibration phase (no history recorded)

for step in range(n_equilibration):
    fluid_pos += fluid_vel * dt_e + 0.5 * forces_f / mass_f * dt**2
    colloid_pos += colloid_vel * dt_e + 0.5 * force_c / mass_c * dt**2
    fluid_pos %= box_size
    colloid_pos %= box_size
    new_forces_f, new_force_c = compute_forces_numba(fluid_pos, colloid_pos, sigma_cf6, sigma_cf12, sigma_f6, sigma_f12, epsilon, box_size, n_fluid)
    fluid_vel += 0.5 * (forces_f + new_forces_f) / mass_f * dt_e
    colloid_vel += 0.5 * (force_c + new_force_c) / mass_c * dt_e
    forces_f = new_forces_f
    force_c = new_force_c
    # if step < 10:  # Log only the first few steps
    #     force_mag = np.linalg.norm(force_c[0])
    #     print(f"Step {step:3d} | Colloid Pos: {colloid_pos[0]} | Vel: {colloid_vel[0]} | |F|: {force_mag:.4e}")

# Production phase (history recorded)

for step in range(n_steps):
    fluid_pos += fluid_vel * dt + 0.5 * forces_f / mass_f * dt**2
    colloid_pos += colloid_vel * dt + 0.5 * force_c / mass_c * dt**2
    fluid_pos %= box_size
    colloid_pos %= box_size
    new_forces_f, new_force_c = compute_forces_numba(fluid_pos, colloid_pos, sigma_cf6, sigma_cf12, sigma_f6, sigma_f12, epsilon, box_size, n_fluid)
    fluid_vel += 0.5 * (forces_f + new_forces_f) / mass_f * dt
    colloid_vel += 0.5 * (force_c + new_force_c) / mass_c * dt
    forces_f = new_forces_f
    force_c = new_force_c
    if step % 10 == 0:
        fluid_history.append(fluid_pos.copy())
        colloid_history.append(colloid_pos.copy())

fig, ax = plt.subplots()
# Calculate figure and plot scale parameters
fig_width_inch = fig.get_size_inches()[0]
dpi = fig.dpi
axis_length_pt = fig_width_inch * dpi
marker_scale = 0.1  # Scale factor for visibility
fluid_marker_size = (marker_scale * axis_length_pt / box_size) ** 2
colloid_marker_size = 0.7*(marker_scale * sigma_c / sigma_f * axis_length_pt / box_size) ** 2
fluid_scatter = ax.scatter([], [], s=fluid_marker_size, c='blue')
colloid_scatter = ax.scatter([], [], s=colloid_marker_size, c='red')
trajectory, = ax.plot([], [], 'r--', linewidth=1, alpha=0.5)
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_aspect('equal')
colloid_traj = []

def init():
    empty_offsets = np.empty((0, 2))
    fluid_scatter.set_offsets(empty_offsets)
    colloid_scatter.set_offsets(empty_offsets)
    trajectory.set_data([], [])
    return fluid_scatter, colloid_scatter, trajectory

def update(frame):
    fluid_scatter.set_offsets(fluid_history[frame])
    colloid_scatter.set_offsets(colloid_history[frame])
    colloid_traj.append(colloid_history[frame][0])
    traj_array = np.array(colloid_traj)
    trajectory.set_data(traj_array[:, 0], traj_array[:, 1])
    return fluid_scatter, colloid_scatter, trajectory

ani = animation.FuncAnimation(fig, update, frames=len(fluid_history), init_func=init, blit=True, interval=20)
ani.save("brownian_colloid.mp4", writer="ffmpeg", fps=30)
print("Simulation complete. Video saved as 'brownian_colloid.mp4'.")
