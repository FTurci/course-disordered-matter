# run with python3 lattice.gas.py --visualize to switch on real time visualization of configurations

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse  # Added for command-line argument parsing
from numba import njit  # For just-in-time compilation to speed up performance-critical functions
import os

# Define nearest-neighbor directions (up, down, left, right)
NEIGHBORS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

# Compute the energy change resulting from swapping two particles
@njit
def delta_energy(lattice, L, i1, j1, i2, j2):
    n1 = lattice[i1, j1]
    n2 = lattice[i2, j2]
    s1 = 0
    s2 = 0
    # Compute the sum of neighbors for each site
    for d in NEIGHBORS:
        ii1 = (i1 + d[0]) % L
        jj1 = (j1 + d[1]) % L
        s1 += lattice[ii1, jj1]
        ii2 = (i2 + d[0]) % L
        jj2 = (j2 + d[1]) % L
        s2 += lattice[ii2, jj2]
    # Return the energy difference caused by swapping
    return (n1 - n2) * (s1 - s2)

# Perform a Metropolis sweep using pre-generated indices of swap sites
@njit
def sweep_with_indices(lattice, L, T, indices):
    accepted = 0
    attempted = 0
    for i in range(indices.shape[0]):
        i1, j1, i2, j2 = indices[i]
        dE = delta_energy(lattice, L, i1, j1, i2, j2)
        attempted += 1
        # Accept the swap if it lowers energy or with Boltzmann probability
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            tmp = lattice[i1, j1]
            lattice[i1, j1] = lattice[i2, j2]
            lattice[i2, j2] = tmp
            accepted += 1
    return accepted, attempted

# Generate valid particle swap pairs avoiding self-swaps and nearest neighbors
def generate_valid_indices(L, n):
    indices = []
    while len(indices) < n:
        i1, j1 = np.random.randint(0, L, 2)
        i2, j2 = np.random.randint(0, L, 2)
        if i1 == i2 and j1 == j2:
            continue
        di = min(abs(i1 - i2), L - abs(i1 - i2))
        dj = min(abs(j1 - j2), L - abs(j1 - j2))
        # Avoid adjacent sites (ensures detailed balance)
        if di + dj == 1:
            continue
        indices.append((i1, j1, i2, j2))
    return np.array(indices, dtype=np.int32)

# Compute the structure factor (square of Fourier transform of density fluctuations)
def compute_structure_factor(config):
    rho = config.astype(np.float32) - np.mean(config)
    F = np.fft.fftshift(np.fft.fft2(rho))
    return np.abs(F) ** 2

# Compute the radial distribution function g(r)
def compute_rdf(config, n_bins=50):
    L = config.shape[0]
    particles = np.argwhere(config)
    # Compute all pairwise distances
    diffs = particles[:, None, :] - particles[None, :, :]
    diffs = (diffs + L/2) % L - L/2  # Apply periodic boundary conditions
    dists = np.sqrt((diffs**2).sum(axis=2)).ravel()
    edges = np.linspace(0, L/np.sqrt(2), n_bins + 1)
    hist, _ = np.histogram(dists, bins=edges)
    r = 0.5 * (edges[:-1] + edges[1:])
    rho = config.mean()
    dr = edges[1] - edges[0]
    norm = 2 * np.pi * r * dr * rho * len(particles)
    g = hist / norm
    return r, g

# Compute radial average of a 2D array (e.g., structure factor)
def radial_average(S, n_bins=50):
    L = S.shape[0]
    y, x = np.indices((L, L))
    r = np.sqrt((x - L//2)**2 + (y - L//2)**2)
    r_flat = r.ravel()
    S_flat = S.ravel()
    bins = np.linspace(0, r.max(), n_bins + 1)
    inds = np.digitize(r_flat, bins)
    radial_S = np.zeros(n_bins)
    for i in range(1, n_bins + 1):
        mask = inds == i
        if np.any(mask):
            radial_S[i-1] = S_flat[mask].mean()
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    return r_centers, radial_S

# Compute total energy of the lattice
@njit
def compute_total_energy(lattice, L):
    energy = 0.0
    for i in range(L):
        for j in range(L):
            n = lattice[i, j]
            for d in NEIGHBORS:
                ni = (i + d[0]) % L
                nj = (j + d[1]) % L
                energy -= 0.5 * n * lattice[ni, nj]  # Avoid double counting
    return energy

# ------------------ Main Simulation ------------------
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Lattice Gas Simulation")
    parser.add_argument('--visualize', action='store_true', help='Enable live visualization')
    args = parser.parse_args()

    live_viz = args.visualize
    print(f"Live visualization: {'ON' if live_viz else 'OFF'}")

    plt.ion()  # Interactive mode on for live visualization

    # Simulation parameters (can tweak here)
    L = 50        # Lattice size
    density = 0.5 # Initial particle density
    n_sweeps_eq = 1000         # Number of sweeps to equilibrate
    n_sweeps_per_meas = 10    # Sweeps between measurements
    n_meas = 200              # Number of measurements

    # Temperature range (edit as desired)
    T_values = np.linspace(0.3, 0.9, 12)  # e.g., 0.4, 0.6, ..., 1.6

    # Output directories
    root_dir = "/Users/phxnw/Dropbox/course-disordered-matter/quarto/phase-transitions/Coursework/"
    os.makedirs(os.path.join(root_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "data"), exist_ok=True)

    # Collectors for aggregated plots
    C_per_site_list = []
    E_per_site_list = []

    print(f"L = {L}, density = {density:.2f}")
    print(f"Temperatures: {', '.join(f'{t:.2f}' for t in T_values)}")

    # -------- Loop over Temperatures --------
    for T in T_values:
        print(f"\n=== Running simulation at T = {T:.2f} ===")

        # Fresh random initial state for each T
        np.random.seed(42)
        lattice = (np.random.rand(L, L) < density).astype(np.uint8)
        print("Initial energy =", compute_total_energy(lattice, L))

        total_accept = 0
        total_attempt = 0
        accum_energy = 0.0
        accum_energy_sq = 0.0
        accum_g = None
        accum_S = None

        # --- Equilibration ---
        for _ in tqdm(range(n_sweeps_eq), desc=f'Equilibrating T={T:.2f}'):
            indices = generate_valid_indices(L, L * L)
            a, b = sweep_with_indices(lattice, L, T, indices)
            total_accept += a
            total_attempt += b

        # --- Measurement ---
        for meas in tqdm(range(n_meas), desc=f'Sampling T={T:.2f}'):
            if live_viz:
                plt.clf()
                plt.imshow(lattice, cmap="gray", vmin=0, vmax=1)
                plt.title(f"Lattice (T={T:.2f}, meas step {meas})")
                plt.pause(0.001)

            for _ in range(n_sweeps_per_meas):
                indices = generate_valid_indices(L, L * L)
                a, b = sweep_with_indices(lattice, L, T, indices)
                total_accept += a
                total_attempt += b

            r_vals, g = compute_rdf(lattice)
            S = compute_structure_factor(lattice)
            if accum_g is None:
                accum_g = np.zeros_like(g)
                accum_S = np.zeros_like(S)
            accum_g += g
            accum_S += S

            energy = compute_total_energy(lattice, L)
            accum_energy += energy
            accum_energy_sq += energy ** 2

        # --- Compute averages ---
        g_avg = accum_g / n_meas
        S_avg = accum_S / n_meas
        r_S, S_rad = radial_average(S_avg)
        E_avg = accum_energy / n_meas
        E2_avg = accum_energy_sq / n_meas

        heat_capacity = (E2_avg - E_avg**2) / (T**2)

        # Per-site quantities
        C_per_site = heat_capacity / (L * L)
        E_per_site = E_avg / (L * L)

        # Collect for aggregated plots
        C_per_site_list.append(C_per_site)
        E_per_site_list.append(E_per_site)

        print(f"E_avg = {E_avg}")
        print(f"E2_avg = {E2_avg}")
        print(f"Heat capacity per site: {C_per_site:.4f}")
        print(f"Acceptance rate: {total_accept / total_attempt:.4f}")

        # --- Output & plots for this T ---
        prefix = f"L{L}_rho{density:.2f}_T{T:.2f}"

        plt.figure()
        plt.plot(r_vals, g_avg)
        plt.xlim(0, L/2)
        plt.xlabel("r")
        plt.ylabel("g(r)")
        plt.title(f"Average radial distribution function: T={T:.2f}, L={L:.0f}")
        plt.savefig(root_dir + f"figures/{prefix}_radial_distribution.pdf")

        plt.figure()
        plt.imshow(S_avg, origin="lower")
        plt.title(f"2D Structure Factor S(k): T={T:.2f}, L={L:.0f}")
        plt.colorbar()
        plt.savefig(root_dir + f"figures/{prefix}_structure_factor_2d.pdf")

        plt.figure()
        plt.plot(r_S, S_rad)
        plt.xlabel("k")
        plt.ylabel("S(k)")
        plt.title(f"Radial Average of Structure Factor: T={T:.2f}, L={L:.0f}")
        plt.savefig(root_dir + f"figures/{prefix}_structure_factor_radial.pdf")
        output_path = root_dir + f"data/{prefix}_structure_factor_radial.csv"
        np.savetxt(output_path, np.column_stack((r_S, S_rad)),
                   delimiter=",", header="k,S(k)", comments='')

        plt.figure()
        plt.imshow(lattice, cmap="gray")
        plt.axis("off")
        plt.title(f"Final Lattice Configuration: T={T:.2f}, L={L:.0f}")
        plt.savefig(root_dir + f"figures/{prefix}_final_lattice.pdf")

        heat_output_path = root_dir + f"data/{prefix}_heat_capacity.txt"
        with open(heat_output_path, "w") as f:
            f.write(f"# Heat capacity per site\n")
            f.write(f"C_per_site = {C_per_site:.6f}\n")
            f.write(f"# Average Energy = {E_avg:.6f}\n")
            f.write(f"# Average Energy^2 = {E2_avg:.6f}\n")

    # -------- Aggregate outputs across temperatures --------
    T_arr = np.array(T_values)
    C_arr = np.array(C_per_site_list)
    E_arr = np.array(E_per_site_list)

    # Save data
    np.savetxt(root_dir + "data/heat_capacity_vs_T.csv",
               np.column_stack((T_arr, C_arr)),
               delimiter=",", header="T,C_per_site", comments='')
    np.savetxt(root_dir + "data/energy_vs_T.csv",
               np.column_stack((T_arr, E_arr)),
               delimiter=",", header="T,E_per_site", comments='')

    # Plot heat capacity vs T
    plt.figure()
    plt.plot(T_arr, C_arr, marker='o')
    plt.xlabel("T")
    plt.ylabel("Heat capacity per site")
    plt.title(f"Heat Capacity vs Temperature (L={L}, rho={density:.2f})")
    plt.grid(True, alpha=0.3)
    plt.savefig(root_dir + "figures/heat_capacity_vs_T.pdf")

    # Plot energy per site vs T
    plt.figure()
    plt.plot(T_arr, E_arr, marker='o')
    plt.xlabel("T")
    plt.ylabel("Energy per site")
    plt.title(f"Energy vs Temperature (L={L}, rho={density:.2f})")
    plt.grid(True, alpha=0.3)
    plt.savefig(root_dir + "figures/energy_vs_T.pdf")

    plt.show()
