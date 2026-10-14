"""Grand-canonical Monte Carlo simulation of the 2D square lattice gas.

The configurational energy is

    E = -epsilon * sum_<i,j> c_i c_j

and configurations are sampled with weight

    exp[-beta (E - mu N)].

A Monte Carlo move selects one lattice site and proposes changing its
occupancy, c_i -> 1-c_i.  The change in the grand-canonical Hamiltonian is

    delta_K = -epsilon * delta_c * n_neighbours - mu * delta_c,

where delta_c is +1 for insertion and -1 for removal.

The program records raw density and energy time series, the density
distribution P(rho), the connected structure factor S(k), and the connected
real-space correlation function C(r).  In particular, it retains the k=0
density fluctuation:

    S(0) = var(N) / V,
    d<rho>/dmu = beta S(0).

Example commands
----------------
Run at coexistence for several temperatures:

    python Full_lattice_gas_GCE.py

Run an equation-of-state scan at one temperature:

    python Full_lattice_gas_GCE.py \
        --temperatures 0.65 \
        --chemical-potentials -2.6 -2.4 -2.2 -2.0 -1.8 -1.6 -1.4

Run the noninteracting validation case:

    python Full_lattice_gas_GCE.py \
        --epsilon 0 --temperatures 1.0 \
        --chemical-potentials -2 -1 0 1 2

Add --visualize to display the lattice while sampling.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

try:
    from numba import njit
except ImportError:
    # The program remains functional without Numba, although large simulations
    # will be much slower.  Numba is available in the teaching environment.
    def njit(function):
        return function


# ---------------------------------------------------------------------------
# Model constants and default run parameters
# ---------------------------------------------------------------------------

NEIGHBORS = np.array(
    [[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int64
)

Q_SQUARE = 4
ISING_TC_OVER_J = 2.0 / np.log(1.0 + np.sqrt(2.0))

DEFAULT_L = 50
DEFAULT_EPSILON = 1.0
DEFAULT_TEMPERATURES = np.array([0.85, 0.68, 0.60, 0.5673, 0.51])
DEFAULT_CHEMICAL_POTENTIALS = np.array([-2.0])
DEFAULT_INITIAL_DENSITY = 0.5
DEFAULT_EQ_SWEEPS = 5000
DEFAULT_SWEEPS_PER_MEASUREMENT = 10
DEFAULT_MEASUREMENTS = 2000
DEFAULT_SEED = 42


def exact_critical_temperature(epsilon):
    """Exact critical temperature of the 2D square lattice gas (k_B=1)."""
    return ISING_TC_OVER_J * epsilon / 4.0


def coexistence_chemical_potential(epsilon):
    """Coexistence chemical potential from particle-hole symmetry."""
    return -0.5 * Q_SQUARE * epsilon


# ---------------------------------------------------------------------------
# Grand-canonical Monte Carlo
# ---------------------------------------------------------------------------

@njit
def seed_numba(seed):
    """Seed Numba's random-number generator."""
    np.random.seed(seed)


@njit
def local_grand_energy_change(lattice, L, i, j, epsilon, mu):
    """Return (delta_K, delta_E, delta_N) for c_ij -> 1-c_ij."""
    # Cast explicitly because lattice has an unsigned integer dtype.
    old_occupancy = np.int64(lattice[i, j])
    delta_N = 1 - 2 * old_occupancy

    occupied_neighbours = 0
    for d in NEIGHBORS:
        ii = (i + d[0]) % L
        jj = (j + d[1]) % L
        occupied_neighbours += lattice[ii, jj]

    delta_E = -epsilon * delta_N * occupied_neighbours
    delta_K = delta_E - mu * delta_N
    return delta_K, delta_E, delta_N


@njit
def grand_canonical_sweep(lattice, L, T, epsilon, mu):
    """Perform L^2 random single-site insertion/removal attempts."""
    accepted = 0
    energy_change = 0.0
    particle_change = 0

    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        delta_K, delta_E, delta_N = local_grand_energy_change(
            lattice, L, i, j, epsilon, mu
        )

        if delta_K <= 0.0 or np.random.random() < np.exp(-delta_K / T):
            lattice[i, j] = 1 - lattice[i, j]
            accepted += 1
            energy_change += delta_E
            particle_change += delta_N

    return accepted, energy_change, particle_change


@njit
def compute_total_energy(lattice, L, epsilon):
    """Compute E, counting every nearest-neighbour bond once."""
    energy = 0.0
    for i in range(L):
        for j in range(L):
            if lattice[i, j]:
                energy -= epsilon * lattice[(i + 1) % L, j]
                energy -= epsilon * lattice[i, (j + 1) % L]
    return energy


# ---------------------------------------------------------------------------
# Correlation functions and output helpers
# ---------------------------------------------------------------------------

def fourier_density(config):
    """Return the unnormalised Fourier transform of the occupancy field."""
    return np.fft.fft2(config.astype(np.float64))


def connected_structure_factor(sum_F, sum_abs_F_sq, n_samples, volume):
    """Return S(k)=(<|n_k|^2>-|<n_k>|^2)/V."""
    mean_F = sum_F / n_samples
    mean_abs_F_sq = sum_abs_F_sq / n_samples
    S = (mean_abs_F_sq - np.abs(mean_F) ** 2) / volume
    # Suppress only roundoff-sized negative values from subtractive cancellation.
    S[np.logical_and(S < 0.0, S > -1.0e-12)] = 0.0
    return np.fft.fftshift(S.real)


def connected_correlation_from_structure_factor(S_shifted):
    """Return the connected real-space correlation C(dx,dy)."""
    S_unshifted = np.fft.ifftshift(S_shifted)
    correlation = np.fft.ifft2(S_unshifted).real
    return np.fft.fftshift(correlation)


def radial_shell_average(field):
    """Average a shifted 2D field over exact square-lattice wavevector shells.

    Returns
    -------
    shell_index : ndarray
        sqrt(n_x^2+n_y^2), where n_x and n_y are integer Fourier indices.
        Multiplication by 2*pi/L converts this to physical wavevector k.
        For a real-space field, the same values are distances in lattice units.
    average : ndarray
        Mean field value on each shell.
    degeneracy : ndarray
        Number of lattice vectors contributing to each shell.
    """
    L = field.shape[0]
    integer_modes = np.fft.fftshift(np.fft.fftfreq(L) * L)
    nx, ny = np.meshgrid(integer_modes, integer_modes, indexing="ij")
    squared_radius = np.rint(nx * nx + ny * ny).astype(np.int64)

    labels, inverse = np.unique(squared_radius.ravel(), return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=field.ravel())
    averages = sums / counts

    return np.sqrt(labels.astype(np.float64)), averages, counts


def density_distribution(particle_samples, volume):
    """Return the exact discrete sampled distribution P(N), expressed vs rho."""
    particle_numbers, counts = np.unique(particle_samples, return_counts=True)
    probability = counts.astype(np.float64) / particle_samples.size
    return particle_numbers / volume, probability, counts


def safe_number(value):
    """Format a floating-point value for use in a filename."""
    return f"{value:.5g}".replace("-", "m").replace(".", "p")


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def simulate_state_point(
    L,
    T,
    mu,
    epsilon,
    initial_density,
    n_sweeps_eq,
    n_sweeps_per_meas,
    n_meas,
    seed,
    live_viz=False,
):
    """Run one state point and return measurements and averaged fields."""
    volume = L * L
    rng = np.random.default_rng(seed)
    lattice = (rng.random((L, L)) < initial_density).astype(np.uint8)
    seed_numba(seed)

    particle_number = int(lattice.sum())
    energy = float(compute_total_energy(lattice, L, epsilon))

    accepted_equilibration = 0
    for _ in tqdm(
        range(n_sweeps_eq),
        desc=f"Equilibrating T={T:.4g}, mu={mu:.4g}",
        leave=False,
    ):
        accepted, delta_E, delta_N = grand_canonical_sweep(
            lattice, L, T, epsilon, mu
        )
        accepted_equilibration += accepted
        energy += delta_E
        particle_number += delta_N

    energy_samples = np.empty(n_meas, dtype=np.float64)
    particle_samples = np.empty(n_meas, dtype=np.int64)
    sum_F = np.zeros((L, L), dtype=np.complex128)
    sum_abs_F_sq = np.zeros((L, L), dtype=np.float64)

    accepted_sampling = 0
    if live_viz:
        plt.ion()
        fig_live, ax_live = plt.subplots()

    for measurement in tqdm(
        range(n_meas),
        desc=f"Sampling T={T:.4g}, mu={mu:.4g}",
        leave=False,
    ):
        for _ in range(n_sweeps_per_meas):
            accepted, delta_E, delta_N = grand_canonical_sweep(
                lattice, L, T, epsilon, mu
            )
            accepted_sampling += accepted
            energy += delta_E
            particle_number += delta_N

        energy_samples[measurement] = energy
        particle_samples[measurement] = particle_number

        F = fourier_density(lattice)
        sum_F += F
        sum_abs_F_sq += np.abs(F) ** 2

        if live_viz:
            ax_live.clear()
            ax_live.imshow(lattice, cmap="gray_r", vmin=0, vmax=1)
            ax_live.set_title(
                f"T={T:.4g}, mu={mu:.4g}, rho={particle_number/volume:.3f}"
            )
            ax_live.set_axis_off()
            plt.pause(0.001)

    if live_viz:
        plt.ioff()
        plt.close(fig_live)

    recomputed_energy = float(compute_total_energy(lattice, L, epsilon))
    recomputed_particle_number = int(lattice.sum())
    if not np.isclose(energy, recomputed_energy, atol=1.0e-9):
        raise RuntimeError(
            "Incremental energy bookkeeping disagrees with a full "
            "recalculation."
        )
    if particle_number != recomputed_particle_number:
        raise RuntimeError(
            "Incremental particle-number bookkeeping disagrees with the "
            "configuration."
        )

    S_shifted = connected_structure_factor(
        sum_F, sum_abs_F_sq, n_meas, volume
    )
    C_shifted = connected_correlation_from_structure_factor(S_shifted)

    attempts_equilibration = n_sweeps_eq * volume
    attempts_sampling = n_meas * n_sweeps_per_meas * volume

    return {
        "lattice": lattice,
        "energy_samples": energy_samples,
        "particle_samples": particle_samples,
        "structure_factor": S_shifted,
        "correlation": C_shifted,
        "acceptance_equilibration": (
            accepted_equilibration / attempts_equilibration
            if attempts_equilibration
            else np.nan
        ),
        "acceptance_sampling": (
            accepted_sampling / attempts_sampling
            if attempts_sampling
            else np.nan
        ),
    }


def save_state_point_results(
    result,
    output_directory,
    L,
    T,
    mu,
    epsilon,
    initial_density,
    seed,
):
    """Save raw data, summary information, and basic diagnostic figures."""
    figures_directory = output_directory / "figures"
    data_directory = output_directory / "data"
    figures_directory.mkdir(parents=True, exist_ok=True)
    data_directory.mkdir(parents=True, exist_ok=True)

    volume = L * L
    energy_samples = result["energy_samples"]
    particle_samples = result["particle_samples"]
    density_samples = particle_samples / volume
    S_shifted = result["structure_factor"]
    C_shifted = result["correlation"]

    prefix = (
        f"L{L}_T{safe_number(T)}_mu{safe_number(mu)}"
        f"_eps{safe_number(epsilon)}_init{safe_number(initial_density)}"
        f"_seed{seed}"
    )

    # Raw time series: these allow students to determine equilibration,
    # autocorrelation times, block errors, and effective sample sizes.
    sample_number = np.arange(energy_samples.size)
    time_series = np.column_stack(
        (sample_number, particle_samples, density_samples, energy_samples)
    )
    np.savetxt(
        data_directory / f"{prefix}_time_series.csv",
        time_series,
        delimiter=",",
        header="measurement,N,rho,E",
        comments="",
    )

    rho_values, probability, counts = density_distribution(
        particle_samples, volume
    )
    np.savetxt(
        data_directory / f"{prefix}_density_distribution.csv",
        np.column_stack((rho_values, probability, counts)),
        delimiter=",",
        header="rho,probability,count",
        comments="",
    )

    shell_index, S_radial, S_degeneracy = radial_shell_average(S_shifted)
    k_values = 2.0 * np.pi * shell_index / L
    np.savetxt(
        data_directory / f"{prefix}_structure_factor_radial.csv",
        np.column_stack((k_values, S_radial, S_degeneracy)),
        delimiter=",",
        header="k,S(k),shell_degeneracy",
        comments="",
    )
    np.save(data_directory / f"{prefix}_structure_factor_2d.npy", S_shifted)

    r_values, C_radial, C_degeneracy = radial_shell_average(C_shifted)
    np.savetxt(
        data_directory / f"{prefix}_correlation_radial.csv",
        np.column_stack((r_values, C_radial, C_degeneracy)),
        delimiter=",",
        header="r,C(r),shell_degeneracy",
        comments="",
    )
    np.save(data_directory / f"{prefix}_correlation_2d.npy", C_shifted)

    mean_density = density_samples.mean()
    density_variance = density_samples.var()
    S_zero = S_shifted[L // 2, L // 2]
    susceptibility_density = density_variance * volume / T
    susceptibility_from_S_zero = S_zero / T
    mean_energy_per_site = energy_samples.mean() / volume

    summary_names = np.array(
        [
            "L",
            "T",
            "mu",
            "epsilon",
            "initial_density",
            "seed",
            "mean_density",
            "density_variance",
            "S_zero",
            "d_rho_d_mu_from_variance",
            "d_rho_d_mu_from_S_zero",
            "mean_energy_per_site",
            "acceptance_equilibration",
            "acceptance_sampling",
        ]
    )
    summary_values = np.array(
        [
            L,
            T,
            mu,
            epsilon,
            initial_density,
            seed,
            mean_density,
            density_variance,
            S_zero,
            susceptibility_density,
            susceptibility_from_S_zero,
            mean_energy_per_site,
            result["acceptance_equilibration"],
            result["acceptance_sampling"],
        ],
        dtype=np.float64,
    )
    np.savetxt(
        data_directory / f"{prefix}_summary.csv",
        np.column_stack((summary_names, summary_values)),
        fmt="%s",
        delimiter=",",
        header="quantity,value",
        comments="",
    )

    fig, ax = plt.subplots()
    ax.imshow(result["lattice"], cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(
        rf"Final configuration: $T={T:.4g}$, $\mu={mu:.4g}$, "
        rf"$\rho={result['lattice'].mean():.3f}$"
    )
    ax.set_axis_off()
    save_figure(fig, figures_directory / f"{prefix}_final_configuration.pdf")

    fig, ax = plt.subplots()
    ax.plot(sample_number, density_samples, lw=0.8)
    ax.set_xlabel("Measurement")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("Density time series")
    save_figure(fig, figures_directory / f"{prefix}_density_time_series.pdf")

    fig, ax = plt.subplots()
    ax.plot(rho_values, probability, drawstyle="steps-mid")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$P_L(\rho)$")
    ax.set_title("Density distribution")
    save_figure(
        fig, figures_directory / f"{prefix}_density_distribution.pdf"
    )

    fig, ax = plt.subplots()
    image = ax.imshow(
        np.log10(1.0 + np.maximum(S_shifted, 0.0)), origin="lower"
    )
    ax.set_title(r"$\log_{10}[1+S(\mathbf{k})]$")
    fig.colorbar(image, ax=ax)
    save_figure(fig, figures_directory / f"{prefix}_structure_factor_2d.pdf")

    fig, ax = plt.subplots()
    ax.plot(k_values, S_radial, marker="o", ms=3, lw=1)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$S(k)$")
    ax.set_title("Radially averaged connected structure factor")
    save_figure(
        fig, figures_directory / f"{prefix}_structure_factor_radial.pdf"
    )

    fig, ax = plt.subplots()
    ax.plot(r_values, C_radial, marker="o", ms=3, lw=1)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlim(0.0, L / 2.0)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$C(r)$")
    ax.set_title("Radially averaged connected density correlation")
    save_figure(fig, figures_directory / f"{prefix}_correlation_radial.pdf")

    return {
        "T": T,
        "mu": mu,
        "mean_density": mean_density,
        "density_variance": density_variance,
        "S_zero": S_zero,
        "susceptibility_density": susceptibility_density,
        "mean_energy_per_site": mean_energy_per_site,
        "acceptance_sampling": result["acceptance_sampling"],
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Grand-canonical 2D lattice-gas simulation"
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--L", type=int, default=DEFAULT_L)
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=DEFAULT_TEMPERATURES,
        help="Temperatures in units with k_B=1",
    )
    parser.add_argument(
        "--chemical-potentials",
        type=float,
        nargs="+",
        default=DEFAULT_CHEMICAL_POTENTIALS,
    )
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument(
        "--initial-density", type=float, default=DEFAULT_INITIAL_DENSITY
    )
    parser.add_argument(
        "--equilibration-sweeps", type=int, default=DEFAULT_EQ_SWEEPS
    )
    parser.add_argument(
        "--sweeps-per-measurement",
        type=int,
        default=DEFAULT_SWEEPS_PER_MEASUREMENT,
    )
    parser.add_argument(
        "--measurements", type=int, default=DEFAULT_MEASUREMENTS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lattice_gas_gce_output"),
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.L < 2:
        raise ValueError("L must be at least 2.")
    if np.any(np.asarray(args.temperatures) <= 0.0):
        raise ValueError("All temperatures must be positive.")
    if not 0.0 <= args.initial_density <= 1.0:
        raise ValueError("The initial density must lie between 0 and 1.")
    if args.epsilon < 0.0:
        raise ValueError("epsilon must be non-negative.")
    if args.equilibration_sweeps < 0:
        raise ValueError("The number of equilibration sweeps cannot be negative.")
    if args.sweeps_per_measurement < 1 or args.measurements < 2:
        raise ValueError(
            "Use at least one sweep per measurement and two measurements."
        )

    args.output.mkdir(parents=True, exist_ok=True)

    print("Grand-canonical 2D lattice gas")
    print(f"L = {args.L}, epsilon = {args.epsilon:g}")
    print(
        "Exact square-lattice values: "
        f"T_c = {exact_critical_temperature(args.epsilon):.6g}, "
        f"mu_coex = {coexistence_chemical_potential(args.epsilon):.6g}"
    )
    print(f"Output directory: {args.output.resolve()}")

    aggregate_rows = []
    state_point_index = 0

    for T in np.asarray(args.temperatures, dtype=float):
        for mu in np.asarray(args.chemical_potentials, dtype=float):
            state_seed = args.seed + state_point_index
            state_point_index += 1

            print(
                f"\nRunning T={T:.6g}, mu={mu:.6g}, seed={state_seed}"
            )
            result = simulate_state_point(
                L=args.L,
                T=T,
                mu=mu,
                epsilon=args.epsilon,
                initial_density=args.initial_density,
                n_sweeps_eq=args.equilibration_sweeps,
                n_sweeps_per_meas=args.sweeps_per_measurement,
                n_meas=args.measurements,
                seed=state_seed,
                live_viz=args.visualize,
            )
            summary = save_state_point_results(
                result=result,
                output_directory=args.output,
                L=args.L,
                T=T,
                mu=mu,
                epsilon=args.epsilon,
                initial_density=args.initial_density,
                seed=state_seed,
            )
            aggregate_rows.append(
                [
                    summary["T"],
                    summary["mu"],
                    summary["mean_density"],
                    summary["density_variance"],
                    summary["S_zero"],
                    summary["susceptibility_density"],
                    summary["mean_energy_per_site"],
                    summary["acceptance_sampling"],
                ]
            )

            print(
                f"<rho> = {summary['mean_density']:.6f}, "
                f"S(0) = {summary['S_zero']:.6f}, "
                f"d<rho>/dmu = {summary['susceptibility_density']:.6f}, "
                f"acceptance = {summary['acceptance_sampling']:.4f}"
            )

    aggregate = np.asarray(aggregate_rows)
    aggregate_path = args.output / "data" / "all_state_points.csv"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        aggregate_path,
        aggregate,
        delimiter=",",
        header=(
            "T,mu,mean_density,density_variance,S_zero,"
            "d_rho_d_mu,mean_energy_per_site,acceptance_sampling"
        ),
        comments="",
    )

    # A convenient equation-of-state plot when several mu values were run.
    if len(args.chemical_potentials) > 1:
        fig, ax = plt.subplots()
        for T in np.unique(aggregate[:, 0]):
            mask = aggregate[:, 0] == T
            order = np.argsort(aggregate[mask, 1])
            rows = aggregate[mask][order]
            ax.plot(
                rows[:, 1],
                rows[:, 2],
                marker="o",
                label=rf"$T={T:.4g}$",
            )
        ax.axvline(
            coexistence_chemical_potential(args.epsilon),
            color="0.5",
            ls="--",
            lw=1,
            label=r"$\mu_{\rm coex}$",
        )
        ax.set_xlabel(r"$\mu$")
        ax.set_ylabel(r"$\langle\rho\rangle$")
        ax.set_title("Grand-canonical equation of state")
        ax.legend()
        save_figure(fig, args.output / "figures" / "equation_of_state.pdf")


if __name__ == "__main__":
    main()
