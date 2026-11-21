#!/usr/bin/env python3
"""
Individual-based simulations of the evolution of a floral investment schedule
(Section 2.6, Appendix C.1, Figure 1).

Each individual carries a schedule u(a) on the season A = [0, 1], discretized on
N_a time points. Each generation: fecundity (eq. C.2), Wright-Fisher sampling of
parents (eq. C.3), and smooth Gaussian mutation of whole schedules (eqs. C.4-C.6).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Parameters of Figure 1
CASES = {
    "stabilizing": dict(beta=0.35, q=0.70),
    "disruptive": dict(beta=0.80, q=1.15),
}
C = 0.80    # linear cost of floral investment
CHI = 1.00  # quadratic cost of floral investment
F0 = 2.00   # baseline fecundity (vegetative propagules)


def season(N_a: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time points a_j, trapezoidal weights omega_j, pollinator availability p(a_j)."""
    a = np.linspace(0.0, 1.0, N_a)
    omega = np.full(N_a, a[1] - a[0])
    omega[[0, -1]] *= 0.5
    p = 1.60 + 0.40 * np.exp(-0.5 * ((a - 0.50) / 0.18) ** 2)
    return a, omega, p


def singular_schedule(p: np.ndarray, beta: float, q: float) -> np.ndarray:
    """u*(a), eq. (26)."""
    return (p - C) / (CHI + p * (q - beta))


def mutation_factor(a: np.ndarray, ell_mut: float) -> np.ndarray:
    """Cholesky factor of the mutational covariance C_jl, eq. (C.5)."""
    cov = np.exp(-0.5 * ((a[:, None] - a[None, :]) / ell_mut) ** 2)
    return np.linalg.cholesky(cov + 1e-10 * np.eye(a.size))


def fecundity(u, p, omega, beta, q):
    """Fecundity f_i of every individual before truncation at zero, eq. (C.2)."""
    u_bar = u.mean(axis=0)
    integrand = p * u * (1.0 + 0.5 * beta * u - q * u_bar) - C * u - 0.5 * CHI * u ** 2
    return F0 + integrand @ omega


def simulate(beta, q, *, N, generations, N_a, record_every, p_mut, sigma_mut, ell_mut, seed):
    rng = np.random.default_rng(seed)
    a, omega, p = season(N_a)
    L = mutation_factor(a, ell_mut)
    u_star = singular_schedule(p, beta, q)
    j_mid = int(np.argmin(np.abs(a - 0.5)))  # a = 0.5

    u = np.tile(0.45 * u_star, (N, 1))
    u_mid_history = [u[:, j_mid].copy()]
    min_f_raw = np.inf

    for t in range(1, generations + 1):
        # Reproduction
        f = fecundity(u, p, omega, beta, q)
        min_f_raw = min(min_f_raw, float(f.min()))
        f = np.maximum(f, 0.0)
        u = u[rng.choice(N, size=N, p=f / f.sum())]

        # Mutation
        mutant = rng.random(N) < p_mut
        if mutant.any():
            xi = rng.normal(size=(int(mutant.sum()), N_a)) @ L.T
            u[mutant] = np.maximum(u[mutant] + sigma_mut * xi, 0.0)

        if t % record_every == 0:
            u_mid_history.append(u[:, j_mid].copy())

    return {
        "a": a,
        "p": p,
        "u_star": u_star,
        "generation": np.arange(0, generations + 1, record_every),
        "u_mid_history": np.array(u_mid_history),
        "u_final": u,
        "min_f_raw": min_f_raw,
    }


def density(history: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Histogram of u(0.5) at each recorded generation, scaled to a maximum of 1."""
    counts = np.array([np.histogram(values, bins=bins)[0] for values in history], dtype=float)
    return counts / np.maximum(counts.max(axis=1, keepdims=True), 1.0)


def make_figure(results: dict, path: Path) -> None:
    plt.rcParams.update({
        "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 13,
        "ytick.labelsize": 13, "legend.fontsize": 13, "mathtext.fontset": "dejavusans",
    })
    blue = "#2864a8"
    all_mid = np.concatenate([r["u_mid_history"].ravel() for r in results.values()])
    bins = np.linspace(0.0, max(1.05 * float(np.quantile(all_mid, 0.9995)), 0.85), 150)

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.8), constrained_layout=True)
    for col, (name, r) in enumerate(results.items()):
        a, u_star, gen = r["a"], r["u_star"], r["generation"]
        j_mid = int(np.argmin(np.abs(a - 0.5)))

        # Top: distribution of u(0.5) through time
        ax = axes[0, col]
        ax.imshow(density(r["u_mid_history"], bins).T, origin="lower", aspect="auto",
                  extent=(gen[0], gen[-1], bins[0], bins[-1]), cmap="Blues", vmin=0.0, vmax=1.0)
        ax.axhline(u_star[j_mid], color="black", lw=1.7, ls="--", label=r"$u^*(0.5)$")
        ax.set_title(f"{name.capitalize()} selection")
        ax.set_xlabel("Generation")
        ax.set_ylabel(r"$u(0.5)$")
        ax.legend(frameon=False, loc="upper left")

        # Bottom: final schedules
        ax = axes[1, col]
        ax.plot(a, r["u_final"].T, color=blue, alpha=0.06, lw=0.8)
        ax.plot(a, u_star, color="black", lw=1.7, ls="--", label=r"$u^*(a)$")
        ax.set_title("Final floral investment schedules")
        ax.set_xlabel(r"Seasonal time $a$")
        ax.set_ylabel(r"$u(a)$")
        ax.legend(frameon=False, loc="upper left")

    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("flowering_schedule_output"))
    parser.add_argument("--N", type=int, default=4000, help="population size")
    parser.add_argument("--generations", type=int, default=50000)
    parser.add_argument("--N-a", type=int, default=1001, help="number of time points")
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--p-mut", type=float, default=0.01)
    parser.add_argument("--sigma-mut", type=float, default=0.005)
    parser.add_argument("--ell-mut", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=8142)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for index, (name, case) in enumerate(CASES.items()):
        # Conditions of Section 2.6: chi + p(q - beta) > 0 (convergence stability);
        # p beta - chi < 0 (uninvadable) or > 0 (invadable) throughout the season.
        p = season(args.N_a)[2]
        print(f"[{name}] min chi+p(q-beta) = {np.min(CHI + p * (case['q'] - case['beta'])):.3f}, "
              f"p beta - chi in [{np.min(p * case['beta'] - CHI):.3f}, {np.max(p * case['beta'] - CHI):.3f}]")

        r = simulate(**case, N=args.N, generations=args.generations, N_a=args.N_a,
                     record_every=args.record_every, p_mut=args.p_mut, sigma_mut=args.sigma_mut,
                     ell_mut=args.ell_mut, seed=args.seed + 1000 * index)
        results[name] = r
        np.savez_compressed(args.output_dir / f"{name}_simulation.npz",
                            **{k: v for k, v in r.items() if isinstance(v, np.ndarray)})

    make_figure(results, args.output_dir / "flowering_schedule.png")

    with (args.output_dir / "simulation_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "beta", "q", "c", "chi", "f0", "N", "p_mut", "sigma_mut",
                         "ell_mut", "generations", "mean_u_mid", "sd_u_mid", "min_f_raw"])
        for name, r in results.items():
            u_mid = r["u_final"][:, int(np.argmin(np.abs(r["a"] - 0.5)))]
            writer.writerow([name, CASES[name]["beta"], CASES[name]["q"], C, CHI, F0, args.N,
                             args.p_mut, args.sigma_mut, args.ell_mut, args.generations,
                             u_mid.mean(), u_mid.std(), r["min_f_raw"]])


if __name__ == "__main__":
    main()
