#!/usr/bin/env python3
"""
Individual-based simulations of the evolution of a flower growth schedule
(Section 3.3, Appendix C.2, Figure 2).

Same structure as flowering_schedule_ibm.py, but u(a) is now a control: investment
in floral growth, which produces floral tissue x(a) following
    dx/da = alpha u(a) - nu x(a),   x(0) = x0          (eq. 47).
Benefits depend on x(a) and costs on u(a) (eq. 49).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp
from scipy.linalg import eigh

# Parameters of Figure 2
CASES = {
    "stabilizing": dict(beta=0.35, q=0.70),
    "disruptive": dict(beta=0.80, q=1.15),
}
ALPHA = 2.00  # floral tissue produced per unit investment
NU = 0.70     # loss rate of floral tissue
X0 = 0.00     # floral tissue at the start of the season
C = 0.00      # linear cost of investment (c = 0 keeps u* > 0 near a = 1, since u*(1) = -c/chi)
CHI = 1.00    # quadratic cost of investment
B0 = 2.00     # baseline fecundity


def pollinators(a):
    """Pollinator availability p(a)."""
    return 1.60 + 0.40 * np.exp(-0.5 * ((a - 0.50) / 0.18) ** 2)


def season(N_a: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time points a_j, trapezoidal weights omega_j, pollinator availability p(a_j)."""
    a = np.linspace(0.0, 1.0, N_a)
    omega = np.full(N_a, a[1] - a[0])
    omega[[0, -1]] *= 0.5
    return a, omega, pollinators(a)


def state_map(a: np.ndarray) -> np.ndarray:
    """Matrix S such that x = S u + x0 e^{-nu a} reproduces the recursion of eq. (C.7)."""
    decay = np.exp(-NU * (a[1] - a[0]))
    lag = np.arange(a.size)[:, None] - np.arange(a.size)[None, :] - 1
    return np.where(lag >= 0, ALPHA * (1.0 - decay) / NU * decay ** np.maximum(lag, 0), 0.0)


def floral_tissue(u: np.ndarray, S: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Floral tissue x(a) of each individual (one schedule per row of u)."""
    return u @ S.T + X0 * np.exp(-NU * a)


def singular_schedule(a: np.ndarray, beta: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """u*(a) and x*(a), from eqs. (53)-(55); lam is the rescaled costate."""
    def odes(s, y):
        x, lam = y
        u = (ALPHA * lam - C) / CHI
        return np.vstack([ALPHA * u - NU * x,
                          NU * lam - pollinators(s) * (1.0 + (beta - q) * x)])

    def boundary(y0, y1):
        return np.array([y0[0] - X0, y1[1]])  # x*(0) = x0, lam*(1) = 0

    s = np.linspace(0.0, 1.0, 801)
    sol = solve_bvp(odes, boundary, s, np.zeros((2, s.size)), max_nodes=60000, tol=1e-9)
    if sol.status != 0:
        raise RuntimeError(sol.message)
    x_star, lam_star = sol.sol(a)
    return (ALPHA * lam_star - C) / CHI, x_star


def leading_eigenvalue(S: np.ndarray, omega: np.ndarray, p: np.ndarray) -> float:
    """Lambda, eq. (70): largest eigenvalue of M, using <v, M v> = int p(a) (S v)(a)^2 da."""
    K = S.T @ np.diag(omega * p) @ S
    return float(eigh(K, np.diag(omega), eigvals_only=True)[-1])


def mutation_factor(a: np.ndarray, ell_mut: float) -> np.ndarray:
    """Cholesky factor of the mutational covariance C_jl, eq. (C.5)."""
    cov = np.exp(-0.5 * ((a[:, None] - a[None, :]) / ell_mut) ** 2)
    return np.linalg.cholesky(cov + 1e-10 * np.eye(a.size))


def fecundity(u, S, a, p, omega, beta, q):
    """Fecundity f_i of every individual before truncation at zero, eq. (C.8)."""
    x = floral_tissue(u, S, a)
    x_bar = x.mean(axis=0)
    integrand = p * x * (1.0 + 0.5 * beta * x - q * x_bar) - C * u - 0.5 * CHI * u ** 2
    return B0 + integrand @ omega


def simulate(beta, q, *, N, generations, N_a, record_every, p_mut, sigma_mut, ell_mut, seed):
    rng = np.random.default_rng(seed)
    a, omega, p = season(N_a)
    S = state_map(a)
    L = mutation_factor(a, ell_mut)
    u_star, x_star = singular_schedule(a, beta, q)
    j_mid = int(np.argmin(np.abs(a - 0.5)))  # a = 0.5

    u = np.tile(0.45 * u_star, (N, 1))
    u_mid_history = [u[:, j_mid].copy()]
    min_f_raw = np.inf

    for t in range(1, generations + 1):
        # Reproduction
        f = fecundity(u, S, a, p, omega, beta, q)
        min_f_raw = min(min_f_raw, float(f.min()))
        f = np.maximum(f, 0.0)
        u = u[rng.choice(N, size=N, p=f / f.sum())]

        # Mutation (u >= 0)
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
        "x_star": x_star,
        "generation": np.arange(0, generations + 1, record_every),
        "u_mid_history": np.array(u_mid_history),
        "u_final": u,
        "x_final": floral_tissue(u, S, a),
        "min_f_raw": min_f_raw,
        "Lambda": leading_eigenvalue(S, omega, p),
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

    fig, axes = plt.subplots(3, 2, figsize=(11.6, 11.4), constrained_layout=True)
    for col, (name, r) in enumerate(results.items()):
        a, u_star, gen = r["a"], r["u_star"], r["generation"]
        j_mid = int(np.argmin(np.abs(a - 0.5)))
        beta, q = CASES[name]["beta"], CASES[name]["q"]

        # Top: distribution of u(0.5) through time
        ax = axes[0, col]
        ax.imshow(density(r["u_mid_history"], bins).T, origin="lower", aspect="auto",
                  extent=(gen[0], gen[-1], bins[0], bins[-1]), cmap="Blues", vmin=0.0, vmax=1.0)
        ax.axhline(u_star[j_mid], color="black", lw=1.7, ls="--", label=r"$u^*(0.5)$")
        ax.set_title(f"{name.capitalize()} selection " rf"($\beta={beta:g},\ q={q:g}$)")
        ax.set_xlabel("Generation")
        ax.set_ylabel(r"$u(0.5)$")
        ax.legend(frameon=False, loc="upper left")

        # Middle: final control schedules u(a)
        ax = axes[1, col]
        ax.plot(a, r["u_final"].T, color=blue, alpha=0.06, lw=0.8)
        ax.plot(a, u_star, color="black", lw=1.7, ls="--", label=r"$u^*(a)$")
        ax.set_title("Final floral investment schedules")
        ax.set_xlabel(r"Seasonal time $a$")
        ax.set_ylabel(r"$u(a)$")
        ax.legend(frameon=False, loc="upper left")

        # Bottom: final floral tissue x(a), with p(a) on a right axis
        ax = axes[2, col]
        ax.plot(a, r["x_final"].T, color=blue, alpha=0.06, lw=0.8)
        line_x, = ax.plot(a, r["x_star"], color="black", lw=1.7, ls="--")
        ax.set_title("Final floral tissue schedules")
        ax.set_xlabel(r"Seasonal time $a$")
        ax.set_ylabel(r"$x(a)$")
        ax_p = ax.twinx()
        line_p, = ax_p.plot(a, r["p"], color="red", lw=1.7, ls="--")
        span = r["p"].max() - r["p"].min()
        ax_p.set_ylim(r["p"].min() - 0.25 * span, r["p"].max() + 0.25 * span)
        ax_p.set_ylabel(r"$p(a)$", color="red")
        ax_p.tick_params(axis="y", colors="red")
        ax_p.spines["right"].set_color("red")
        ax.legend([line_x, line_p], [r"$x^*(a)$", r"$p(a)$"], frameon=False, loc="upper left")

    # Common axis limits within each row
    for row in axes:
        xlims = [ax.get_xlim() for ax in row]
        ylims = [ax.get_ylim() for ax in row]
        for ax in row:
            ax.set_xlim(min(l[0] for l in xlims), max(l[1] for l in xlims))
            ax.set_ylim(min(l[0] for l in ylims), max(l[1] for l in ylims))

    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("flower_growth_output"))
    parser.add_argument("--N", type=int, default=4000, help="population size")
    parser.add_argument("--generations", type=int, default=100000)
    parser.add_argument("--N-a", type=int, default=1001, help="number of time points")
    parser.add_argument("--record-every", type=int, default=5)
    parser.add_argument("--p-mut", type=float, default=0.01)
    parser.add_argument("--sigma-mut", type=float, default=0.005)
    parser.add_argument("--ell-mut", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=8142)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for index, (name, case) in enumerate(CASES.items()):
        r = simulate(**case, N=args.N, generations=args.generations, N_a=args.N_a,
                     record_every=args.record_every, p_mut=args.p_mut, sigma_mut=args.sigma_mut,
                     ell_mut=args.ell_mut, seed=args.seed + 1000 * index)
        results[name] = r
        # Section 3.3: convergence stable if (beta - q) Lambda < chi; uninvadable if beta Lambda < chi
        Lam = r["Lambda"]
        print(f"[{name}] Lambda = {Lam:.3f}, (beta-q) Lambda = {(case['beta'] - case['q']) * Lam:.3f}, "
              f"beta Lambda = {case['beta'] * Lam:.3f}, chi = {CHI}")
        np.savez_compressed(args.output_dir / f"{name}_simulation.npz",
                            **{k: v for k, v in r.items() if isinstance(v, np.ndarray)})

    make_figure(results, args.output_dir / "flower_growth_schedule.png")

    with (args.output_dir / "simulation_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "beta", "q", "alpha", "nu", "x0", "c", "chi", "b0", "N", "p_mut",
                         "sigma_mut", "ell_mut", "generations", "Lambda", "mean_u_mid", "sd_u_mid",
                         "min_f_raw"])
        for name, r in results.items():
            u_mid = r["u_final"][:, int(np.argmin(np.abs(r["a"] - 0.5)))]
            writer.writerow([name, CASES[name]["beta"], CASES[name]["q"], ALPHA, NU, X0, C, CHI, B0,
                             args.N, args.p_mut, args.sigma_mut, args.ell_mut, args.generations,
                             r["Lambda"], u_mid.mean(), u_mid.std(), r["min_f_raw"]])


if __name__ == "__main__":
    main()
