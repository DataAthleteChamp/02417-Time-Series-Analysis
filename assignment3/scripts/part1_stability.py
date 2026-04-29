"""Assignment 3 — Part 1: Stability of AR(2) processes.

Model:  X_t + phi1 X_{t-1} + phi2 X_{t-2} = eps_t,   eps_t ~ WN(0, 1).

For each of 5 (phi1, phi2) pairs:
  * report roots of 1 + phi1 z + phi2 z^2 and stationarity check
  * simulate 5 realisations with n=200 (seeded)
  * plot the 5 simulations overlaid
  * compute empirical ACF to lag 30 and overlay theoretical rho(k)
    (theoretical rho only plotted for stationary cases)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

from common import sample_acf, acf_ci, savefig


N = 200
N_REPS = 5
N_LAGS = 30
BURNIN = 200
SEED = 2026


def stationarity_check(phi1, phi2):
    """Return (roots, |roots|, conds dict, is_stationary)."""
    roots = np.roots([phi2, phi1, 1.0])  # highest power first
    absr = np.abs(roots)
    conds = {
        "|phi2|<1": abs(phi2) < 1,
        "phi1+phi2 > -1": phi1 + phi2 > -1,
        "phi1-phi2 < 1": phi1 - phi2 < 1,
    }
    stationary = all(conds.values()) and np.all(absr > 1)
    return roots, absr, conds, stationary


def theoretical_acf(phi1, phi2, n_lags):
    """Yule-Walker ACF, valid only for stationary case."""
    rho = np.zeros(n_lags + 1)
    rho[0] = 1.0
    rho[1] = -phi1 / (1 + phi2)
    for k in range(2, n_lags + 1):
        rho[k] = -phi1 * rho[k - 1] - phi2 * rho[k - 2]
    return rho


def simulate(phi1, phi2, n, burnin, rng):
    eps = rng.normal(0.0, 1.0, n + burnin)
    x = lfilter([1.0], [1.0, phi1, phi2], eps)
    return x[burnin:]


PARAM_GRID = [
    ("1.1", -0.6,  0.5),
    ("1.3", -0.6, -0.3),
    ("1.4",  0.6, -0.3),
    ("1.5", -0.7, -0.3),
    ("1.6", -0.75, -0.3),
]


def run_case(tag, phi1, phi2):
    print(f"\n── Case {tag}: phi1={phi1}, phi2={phi2} ──")
    roots, absr, conds, stationary = stationarity_check(phi1, phi2)
    print(f"  roots        = {roots}")
    print(f"  |roots|      = {absr}")
    for k, v in conds.items():
        print(f"  {k}: {v}")
    print(f"  stationary   = {stationary}")

    rng = np.random.default_rng(SEED)
    sims = np.stack([simulate(phi1, phi2, N, BURNIN, rng) for _ in range(N_REPS)])

    # --- figure: 5 simulations ---
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i, s in enumerate(sims):
        ax.plot(np.arange(1, N + 1), s, linewidth=0.8, label=f"rep {i+1}")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$X_t$")
    ax.set_title(
        rf"AR(2) simulations — $\phi_1={phi1}$, $\phi_2={phi2}$"
        + ("" if stationary else "   [NON-STATIONARY]")
    )
    ax.legend(fontsize=7, ncol=5, loc="upper right")
    savefig(fig, f"p1_{tag.replace('.', '_')}_sims.pdf")

    # --- figure: ACFs ---
    fig, ax = plt.subplots(figsize=(8, 3.6))
    lags = np.arange(N_LAGS + 1)
    for i, s in enumerate(sims):
        acf = sample_acf(s, N_LAGS)
        ax.plot(lags, acf, marker="o", markersize=2.5, linewidth=0.6,
                alpha=0.55, label=f"rep {i+1}")
    if stationary:
        rho = theoretical_acf(phi1, phi2, N_LAGS)
        ax.plot(lags, rho, color="k", linewidth=2.0, label=r"theoretical $\rho(k)$")
    ci = acf_ci(N)
    ax.axhline(ci, color="b", linestyle="--", linewidth=0.6)
    ax.axhline(-ci, color="b", linestyle="--", linewidth=0.6)
    ax.axhline(0, color="k", linewidth=0.4)
    ax.set_xlabel("Lag k")
    ax.set_ylabel(r"$\hat{\rho}(k)$")
    title = rf"Empirical ACF — $\phi_1={phi1}$, $\phi_2={phi2}$"
    if not stationary:
        title += "   [NON-STATIONARY, no theoretical ACF]"
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    savefig(fig, f"p1_{tag.replace('.', '_')}_acf.pdf")

    # sample ACF values printed (useful for report commentary)
    acfs = np.stack([sample_acf(s, N_LAGS) for s in sims])
    print(f"  mean sample rho(1) over reps = {acfs[:, 1].mean():.3f}")
    print(f"  mean sample rho(5)            = {acfs[:, 5].mean():.3f}")
    print(f"  mean sample rho(10)           = {acfs[:, 10].mean():.3f}")
    print(f"  simulation variance (mean)    = {sims.var(axis=1).mean():.3f}")


def main():
    for tag, p1, p2 in PARAM_GRID:
        run_case(tag, p1, p2)


if __name__ == "__main__":
    main()
