"""Part 1 Q1.1 and Q1.2: simulation plots."""

import numpy as np
import matplotlib.pyplot as plt
from common import save

A_TRUE, B_TRUE, SIGMA1_TRUE, X0_TRUE = 0.9, 1.0, 1.0, 5.0
N = 100


def simulate(n, a, b, sigma1, x0, rng):
    X = np.empty(n)
    X[0] = a * x0 + b + sigma1 * rng.standard_normal()
    for t in range(1, n):
        X[t] = a * X[t - 1] + b + sigma1 * rng.standard_normal()
    return X


def q11():
    rng = np.random.default_rng(2026)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for k in range(5):
        X = simulate(N, A_TRUE, B_TRUE, SIGMA1_TRUE, X0_TRUE, rng)
        ax.plot(X, lw=1.0, label=f"realization {k + 1}")
    ax.axhline(B_TRUE / (1 - A_TRUE), color="k", ls=":", lw=0.8,
               label=r"stationary mean $b/(1-a)$")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$X_t$")
    ax.set_title("Five independent realizations of the latent process")
    ax.legend(ncol=3, fontsize=8, loc="lower right")
    save(fig, "p1_trajectories")


def q12():
    rng = np.random.default_rng(42)
    X = simulate(N, A_TRUE, B_TRUE, SIGMA1_TRUE, X0_TRUE, rng)
    Y = X + rng.standard_normal(N)  # sigma2 = 1
    np.savez(
        "_p1_q12_data.npz", X=X, Y=Y,
        a=A_TRUE, b=B_TRUE, sigma1=SIGMA1_TRUE, sigma2=1.0, X0=X0_TRUE,
    )
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(X, color="C0", lw=1.2, label=r"latent $X_t$")
    ax.plot(Y, "o", color="C3", ms=3, alpha=0.7, label=r"observation $Y_t$")
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.set_title(r"Latent state $X_t$ and noisy observation $Y_t=X_t+e_{2,t}$")
    ax.legend()
    save(fig, "p1_latent_vs_obs")


if __name__ == "__main__":
    q11()
    q12()
    print("Part 1.1 and 1.2 figures saved.")
