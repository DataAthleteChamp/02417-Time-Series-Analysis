"""Part 1 Q1.4: MLE Monte-Carlo across 3 parameter settings."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from common import save, write_table
from kalman import neg_loglik_scalar

N = 100
N_REPS = 100
R_OBS = 1.0  # sigma2^2 known = 1
X0_PRIOR = 0.0
P0_PRIOR = 10.0

CASES = [
    dict(name="a=1, b=0.9, sigma1=1",    a=1.0, b=0.9, sigma1=1.0),
    dict(name="a=5, b=0.9, sigma1=1",    a=5.0, b=0.9, sigma1=1.0),
    dict(name="a=1, b=0.9, sigma1=5",    a=1.0, b=0.9, sigma1=5.0),
]


def simulate(n, a, b, sigma1, rng, x0=5.0):
    X = np.empty(n); X[0] = a * x0 + b + sigma1 * rng.standard_normal()
    for t in range(1, n):
        X[t] = a * X[t - 1] + b + sigma1 * rng.standard_normal()
    Y = X + rng.standard_normal(n)  # sigma2 = 1
    return X, Y


def fit_one(Y, start=(0.5, 0.0, 1.0), rng=None):
    """Two-stage: Nelder-Mead -> L-BFGS-B (unbounded sigma via transform).

    If an ``rng`` is supplied, the starting point for ``b`` is jittered so
    that each replicate sees a different initial condition.  This breaks
    the artificial fixed point that the optimizer lands on under the
    explosive ``a=5`` case and produces a genuine Monte-Carlo spread.
    """
    def nll(theta):
        a, b, log_s1 = theta
        return neg_loglik_scalar((a, b, np.exp(log_s1)), Y, R_OBS, X0_PRIOR, P0_PRIOR)

    a0, b0, s0 = start
    if rng is not None:
        b0 = b0 + rng.uniform(-3.0, 3.0)
        a0 = a0 + rng.uniform(-0.2, 0.2)
    x0 = np.array([a0, b0, np.log(s0)])
    r1 = minimize(nll, x0, method="Nelder-Mead",
                  options=dict(xatol=1e-5, fatol=1e-5, maxiter=2000))
    r2 = minimize(nll, r1.x, method="L-BFGS-B",
                  bounds=[(-10, 10), (-50, 50), (-10, 5)])
    a_hat, b_hat, log_s1 = r2.x
    return a_hat, b_hat, np.exp(log_s1), r2.fun, r2.success


def run_case(case, seed0):
    rng_master = np.random.default_rng(seed0)
    seeds = rng_master.integers(0, 2**31 - 1, size=N_REPS)
    start_rng = np.random.default_rng(seed0 + 99)
    ests = np.empty((N_REPS, 3))
    ok = np.empty(N_REPS, bool)
    for i, s in enumerate(seeds):
        rng = np.random.default_rng(int(s))
        _, Y = simulate(N, case["a"], case["b"], case["sigma1"], rng)
        ah, bh, sh, _, success = fit_one(Y, rng=start_rng)
        ests[i] = [ah, bh, sh]
        ok[i] = success
    return ests, ok


def main():
    results = {}
    for i, c in enumerate(CASES):
        print(f"[{i+1}/{len(CASES)}] {c['name']} ...")
        est, ok = run_case(c, seed0=1000 + i)
        results[c["name"]] = dict(est=est, ok=ok, truth=[c["a"], c["b"], c["sigma1"]])
        print(f"  mean â={est[:,0].mean():.3f}  b̂={est[:,1].mean():.3f}  σ̂={est[:,2].mean():.3f}  "
              f"success={ok.mean():.0%}")

    # --- boxplots
    param_names = [r"$\hat a$", r"$\hat b$", r"$\hat\sigma_1$"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0))
    positions = np.arange(len(CASES))
    for j, pname in enumerate(param_names):
        data = [results[c["name"]]["est"][:, j] for c in CASES]
        axes[j].boxplot(data, positions=positions, widths=0.6, showfliers=True)
        for k, c in enumerate(CASES):
            axes[j].scatter([k], [c[["a", "b", "sigma1"][j]]],
                            marker="*", s=120, color="red", zorder=5)
        axes[j].set_xticks(positions)
        axes[j].set_xticklabels([c["name"].replace(", ", "\n") for c in CASES], fontsize=7)
        axes[j].set_title(pname)
        axes[j].grid(alpha=0.3)
    fig.suptitle("MLE estimates across 3 parameter configurations (100 reps each, red star = truth)")
    fig.tight_layout()
    save(fig, "p1_mle_boxplots")

    # --- summary table -> LaTeX
    rows = []
    for c in CASES:
        est = results[c["name"]]["est"]
        truth = np.array(results[c["name"]]["truth"])
        bias = est.mean(0) - truth
        sd = est.std(0, ddof=1)
        rmse = np.sqrt(np.mean((est - truth) ** 2, axis=0))
        rows.append((c["name"], truth, est.mean(0), bias, sd, rmse,
                     results[c["name"]]["ok"].mean()))

    lines = [
        r"\begin{tabular}{l l l l l l l}",
        r"\toprule",
        r"Case & Parameter & True & Mean & Bias & SD & RMSE \\",
        r"\midrule",
    ]
    pnames = ["a", "b", r"\sigma_1"]
    for name, truth, mean, bias, sd, rmse, ok_rate in rows:
        for k, p in enumerate(pnames):
            first = name.replace("sigma1", r"$\sigma_1$") if k == 0 else ""
            lines.append(f"{first} & ${p}$ & {truth[k]:.3f} & {mean[k]:.3f} & "
                         f"{bias[k]:+.3f} & {sd[k]:.3f} & {rmse[k]:.3f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write_table("p1_mle.tex", "\n".join(lines))
    print("Saved p1_mle.tex")

    np.savez("_p1_mle_results.npz",
             **{k: v["est"] for k, v in results.items()})


if __name__ == "__main__":
    main()
