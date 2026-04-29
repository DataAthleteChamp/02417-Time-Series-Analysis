"""Part 1 Q1.5: Student-t process noise robustness study."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import t as student_t, norm
from common import save, write_table
from kalman import neg_loglik_scalar

N = 100
N_REPS = 100
A_TRUE, B_TRUE, SIGMA1_TRUE = 1.0, 0.9, 1.0
NUS = [100, 5, 2, 1]
R_OBS = 1.0


def simulate_t(n, a, b, sigma1, nu, rng, x0=5.0):
    X = np.empty(n)
    eps = student_t.rvs(df=nu, size=n, random_state=rng)
    X[0] = a * x0 + b + sigma1 * eps[0]
    for t in range(1, n):
        X[t] = a * X[t - 1] + b + sigma1 * eps[t]
    Y = X + rng.standard_normal(n)
    return X, Y


def fit(Y, start=(0.5, 0.0, 1.0)):
    def nll(theta):
        a, b, log_s1 = theta
        return neg_loglik_scalar((a, b, np.exp(log_s1)), Y, R_OBS, 0.0, 10.0)
    x0 = np.array([start[0], start[1], np.log(start[2])])
    r1 = minimize(nll, x0, method="Nelder-Mead",
                  options=dict(xatol=1e-5, fatol=1e-5, maxiter=2000))
    r2 = minimize(nll, r1.x, method="L-BFGS-B",
                  bounds=[(-10, 10), (-100, 100), (-10, 8)])
    a_hat, b_hat, ls = r2.x
    return a_hat, b_hat, np.exp(ls), np.isfinite(r2.fun)


def density_plot():
    x = np.linspace(-6, 6, 400)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(x, norm.pdf(x), "k--", lw=1.4, label=r"$\mathcal{N}(0,1)$")
    for nu in NUS:
        ax.plot(x, student_t.pdf(x, df=nu), lw=1.2, label=rf"$t_\nu,\; \nu={nu}$")
    ax.set_xlabel("x"); ax.set_ylabel("density")
    ax.set_title("Standard normal vs Student-$t$ densities")
    ax.legend()
    save(fig, "p1_tdens")


def main():
    density_plot()

    results = {}
    for nu in NUS:
        rng_master = np.random.default_rng(5000 + nu)
        seeds = rng_master.integers(0, 2**31 - 1, size=N_REPS)
        ests = np.full((N_REPS, 3), np.nan)
        ok = np.zeros(N_REPS, bool)
        for i, s in enumerate(seeds):
            rng = np.random.default_rng(int(s))
            try:
                _, Y = simulate_t(N, A_TRUE, B_TRUE, SIGMA1_TRUE, nu, rng)
                ah, bh, sh, fin = fit(Y)
                if fin and np.isfinite([ah, bh, sh]).all():
                    ests[i] = [ah, bh, sh]
                    ok[i] = True
            except Exception:
                pass
        results[nu] = dict(est=ests, ok=ok)
        m = np.nanmean(ests, axis=0)
        print(f"nu={nu}: â={m[0]:.3f}  b̂={m[1]:.3f}  σ̂={m[2]:.3f}  success={ok.mean():.0%}")

    # --- boxplots ---
    param_names = [r"$\hat a$", r"$\hat b$", r"$\hat\sigma_1$"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    pos = np.arange(len(NUS))
    truths = [A_TRUE, B_TRUE, SIGMA1_TRUE]
    for j in range(3):
        data = [results[nu]["est"][:, j][np.isfinite(results[nu]["est"][:, j])]
                for nu in NUS]
        axes[j].boxplot(data, positions=pos, widths=0.6, showfliers=True)
        axes[j].axhline(truths[j], color="red", ls=":", lw=1.0, label="truth")
        axes[j].set_xticks(pos)
        axes[j].set_xticklabels([rf"$\nu={nu}$" for nu in NUS])
        axes[j].set_title(param_names[j])
        axes[j].grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    # Trim sigma1 axis if Cauchy gives huge outliers
    axes[2].set_yscale("log")
    fig.suptitle(r"Gaussian-KF MLE estimates under Student-$t$ process noise")
    fig.tight_layout()
    save(fig, "p1_robustness_box")

    # --- summary table ---
    lines = [
        r"\begin{tabular}{l l l l l}",
        r"\toprule",
        r"$\nu$ & Parameter & Mean & Bias & Success rate \\",
        r"\midrule",
    ]
    pnames = ["a", "b", r"\sigma_1"]
    for nu in NUS:
        est = results[nu]["est"]
        m = np.nanmean(est, axis=0)
        bias = m - np.array(truths)
        for k, pn in enumerate(pnames):
            first = f"{nu}" if k == 0 else ""
            lines.append(f"{first} & ${pn}$ & {m[k]:.3f} & {bias[k]:+.3f} & "
                         f"{results[nu]['ok'].mean()*100:.0f}\\% \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write_table("p1_robustness.tex", "\n".join(lines))
    print("Saved p1_robustness.tex")


if __name__ == "__main__":
    main()
