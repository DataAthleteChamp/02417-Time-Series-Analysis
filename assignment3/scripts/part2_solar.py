"""Assignment 3 — Part 2: Predicting monthly solar power.

Seasonal AR model:
    (1 + phi1 B)(1 + Phi1 B^12) (log Y_t - mu) = eps_t
  with phi1 = -0.38, Phi1 = -0.94, mu = 5.72, sigma_eps = 0.22.

Let X_t = log Y_t - mu.  Expanding:
    X_t + phi1 X_{t-1} + Phi1 X_{t-12} + phi1*Phi1 X_{t-13} = eps_t
==> X_t = - phi1 X_{t-1} - Phi1 X_{t-12} - phi1*Phi1 X_{t-13} + eps_t
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from common import sample_acf, acf_ci, savefig, DATA_DIR

PHI1 = -0.38
PHIS1 = -0.94
MU = 5.72
SIGMA = 0.22
SIGMA2 = SIGMA ** 2
K_MAX = 12


def load_data():
    df = pd.read_csv(DATA_DIR / "datasolar.csv")
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_residuals(X):
    """eps_hat_t for t = 13 .. n-1 (0-indexed), needing X_{t-13}."""
    n = len(X)
    eps = np.full(n, np.nan)
    for t in range(13, n):
        x_hat = -PHI1 * X[t - 1] - PHIS1 * X[t - 12] - PHI1 * PHIS1 * X[t - 13]
        eps[t] = X[t] - x_hat
    return eps


def forecast(X, k_max):
    """Recursive k-step-ahead X_hat for k=1..k_max starting at t=n-1 (last obs)."""
    n = len(X)
    Xext = np.concatenate([X, np.full(k_max, np.nan)])
    for k in range(1, k_max + 1):
        idx = n - 1 + k  # position of X_{t+k}
        x1 = Xext[idx - 1]                   # may be forecasted (if k>=2)
        x12 = Xext[idx - 12]                 # observed for k<=12
        x13 = Xext[idx - 13]                 # observed for k<=12
        Xext[idx] = -PHI1 * x1 - PHIS1 * x12 - PHI1 * PHIS1 * x13
    return Xext[n:]


def ar1_forecast_variance(k_max, a, sigma2):
    """Var(e_{t+k|t}) for AR(1) with X_t = a X_{t-1} + eps_t, eps ~ (0, sigma2)."""
    var = np.empty(k_max)
    for k in range(1, k_max + 1):
        var[k - 1] = sigma2 * (1 - a ** (2 * k)) / (1 - a ** 2)
    return var


def main():
    df = load_data()
    Y = df["power"].to_numpy(dtype=float)
    X = np.log(Y) - MU
    n = len(Y)
    print(f"Loaded {n} monthly observations ({df['date'].iloc[0].date()} .. {df['date'].iloc[-1].date()})")

    # ============================================================
    # 2.1  residuals + diagnostics
    # ============================================================
    eps = compute_residuals(X)
    eps_clean = eps[~np.isnan(eps)]
    print(f"\n2.1  residuals: n={len(eps_clean)}, "
          f"mean={eps_clean.mean():.4f}, std={eps_clean.std(ddof=1):.4f}")

    lb = acorr_ljungbox(eps_clean, lags=[5, 10], return_df=True)
    print(lb)
    sh = stats.shapiro(eps_clean)
    print(f"Shapiro-Wilk: W={sh.statistic:.4f}, p={sh.pvalue:.4f}")

    # diagnostics figure
    n_res = len(eps_clean)
    n_lags_acf = min(15, n_res - 1)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes[0, 0].plot(df["date"].iloc[13:], eps_clean, "-o", markersize=3)
    axes[0, 0].axhline(0, color="k", linewidth=0.5)
    axes[0, 0].set_title(r"Residuals $\hat\varepsilon_t$")
    axes[0, 0].set_xlabel("date")

    acf = sample_acf(eps_clean, n_lags_acf)
    lags = np.arange(n_lags_acf + 1)
    axes[0, 1].vlines(lags, 0, acf, linewidth=1.2)
    axes[0, 1].plot(lags, acf, "o", markersize=3)
    ci = acf_ci(n_res)
    axes[0, 1].axhline(ci, color="b", linestyle="--", linewidth=0.6)
    axes[0, 1].axhline(-ci, color="b", linestyle="--", linewidth=0.6)
    axes[0, 1].axhline(0, color="k", linewidth=0.4)
    axes[0, 1].set_title("Residual ACF")
    axes[0, 1].set_xlabel("lag")

    stats.probplot(eps_clean, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("QQ plot (Normal)")

    axes[1, 1].hist(eps_clean, bins=8, edgecolor="k")
    axes[1, 1].set_title("Residual histogram")
    axes[1, 1].set_xlabel(r"$\hat\varepsilon_t$")

    fig.tight_layout()
    savefig(fig, "p2_residual_diagnostics.pdf")

    # ============================================================
    # 2.2  forecast Y_hat for k = 1..12
    # ============================================================
    X_fc = forecast(X, K_MAX)
    Y_fc = np.exp(X_fc + MU)

    # ============================================================
    # 2.3  95% PI using AR(1) variance
    # ============================================================
    a = -PHI1  # = 0.38
    var_k = ar1_forecast_variance(K_MAX, a, SIGMA2)
    se_k = np.sqrt(var_k)
    lo_X = X_fc - 1.96 * se_k
    hi_X = X_fc + 1.96 * se_k
    lo_Y = np.exp(lo_X + MU)
    hi_Y = np.exp(hi_X + MU)

    last_date = df["date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1),
                                 periods=K_MAX, freq="MS")

    table = pd.DataFrame({
        "k": np.arange(1, K_MAX + 1),
        "date": future_dates.strftime("%Y-%m"),
        "X_hat": X_fc.round(4),
        "Y_hat_MWh": Y_fc.round(1),
        "PI_low": lo_Y.round(1),
        "PI_high": hi_Y.round(1),
        "SE_X": se_k.round(4),
    })
    print("\n2.2 / 2.3  12-month forecast:")
    print(table.to_string(index=False))
    table.to_csv(DATA_DIR / "plots" / "p2_forecast_table.csv", index=False)

    # ============================================================
    # plot: observed + forecast + PI band
    # ============================================================
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["date"], Y, "-o", color="C0", markersize=4, label="observed")
    ax.plot(future_dates, Y_fc, "--s", color="C3", markersize=5, label=r"$\hat Y_{t+k|t}$")
    ax.fill_between(future_dates, lo_Y, hi_Y, color="C3", alpha=0.2, label="95% PI (AR(1))")
    ax.set_xlabel("date")
    ax.set_ylabel("Generation (MWh)")
    ax.set_title("Monthly solar power: observed + 12-step forecast with 95% PI")
    ax.legend()
    savefig(fig, "p2_forecast.pdf")


if __name__ == "__main__":
    main()
