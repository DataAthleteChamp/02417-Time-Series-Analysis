"""Assignment 3 — Part 3: ARX model for heating of a box.

Data: box_data_60min.csv (231 hourly obs, columns Ph, Tdelta, Gv + lags .l0..l10).
Split: first 167 rows = train; last 64 rows = test (matches RMSE divisor 1/64).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import sample_acf, sample_ccf, acf_ci, savefig, DATA_DIR
from scipy.stats import chi2

N_TRAIN = 167
N_TEST = 64
P_MAX = 10


# ------------------------------------------------------------
# data loading
# ------------------------------------------------------------
def load():
    df = pd.read_csv(DATA_DIR / "box_data_60min.csv")
    df["tdate"] = pd.to_datetime(df["tdate"])
    df = df.sort_values("thour").reset_index(drop=True)
    df["t"] = np.arange(1, len(df) + 1)
    return df


# ------------------------------------------------------------
# OLS helpers
# ------------------------------------------------------------
def fit_ols(y, X):
    """Returns dict with beta, yhat, resid, rss, n, k, aic, bic, sigma2."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    n, k = X.shape
    rss = float(resid @ resid)
    sigma2 = rss / n
    aic = n * np.log(sigma2) + 2 * k
    bic = n * np.log(sigma2) + k * np.log(n)
    # parameter standard errors: sigma_hat * sqrt(diag((X'X)^{-1}))
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
        se = float(np.sqrt(sigma2)) * np.sqrt(np.maximum(np.diag(xtx_inv), 0))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    return dict(beta=beta, se=se, yhat=yhat, resid=resid, rss=rss, n=n, k=k,
                aic=aic, bic=bic, sigma2=sigma2)


def build_arx_design(df, p):
    """Return (y, X, colnames) for ARX(p):
       Ph_t = -sum_{i=1..p} phi_i Ph_{t-i} + sum_{j=0..p-1} w1_j Tdelta_{t-j}
                                             + sum_{j=0..p-1} w2_j Gv_{t-j} + eps
       Rows: t starts at max(p, p) = p (need Ph.l1..lp and inputs l0..l(p-1)).
    """
    n = len(df)
    # Need indices t such that Ph.l{p} exists (i.e. t-p >= 1). For row-based lags
    # the CSV provides them precomputed; but because the CSV starts mid-series
    # the .l* columns are ALREADY aligned — just use them.
    Ph_lags = [df[f"Ph.l{i}"].to_numpy(dtype=float) for i in range(1, p + 1)]
    Td_lags = [df[f"Tdelta.l{j}"].to_numpy(dtype=float) for j in range(0, p)]
    Gv_lags = [df[f"Gv.l{j}"].to_numpy(dtype=float) for j in range(0, p)]
    cols = (
        [f"Ph.l{i}" for i in range(1, p + 1)]
        + [f"Tdelta.l{j}" for j in range(0, p)]
        + [f"Gv.l{j}" for j in range(0, p)]
    )
    X = np.column_stack(Ph_lags + Td_lags + Gv_lags)
    y = df["Ph"].to_numpy(dtype=float)
    # Sign convention: model has -phi_i on lags. Absorb by flipping sign of Ph-lag columns
    X[:, :p] = -X[:, :p]
    return y, X, cols


# ------------------------------------------------------------
# 3.1 time-series plot
# ------------------------------------------------------------
def plot_timeseries(df):
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(df["tdate"], df["Ph"], color="C3")
    axes[0].set_ylabel("Ph [W]")
    axes[1].plot(df["tdate"], df["Tdelta"], color="C0")
    axes[1].set_ylabel(r"$T_{\Delta}$ [°C]")
    axes[2].plot(df["tdate"], df["Gv"], color="C1")
    axes[2].set_ylabel(r"$G_v$ [W/m²]")
    axes[2].set_xlabel("date")
    axes[0].set_title("3.1  Observed time series (full period)")
    fig.tight_layout()
    savefig(fig, "p3_timeseries.pdf")


# ------------------------------------------------------------
# 3.3 EDA: scatter + ACF + CCF (train only)
# ------------------------------------------------------------
def plot_eda(train):
    Ph, Td, Gv = (train[c].to_numpy(dtype=float) for c in ["Ph", "Tdelta", "Gv"])
    n = len(Ph)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    axes[0, 0].scatter(Td, Ph, s=8, alpha=0.6)
    axes[0, 0].set_xlabel(r"$T_\Delta$"); axes[0, 0].set_ylabel("Ph")
    axes[0, 0].set_title(r"Ph vs $T_\Delta$")

    axes[0, 1].scatter(Gv, Ph, s=8, alpha=0.6, color="C1")
    axes[0, 1].set_xlabel(r"$G_v$"); axes[0, 1].set_ylabel("Ph")
    axes[0, 1].set_title(r"Ph vs $G_v$")

    axes[0, 2].scatter(Gv, Td, s=8, alpha=0.6, color="C2")
    axes[0, 2].set_xlabel(r"$G_v$"); axes[0, 2].set_ylabel(r"$T_\Delta$")
    axes[0, 2].set_title(r"$T_\Delta$ vs $G_v$")

    nlag = 30
    acf_Ph = sample_acf(Ph, nlag)
    lags = np.arange(nlag + 1)
    axes[1, 0].vlines(lags, 0, acf_Ph, linewidth=1.1)
    axes[1, 0].plot(lags, acf_Ph, "o", markersize=3)
    axes[1, 0].axhline(0, color="k", linewidth=0.4)
    ci = acf_ci(n)
    axes[1, 0].axhline(ci, color="b", linestyle="--", linewidth=0.6)
    axes[1, 0].axhline(-ci, color="b", linestyle="--", linewidth=0.6)
    axes[1, 0].set_title("ACF of Ph")
    axes[1, 0].set_xlabel("lag")

    lg, cc = sample_ccf(Td, Ph, 20)
    axes[1, 1].vlines(lg, 0, cc, linewidth=1.0)
    axes[1, 1].axhline(ci, color="b", linestyle="--", linewidth=0.6)
    axes[1, 1].axhline(-ci, color="b", linestyle="--", linewidth=0.6)
    axes[1, 1].axhline(0, color="k", linewidth=0.4)
    axes[1, 1].set_title(r"CCF($T_\Delta \to$ Ph)")
    axes[1, 1].set_xlabel("lag h")

    lg, cc = sample_ccf(Gv, Ph, 20)
    axes[1, 2].vlines(lg, 0, cc, linewidth=1.0, color="C1")
    axes[1, 2].axhline(ci, color="b", linestyle="--", linewidth=0.6)
    axes[1, 2].axhline(-ci, color="b", linestyle="--", linewidth=0.6)
    axes[1, 2].axhline(0, color="k", linewidth=0.4)
    axes[1, 2].set_title(r"CCF($G_v \to$ Ph)")
    axes[1, 2].set_xlabel("lag h")

    fig.tight_layout()
    savefig(fig, "p3_eda.pdf")


# ------------------------------------------------------------
# 3.4 impulse response (FIR OLS, lags 0..10)
# ------------------------------------------------------------
def plot_impulse(train, H=10):
    y = train["Ph"].to_numpy(dtype=float)
    cols_Td = [f"Tdelta.l{h}" for h in range(H + 1)]
    cols_Gv = [f"Gv.l{h}" for h in range(H + 1)]
    X = np.column_stack(
        [np.ones(len(train))]
        + [train[c].to_numpy(dtype=float) for c in cols_Td]
        + [train[c].to_numpy(dtype=float) for c in cols_Gv]
    )
    res = fit_ols(y, X)
    beta = res["beta"]
    b_Td = beta[1: H + 2]
    b_Gv = beta[H + 2:]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].stem(range(H + 1), b_Td, basefmt=" ")
    axes[0].set_title(r"Impulse response  $T_\Delta \to$ Ph")
    axes[0].set_xlabel("lag h"); axes[0].set_ylabel(r"$\hat\beta_h$")
    axes[0].axhline(0, color="k", linewidth=0.4)

    axes[1].stem(range(H + 1), b_Gv, basefmt=" ", linefmt="C1-", markerfmt="C1o")
    axes[1].set_title(r"Impulse response  $G_v \to$ Ph")
    axes[1].set_xlabel("lag h"); axes[1].set_ylabel(r"$\hat\beta_h$")
    axes[1].axhline(0, color="k", linewidth=0.4)

    fig.tight_layout()
    savefig(fig, "p3_impulse.pdf")

    print("\n3.4 Impulse responses")
    print("  h  Tdelta->Ph   Gv->Ph")
    for h in range(H + 1):
        print(f"  {h:2d}  {b_Td[h]:+.3f}     {b_Gv[h]:+.4f}")


# ------------------------------------------------------------
# 3.3b pre-whitened CCF (Box-Jenkins identification)
# ------------------------------------------------------------
def fit_ar_aic(x, p_max=10):
    """Fit AR(p) by OLS for p = 1..p_max and return best (by AIC)."""
    x = np.asarray(x, dtype=float)
    n_tot = len(x)
    best = None
    for p in range(1, p_max + 1):
        y = x[p:]
        X = np.column_stack([x[p - 1 - i:n_tot - 1 - i] for i in range(p)])
        res = fit_ols(y, X)
        if best is None or res["aic"] < best["aic"]:
            res["p"] = p
            best = res
    return best


def ar_filter(x, phi):
    """Apply pi(B) = 1 - phi_1 B - ... - phi_p B^p to x.
    Returns residuals of length len(x) - p (aligned to t = p, p+1, ...)."""
    x = np.asarray(x, dtype=float)
    p = len(phi)
    n = len(x)
    out = np.empty(n - p)
    for t in range(p, n):
        out[t - p] = x[t] - sum(phi[i] * x[t - 1 - i] for i in range(p))
    return out


def plot_prewhitened_ccf(train, p_max=10, H=20):
    """For each input x ∈ {Tdelta, Gv}:
       1. Fit AR(p_x) to x with AIC order selection.
       2. Filter both x and Ph by the fitted pi(B).
       3. Plot CCF of filtered series with 95% band.
    """
    Ph = train["Ph"].to_numpy(dtype=float)
    inputs = {"Tdelta": train["Tdelta"].to_numpy(dtype=float),
              "Gv": train["Gv"].to_numpy(dtype=float)}

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    summary = {}
    for ax, (name, x) in zip(axes, inputs.items()):
        ar = fit_ar_aic(x, p_max=p_max)
        phi = ar["beta"]
        u_tilde = ar_filter(x, phi)
        v_tilde = ar_filter(Ph, phi)
        lags, cc = sample_ccf(u_tilde, v_tilde, H)
        ci = acf_ci(len(u_tilde))
        peak_lag = int(lags[np.argmax(np.abs(cc))])
        peak_val = float(cc[np.argmax(np.abs(cc))])
        summary[name] = dict(p=ar["p"], peak_lag=peak_lag, peak_val=peak_val)

        color = "C0" if name == "Tdelta" else "C1"
        ax.vlines(lags, 0, cc, linewidth=1.0, color=color)
        ax.axhline(ci, color="b", linestyle="--", linewidth=0.6)
        ax.axhline(-ci, color="b", linestyle="--", linewidth=0.6)
        ax.axhline(0, color="k", linewidth=0.4)
        label = r"$T_\Delta$" if name == "Tdelta" else r"$G_v$"
        ax.set_title(f"Pre-whitened CCF ({label} → Ph),  AR({ar['p']})")
        ax.set_xlabel("lag h")

    fig.tight_layout()
    savefig(fig, "p3_prewhitened_ccf.pdf")

    print("\n3.3b  Pre-whitened CCF summary")
    for name, s in summary.items():
        print(f"  {name:<7s}  AR order p={s['p']}  peak at lag {s['peak_lag']:+d} "
              f"(|ccf|={abs(s['peak_val']):.3f})")
    return summary


# ------------------------------------------------------------
# 3.9a residual diagnostics for the final ARX(p*) model
# ------------------------------------------------------------
def ljung_box(resid, lags_list=(10, 20)):
    """Return list of (lag, Q, p-value). Uses biased ACF as in course book."""
    r = np.asarray(resid, dtype=float)
    n = len(r)
    acf = sample_acf(r, max(lags_list))
    out = []
    for L in lags_list:
        Q = n * (n + 2) * float(np.sum(acf[1:L + 1] ** 2 / (n - np.arange(1, L + 1))))
        p = float(chi2.sf(Q, df=L))
        out.append((L, Q, p))
    return out


def residual_analysis_arx(train, p):
    """Fit ARX(p), generate residual panel, run Ljung–Box tests."""
    y, X, cols = build_arx_design(train, p)
    res = fit_ols(y, X)
    residual_panel(train, res["resid"],
                   f"3.9  Residuals of ARX({p}) (final model)",
                   f"p3_arx{p}_resid.pdf")
    lb = ljung_box(res["resid"], lags_list=(10, 20))
    print(f"\n3.9a  ARX({p}) residual diagnostics")
    print(f"  sigma_train = {np.sqrt(res['sigma2']):.3f}")
    for L, Q, p_val in lb:
        print(f"  Ljung-Box  Q({L}) = {Q:6.2f}    p = {p_val:.3f}")
    # max |CCF| with inputs over lags -20..+20
    for name in ("Tdelta", "Gv"):
        _, cc = sample_ccf(train[name].to_numpy(dtype=float), res["resid"], 20)
        print(f"  max |CCF({name}, resid)| within ±20 = {np.max(np.abs(cc)):.3f}")
    return res, lb


# ------------------------------------------------------------
# 3.5 static OLS + 3.6 ARX(1)  residual analysis
# ------------------------------------------------------------
def residual_panel(train, resid, title, filename):
    n = len(resid)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes[0, 0].plot(train["tdate"], resid)
    axes[0, 0].axhline(0, color="k", linewidth=0.4)
    axes[0, 0].set_title("Residuals over time")

    nlag = 30
    acf = sample_acf(resid, nlag)
    lags = np.arange(nlag + 1)
    ci = acf_ci(n)
    axes[0, 1].vlines(lags, 0, acf, linewidth=1.0)
    axes[0, 1].axhline(ci, color="b", linestyle="--", linewidth=0.6)
    axes[0, 1].axhline(-ci, color="b", linestyle="--", linewidth=0.6)
    axes[0, 1].axhline(0, color="k", linewidth=0.4)
    axes[0, 1].set_title("ACF of residuals")
    axes[0, 1].set_xlabel("lag")

    lg, cc = sample_ccf(train["Tdelta"].to_numpy(dtype=float), resid, 20)
    axes[1, 0].vlines(lg, 0, cc, linewidth=1.0)
    axes[1, 0].axhline(ci, color="b", linestyle="--", linewidth=0.6)
    axes[1, 0].axhline(-ci, color="b", linestyle="--", linewidth=0.6)
    axes[1, 0].axhline(0, color="k", linewidth=0.4)
    axes[1, 0].set_title(r"CCF($T_\Delta$, resid)")
    axes[1, 0].set_xlabel("lag h")

    lg, cc = sample_ccf(train["Gv"].to_numpy(dtype=float), resid, 20)
    axes[1, 1].vlines(lg, 0, cc, linewidth=1.0, color="C1")
    axes[1, 1].axhline(ci, color="b", linestyle="--", linewidth=0.6)
    axes[1, 1].axhline(-ci, color="b", linestyle="--", linewidth=0.6)
    axes[1, 1].axhline(0, color="k", linewidth=0.4)
    axes[1, 1].set_title(r"CCF($G_v$, resid)")
    axes[1, 1].set_xlabel("lag h")

    fig.suptitle(title)
    fig.tight_layout()
    savefig(fig, filename)


def fit_static(train):
    y = train["Ph"].to_numpy(dtype=float)
    X = np.column_stack([
        train["Tdelta"].to_numpy(dtype=float),
        train["Gv"].to_numpy(dtype=float),
    ])
    res = fit_ols(y, X)
    print("\n3.5 Static regression  Ph = ω1 Tdelta + ω2 Gv + eps")
    print(f"  ω1 (Tdelta) = {res['beta'][0]:+.4f}   SE = {res['se'][0]:.4f}")
    print(f"  ω2 (Gv)     = {res['beta'][1]:+.4f}   SE = {res['se'][1]:.4f}")
    print(f"  sigma       = {np.sqrt(res['sigma2']):.3f},   RSS = {res['rss']:.1f}")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    print(f"  R²          = {1 - res['rss'] / ss_tot:.3f}")
    residual_panel(train, res["resid"], "3.5 Static regression residuals",
                   "p3_static_resid.pdf")
    return res


def fit_arx1(train):
    y, X, cols = build_arx_design(train, p=1)
    res = fit_ols(y, X)
    print("\n3.6 ARX(1)")
    for c, b, s in zip(cols, res["beta"], res["se"]):
        print(f"  {c:<12s}  {b:+.4f}   SE = {s:.4f}")
    print(f"  sigma       = {np.sqrt(res['sigma2']):.3f},   RSS = {res['rss']:.1f}")
    print(f"  AIC = {res['aic']:.2f}, BIC = {res['bic']:.2f}")
    residual_panel(train, res["resid"], "3.6 ARX(1) residuals", "p3_arx1_resid.pdf")
    return res


# ------------------------------------------------------------
# 3.7 model order search
# ------------------------------------------------------------
def order_search(train, test):
    results = []
    for p in range(1, P_MAX + 1):
        y_tr, X_tr, _ = build_arx_design(train, p)
        fit = fit_ols(y_tr, X_tr)

        # one-step predictions on test: use TRUE lagged Ph (no simulation)
        y_te, X_te, _ = build_arx_design(test, p)
        yhat_te = X_te @ fit["beta"]
        eps_te = y_te - yhat_te
        rmse = float(np.sqrt(np.mean(eps_te ** 2)))

        results.append({"p": p, "k": fit["k"], "AIC": fit["aic"], "BIC": fit["bic"],
                        "sigma_train": np.sqrt(fit["sigma2"]), "RMSE_test": rmse})
    res_df = pd.DataFrame(results)
    print("\n3.7/3.8 Model order search")
    print(res_df.to_string(index=False))
    res_df.to_csv(DATA_DIR / "plots" / "p3_order_table.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(res_df["p"], res_df["AIC"], "-o", label="AIC")
    axes[0].plot(res_df["p"], res_df["BIC"], "-s", label="BIC")
    axes[0].set_xlabel("ARX order p"); axes[0].set_ylabel("IC")
    axes[0].set_title("AIC / BIC vs ARX order")
    axes[0].legend()

    axes[1].plot(res_df["p"], res_df["RMSE_test"], "-o", color="C3")
    axes[1].set_xlabel("ARX order p"); axes[1].set_ylabel("test RMSE")
    axes[1].set_title("One-step test RMSE vs ARX order")

    fig.tight_layout()
    savefig(fig, "p3_order_selection.pdf")
    return res_df


# ------------------------------------------------------------
# 3.9 multi-step simulation using chosen order
# ------------------------------------------------------------
def multistep_simulation(df, p_star):
    """Fit on train, simulate through entire period using observed inputs
    and recursively computed Ph lags."""
    train = df.iloc[:N_TRAIN].copy()
    y_tr, X_tr, _ = build_arx_design(train, p_star)
    fit = fit_ols(y_tr, X_tr)
    beta = fit["beta"]
    # layout: [-Ph.l1 .. -Ph.lp, Tdelta.l0..l(p-1), Gv.l0..l(p-1)]
    phi = -beta[:p_star]                          # positive coefs on Ph_{t-i}
    w1 = beta[p_star: 2 * p_star]
    w2 = beta[2 * p_star: 3 * p_star]

    Ph_obs = df["Ph"].to_numpy(dtype=float)
    Td = df["Tdelta"].to_numpy(dtype=float)
    Gv = df["Gv"].to_numpy(dtype=float)
    n = len(df)

    Ph_sim = np.empty(n)
    Ph_sim[:p_star] = Ph_obs[:p_star]  # warm-up with observed
    for t in range(p_star, n):
        ar_part = sum(phi[i] * Ph_sim[t - 1 - i] for i in range(p_star))
        inp_part = sum(w1[j] * Td[t - j] + w2[j] * Gv[t - j]
                       for j in range(p_star))
        Ph_sim[t] = ar_part + inp_part

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["tdate"], Ph_obs, label="observed", color="k", linewidth=1.0)
    ax.plot(df["tdate"], Ph_sim, label=f"simulated ARX({p_star})",
            color="C3", linewidth=1.0)
    ax.axvline(df["tdate"].iloc[N_TRAIN - 1], color="gray", linestyle="--",
               label="train/test split")
    ax.set_ylabel("Ph [W]"); ax.set_xlabel("date")
    ax.set_title(f"3.9  Multi-step simulation (ARX({p_star}), inputs observed)")
    ax.legend()
    savefig(fig, "p3_multistep_sim.pdf")

    rmse_train = float(np.sqrt(np.mean((Ph_obs[p_star:N_TRAIN] - Ph_sim[p_star:N_TRAIN]) ** 2)))
    rmse_test = float(np.sqrt(np.mean((Ph_obs[N_TRAIN:] - Ph_sim[N_TRAIN:]) ** 2)))
    print(f"\n3.9 Multi-step simulation RMSE (p={p_star}):")
    print(f"  train RMSE = {rmse_train:.2f} W")
    print(f"  test  RMSE = {rmse_test:.2f} W")


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    df = load()
    print(f"Loaded {len(df)} rows;  first={df['tdate'].iloc[0]}  last={df['tdate'].iloc[-1]}")
    print(f"Training row 167 tdate = {df['tdate'].iloc[N_TRAIN - 1]} (should be 2013-02-06 00:00)")
    train = df.iloc[:N_TRAIN].reset_index(drop=True)
    test = df.iloc[N_TRAIN:].reset_index(drop=True)

    plot_timeseries(df)       # 3.1
    plot_eda(train)           # 3.3
    plot_prewhitened_ccf(train)  # 3.3b
    plot_impulse(train)       # 3.4
    fit_static(train)         # 3.5
    fit_arx1(train)           # 3.6
    res_df = order_search(train, test)  # 3.7 / 3.8

    # pick order: min BIC
    p_star = int(res_df.loc[res_df["BIC"].idxmin(), "p"])
    print(f"\nChosen ARX order by BIC: p* = {p_star}")
    p_rmse = int(res_df.loc[res_df["RMSE_test"].idxmin(), "p"])
    print(f"(RMSE-optimal order: p = {p_rmse})")

    residual_analysis_arx(train, p_star)  # 3.9a
    multistep_simulation(df, p_star)      # 3.9


if __name__ == "__main__":
    main()
