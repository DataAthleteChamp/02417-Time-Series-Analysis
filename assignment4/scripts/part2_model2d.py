"""Part 2 Q2.3: 2-D state-space model for transformer temperature."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from common import save, residual_diag_fig, ljung_box, write_table
from kalman import kf_mv, build_chol_lower

INPUT_COLS = ["Ta", "S", "I"]
OBS_COL = "Y"

# Parameter layout (16 params):
#   A  [0:4]   = a11, a12, a21, a22
#   B  [4:10]  = b11, b12, b13, b21, b22, b23
#   Qlt[10:13] = q11, q21, q22 (lower triangular; Q = Qlt Qlt^T)
#   logR[13]
#   x0 [14:16] = x0_1, x0_2


def unpack_2d(par):
    A  = np.array([[par[0], par[1]],
                   [par[2], par[3]]])
    B  = np.array([[par[4], par[5], par[6]],
                   [par[7], par[8], par[9]]])
    Qlt = build_chol_lower(par[10:13], 2)
    Q   = Qlt @ Qlt.T
    C   = np.array([[1.0, 0.0]])
    R   = np.array([[np.exp(par[13]) ** 2]])
    x0  = np.array([par[14], par[15]])
    return A, B, C, Q, R, x0


def nll_2d(par, Y, U, P0):
    A, B, C, Q, R, x0 = unpack_2d(par)
    res = kf_mv(Y, U, A, B, C, Q, R, x0, P0)
    if not np.isfinite(res["logLik"]):
        return 1e10
    return -res["logLik"]


def fit(df, start, P0):
    Y = df[[OBS_COL]].values
    U = df[INPUT_COLS].values
    r1 = minimize(nll_2d, start, args=(Y, U, P0),
                  method="Nelder-Mead",
                  options=dict(xatol=1e-4, fatol=1e-4, maxiter=3000, adaptive=True))
    r2 = minimize(nll_2d, r1.x, args=(Y, U, P0),
                  method="L-BFGS-B",
                  options=dict(maxiter=500))
    return r2, Y, U


def multi_start_fit(df, n_starts=3, seed=123):
    rng = np.random.default_rng(seed)
    P0 = np.eye(2) * 10.0
    y0 = df[OBS_COL].iloc[0]

    base_starts = [
        # near-identity A, small B from 1-D fit, moderate noise
        [0.9, 0.0, 0.0, 0.8,
         0.05, 0.001, 0.1, 0.05, 0.001, 0.05,
         0.5, 0.0, 0.5,
         np.log(0.3),
         y0, y0],
        # coupled states
        [0.8, 0.1, 0.1, 0.7,
         0.05, 0.001, 0.15, 0.02, 0.0005, 0.05,
         0.3, 0.1, 0.3,
         np.log(0.3),
         y0, y0 * 0.9],
        # slow + fast mode
        [0.6, 0.2, 0.1, 0.95,
         0.1, 0.002, 0.2, 0.02, 0.0005, 0.02,
         0.3, 0.0, 0.2,
         np.log(0.3),
         y0, y0],
    ]

    best = None
    for i, s in enumerate(base_starts):
        try:
            r, Y, U = fit(df, np.asarray(s, float), P0)
            print(f"  start {i}: nll={r.fun:.3f}")
            if best is None or r.fun < best[0].fun:
                best = (r, Y, U)
        except Exception as e:
            print(f"  start {i} failed: {e}")

    # random perturbations of best so far
    for i in range(n_starts):
        s = best[0].x + 0.1 * rng.standard_normal(len(best[0].x))
        try:
            r, Y, U = fit(df, s, P0)
            print(f"  perturb {i}: nll={r.fun:.3f}")
            if r.fun < best[0].fun:
                best = (r, Y, U)
        except Exception:
            pass
    return best, P0


def main():
    df = pd.read_pickle(Path(__file__).parent / "_p2_df.pkl")
    (best, P0) = multi_start_fit(df)
    r, Y, U = best
    par = r.x
    A, B, C, Q, R, x0 = unpack_2d(par)
    res = kf_mv(Y, U, A, B, C, Q, R, x0, P0)
    logL = -r.fun
    N = len(df)
    k = len(par)
    AIC = 2 * k - 2 * logL
    BIC = k * np.log(N) - 2 * logL
    eig = np.linalg.eigvals(A)

    print("\n=== 2-D state-space estimates ===")
    print("A =\n", np.round(A, 4))
    print("B =\n", np.round(B, 5))
    print("Q =\n", np.round(Q, 5))
    print("R =", float(R[0, 0]))
    print("x0 =", x0)
    print(f"eig(A) = {eig}")
    print(f"logL={logL:.2f}  AIC={AIC:.2f}  BIC={BIC:.2f}")

    innov = res["innov"][:, 0]
    S = res["S"][:, 0, 0]
    std_innov = innov / np.sqrt(S)
    Q_lb, p_lb = ljung_box(std_innov, lags=10)
    print(f"Ljung-Box(10): Q={Q_lb:.2f}  p={p_lb:.4f}")

    residual_diag_fig(std_innov, "2-D SS residual diagnostics", "p2_2d_diag")

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    t = df["time"].values
    ax.plot(t, df["Y"], "o", ms=3, color="C3", alpha=0.6, label=r"$Y_t$")
    ax.plot(t, res["x_pred"][:, 0] * 0 + (C @ res["x_pred"].T).ravel(),
            color="C2", lw=1.2, label=r"$\hat Y_{t|t-1}$")
    ax.fill_between(t, (C @ res["x_pred"].T).ravel() - 1.96 * np.sqrt(S),
                    (C @ res["x_pred"].T).ravel() + 1.96 * np.sqrt(S),
                    color="C2", alpha=0.2, label="95% CI")
    ax.set_xlabel("time [h]"); ax.set_ylabel(r"$Y_t$ [$^\circ$C]")
    ax.set_title("2-D SS model 1-step predictions")
    ax.legend(fontsize=8)
    save(fig, "p2_2d_pred")

    tex = [
        r"\begin{tabular}{l r r}",
        r"\toprule",
        r"Parameter & Row 1 (state 1) & Row 2 (state 2) \\",
        r"\midrule",
        rf"$A$ (col 1) & {A[0,0]:+.4f} & {A[1,0]:+.4f} \\",
        rf"$A$ (col 2) & {A[0,1]:+.4f} & {A[1,1]:+.4f} \\",
        rf"$B_{{T_a}}$ & {B[0,0]:+.4f} & {B[1,0]:+.4f} \\",
        rf"$B_{{S}}$ & {B[0,1]:+.6f} & {B[1,1]:+.6f} \\",
        rf"$B_{{I}}$ & {B[0,2]:+.4f} & {B[1,2]:+.4f} \\",
        rf"$x_0$ & {x0[0]:+.3f} & {x0[1]:+.3f} \\",
        r"\midrule",
        rf"$Q_{{11}}$ & \multicolumn{{2}}{{r}}{{{Q[0,0]:.4f}}} \\",
        rf"$Q_{{22}}$ & \multicolumn{{2}}{{r}}{{{Q[1,1]:.4f}}} \\",
        rf"$Q_{{12}}$ & \multicolumn{{2}}{{r}}{{{Q[0,1]:.4f}}} \\",
        rf"$\sigma_y$ & \multicolumn{{2}}{{r}}{{{np.sqrt(R[0,0]):.4f}}} \\",
        rf"eig$(A)$ & \multicolumn{{2}}{{r}}{{{eig[0].real:.4f},\; {eig[1].real:.4f}}} \\",
        rf"$\log L$ & \multicolumn{{2}}{{r}}{{{logL:.2f}}} \\",
        rf"AIC & \multicolumn{{2}}{{r}}{{{AIC:.2f}}} \\",
        rf"BIC & \multicolumn{{2}}{{r}}{{{BIC:.2f}}} \\",
        rf"Ljung--Box $p$ (lag 10) & \multicolumn{{2}}{{r}}{{{p_lb:.3f}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write_table("p2_2d.tex", "\n".join(tex))
    np.savez("_p2_2d_fit.npz", par=par, A=A, B=B, Q=Q, R=R, x0=x0,
             x_filt=res["x_filt"], x_pred=res["x_pred"],
             logL=logL, AIC=AIC, BIC=BIC)


if __name__ == "__main__":
    main()
