"""Part 2 Q2.2: 1-D state-space model for transformer temperature."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from common import save, residual_diag_fig, ljung_box, write_table
from kalman import kf_mv

INPUT_COLS = ["Ta", "S", "I"]
OBS_COL = "Y"


def unpack_1d(par):
    # par = [A, B1, B2, B3, log_sigma_x, log_sigma_y]
    A = np.array([[par[0]]])
    B = np.array([[par[1], par[2], par[3]]])
    C = np.array([[1.0]])
    Q = np.array([[np.exp(par[4]) ** 2]])
    R = np.array([[np.exp(par[5]) ** 2]])
    return A, B, C, Q, R


def nll_1d(par, Y, U, x0, P0):
    A, B, C, Q, R = unpack_1d(par)
    res = kf_mv(Y, U, A, B, C, Q, R, x0, P0)
    if not np.isfinite(res["logLik"]):
        return 1e10
    return -res["logLik"]


def fit(df, start=None):
    Y = df[[OBS_COL]].values
    U = df[INPUT_COLS].values
    x0 = np.array([Y[0, 0]])
    P0 = np.array([[10.0]])

    if start is None:
        # signs: Ta (+ weak), S (+ weak), I (+) tends to heat transformer
        start = [0.95, 0.02, 0.0005, 0.1, np.log(0.5), np.log(0.3)]
    start = np.asarray(start, float)

    # stage 1: Nelder-Mead
    r1 = minimize(nll_1d, start, args=(Y, U, x0, P0),
                  method="Nelder-Mead",
                  options=dict(xatol=1e-6, fatol=1e-6, maxiter=8000))
    # stage 2: L-BFGS-B with lower bound on sigma_y to avoid Q/R
    # identifiability boundary (the unconstrained fit collapses sigma_y to
    # essentially zero and dumps all variance into Q).
    r2 = minimize(nll_1d, r1.x, args=(Y, U, x0, P0),
                  method="L-BFGS-B",
                  bounds=[(0.0, 1.2), (-5, 5), (-0.1, 0.1), (-5, 5),
                          (-6, 3), (np.log(0.1), 3)])
    par = r2.x
    A, B, C, Q, R = unpack_1d(par)
    result = kf_mv(Y, U, A, B, C, Q, R, x0, P0)
    return dict(par=par, A=A, B=B, Q=Q, R=R, res=result, nll=r2.fun,
                Y=Y, U=U, x0=x0, P0=P0, opt=r2)


def main():
    df = pd.read_pickle(Path(__file__).parent / "_p2_df.pkl")
    out = fit(df)
    par = out["par"]
    logL = -out["nll"]
    N = len(df)
    k = len(par)
    AIC = 2 * k - 2 * logL
    BIC = k * np.log(N) - 2 * logL

    print("\n=== 1-D state-space estimates ===")
    print(f"A  = {out['A'][0,0]:+.5f}")
    print(f"B  = [{out['B'][0,0]:+.5f} (Ta), {out['B'][0,1]:+.6f} (S), {out['B'][0,2]:+.5f} (I)]")
    print(f"sigma_x = {np.sqrt(out['Q'][0,0]):.4f}")
    print(f"sigma_y = {np.sqrt(out['R'][0,0]):.4f}")
    print(f"logL={logL:.2f}  AIC={AIC:.2f}  BIC={BIC:.2f}")

    res = out["res"]
    innov = res["innov"][:, 0]
    S = res["S"][:, 0, 0]
    std_innov = innov / np.sqrt(S)
    Q_lb, p_lb = ljung_box(std_innov, lags=10)
    print(f"Ljung-Box(10): Q={Q_lb:.2f}  p={p_lb:.4f}")

    residual_diag_fig(std_innov, "1-D SS residual diagnostics", "p2_1d_diag")

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    t = df["time"].values
    ax.plot(t, df["Y"], "o", ms=3, color="C3", alpha=0.6, label=r"$Y_t$")
    ax.plot(t, res["x_pred"][:, 0], color="C2", lw=1.2, label=r"$\hat Y_{t|t-1}$")
    ax.fill_between(t, res["x_pred"][:, 0] - 1.96 * np.sqrt(S),
                    res["x_pred"][:, 0] + 1.96 * np.sqrt(S),
                    color="C2", alpha=0.2, label="95% CI")
    ax.set_xlabel("time [h]"); ax.set_ylabel(r"$Y_t$ [$^\circ$C]")
    ax.set_title("1-D SS model 1-step predictions")
    ax.legend(fontsize=8)
    save(fig, "p2_1d_pred")

    # -- LaTeX param table
    tex = [
        r"\begin{tabular}{l r}",
        r"\toprule",
        r"Parameter & Estimate \\",
        r"\midrule",
        rf"$A$ & {out['A'][0,0]:+.4f} \\",
        rf"$B_{{T_a}}$ & {out['B'][0,0]:+.4f} \\",
        rf"$B_{{S}}$ & {out['B'][0,1]:+.6f} \\",
        rf"$B_{{I}}$ & {out['B'][0,2]:+.4f} \\",
        rf"$\sigma_x$ & {np.sqrt(out['Q'][0,0]):.4f} \\",
        rf"$\sigma_y$ & {np.sqrt(out['R'][0,0]):.4f} \\",
        r"\midrule",
        rf"$\log L$ & {logL:.2f} \\",
        rf"AIC & {AIC:.2f} \\",
        rf"BIC & {BIC:.2f} \\",
        rf"Ljung--Box $p$ (lag 10) & {p_lb:.3f} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write_table("p2_1d.tex", "\n".join(tex))
    np.savez("_p2_1d_fit.npz", par=par, logL=logL, AIC=AIC, BIC=BIC)


if __name__ == "__main__":
    main()
