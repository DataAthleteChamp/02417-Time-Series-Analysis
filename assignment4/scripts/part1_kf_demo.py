"""Part 1 Q1.3: Kalman filter applied to the simulated series from Q1.2."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from common import save, residual_diag_fig, ljung_box
from kalman import kf_scalar


def main():
    data = np.load(Path(__file__).parent / "_p1_q12_data.npz")
    X, Y = data["X"], data["Y"]
    a, b, s1, s2, X0 = float(data["a"]), float(data["b"]), float(data["sigma1"]), float(data["sigma2"]), float(data["X0"])

    res = kf_scalar(Y, a, b, s1, R=s2 ** 2, x0=X0, P0=10.0)
    xp = res["x_pred"]
    Pp = res["P_pred"]
    ci = 1.96 * np.sqrt(Pp)
    t = np.arange(len(Y))

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.fill_between(t, xp - ci, xp + ci, alpha=0.25, color="C2",
                    label=r"95% CI for $\hat X_{t+1|t}$")
    ax.plot(t, X, color="C0", lw=1.2, label=r"true state $X_t$")
    ax.plot(t, Y, "o", color="C3", ms=3, alpha=0.6, label=r"observation $Y_t$")
    ax.plot(t, xp, color="C2", lw=1.3, label=r"prediction $\hat X_{t+1|t}$")
    ax.set_xlabel("t"); ax.set_ylabel("value")
    ax.set_title("Kalman filter tracking of the hidden state")
    ax.legend(ncol=2, fontsize=8)
    save(fig, "p1_kf_tracking")

    # Standardized innovations diagnostics (extra support, may not be in report)
    std_innov = res["innov"] / np.sqrt(res["S"])
    residual_diag_fig(std_innov, "Standardized innovations (Q1.3)", "p1_kf_diag")
    Q, p = ljung_box(std_innov, lags=10)
    print(f"logL={res['logLik']:.3f}  LB Q={Q:.2f} p={p:.3f}")


if __name__ == "__main__":
    main()
