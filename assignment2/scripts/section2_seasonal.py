"""
Section 2: Simulating seasonal processes.

Multiplicative seasonal ARMA:  φ(B) Φ(B^s) Y_t = θ(B) Θ(B^s) ε_t

Convention: polynomials use PLUS signs, matching scipy.signal.lfilter directly.
  lfilter(b, a, x) solves  a[0]y[n] + a[1]y[n-1] + ... = b[0]x[n] + b[1]x[n-1] + ...
"""

import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── Constants ──────────────────────────────────────────────
N_SIM = 300
N_BURNIN = 500
N_LAGS_ACF = 50
SEED = 42
SIGMA = 1.0
S = 12  # seasonal period


# ── Helper functions ──────────────────────────────────────

def expand_seasonal_poly(poly, s):
    """Expand a seasonal polynomial from B^s domain to B domain.

    E.g. [1, Φ₁] with s=12  →  array of length 13 with coeff at index 0 and 12.
    """
    full = np.zeros(1 + s * (len(poly) - 1))
    for i, c in enumerate(poly):
        full[i * s] = c
    return full


def build_combined_poly(nonseasonal, seasonal, s=12):
    """Multiply non-seasonal and seasonal polynomials via convolution."""
    seasonal_expanded = expand_seasonal_poly(seasonal, s)
    return np.convolve(nonseasonal, seasonal_expanded)


def simulate_arma(ar_poly, ma_poly, n, n_burnin, sigma, rng):
    """Simulate ARMA process using lfilter.

    ar_poly, ma_poly: coefficient arrays in assignment convention (plus signs).
    """
    total = n + n_burnin
    eps = rng.normal(0, sigma, total)
    y = lfilter(ma_poly, ar_poly, eps)
    return y[n_burnin:]


def compute_sample_acf(x, n_lags):
    """Biased sample ACF estimator (matches R's acf())."""
    n = len(x)
    xc = x - x.mean()
    gamma0 = np.sum(xc ** 2) / n
    acf = np.zeros(n_lags + 1)
    acf[0] = 1.0
    for k in range(1, n_lags + 1):
        acf[k] = np.sum(xc[:n - k] * xc[k:]) / n / gamma0
    return acf


def compute_sample_pacf(x, n_lags):
    """PACF via Durbin-Levinson recursion."""
    acf = compute_sample_acf(x, n_lags)
    pacf = np.zeros(n_lags + 1)
    pacf[0] = 1.0

    # Durbin-Levinson
    # phi_prev[j] = φ_{k-1,j}
    phi_prev = np.zeros(n_lags + 1)

    for k in range(1, n_lags + 1):
        # Compute φ_{k,k}
        num = acf[k]
        for j in range(1, k):
            num -= phi_prev[j] * acf[k - j]
        denom = 1.0
        for j in range(1, k):
            denom -= phi_prev[j] * acf[j]
        if abs(denom) < 1e-15:
            pacf[k] = 0.0
            break
        phi_kk = num / denom
        pacf[k] = phi_kk

        # Update φ coefficients
        phi_new = np.zeros(n_lags + 1)
        phi_new[k] = phi_kk
        for j in range(1, k):
            phi_new[j] = phi_prev[j] - phi_kk * phi_prev[k - j]
        phi_prev = phi_new

    return pacf


def plot_three_panel(x, acf_vals, pacf_vals, title, filename, n_lags_plot=50):
    """Three-panel plot: time series (top), ACF (bottom-left), PACF (bottom-right)."""
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30)

    n = len(x)
    ci = 1.96 / np.sqrt(n)

    # Top panel — time series
    ax_ts = fig.add_subplot(gs[0, :])
    ax_ts.plot(np.arange(n), x, linewidth=0.6, color="C0")
    ax_ts.set_xlabel("Time")
    ax_ts.set_ylabel("$Y_t$")
    ax_ts.set_title(title)

    # Bottom-left — ACF
    ax_acf = fig.add_subplot(gs[1, 0])
    lags = np.arange(0, n_lags_plot + 1)
    ax_acf.vlines(lags, 0, acf_vals[: n_lags_plot + 1], linewidth=1.2)
    ax_acf.plot(lags, acf_vals[: n_lags_plot + 1], "o", markersize=2.5, color="C0")
    ax_acf.axhline(0, color="k", linewidth=0.5)
    ax_acf.axhline(ci, color="b", linestyle="--", linewidth=0.7)
    ax_acf.axhline(-ci, color="b", linestyle="--", linewidth=0.7)
    ax_acf.set_xlabel("Lag")
    ax_acf.set_ylabel("ACF")
    ax_acf.set_xlim(-0.5, n_lags_plot + 0.5)

    # Bottom-right — PACF (skip lag 0)
    ax_pacf = fig.add_subplot(gs[1, 1])
    lags_p = np.arange(1, n_lags_plot + 1)
    ax_pacf.vlines(lags_p, 0, pacf_vals[1: n_lags_plot + 1], linewidth=1.2)
    ax_pacf.plot(lags_p, pacf_vals[1: n_lags_plot + 1], "o", markersize=2.5, color="C0")
    ax_pacf.axhline(0, color="k", linewidth=0.5)
    ax_pacf.axhline(ci, color="b", linestyle="--", linewidth=0.7)
    ax_pacf.axhline(-ci, color="b", linestyle="--", linewidth=0.7)
    ax_pacf.set_xlabel("Lag")
    ax_pacf.set_ylabel("PACF")
    ax_pacf.set_xlim(0.5, n_lags_plot + 0.5)

    fig.savefig(filename, bbox_inches="tight")
    print(f"  Saved → {filename}")
    plt.close(fig)


# ── Model definitions ─────────────────────────────────────
# Each model: (label, ar_nonseasonal, ar_seasonal, ma_nonseasonal, ma_seasonal)
models = [
    (
        "2.1",
        r"$(1,0,0)\times(0,0,0)_{12}$, $\phi_1=0.6$",
        [1, 0.6],   # ar nonseasonal
        [1],         # ar seasonal
        [1],         # ma nonseasonal
        [1],         # ma seasonal
    ),
    (
        "2.2",
        r"$(0,0,0)\times(1,0,0)_{12}$, $\Phi_1=-0.9$",
        [1],
        [1, -0.9],
        [1],
        [1],
    ),
    (
        "2.3",
        r"$(1,0,0)\times(0,0,1)_{12}$, $\phi_1=0.9$, $\Theta_1=-0.7$",
        [1, 0.9],
        [1],
        [1],
        [1, -0.7],
    ),
    (
        "2.4",
        r"$(1,0,0)\times(1,0,0)_{12}$, $\phi_1=-0.6$, $\Phi_1=-0.8$",
        [1, -0.6],
        [1, -0.8],
        [1],
        [1],
    ),
    (
        "2.5",
        r"$(0,0,1)\times(0,0,1)_{12}$, $\theta_1=0.4$, $\Theta_1=-0.8$",
        [1],
        [1],
        [1, 0.4],
        [1, -0.8],
    ),
    (
        "2.6",
        r"$(0,0,1)\times(1,0,0)_{12}$, $\theta_1=-0.4$, $\Phi_1=0.7$",
        [1],
        [1, 0.7],
        [1, -0.4],
        [1],
    ),
]


# ── Main loop ─────────────────────────────────────────────
rng = np.random.default_rng(SEED)

for num, label, ar_ns, ar_s, ma_ns, ma_s in models:
    print(f"Model {num}: {label}")

    ar_poly = build_combined_poly(ar_ns, ar_s, S)
    ma_poly = build_combined_poly(ma_ns, ma_s, S)
    print(f"  AR poly (len {len(ar_poly)}): {ar_poly}")
    print(f"  MA poly (len {len(ma_poly)}): {ma_poly}")

    y = simulate_arma(ar_poly, ma_poly, N_SIM, N_BURNIN, SIGMA, rng)
    acf_vals = compute_sample_acf(y, N_LAGS_ACF)
    pacf_vals = compute_sample_pacf(y, N_LAGS_ACF)

    print(f"  Sample ACF[1] = {acf_vals[1]:.4f}")
    if N_LAGS_ACF >= 12:
        print(f"  Sample ACF[12] = {acf_vals[12]:.4f}")

    title = f"Model {num}: {label}"
    fname = PLOT_DIR / f"sec2_model_{num.replace('.', '_')}.pdf"
    plot_three_panel(y, acf_vals, pacf_vals, title, fname, n_lags_plot=N_LAGS_ACF)
    print()
