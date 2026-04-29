"""Shared helpers for Assignment 3 scripts."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

DATA_DIR = Path(__file__).resolve().parent.parent


def sample_acf(x, n_lags):
    """Biased sample ACF estimator (matches R's acf())."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    xc = x - x.mean()
    gamma0 = np.sum(xc ** 2) / n
    acf = np.empty(n_lags + 1)
    acf[0] = 1.0
    for k in range(1, n_lags + 1):
        acf[k] = np.sum(xc[:n - k] * xc[k:]) / n / gamma0
    return acf


def sample_ccf(x, y, n_lags):
    """Cross-correlation of x leading y by h (h=-n_lags..n_lags).

    Returns (lags, ccf). Positive lag h: corr(x_t, y_{t+h}).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    xc = x - x.mean()
    yc = y - y.mean()
    sx = np.sqrt(np.sum(xc ** 2))
    sy = np.sqrt(np.sum(yc ** 2))
    lags = np.arange(-n_lags, n_lags + 1)
    out = np.empty_like(lags, dtype=float)
    for i, h in enumerate(lags):
        if h >= 0:
            out[i] = np.sum(xc[: n - h] * yc[h:]) / (sx * sy)
        else:
            out[i] = np.sum(xc[-h:] * yc[: n + h]) / (sx * sy)
    return lags, out


def acf_ci(n, alpha=0.05):
    from scipy.stats import norm
    return norm.ppf(1 - alpha / 2) / np.sqrt(n)


def stem_acf(ax, acf_vals, n, lag_start=0, label=None, color="C0"):
    lags = np.arange(lag_start, len(acf_vals))
    vals = acf_vals[lag_start:]
    ax.vlines(lags, 0, vals, linewidth=1.2, color=color)
    ax.plot(lags, vals, "o", markersize=3, color=color, label=label)
    ax.axhline(0, color="k", linewidth=0.5)
    ci = acf_ci(n)
    ax.axhline(ci, color="b", linestyle="--", linewidth=0.6)
    ax.axhline(-ci, color="b", linestyle="--", linewidth=0.6)
    ax.set_xlabel("Lag")


def savefig(fig, name):
    path = PLOT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.relative_to(DATA_DIR.parent)}")
    return path
