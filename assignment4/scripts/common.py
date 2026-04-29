"""Shared helpers for Assignment 4 scripts."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, probplot

ROOT = Path(__file__).resolve().parent.parent
PLOT_DIR = ROOT / "plots"
DATA_DIR = ROOT / "data"
TABLE_DIR = ROOT / "report" / "tables"
PLOT_DIR.mkdir(exist_ok=True, parents=True)
TABLE_DIR.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.2,
})


def save(fig, name):
    """Save both PDF (for LaTeX) and PNG (for preview)."""
    path = PLOT_DIR / name
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def sample_acf(x, n_lags):
    x = np.asarray(x, float)
    n = len(x)
    xc = x - x.mean()
    g0 = np.sum(xc ** 2) / n
    out = np.empty(n_lags + 1)
    out[0] = 1.0
    for k in range(1, n_lags + 1):
        out[k] = np.sum(xc[:n - k] * xc[k:]) / n / g0
    return out


def sample_pacf(x, n_lags):
    """Durbin-Levinson recursion for PACF."""
    r = sample_acf(x, n_lags)
    pacf = np.zeros(n_lags + 1)
    pacf[0] = 1.0
    phi = np.zeros((n_lags + 1, n_lags + 1))
    phi[1, 1] = r[1]
    pacf[1] = phi[1, 1]
    for k in range(2, n_lags + 1):
        num = r[k] - np.sum(phi[k - 1, 1:k] * r[1:k][::-1])
        den = 1.0 - np.sum(phi[k - 1, 1:k] * r[1:k])
        phi[k, k] = num / den
        phi[k, 1:k] = phi[k - 1, 1:k] - phi[k, k] * phi[k - 1, 1:k][::-1]
        pacf[k] = phi[k, k]
    return pacf


def acf_ci(n, alpha=0.05):
    return norm.ppf(1 - alpha / 2) / np.sqrt(n)


def stem_on(ax, vals, n, color="C0", title=None, ylabel=None, lag_start=0):
    lags = np.arange(lag_start, len(vals))
    v = vals[lag_start:]
    ax.vlines(lags, 0, v, colors=color, linewidth=1.1)
    ax.plot(lags, v, "o", ms=3, color=color)
    ax.axhline(0, color="k", lw=0.5)
    ci = acf_ci(n)
    ax.axhline(ci, color="b", ls="--", lw=0.6)
    ax.axhline(-ci, color="b", ls="--", lw=0.6)
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("Lag")


def ljung_box(resid, lags=10):
    """Q-statistic and p-value for Ljung-Box."""
    from scipy.stats import chi2
    resid = np.asarray(resid, float)
    n = len(resid)
    r = sample_acf(resid, lags)[1:]
    Q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, lags + 1)))
    p = 1 - chi2.cdf(Q, df=lags)
    return Q, p


def residual_diag_fig(std_res, title, fname, n_lags=30):
    """2x2 panel: residuals, ACF, PACF, QQ."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    n = len(std_res)
    axes[0, 0].plot(std_res, lw=0.8, color="C0")
    axes[0, 0].axhline(1.96, color="r", ls="--", lw=0.6)
    axes[0, 0].axhline(-1.96, color="r", ls="--", lw=0.6)
    axes[0, 0].axhline(0, color="k", lw=0.4)
    axes[0, 0].set_title("Standardized residuals")
    axes[0, 0].set_xlabel("t")

    stem_on(axes[0, 1], sample_acf(std_res, n_lags), n, title="ACF", lag_start=1)
    stem_on(axes[1, 0], sample_pacf(std_res, n_lags), n, title="PACF", lag_start=1)

    probplot(std_res, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Normal QQ plot")
    axes[1, 1].get_lines()[0].set_markersize(3)

    fig.suptitle(title)
    fig.tight_layout()
    save(fig, fname)


def write_table(path_name, tex):
    (TABLE_DIR / path_name).write_text(tex)
