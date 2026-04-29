"""Part 2 Q2.4: Interpret the two reconstructed latent states."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from common import save


def main():
    df = pd.read_pickle(Path(__file__).parent / "_p2_df.pkl")
    fit = np.load(Path(__file__).parent / "_p2_2d_fit.npz")
    x_filt = fit["x_filt"]  # (T, 2)
    A = fit["A"]; B = fit["B"]
    eig = np.linalg.eigvals(A)
    print("eig(A) =", eig)
    print("A =\n", A)
    print("B =\n", B)

    t = df["time"].values
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(t, df["Y"], "o", ms=2.5, alpha=0.4, color="k",
                 label=r"$Y_t$ (observed)")
    axes[0].plot(t, x_filt[:, 0], color="C3", lw=1.3,
                 label=r"state 1 $\hat X_{1,t|t}$")
    axes[0].plot(t, x_filt[:, 1], color="C0", lw=1.3,
                 label=r"state 2 $\hat X_{2,t|t}$")
    axes[0].set_ylabel(r"temperature [$^\circ$C]")
    axes[0].set_title("Reconstructed latent states")
    axes[0].legend(fontsize=8)

    # Normalized inputs (z-score) on single shared axis
    for col, c, lbl in zip(["Ta", "S", "I"],
                           ["C1", "C2", "C4"],
                           [r"$T_{a,t}$", r"$\Phi_{s,t}$", r"$\Phi_{I,t}$"]):
        z = (df[col] - df[col].mean()) / df[col].std()
        axes[1].plot(t, z, color=c, lw=1.0, label=lbl)
    axes[1].set_xlabel("time [h]")
    axes[1].set_ylabel("input (z-score)")
    axes[1].set_title("Inputs (standardized)")
    axes[1].legend(fontsize=8, ncol=3)
    fig.tight_layout()
    save(fig, "p2_2d_states")

    # Correlations between each state and inputs/observation
    print("\nCorrelations:")
    for k in [0, 1]:
        for col in ["Y", "Ta", "S", "I"]:
            r = np.corrcoef(x_filt[:, k], df[col].values)[0, 1]
            print(f"  state {k+1} ~ {col}: r = {r:+.3f}")


if __name__ == "__main__":
    main()
