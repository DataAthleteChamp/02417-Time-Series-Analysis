"""Part 2 Q2.1: Exploratory analysis of transformer data."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import save, DATA_DIR, sample_acf


def main():
    df = pd.read_csv(DATA_DIR / "transformer_data.csv")
    print(df.describe().round(2))
    print("shape:", df.shape)

    fig, axes = plt.subplots(4, 1, figsize=(9.0, 7.2), sharex=True)
    variables = [
        ("Y",  r"$Y_t$ [$^\circ$C]",             "C3"),
        ("Ta", r"$T_{a,t}$ [$^\circ$C]",         "C0"),
        ("S",  r"$\Phi_{s,t}$ [W/m$^2$]",        "C1"),
        ("I",  r"$\Phi_{I,t}$ [kA]",              "C2"),
    ]
    for ax, (col, lab, c) in zip(axes, variables):
        ax.plot(df["time"], df[col], color=c, lw=1.0)
        ax.set_ylabel(lab, fontsize=10)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time [h]")
    fig.suptitle("Transformer station time series (hourly)")
    fig.tight_layout(rect=(0.02, 0.0, 1.0, 0.96))
    save(fig, "p2_eda")

    # cross-correlations Y vs inputs
    from scipy.stats import pearsonr
    print("\nContemporaneous correlations with Y:")
    for col in ["Ta", "S", "I"]:
        r, p = pearsonr(df["Y"], df[col])
        print(f"  Y ~ {col}: r={r:+.3f} (p={p:.2e})")

    df.to_pickle("_p2_df.pkl")


if __name__ == "__main__":
    main()
