# Assignment 4 — Kalman Filtering and State-Space Modeling

DTU 02417, Spring 2026.

## Task

- Part 1: Kalman filter on a simulated 1-D state-space system, MLE estimation of parameters, robustness study under model misspecification
- Part 2: state-space modeling (1-D and 2-D) of transformer temperature data (`transformer_data.csv`)

See [`assignment4.pdf`](./assignment4.pdf) for the full task description.

## Files

| File | Purpose |
|---|---|
| `assignment4.pdf` | Task description from instructor |
| `data/transformer_data.csv` | Real transformer temperature data (Part 2) |
| `scripts/common.py` | Shared plotting / diagnostics helpers |
| `scripts/kalman.py` | Reusable Kalman filter implementation |
| `scripts/part1_simulation.py` | Part 1 — generate simulated state-space data |
| `scripts/part1_kf_demo.py` | Part 1 — Kalman filter demonstration |
| `scripts/part1_mle.py` | Part 1 — MLE estimation |
| `scripts/part1_robustness.py` | Part 1 — robustness experiments |
| `scripts/part2_eda.py` | Part 2 — exploratory data analysis |
| `scripts/part2_model1d.py` | Part 2 — 1-D state-space model fitting |
| `scripts/part2_model2d.py` | Part 2 — 2-D state-space model fitting |
| `scripts/part2_extras.py` | Part 2 — extra diagnostics |
| `scripts/part2_interpret.py` | Part 2 — interpretation / final figures |
| `functions_chant_R_provided/` | R helper functions provided by instructor (reference only) |
| `plots/` | Generated PDF + PNG figures |
| `report/report.tex` / `report.pdf` | Written report (LaTeX source + compiled) |

## Running — order matters!

Several `part2_*.py` scripts depend on cached fits. Run in this order:

```bash
cd assignment4

# Part 1 — simulation must run first
python scripts/part1_simulation.py    # → _p1_q12_data.npz
python scripts/part1_kf_demo.py
python scripts/part1_mle.py            # → _p1_mle_results.npz
python scripts/part1_robustness.py

# Part 2
python scripts/part2_eda.py            # → _p2_df.pkl
python scripts/part2_model1d.py        # → _p2_1d_fit.npz
python scripts/part2_model2d.py        # → _p2_2d_fit.npz
python scripts/part2_extras.py
python scripts/part2_interpret.py
```

The cache files (`_p1_*.npz`, `_p2_*.npz`, `_p2_*.pkl`) are regenerable and are gitignored.
