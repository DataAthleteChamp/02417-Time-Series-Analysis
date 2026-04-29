# Assignment 2 — ARMA Stability and Seasonal Processes

DTU 02417, Spring 2026.

## Task

- Section 1: AR/ARMA stability via root location and ACF analysis
- Section 2: simulation and identification of six seasonal ARMA models
- Section 3: model identification on a given series

See [`assignment2.pdf`](./assignment2.pdf) for the full task description.

## Files

| File | Purpose |
|---|---|
| `assignment2.pdf` | Task description from instructor |
| `presentation.pdf` | Course/group presentation slides |
| `scripts/section1_stability.py` | Section 1 — stability/ACF |
| `scripts/section2_seasonal.py` | Section 2 — seasonal ARMA simulations |
| `scripts/section3_identification.py` | Section 3 — model identification |
| `plots/` | Generated PDF figures (`sec1_acf.pdf`, `sec2_model_*.pdf`) |
| `report/main.tex` | Written report (LaTeX source) |

## Running

All scripts are self-contained and produce PDFs under `plots/`.

```bash
cd assignment2
python scripts/section1_stability.py
python scripts/section2_seasonal.py
python scripts/section3_identification.py
```
