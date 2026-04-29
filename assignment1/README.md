# Assignment 1 — Linear, Weighted, and Recursive Trend Models

DTU 02417, Spring 2026.

## Task

Analyse Danish vehicle registration data (`DST_BIL54.csv`) using
- ordinary least squares (OLS) trend fitting
- weighted least squares (WLS) with exponential forgetting
- recursive least squares (RLS) with adaptive forgetting

See [`assignment1.pdf`](./assignment1.pdf) for the full task description.

## Files

| File | Purpose |
|---|---|
| `assignment1.pdf` | Task description from instructor |
| `DST_BIL54.csv` | Raw monthly vehicle registration counts |
| `assignment1.py` | Solution script (reproduces all figures and printed numbers) |
| `read_data.R` | R helper provided by instructor (not used by Python solution) |
| `figures/` | Generated PNGs (`fig_1_1_*` … `fig_5_7_*`) |
| `report/report.tex` / `report.pdf` | Written report (LaTeX source + compiled) |
| `report/handcalc_guide.tex` / `.pdf` | Hand-calculation appendix |
| `report/final_report.pdf` | Polished final version of the report |

## Running

```bash
cd assignment1
python assignment1.py
```

The script reads `DST_BIL54.csv` from the **current working directory**, so run it from inside `assignment1/`.
