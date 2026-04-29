# Assignment 3 — ARMAX / ARX Modeling

DTU 02417, Spring 2026.

## Task

- Part 1: AR(p) stability and ACF for simulated processes
- Part 2: ARMAX forecasting on solar generation data (`datasolar.csv`)
- Part 3: ARX model order selection and impulse response on hourly transformer-box data (`box_data_60min.csv`)

See [`assignment3.pdf`](./assignment3.pdf) for the full task description.

## Files

| File | Purpose |
|---|---|
| `assignment3.pdf` | Task description from instructor |
| `datasolar.csv` | Solar dataset (Part 2) — must stay at folder root |
| `box_data_60min.csv` | Transformer-box dataset (Part 3) — must stay at folder root |
| `scripts/common.py` | Shared plotting / ACF helpers |
| `scripts/part1_stability.py` | Part 1 — AR stability + ACF |
| `scripts/part2_solar.py` | Part 2 — ARMAX forecasting |
| `scripts/part3_arx_box.py` | Part 3 — ARX order selection on box data |
| `plots/` | Generated PDFs and CSV summary tables |
| `report/report.tex` | Written report (LaTeX source) |

## Running

The scripts use `__file__`-relative paths and expect the CSVs at the assignment folder root (one level above `scripts/`).

```bash
cd assignment3
python scripts/part1_stability.py
python scripts/part2_solar.py
python scripts/part3_arx_box.py
```
