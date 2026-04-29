# 02417 — Time Series Analysis (DTU, Spring 2026)

Solutions and reports for the four assignments of DTU course **02417 Time Series Analysis**.

Each assignment lives in its own subfolder and is self-contained: task description PDF, raw data, Python scripts that reproduce the results, generated figures, and the written report (LaTeX source + compiled PDF).

## Contents

| Folder | Topic | Data | Main scripts |
|---|---|---|---|
| [`assignment1/`](./assignment1) | Linear / weighted / recursive trend models | `DST_BIL54.csv` (Danish vehicle registrations) | `assignment1.py` |
| [`assignment2/`](./assignment2) | ARMA stability and seasonal process analysis | *(simulated)* | `scripts/section{1,2,3}_*.py` |
| [`assignment3/`](./assignment3) | ARMAX / ARX modeling and prediction | `datasolar.csv`, `box_data_60min.csv` | `scripts/part{1,2,3}_*.py` |
| [`assignment4/`](./assignment4) | Kalman filtering and state-space modeling | `transformer_data.csv` | `scripts/part{1,2}_*.py` |

## Running the code

Each assignment's `scripts/` (or root `.py` for A1) is designed to be run from inside that assignment's folder. Common dependencies:

```bash
python -m pip install numpy scipy pandas matplotlib statsmodels
```

A1 expects to be run from the folder containing `DST_BIL54.csv`:
```bash
cd assignment1 && python assignment1.py
```

A2/A3/A4 use `Path(__file__)`-relative paths, so they can be run from anywhere:
```bash
cd assignment3 && python scripts/part2_solar.py
```

A4's `part2_*.py` scripts depend on each other through cached `.npz`/`.pkl` artifacts. Run order:
```
part1_simulation.py → part1_kf_demo.py → part1_mle.py → part1_robustness.py
part2_eda.py       → part2_model1d.py → part2_model2d.py → part2_extras.py → part2_interpret.py
```

## License & attribution

This is coursework. Task descriptions (`assignmentN.pdf`) and any provided helper code (`functions_chant_R_provided/`, `read_data.R`) are © DTU course staff and reproduced here for context only.
