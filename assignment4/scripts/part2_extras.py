"""Compute det(Q), eig(Q), eig(A), and steady-state gain C(I-A)^{-1} B
from the stored 2-D fit, and write an auxiliary LaTeX table.

This avoids re-running the 10-minute 2-D optimisation.
"""

import numpy as np
from pathlib import Path
from common import write_table

HERE = Path(__file__).resolve().parent
fit_path = HERE / "_p2_2d_fit.npz"

if fit_path.exists():
    f = np.load(fit_path)
    A = f["A"]; B = f["B"]; Q = f["Q"]; R = f["R"]
else:
    # values as reported in the current Table 4 of the report
    A = np.array([[0.6993, 0.1860],
                  [0.0849, 0.0957]])
    B = np.array([[-0.0297, 0.001281, 0.0695],
                  [ 0.8391, 0.006653, 1.2277]])
    Q = np.array([[0.2361, 0.7693],
                  [0.7693, 2.5213]])
    R = np.array([[0.0770 ** 2]])

C = np.array([[1.0, 0.0]])
eigA = np.linalg.eigvals(A).real
eigQ = np.linalg.eigvalsh(Q)
detQ = float(np.linalg.det(Q))
condQ = float(np.linalg.cond(Q))
G_ss = np.linalg.solve(np.eye(2) - A, B)
dY = (C @ G_ss).ravel()

print("eig(A) =", eigA)
print("eig(Q) =", eigQ, "det=", detQ, "cond=", condQ)
print("Steady-state dY/du =", dY)

tex = [
    r"\begin{tabular}{l r}",
    r"\toprule",
    r"Quantity & Value \\",
    r"\midrule",
    rf"eig$(A)$                           & ${eigA[0]:+.4f},\; {eigA[1]:+.4f}$ \\",
    rf"eig$(Q)$                           & ${eigQ[0]:.3e},\; {eigQ[1]:.3f}$ \\",
    rf"$\det(Q)$                          & ${detQ:.3e}$ \\",
    rf"$\mathrm{{cond}}(Q)$               & ${condQ:.0f}$ \\",
    rf"$\partial Y_\infty/\partial T_a$   & $+{dY[0]:.3f}$ \si{{\celsius\per\celsius}} \\",
    rf"$\partial Y_\infty/\partial \Phi_s$ & $+{dY[1]:.4f}$ \si{{\celsius\per\watt\per\meter\squared}} \\",
    rf"$\partial Y_\infty/\partial \Phi_I$ & $+{dY[2]:.3f}$ \si{{\celsius\per\kilo\ampere}} \\",
    r"\bottomrule",
    r"\end{tabular}",
]
write_table("p2_2d_extra.tex", "\n".join(tex))
print("Saved p2_2d_extra.tex")
