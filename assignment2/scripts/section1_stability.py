"""
Section 1: Stability — AR(2) process analysis.

Model: X_t + φ₁ X_{t-1} + φ₂ X_{t-2} = ε_t
with φ₁ = -0.7, φ₂ = -0.2, σ_ε = 1.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# --- Parameters ---
phi1 = -0.7
phi2 = -0.2

# ============================================================
# 1.1  Stationarity — roots of characteristic polynomial
# ============================================================
# Characteristic equation: 1 + φ₁z + φ₂z² = 0
# np.roots expects [highest_power_coeff, ..., constant]
roots = np.roots([phi2, phi1, 1])
print("1.1 Stationarity")
print(f"  Characteristic polynomial: 1 + ({phi1})z + ({phi2})z² = 0")
print(f"  Roots: {roots}")
print(f"  |roots|: {np.abs(roots)}")
print(f"  All |root| > 1? {all(np.abs(roots) > 1)}  → Process is stationary.\n")

# ============================================================
# 1.2  Invertibility
# ============================================================
print("1.2 Invertibility")
print("  A pure AR process has no MA polynomial (θ(B) = 1).")
print("  Invertibility requires all roots of θ(z) = 0 to lie outside the unit circle.")
print("  Since θ(z) = 1 has no roots, the process is trivially invertible.\n")

# ============================================================
# 1.3  ACF via Yule–Walker recursion
# ============================================================
# For AR(2): X_t + φ₁ X_{t-1} + φ₂ X_{t-2} = ε_t
# Yule-Walker equations:
#   ρ(1) = -φ₁ - φ₂ ρ(1)   →  ρ(1)(1 + φ₂) = -φ₁  →  ρ(1) = -φ₁/(1+φ₂)
#   ρ(k) = -φ₁ ρ(k-1) - φ₂ ρ(k-2)  for k ≥ 2

n_lags = 30
rho = np.zeros(n_lags + 1)
rho[0] = 1.0
rho[1] = -phi1 / (1 + phi2)

for k in range(2, n_lags + 1):
    rho[k] = -phi1 * rho[k - 1] - phi2 * rho[k - 2]

print("1.3 ACF (Yule-Walker)")
print(f"  ρ(0) = {rho[0]:.4f}")
print(f"  ρ(1) = -φ₁/(1+φ₂) = {-phi1}/(1+{phi2}) = {rho[1]:.4f}")
print(f"  ρ(2) = {rho[2]:.4f}")
print(f"  ρ(3) = {rho[3]:.4f}")
print(f"  ρ(5) = {rho[5]:.4f}")
print(f"  ρ(10) = {rho[10]:.4f}\n")

# ============================================================
# 1.4  ACF plot
# ============================================================
fig, ax = plt.subplots(figsize=(7, 3.5))
lags = np.arange(0, n_lags + 1)
ax.vlines(lags, 0, rho, linewidth=1.5)
ax.plot(lags, rho, "o", markersize=4, color="C0")
ax.axhline(0, color="k", linewidth=0.5)
ax.set_xlabel("Lag $k$")
ax.set_ylabel(r"$\rho(k)$")
ax.set_title(
    rf"Theoretical ACF of AR(2) with $\phi_1={phi1}$, $\phi_2={phi2}$"
)
ax.set_xlim(-0.5, n_lags + 0.5)
fig.tight_layout()
fig.savefig(PLOT_DIR / "sec1_acf.pdf")
print(f"Saved plot → {PLOT_DIR / 'sec1_acf.pdf'}")
plt.close(fig)
