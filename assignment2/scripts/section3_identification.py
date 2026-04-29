"""
Section 3: Identifying ARMA models from ACF/PACF plots.

This script documents the reasoning for each process identification.
No simulation is needed — we analyse the plots given in the assignment.
"""

print("=" * 60)
print("Section 3: Identifying ARMA Models")
print("=" * 60)

print("""
3.1  Process 1
──────────────
  ACF:  Sharp cutoff after lag 1 (only ρ(1) significant, ~0.8).
        For lags ≥ 2, all values near zero within confidence bands.
  PACF: Decays/tails off gradually.

  → This is the signature of an MA(1) process.
    The ACF of MA(q) cuts off after lag q, while PACF tails off.
    Here q = 1, so the model is MA(1).

3.2  Process 2
──────────────
  ACF:  Decays slowly (geometrically), remaining significant for many lags.
  PACF: Single significant spike at lag 1 (~0.65), then cuts off.

  → This is the signature of an AR(1) process.
    The PACF of AR(p) cuts off after lag p, while ACF tails off.
    Here p = 1, so the model is AR(1).

3.3  Process 3
──────────────
  ACF:  Tails off — oscillating decay, significant for many lags.
  PACF: Two significant spikes at lags 1 (~0.85) and 2 (~-0.45),
        then cuts off (all subsequent values within bands).

  → This is the signature of an AR(2) process.
    PACF cuts off after lag 2, ACF tails off.
    Here p = 2, so the model is AR(2).
""")
