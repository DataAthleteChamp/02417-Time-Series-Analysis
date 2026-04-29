"""
Assignment 1 — 02417 Time Series Analysis (DTU, Spring 2025)
Danish Vehicle Registration Data: OLS, WLS, and RLS Regression

All methods implemented from scratch using numpy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# Data Loading and Preprocessing
# ============================================================

df = pd.read_csv('DST_BIL54.csv')

# Create decimal year: 2018-Jan -> 2018.0, 2018-Feb -> 2018 + 1/12, etc.
dates = pd.to_datetime(df['time'] + '-01')
x_all = (dates.dt.year + (dates.dt.month - 1) / 12).values
y_all = df['total'].values / 1E6  # millions of vehicles

# Train/test split: train = 2018-01 to 2023-12, test = 2024-01 to 2024-12
train_mask = dates < pd.Timestamp('2024-01-01')
test_mask = dates >= pd.Timestamp('2024-01-01')

x_train = x_all[train_mask.values]
y_train = y_all[train_mask.values]
x_test = x_all[test_mask.values]
y_test = y_all[test_mask.values]

N = len(x_train)       # 72
N_test = len(x_test)   # 12
p = 2                  # intercept + slope

# Design matrices: X = [1, x]
X_train = np.column_stack([np.ones(N), x_train])          # (72, 2)
X_test = np.column_stack([np.ones(N_test), x_test])        # (12, 2)

print(f"Training set: N = {N} observations ({x_train[0]:.0f}-Jan to {x_train[-1]:.4f})")
print(f"Test set:     {N_test} observations ({x_test[0]:.0f}-Jan to {x_test[-1]:.4f})")
print(f"y range: [{y_train.min():.3f}, {y_train.max():.3f}] million vehicles")
print()

# ============================================================
# Section 1: Plot Data
# ============================================================

# 1.1 — Scatter plot of training data
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x_train, y_train, c='steelblue', s=25, zorder=3, label='Training data')
ax.set_xlabel('Year')
ax.set_ylabel('Total vehicles (millions)')
ax.set_title('Section 1.1: Training Data — Danish Vehicle Registrations')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_1_1_training_data.png', dpi=150)
plt.show()

# 1.2 — Description
print("=" * 70)
print("Section 1.2: Time Series Description")
print("=" * 70)
print(f"The training data contains {N} monthly observations from Jan 2018 to Dec 2023.")
print(f"Total registered vehicles range from {y_train.min():.3f} to {y_train.max():.3f} million.")
print("The series shows a generally upward trend over the period.")
print("There appears to be a slight slowdown or dip around 2019-2020, possibly")
print("related to the COVID-19 pandemic, followed by a recovery and continued growth.")
print("The growth rate appears roughly linear overall, though with some month-to-month")
print("fluctuations (slight seasonal patterns with dips in autumn/winter months).")
print()

# ============================================================
# Section 2: Linear Trend Model — Matrix Form
# ============================================================

print("=" * 70)
print("Section 2.1: Matrix Form for First 3 Time Points")
print("=" * 70)
print()
print("Model: Y_t = theta_1 + theta_2 * x_t + epsilon_t")
print()
print("In matrix form: y = X @ theta + epsilon")
print()
print("For t = 1, 2, 3:")
print()
print(f"y = [{y_train[0]:.3f}]    X = [[1  {x_train[0]:.3f}]    theta = [theta_1]    eps = [eps_1]")
print(f"    [{y_train[1]:.3f}]        [1  {x_train[1]:.3f}]            [theta_2]          [eps_2]")
print(f"    [{y_train[2]:.3f}]        [1  {x_train[2]:.3f}]]                              [eps_3]")
print()
print("With actual numerical values:")
print(f"y = [{y_train[0]:.6f}]    X = [[1  {x_train[0]:.6f}]")
print(f"    [{y_train[1]:.6f}]        [1  {x_train[1]:.6f}]")
print(f"    [{y_train[2]:.6f}]        [1  {x_train[2]:.6f}]]")
print()

# ============================================================
# Section 3: OLS — Global Linear Trend
# ============================================================

# 3.1 — Estimate parameters: theta_hat = (X'X)^{-1} X'Y
XtX = X_train.T @ X_train
XtX_inv = np.linalg.inv(XtX)
theta_ols = XtX_inv @ (X_train.T @ y_train)

print("=" * 70)
print("Section 3.1: OLS Parameter Estimation")
print("=" * 70)
print()
print("theta_hat = (X^T X)^{-1} X^T Y")
print()
print(f"  theta_1 (intercept) = {theta_ols[0]:.6f}")
print(f"  theta_2 (slope)     = {theta_ols[1]:.6f}")
print()

# 3.2 — Standard errors and fitted line plot
residuals_ols = y_train - X_train @ theta_ols
sigma2_ols = (residuals_ols @ residuals_ols) / (N - p)
sigma_ols = np.sqrt(sigma2_ols)
V_theta_ols = sigma2_ols * XtX_inv
se_ols = np.sqrt(np.diag(V_theta_ols))

print("=" * 70)
print("Section 3.2: Parameter Estimates with Standard Errors")
print("=" * 70)
print()
print(f"  sigma^2 = {sigma2_ols:.6e}")
print(f"  sigma   = {sigma_ols:.6f}")
print()
print(f"  theta_1 = {theta_ols[0]:12.6f}  (SE = {se_ols[0]:.6f})")
print(f"  theta_2 = {theta_ols[1]:12.6f}  (SE = {se_ols[1]:.6f})")
print()

# Fitted line
x_line = np.linspace(x_train[0], x_train[-1], 200)
y_line = theta_ols[0] + theta_ols[1] * x_line

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x_train, y_train, c='steelblue', s=25, zorder=3, label='Observations')
ax.plot(x_line, y_line, 'r-', linewidth=2, label='OLS fitted line')
ax.set_xlabel('Year')
ax.set_ylabel('Total vehicles (millions)')
ax.set_title('Section 3.2: OLS Fitted Line')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_3_2_ols_fit.png', dpi=150)
plt.show()

# 3.3 — Forecast with prediction intervals
y_pred_ols = X_test @ theta_ols
t_crit = stats.t.ppf(0.975, N - p)  # t_{0.025} with N-p = 70 df

pi_half_ols = np.zeros(N_test)
for i in range(N_test):
    x_new = X_test[i]
    pi_half_ols[i] = t_crit * sigma_ols * np.sqrt(1 + x_new @ XtX_inv @ x_new)

print("=" * 70)
print("Section 3.3: OLS Forecast for Test Set (2024)")
print("=" * 70)
print()
print(f"  t-critical (alpha=0.05, df={N-p}): {t_crit:.4f}")
print()
print(f"  {'Month':>8s}  {'Predicted':>10s}  {'Lower PI':>10s}  {'Upper PI':>10s}  {'Actual':>10s}")
print(f"  {'-----':>8s}  {'--------':>10s}  {'--------':>10s}  {'--------':>10s}  {'------':>10s}")
months_2024 = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for i in range(N_test):
    lo = y_pred_ols[i] - pi_half_ols[i]
    hi = y_pred_ols[i] + pi_half_ols[i]
    print(f"  {months_2024[i]:>8s}  {y_pred_ols[i]:10.4f}  {lo:10.4f}  {hi:10.4f}  {y_test[i]:10.4f}")
print()

# 3.4 — Full plot: training + forecast + PI + test data
x_forecast_line = np.linspace(x_train[0], x_test[-1], 300)
y_forecast_line = theta_ols[0] + theta_ols[1] * x_forecast_line

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(x_train, y_train, c='steelblue', s=25, zorder=3, label='Training data')
ax.scatter(x_test, y_test, c='green', s=40, marker='D', zorder=3, label='Test data (actual)')
ax.plot(x_forecast_line, y_forecast_line, 'r-', linewidth=1.5, label='OLS fitted line')
ax.plot(x_test, y_pred_ols, 'ro', markersize=6, label='OLS forecast')
ax.fill_between(x_test, y_pred_ols - pi_half_ols, y_pred_ols + pi_half_ols,
                color='red', alpha=0.15, label='95% prediction interval')
ax.axvline(x=2024.0, color='gray', linestyle='--', alpha=0.5, label='Train/test split')
ax.set_xlabel('Year')
ax.set_ylabel('Total vehicles (millions)')
ax.set_title('Section 3.4: OLS Forecast with Prediction Intervals')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_3_4_ols_forecast.png', dpi=150)
plt.show()

# 3.5 — Commentary on forecast quality
print("=" * 70)
print("Section 3.5: Commentary on OLS Forecast")
print("=" * 70)
print()
ols_test_rmse = np.sqrt(np.mean((y_pred_ols - y_test)**2))
print(f"  Test RMSE: {ols_test_rmse:.4f} million vehicles")
in_pi = np.sum((y_test >= y_pred_ols - pi_half_ols) & (y_test <= y_pred_ols + pi_half_ols))
print(f"  Test points inside 95% PI: {in_pi}/{N_test}")
print()
print("  The OLS forecast extrapolates the global linear trend estimated from all 72")
print("  training observations. Since this is a global model, it cannot adapt to recent")
print("  changes in the growth rate. The prediction intervals widen for points further")
print("  from the training data center, which is expected.")
print()

# 3.6 — Residual diagnostics (2x2 subplot)
def compute_acf(x, max_lag):
    """Compute sample autocorrelation function."""
    n = len(x)
    xm = x - np.mean(x)
    c0 = np.sum(xm**2)
    acf_vals = np.zeros(max_lag + 1)
    for h in range(max_lag + 1):
        acf_vals[h] = np.sum(xm[:n-h] * xm[h:]) / c0
    return acf_vals

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# (a) Residuals vs time
axes[0, 0].scatter(x_train, residuals_ols, c='steelblue', s=15)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Residual')
axes[0, 0].set_title('(a) Residuals vs Time')
axes[0, 0].grid(True, alpha=0.3)

# (b) Histogram + normal curve
axes[0, 1].hist(residuals_ols, bins=15, density=True, color='steelblue', alpha=0.7, edgecolor='white')
x_norm = np.linspace(residuals_ols.min(), residuals_ols.max(), 100)
axes[0, 1].plot(x_norm, stats.norm.pdf(x_norm, np.mean(residuals_ols), np.std(residuals_ols)),
                'r-', linewidth=2)
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('(b) Histogram of Residuals')
axes[0, 1].grid(True, alpha=0.3)

# (c) QQ-plot
(osm, osr), (slope, intercept, r) = stats.probplot(residuals_ols, dist="norm")
axes[1, 0].scatter(osm, osr, c='steelblue', s=15)
axes[1, 0].plot(osm, slope * np.array(osm) + intercept, 'r-', linewidth=2)
axes[1, 0].set_xlabel('Theoretical Quantiles')
axes[1, 0].set_ylabel('Sample Quantiles')
axes[1, 0].set_title('(c) Normal QQ-Plot')
axes[1, 0].grid(True, alpha=0.3)

# (d) ACF
max_lag = 20
acf_vals = compute_acf(residuals_ols, max_lag)
lags = np.arange(max_lag + 1)
ci_bound = 1.96 / np.sqrt(N)
axes[1, 1].bar(lags, acf_vals, color='steelblue', width=0.6)
axes[1, 1].axhline(y=ci_bound, color='red', linestyle='--', label=f'95% bounds (±{ci_bound:.3f})')
axes[1, 1].axhline(y=-ci_bound, color='red', linestyle='--')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('ACF')
axes[1, 1].set_title('(d) Autocorrelation Function')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle('Section 3.6: OLS Residual Diagnostics', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('fig_3_6_ols_residuals.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 70)
print("Section 3.6: Residual Analysis Commentary")
print("=" * 70)
print()
print("  (a) Residuals vs time show a clear non-random pattern — they are negative at")
print("      the start, positive in the middle, and negative again at the end. This")
print("      suggests the linear trend model does not fully capture the data dynamics.")
print("  (b) The histogram appears roughly bell-shaped but may show slight skewness.")
print("  (c) The QQ-plot shows some deviation from the normal line at the tails,")
print("      suggesting the residuals may not be perfectly normally distributed.")
print("  (d) The ACF shows significant autocorrelation at several lags, indicating")
print("      the residuals are NOT independent — a key OLS assumption is violated.")
print("      The model is missing some systematic structure in the data.")
print()

# ============================================================
# Section 4: WLS — Local Linear Trend (lambda = 0.9)
# ============================================================

lam = 0.9

# Weights: w[i] = lam^{N-1-i}, so w[0] = lam^71 (oldest, smallest)
#                                    w[N-1] = lam^0 = 1 (newest, largest)
w = np.array([lam**(N - 1 - i) for i in range(N)])
W = np.diag(w)

# 4.1 — Describe the variance-covariance matrix Sigma
print("=" * 70)
print("Section 4.1: Variance-Covariance Matrix Sigma (WLS, lambda=0.9)")
print("=" * 70)
print()
print("  In WLS, we model V[epsilon] = sigma^2 * Sigma, where Sigma^{-1} = W = diag(w).")
print("  The weights are: w_i = lambda^{N-1-i} for i = 0, ..., N-1 (0-indexed).")
print("  This gives the newest observation weight 1 and the oldest weight lambda^{N-1}.")
print()
print("  Sigma = diag(1/w_i) = diag(lambda^{-(N-1)}, lambda^{-(N-2)}, ..., 1)")
print()
print("  Corner values of Sigma (72 x 72 diagonal matrix):")
Sigma_diag = 1.0 / w
print(f"    Sigma[0,0]   = lambda^{{-71}} = {Sigma_diag[0]:.4f}  (oldest, highest variance)")
print(f"    Sigma[1,1]   = lambda^{{-70}} = {Sigma_diag[1]:.4f}")
print(f"    Sigma[70,70] = lambda^{{-1}}  = {Sigma_diag[-2]:.4f}")
print(f"    Sigma[71,71] = lambda^0    = {Sigma_diag[-1]:.4f}  (newest, lowest variance)")
print()
print("  For comparison, in OLS: Sigma = I (identity matrix), all diagonal entries = 1.")
print("  In OLS, all observations are equally weighted/trusted.")
print("  In WLS with lambda=0.9, recent observations have much lower assumed variance,")
print("  meaning we trust them more and they have more influence on the estimates.")
print()

# 4.2 — Bar plot of weights
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x_train, w, width=0.06, color='steelblue')
ax.set_xlabel('Year')
ax.set_ylabel('Weight')
ax.set_title(f'Section 4.2: Lambda-Weights vs Time (lambda={lam})')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_4_2_weights.png', dpi=150)
plt.show()

print("=" * 70)
print("Section 4.2: Weight Distribution")
print("=" * 70)
print(f"  The latest time point ({x_train[-1]:.4f}) has the highest weight = {w[-1]:.4f}.")
print(f"  The oldest time point ({x_train[0]:.4f}) has the lowest weight = {w[0]:.6f}.")
print()

# 4.3 — Sum of weights
sum_w = np.sum(w)
sum_w_formula = (1 - lam**N) / (1 - lam)

print("=" * 70)
print("Section 4.3: Sum of Weights")
print("=" * 70)
print(f"  Sum of weights (computed):  {sum_w:.4f}")
print(f"  Sum of weights (formula):   (1 - lambda^N)/(1 - lambda) = {sum_w_formula:.4f}")
print(f"  For OLS, all weights = 1, so sum = N = {N}")
print(f"  The effective sample size for WLS is ~{sum_w:.1f}, much smaller than {N}.")
print()

# 4.4 — WLS parameter estimates
XtW = (X_train * w[:, None]).T                # (2, 72): X^T W  (efficient diagonal mult)
XtWX = XtW @ X_train                          # (2, 2)
XtWX_inv = np.linalg.inv(XtWX)
theta_wls = XtWX_inv @ (XtW @ y_train)        # (2,)

print("=" * 70)
print(f"Section 4.4: WLS Parameter Estimates (lambda={lam})")
print("=" * 70)
print()
print(f"  theta_1 (intercept) = {theta_wls[0]:.6f}")
print(f"  theta_2 (slope)     = {theta_wls[1]:.6f}")
print()
print(f"  Compare with OLS:")
print(f"    OLS:  theta_1 = {theta_ols[0]:.6f},  theta_2 = {theta_ols[1]:.6f}")
print(f"    WLS:  theta_1 = {theta_wls[0]:.6f},  theta_2 = {theta_wls[1]:.6f}")
print()

# 4.5 — WLS Forecast
y_pred_wls = X_test @ theta_wls
res_wls = y_train - X_train @ theta_wls
sigma2_wls = np.sum(w * res_wls**2) / (N - p)
sigma_wls = np.sqrt(sigma2_wls)

pi_half_wls = np.zeros(N_test)
for i in range(N_test):
    x_new = X_test[i]
    pi_half_wls[i] = t_crit * sigma_wls * np.sqrt(1 + x_new @ XtWX_inv @ x_new)

print("=" * 70)
print(f"Section 4.5: WLS Forecast (lambda={lam})")
print("=" * 70)
print()
print(f"  sigma^2_WLS = {sigma2_wls:.6e},  sigma_WLS = {sigma_wls:.6f}")
print()
print(f"  {'Month':>8s}  {'OLS pred':>10s}  {'WLS pred':>10s}  {'Actual':>10s}")
print(f"  {'-----':>8s}  {'--------':>10s}  {'--------':>10s}  {'------':>10s}")
for i in range(N_test):
    print(f"  {months_2024[i]:>8s}  {y_pred_ols[i]:10.4f}  {y_pred_wls[i]:10.4f}  {y_test[i]:10.4f}")
print()

# Plot: OLS vs WLS forecasts
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(x_train, y_train, c='steelblue', s=20, zorder=3, label='Training data')
ax.scatter(x_test, y_test, c='green', s=40, marker='D', zorder=3, label='Test data (actual)')

# OLS
ax.plot(x_test, y_pred_ols, 'r-o', markersize=5, linewidth=1.5, label='OLS forecast')
ax.fill_between(x_test, y_pred_ols - pi_half_ols, y_pred_ols + pi_half_ols,
                color='red', alpha=0.1, label='OLS 95% PI')

# WLS
ax.plot(x_test, y_pred_wls, 'm-s', markersize=5, linewidth=1.5, label=f'WLS forecast (λ={lam})')
ax.fill_between(x_test, y_pred_wls - pi_half_wls, y_pred_wls + pi_half_wls,
                color='purple', alpha=0.1, label='WLS 95% PI')

# Fitted lines extending into test period
x_ext = np.linspace(x_train[0], x_test[-1], 300)
ax.plot(x_ext, theta_ols[0] + theta_ols[1] * x_ext, 'r--', alpha=0.4, linewidth=1)
ax.plot(x_ext, theta_wls[0] + theta_wls[1] * x_ext, 'm--', alpha=0.4, linewidth=1)

ax.axvline(x=2024.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Total vehicles (millions)')
ax.set_title('Section 4.5: OLS vs WLS Forecasts')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_4_5_ols_vs_wls.png', dpi=150)
plt.show()

wls_test_rmse = np.sqrt(np.mean((y_pred_wls - y_test)**2))
print(f"  OLS Test RMSE: {ols_test_rmse:.4f}")
print(f"  WLS Test RMSE: {wls_test_rmse:.4f}")
print()
print("  Comment: The WLS forecast puts more weight on recent observations, so its")
print("  trend line is more influenced by the latest training data. This may provide")
print("  a better or worse forecast depending on whether the recent trend continues.")
print()

# 4.6 (Optional) — Multiple lambda values
lambdas_opt = [0.99, 0.8, 0.7, 0.6]

print("=" * 70)
print("Section 4.6 (Optional): WLS for Multiple Lambda Values")
print("=" * 70)
print()

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(x_train, y_train, c='steelblue', s=20, zorder=3, label='Training data')
ax.scatter(x_test, y_test, c='green', s=40, marker='D', zorder=3, label='Test data')

# Plot OLS and WLS (lam=0.9)
ax.plot(x_test, y_pred_ols, 'r-o', markersize=4, linewidth=1.5, label='OLS')
ax.plot(x_test, y_pred_wls, 'm-s', markersize=4, linewidth=1.5, label=f'WLS λ={lam}')

colors_opt = ['darkorange', 'teal', 'brown', 'navy']
for lam_i, col in zip(lambdas_opt, colors_opt):
    w_i = np.array([lam_i**(N - 1 - j) for j in range(N)])
    XtW_i = (X_train * w_i[:, None]).T
    XtWX_i = XtW_i @ X_train
    theta_i = np.linalg.inv(XtWX_i) @ (XtW_i @ y_train)
    y_pred_i = X_test @ theta_i
    ax.plot(x_test, y_pred_i, marker='^', markersize=4, linewidth=1.5, color=col,
            label=f'WLS λ={lam_i}')
    rmse_i = np.sqrt(np.mean((y_pred_i - y_test)**2))
    print(f"  lambda={lam_i}: theta_1={theta_i[0]:.4f}, theta_2={theta_i[1]:.6f}, RMSE={rmse_i:.4f}")

ax.axvline(x=2024.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Total vehicles (millions)')
ax.set_title('Section 4.6: WLS Forecasts for Multiple Lambda Values')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_4_6_multi_lambda_wls.png', dpi=150)
plt.show()

print()
print("  Comment: Smaller lambda gives more weight to recent data, resulting in a")
print("  steeper or shallower slope depending on the recent trend. Lambda close to 1")
print("  approaches OLS. Very small lambda (e.g. 0.6) relies heavily on the most")
print("  recent observations and can be unstable.")
print()

# ============================================================
# Section 5: RLS — Recursive Estimation
# ============================================================

def rls(X, y, lam=1.0, R0_scale=0.1, theta0=None):
    """
    Recursive Least Squares.

    R_t = lam * R_{t-1} + x_t x_t^T
    theta_t = theta_{t-1} + R_t^{-1} x_t (Y_t - x_t^T theta_{t-1})

    Parameters
    ----------
    X : (N, p) design matrix
    y : (N,) observations
    lam : forgetting factor (1.0 = no forgetting)
    R0_scale : scalar for R_0 = R0_scale * I
    theta0 : (p,) initial parameter estimate

    Returns
    -------
    theta_history : (N+1, p) — theta_history[0] = theta_0, theta_history[t] = theta_t
    R_history : (N+1, p, p) — R matrices
    """
    N_obs, p_dim = X.shape
    if theta0 is None:
        theta0 = np.zeros(p_dim)

    theta = theta0.copy().astype(float)
    R = R0_scale * np.eye(p_dim)

    theta_history = np.zeros((N_obs + 1, p_dim))
    R_history = np.zeros((N_obs + 1, p_dim, p_dim))
    theta_history[0] = theta
    R_history[0] = R

    for t in range(N_obs):
        x_t = X[t]                                      # (p,)
        R = lam * R + np.outer(x_t, x_t)                # (p, p)
        R_inv = np.linalg.inv(R)                         # (p, p)
        innovation = y[t] - x_t @ theta                  # scalar
        theta = theta + R_inv @ x_t * innovation          # (p,)
        theta_history[t + 1] = theta
        R_history[t + 1] = R

    return theta_history, R_history


# 5.1 — Hand calculations for R_1, theta_1, R_2
print("=" * 70)
print("Section 5.1: RLS Update Equations and Hand Calculations")
print("=" * 70)
print()
print("  RLS update equations (with forgetting factor lambda):")
print("    R_t = lambda * R_{t-1} + x_t x_t^T")
print("    theta_t = theta_{t-1} + R_t^{-1} x_t (Y_t - x_t^T theta_{t-1})")
print()
print("  Initialize: R_0 = 0.1 * I,  theta_0 = [0, 0]^T")
print()

# Compute first 2 iterations by hand (with lam=1)
R0 = 0.1 * np.eye(2)
theta0_vec = np.zeros(2)
x1 = X_train[0]  # [1, 2018.000]
y1 = y_train[0]  # 2.930483

# Step 1
R1 = 1.0 * R0 + np.outer(x1, x1)
R1_inv = np.linalg.inv(R1)
innov1 = y1 - x1 @ theta0_vec
theta1 = theta0_vec + R1_inv @ x1 * innov1

print("  --- Step t=1 (lambda=1) ---")
print(f"  x_1 = [{x1[0]:.3f}, {x1[1]:.3f}]^T")
print(f"  y_1 = {y1:.6f}")
print()
print(f"  R_1 = lambda * R_0 + x_1 x_1^T")
print(f"       = [[{R1[0,0]:.4f}, {R1[0,1]:.4f}],")
print(f"          [{R1[1,0]:.4f}, {R1[1,1]:.4f}]]")
print()
print(f"  Innovation: y_1 - x_1^T theta_0 = {innov1:.6f}")
print(f"  theta_1 = theta_0 + R_1^{{-1}} x_1 * {innov1:.6f}")
print(f"          = [{theta1[0]:.6f}, {theta1[1]:.6f}]^T")
print()

# Step 2
x2 = X_train[1]
y2 = y_train[1]
R2 = 1.0 * R1 + np.outer(x2, x2)
R2_inv = np.linalg.inv(R2)
innov2 = y2 - x2 @ theta1
theta2 = theta1 + R2_inv @ x2 * innov2

print("  --- Step t=2 (lambda=1) ---")
print(f"  x_2 = [{x2[0]:.3f}, {x2[1]:.3f}]^T")
print(f"  y_2 = {y2:.6f}")
print()
print(f"  R_2 = R_1 + x_2 x_2^T")
print(f"       = [[{R2[0,0]:.4f}, {R2[0,1]:.4f}],")
print(f"          [{R2[1,0]:.4f}, {R2[1,1]:.4f}]]")
print()
print(f"  Innovation: y_2 - x_2^T theta_1 = {innov2:.6f}")
print(f"  theta_2 = [{theta2[0]:.6f}, {theta2[1]:.6f}]^T")
print()

# 5.2 — RLS loop, present theta up to t=3
theta_hist_1, R_hist_1 = rls(X_train, y_train, lam=1.0, R0_scale=0.1)

print("=" * 70)
print("Section 5.2: RLS Estimates for t = 1, 2, 3 (lambda=1)")
print("=" * 70)
print()
for t in range(1, 4):
    print(f"  t={t}: theta = [{theta_hist_1[t, 0]:.6f}, {theta_hist_1[t, 1]:.6f}]")
print()
print("  The parameters change rapidly at first as the algorithm adjusts from the")
print("  initial guess [0,0]. As more data arrives, the updates become smaller and")
print("  the estimates stabilize. This is intuitive: early observations carry a lot")
print("  of information relative to the prior, while later ones refine the estimates.")
print()

# 5.3 — Compare RLS(t=N) with OLS, sensitivity to R_0
print("=" * 70)
print("Section 5.3: RLS at t=N vs OLS, Initialization Sensitivity")
print("=" * 70)
print()
print(f"  RLS (R0=0.1*I):  theta_N = [{theta_hist_1[N, 0]:.6f}, {theta_hist_1[N, 1]:.6f}]")
print(f"  OLS:              theta   = [{theta_ols[0]:.6f}, {theta_ols[1]:.6f}]")
diff = theta_hist_1[N] - theta_ols
print(f"  Difference:                  [{diff[0]:.6e}, {diff[1]:.6e}]")
print()
print("  The difference is large! This is because X^T X is extremely ill-conditioned")
print("  (condition number ~ 10^13) due to x-values being around 2018-2024.")
print("  RLS computes (R_0 + X^T X)^{-1} X^T Y, and adding R_0=0.1*I changes the")
print("  smallest eigenvalue of X^T X from ~3e-5 to ~0.1, drastically altering the inverse.")
print()
print("  Note: both parameterizations give similar FITTED values (~3.1 at x=2020),")
print("  but very different intercept/slope decompositions due to collinearity.")
print()

# Try different R_0 scales
for r0s in [1000.0, 1e-6, 1e-8]:
    th_test, _ = rls(X_train, y_train, lam=1.0, R0_scale=r0s)
    d = th_test[N] - theta_ols
    print(f"  RLS (R0={r0s:.0e}*I): theta_N = [{th_test[N, 0]:.6f}, {th_test[N, 1]:.6f}]  "
          f"diff = [{d[0]:.4e}, {d[1]:.4e}]")

print()
print("  With R0=1e-8*I, the regularization is negligible relative to the smallest")
print("  eigenvalue of X^T X (~3e-5), so RLS converges very close to OLS.")
print("  Larger R0 values (0.1 or 1000) distort the solution because they dominate")
print("  the ill-conditioned direction of X^T X.")
print("  Lesson: for ill-conditioned problems, the initial R_0 must be much smaller")
print("  than the smallest eigenvalue of X^T X for RLS(lambda=1) to match OLS.")
print()

# 5.4 — RLS with forgetting: lambda=0.7 and lambda=0.99
theta_hist_07, _ = rls(X_train, y_train, lam=0.7, R0_scale=0.1)
theta_hist_099, _ = rls(X_train, y_train, lam=0.99, R0_scale=0.1)

burn_in = 4  # skip first 4 points

# Plot theta_1 over time
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_train[burn_in:], theta_hist_07[burn_in+1:N+1, 0], 'b-', linewidth=1.5,
        label='λ=0.7')
ax.plot(x_train[burn_in:], theta_hist_099[burn_in+1:N+1, 0], 'r-', linewidth=1.5,
        label='λ=0.99')
ax.set_xlabel('Year')
ax.set_ylabel('theta_1 (intercept)')
ax.set_title('Section 5.4: RLS theta_1 Over Time')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_5_4a_theta1.png', dpi=150)
plt.show()

# Plot theta_2 over time
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_train[burn_in:], theta_hist_07[burn_in+1:N+1, 1], 'b-', linewidth=1.5,
        label='λ=0.7')
ax.plot(x_train[burn_in:], theta_hist_099[burn_in+1:N+1, 1], 'r-', linewidth=1.5,
        label='λ=0.99')
ax.set_xlabel('Year')
ax.set_ylabel('theta_2 (slope)')
ax.set_title('Section 5.4: RLS theta_2 Over Time')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_5_4b_theta2.png', dpi=150)
plt.show()

print("=" * 70)
print("Section 5.4: RLS with Forgetting — Commentary")
print("=" * 70)
print()
print("  With lambda=0.7 (strong forgetting), the parameter estimates fluctuate more")
print("  and react strongly to recent data changes. The effective memory is short.")
print("  With lambda=0.99 (weak forgetting), the estimates are smoother and change")
print("  slowly, as the algorithm retains most of its history.")
print()
print("  Comparison of RLS at t=N with WLS (same lambda):")
for lam_cmp, th_hist in [(0.7, theta_hist_07), (0.99, theta_hist_099)]:
    w_cmp = np.array([lam_cmp**(N-1-i) for i in range(N)])
    XtW_cmp = (X_train * w_cmp[:, None]).T
    theta_wls_cmp = np.linalg.inv(XtW_cmp @ X_train) @ (XtW_cmp @ y_train)
    print(f"    lambda={lam_cmp}:")
    print(f"      RLS at t=N: [{th_hist[N, 0]:12.6f}, {th_hist[N, 1]:.6f}]")
    print(f"      WLS:        [{theta_wls_cmp[0]:12.6f}, {theta_wls_cmp[1]:.6f}]")
print()
print("  For lambda=0.7, RLS and WLS match closely because strong forgetting causes")
print("  the initial conditions to be 'forgotten'. For lambda=0.99, the mismatch is")
print("  larger because the initial R_0 persists longer in the R_t accumulation.")
print()

# 5.5 — One-step prediction residuals
# eps_{t|t-1} = x_t^T theta_{t-1} - y_t, plotted for t=5,...,N (1-indexed)
# In 0-indexed: for t_code=4,...,N-1, eps = X[t_code] @ theta_hist[t_code] - y[t_code]

eps_1step_07 = np.array([X_train[t] @ theta_hist_07[t] - y_train[t] for t in range(N)])
eps_1step_099 = np.array([X_train[t] @ theta_hist_099[t] - y_train[t] for t in range(N)])

t_plot = range(burn_in, N)  # skip first 4 (burn-in)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_train[burn_in:], eps_1step_07[burn_in:], 'b-', linewidth=1, alpha=0.8,
        label='λ=0.7')
ax.plot(x_train[burn_in:], eps_1step_099[burn_in:], 'r-', linewidth=1, alpha=0.8,
        label='λ=0.99')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('One-step residual')
ax.set_title('Section 5.5: One-Step Prediction Residuals')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_5_5_one_step_residuals.png', dpi=150)
plt.show()

print("=" * 70)
print("Section 5.5: One-Step Residuals Commentary")
print("=" * 70)
print()
print("  The one-step residuals for lambda=0.7 are generally smaller in magnitude")
print("  because the model adapts quickly to recent data. However, they may also")
print("  show more noise. Lambda=0.99 produces smoother but potentially larger")
print("  residuals, as the model is slower to adapt to changes in the data pattern.")
print()

# 5.6 — Optimize lambda: k-step RMSE heatmap
print("=" * 70)
print("Section 5.6: Lambda Optimization — k-step RMSE")
print("=" * 70)
print()

lambdas_grid = np.arange(0.50, 1.00, 0.01)  # 0.50, 0.51, ..., 0.99
ks = np.arange(1, 13)                        # 1, 2, ..., 12
rmse_matrix = np.zeros((len(lambdas_grid), len(ks)))

# Burn-in: skip predictions using theta estimates from the first few time steps,
# because the initialization (theta_0=[0,0], R_0=0.1*I) produces very poor
# estimates initially due to the ill-conditioned design matrix.
rmse_burn_in = 10

for i, lam_i in enumerate(lambdas_grid):
    th, _ = rls(X_train, y_train, lam=lam_i, R0_scale=0.1)
    for j, k in enumerate(ks):
        # eps_{t|t-k} = x_t^T theta_{t-k} - y_t  for t=k,...,N (1-indexed)
        # In code: t_code = k-1,...,N-1; theta index = t_code+1-k
        # We require theta index >= rmse_burn_in, so t_code+1-k >= rmse_burn_in
        # => t_code >= k + rmse_burn_in - 1
        t_start = k + rmse_burn_in - 1
        errors = []
        for t_code in range(t_start, N):
            theta_idx = t_code + 1 - k  # theta_history index
            pred = X_train[t_code] @ th[theta_idx]
            errors.append(pred - y_train[t_code])
        errors = np.array(errors)
        n_terms = len(errors)
        rmse_matrix[i, j] = np.sqrt(np.sum(errors**2) / n_terms)

# Find best lambda for each k
best_lam_per_k = np.zeros(len(ks))
best_rmse_per_k = np.zeros(len(ks))
for j, k in enumerate(ks):
    best_idx = np.argmin(rmse_matrix[:, j])
    best_lam_per_k[j] = lambdas_grid[best_idx]
    best_rmse_per_k[j] = rmse_matrix[best_idx, j]
    print(f"  k={k:2d}: best lambda = {best_lam_per_k[j]:.2f}, RMSE = {best_rmse_per_k[j]:.6f}")

# Overall best lambda (min average RMSE across all k)
avg_rmse = np.mean(rmse_matrix, axis=1)
best_overall_idx = np.argmin(avg_rmse)
best_overall_lam = lambdas_grid[best_overall_idx]
print(f"\n  Overall best lambda (min avg RMSE): {best_overall_lam:.2f}")
print()

# Heatmap
fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(rmse_matrix.T, aspect='auto', origin='lower',
               extent=[lambdas_grid[0] - 0.005, lambdas_grid[-1] + 0.005,
                       ks[0] - 0.5, ks[-1] + 0.5],
               cmap='viridis')
ax.set_xlabel('Lambda')
ax.set_ylabel('Forecast horizon k')
ax.set_title('Section 5.6: k-step RMSE (lambda x horizon)')
ax.set_yticks(ks)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('RMSE')
plt.tight_layout()
plt.savefig('fig_5_6_rmse_heatmap.png', dpi=150)
plt.show()

print("  Comment: There appears to be a pattern where the optimal lambda depends on")
print("  the forecast horizon. For short horizons (k=1,2), smaller lambda (more")
print("  forgetting) may be better as it adapts quickly. For longer horizons, a")
print("  larger lambda may be more stable. One could let lambda depend on the horizon.")
print()

# 5.7 — Predict test set using best RLS lambda; compare all methods
print("=" * 70)
print("Section 5.7: RLS Test Set Predictions — Comparison")
print("=" * 70)
print()

# Use overall best lambda
theta_hist_best, _ = rls(X_train, y_train, lam=best_overall_lam, R0_scale=0.1)
theta_rls_final = theta_hist_best[N]
y_pred_rls = X_test @ theta_rls_final
rls_test_rmse = np.sqrt(np.mean((y_pred_rls - y_test)**2))

print(f"  Best overall lambda: {best_overall_lam:.2f}")
print(f"  RLS theta at t=N: [{theta_rls_final[0]:.6f}, {theta_rls_final[1]:.6f}]")
print()
print(f"  {'Month':>8s}  {'OLS':>9s}  {'WLS 0.9':>9s}  {'RLS':>9s}  {'Actual':>9s}")
print(f"  {'-----':>8s}  {'---':>9s}  {'-------':>9s}  {'---':>9s}  {'------':>9s}")
for i in range(N_test):
    print(f"  {months_2024[i]:>8s}  {y_pred_ols[i]:9.4f}  {y_pred_wls[i]:9.4f}  "
          f"{y_pred_rls[i]:9.4f}  {y_test[i]:9.4f}")
print()
print(f"  Test RMSE — OLS: {ols_test_rmse:.4f}, WLS(0.9): {wls_test_rmse:.4f}, "
      f"RLS({best_overall_lam:.2f}): {rls_test_rmse:.4f}")
print()

# Also: per-horizon optimal lambda predictions
y_pred_rls_per_k = np.zeros(N_test)
for j in range(N_test):
    k = j + 1  # horizon
    lam_k = best_lam_per_k[j]
    th_k, _ = rls(X_train, y_train, lam=lam_k, R0_scale=0.1)
    y_pred_rls_per_k[j] = X_test[j] @ th_k[N]
rls_perk_rmse = np.sqrt(np.mean((y_pred_rls_per_k - y_test)**2))
print(f"  RLS (per-horizon optimal lambda) Test RMSE: {rls_perk_rmse:.4f}")
print()

# Comparison plot: all methods
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(x_train, y_train, c='steelblue', s=20, zorder=3, label='Training data')
ax.scatter(x_test, y_test, c='green', s=50, marker='D', zorder=4, label='Test data (actual)')

ax.plot(x_test, y_pred_ols, 'r-o', markersize=5, linewidth=1.5, label='OLS')
ax.plot(x_test, y_pred_wls, 'm-s', markersize=5, linewidth=1.5, label=f'WLS (λ={lam})')
ax.plot(x_test, y_pred_rls, 'b-^', markersize=5, linewidth=1.5,
        label=f'RLS (λ={best_overall_lam:.2f})')

# Fitted/forecast lines
x_ext = np.linspace(x_train[0], x_test[-1], 300)
ax.plot(x_ext, theta_ols[0] + theta_ols[1] * x_ext, 'r--', alpha=0.3, linewidth=1)
ax.plot(x_ext, theta_wls[0] + theta_wls[1] * x_ext, 'm--', alpha=0.3, linewidth=1)
ax.plot(x_ext, theta_rls_final[0] + theta_rls_final[1] * x_ext, 'b--', alpha=0.3, linewidth=1)

ax.axvline(x=2024.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Total vehicles (millions)')
ax.set_title('Section 5.7: All Methods Comparison — Test Set Predictions')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_5_7_all_methods.png', dpi=150)
plt.show()

# 5.8 — Reflections
print("=" * 70)
print("Section 5.8: Reflections on Time-Adaptive Models")
print("=" * 70)
print()
print("  1. Overfitting vs. Underfitting:")
print("     Small lambda (strong forgetting) makes the model very reactive to recent")
print("     data, which can lead to overfitting short-term noise. Large lambda (weak")
print("     forgetting) produces smoother estimates but may underfit if the underlying")
print("     process has changed significantly over time.")
print()
print("  2. Challenges with time-dependent test sets:")
print("     Unlike cross-sectional data where random train/test splits work, time")
print("     series data requires chronological splits. The test set comes AFTER the")
print("     training set, so the model must extrapolate. If the data-generating process")
print("     changes (structural breaks), the test set may not reflect training patterns.")
print()
print("  3. Recursive estimation and test set challenges:")
print("     RLS can partially alleviate test set challenges because it adapts to changing")
print("     patterns. By choosing an appropriate forgetting factor, the model focuses on")
print("     the most relevant recent data. However, it still relies on the assumption")
print("     that recent trends continue into the future.")
print()
print("  4. Other techniques for time-adaptive estimation:")
print("     - Kalman filtering (state-space models) — provides optimal recursive estimation")
print("       under Gaussian assumptions with separate process and observation noise.")
print("     - Exponential smoothing methods (Holt, Holt-Winters).")
print("     - Rolling/expanding window OLS.")
print("     - ARIMA/SARIMA models that capture autocorrelation and seasonality.")
print("     - Bayesian online learning with time-varying priors.")
print()
print("  5. Additional thoughts:")
print("     The optimal lambda may depend on the forecast horizon, suggesting that")
print("     a single forgetting factor may not be optimal for all purposes. An adaptive")
print("     forgetting factor that changes over time could be beneficial. The choice of")
print("     initialization (R_0, theta_0) matters for short time series but becomes less")
print("     important as more data is observed.")
print()
print("=" * 70)
print("Assignment 1 complete.")
print("=" * 70)
