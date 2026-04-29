"""Kalman filter implementations (scalar + multivariate) for Assignment 4."""

import numpy as np
from scipy.linalg import solve, cho_factor, cho_solve

LOG2PI = np.log(2 * np.pi)


# ---------------------------------------------------------------------------
# Scalar Kalman filter (Part 1)
# ---------------------------------------------------------------------------

def kf_scalar(y, a, b, sigma1, R, x0=0.0, P0=10.0):
    """Kalman filter for X_{t+1}=a X_t + b + e1, Y_t = X_t + e2.

    Returns dict with predicted mean/var, filtered mean/var, innovation
    and its variance, plus total log-likelihood.
    """
    y = np.asarray(y, float)
    N = len(y)
    x_pred = np.empty(N)
    P_pred = np.empty(N)
    x_filt = np.empty(N)
    P_filt = np.empty(N)
    innov = np.empty(N)
    S = np.empty(N)
    logL = 0.0
    sigma1_sq = sigma1 ** 2

    for t in range(N):
        # prediction
        if t == 0:
            x_pred[t] = a * x0 + b
            P_pred[t] = a * a * P0 + sigma1_sq
        else:
            x_pred[t] = a * x_filt[t - 1] + b
            P_pred[t] = a * a * P_filt[t - 1] + sigma1_sq

        # innovation
        innov[t] = y[t] - x_pred[t]
        S[t] = P_pred[t] + R

        # likelihood contribution
        logL += -0.5 * (LOG2PI + np.log(S[t]) + innov[t] ** 2 / S[t])

        # update
        K = P_pred[t] / S[t]
        x_filt[t] = x_pred[t] + K * innov[t]
        P_filt[t] = (1.0 - K) * P_pred[t]

    return dict(x_pred=x_pred, P_pred=P_pred,
                x_filt=x_filt, P_filt=P_filt,
                innov=innov, S=S, logLik=logL)


def neg_loglik_scalar(theta, y, R=1.0, x0=0.0, P0=10.0):
    """Negative log-likelihood for (a, b, sigma1)."""
    a, b, sigma1 = theta
    if sigma1 <= 0:
        return 1e10
    try:
        res = kf_scalar(y, a, b, sigma1, R, x0, P0)
    except (FloatingPointError, OverflowError):
        return 1e10
    if not np.isfinite(res["logLik"]):
        return 1e10
    return -res["logLik"]


# ---------------------------------------------------------------------------
# Multivariate Kalman filter (Part 2)
# ---------------------------------------------------------------------------

def kf_mv(Y, U, A, B, C, Q, R, x0, P0):
    """Standard linear KF.

    Y : (T, m) observations
    U : (T, p) inputs (use zeros if no input)
    A : (n, n), B : (n, p), C : (m, n)
    Q : (n, n) system noise covariance
    R : (m, m) observation noise covariance
    x0: (n,) or (n,1); P0: (n, n)

    Returns arrays of shape (T, ...) and total log-likelihood.
    """
    Y = np.atleast_2d(Y)
    if Y.shape[0] < Y.shape[1] and Y.ndim == 2 and Y.shape[0] in (1,):
        Y = Y.T
    U = np.atleast_2d(U)
    T = Y.shape[0]
    n = A.shape[0]
    m = C.shape[0]

    x_pred = np.zeros((T, n))
    P_pred = np.zeros((T, n, n))
    x_filt = np.zeros((T, n))
    P_filt = np.zeros((T, n, n))
    innov = np.zeros((T, m))
    S_all = np.zeros((T, m, m))
    logL = 0.0

    x = np.asarray(x0, float).reshape(n)
    P = np.asarray(P0, float).reshape(n, n)

    for t in range(T):
        # ---- prediction
        xp = A @ x + B @ U[t]
        Pp = A @ P @ A.T + Q
        x_pred[t] = xp
        P_pred[t] = Pp

        # ---- innovation
        yp = C @ xp
        e = Y[t] - yp
        S = C @ Pp @ C.T + R
        innov[t] = e
        S_all[t] = S

        # ---- log-likelihood
        try:
            cho, low = cho_factor(S, lower=True)
            sol = cho_solve((cho, low), e)
            logdet = 2.0 * np.sum(np.log(np.diag(cho)))
        except np.linalg.LinAlgError:
            return dict(logLik=-np.inf)
        logL += -0.5 * (m * LOG2PI + logdet + e @ sol)

        # ---- update
        K = solve(S.T, (Pp @ C.T).T).T  # P C^T S^{-1}
        x = xp + K @ e
        P = Pp - K @ C @ Pp
        # symmetrize for numerical stability
        P = 0.5 * (P + P.T)
        x_filt[t] = x
        P_filt[t] = P

    return dict(x_pred=x_pred, P_pred=P_pred,
                x_filt=x_filt, P_filt=P_filt,
                innov=innov, S=S_all, logLik=logL)


# ---------- helpers to build parameter matrices with positivity guarantees ----------

def build_chol_lower(vals, n):
    """Lower-triangular matrix from length n*(n+1)/2 vector (row-major)."""
    L = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1):
            L[i, j] = vals[idx]
            idx += 1
    return L


def n_chol_params(n):
    return n * (n + 1) // 2
