import numpy as np
from scipy.linalg import toeplitz

N, SIGMA, NOISE = 32, 2.5, 0.04


def build():
    theta_true = np.ones((N, N)) * 0.5
    theta_true[3:13, 3:13] = 5.0
    theta_true[19:29, 19:29] = 4.0
    theta_true[3:10, 20:28] = 3.0
    t = np.arange(N)
    row = np.exp(-(t**2) / (2 * SIGMA**2))
    row /= row.sum()
    K1 = toeplitz(row, row)
    K2D = np.kron(K1, K1)
    cond = np.linalg.cond(K2D)
    y_clean = K2D @ theta_true.ravel()
    y_noisy = y_clean + NOISE * y_clean.std() * np.random.randn(N * N)
    return theta_true, K2D, y_noisy, cond


IMK = dict(origin="lower", cmap="inferno", interpolation="nearest", vmin=0.0, vmax=5.5)

