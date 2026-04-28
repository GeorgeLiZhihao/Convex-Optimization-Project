import numpy as np
from scipy.linalg import toeplitz

N_H, N_W, SIGMA, NOISE = 40, 32, 2.5, 0.04
N = N_H * N_W


def _create_ellipse_mask(h, w, center_h=None, center_w=None, semi_h=None, semi_w=None):
    if center_h is None:
        center_h = h / 2
    if center_w is None:
        center_w = w / 2
    if semi_h is None:
        semi_h = h / 2 - 2
    if semi_w is None:
        semi_w = w / 2 - 2
    
    yy, xx = np.ogrid[:h, :w]
    ellipse = ((xx - center_w) ** 2 / semi_w ** 2 + 
               (yy - center_h) ** 2 / semi_h ** 2) <= 1
    return ellipse


def build():
    mask = _create_ellipse_mask(N_H, N_W)
    
    theta_true = np.ones((N_H, N_W)) * 0.5 
    theta_true[6:16, 6:14] = 5.0
    theta_true[24:34, 20:28] = 4.0
    theta_true[8:16, 22:30] = 3.0
    theta_true[~mask] = 0
    
    t_h = np.arange(N_H)
    t_w = np.arange(N_W)
    row_h = np.exp(-(t_h**2) / (2 * SIGMA**2))
    row_h /= row_h.sum()
    row_w = np.exp(-(t_w**2) / (2 * SIGMA**2))
    row_w /= row_w.sum()
    
    K_h = toeplitz(row_h, row_h)
    K_w = toeplitz(row_w, row_w)
    K2D = np.kron(K_h, K_w)
    
    cond = np.linalg.cond(K2D)
    y_clean = K2D @ theta_true.ravel()
    y_noisy = y_clean + NOISE * y_clean.std() * np.random.randn(N_H * N_W)
    y_noisy[~mask.ravel()] = 0
    
    return theta_true, K2D, y_noisy, cond, mask


IMK = dict(origin="lower", cmap="inferno", interpolation="nearest", vmin=0.0, vmax=5.5)
