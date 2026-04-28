import sys
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import importlib
setup_2d = importlib.import_module("setup_2d(1)")
IMK, N_H, N_W, SIGMA, NOISE, _create_ellipse_mask = setup_2d.IMK, setup_2d.N_H, setup_2d.N_W, setup_2d.SIGMA, setup_2d.NOISE, setup_2d._create_ellipse_mask
from scipy.linalg import toeplitz

np.random.seed(42)

def build_smooth():
    mask = _create_ellipse_mask(N_H, N_W)
    
    yy, xx = np.mgrid[:N_H, :N_W]
    theta_true = np.ones((N_H, N_W)) * 0.5
    
    theta_true += 4.5 * np.exp(-((yy - 11)**2 + (xx - 10)**2) / (2 * 4.0**2))
    theta_true += 3.5 * np.exp(-((yy - 29)**2 + (xx - 24)**2) / (2 * 4.0**2))
    theta_true += 2.5 * np.exp(-((yy - 12)**2 + (xx - 26)**2) / (2 * 4.0**2))
    
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

theta_true, K2D, y_noisy, cond, mask = build_smooth()

# 1. Baseline
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
im0 = axes[0].imshow(theta_true, **IMK)
axes[0].set_title("Ground truth  θ*(x,y,0)\nSmooth Gaussian sources", fontsize=11)
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(y_noisy.reshape(N_H, N_W), **IMK)
axes[1].set_title(f"Observation  y = K_T · θ* + noise\nHeat diffusion σ={SIGMA}", fontsize=11)
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046)
plt.suptitle("Problem: recover smooth initial temperature from blurry noisy observation", fontsize=11, y=1.03)
plt.tight_layout()
plt.savefig("smooth_baseline.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Direct Solve
result, _, _, _ = np.linalg.lstsq(K2D, y_noisy, rcond=None)
theta_ls = result.reshape(N_H, N_W)
err_ds = np.linalg.norm(theta_ls - theta_true) / np.linalg.norm(theta_true)

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
im0 = axes[0].imshow(theta_true, **IMK)
axes[0].set_title("Ground truth  θ*", fontsize=11)
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(theta_ls, cmap="inferno", origin="lower", interpolation="nearest")
axes[1].set_title(f"Direct LS deconvolution\nerror = {err_ds:.3f}", fontsize=11, color="#c0392b", fontweight="bold")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046)
plt.suptitle("Direct LS on Smooth GT", fontsize=11, y=1.03)
plt.tight_layout()
plt.savefig("smooth_direct_solve.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. L2 Solve
LAM_L2 = 0.5
KtK = K2D.T @ K2D
Kty = K2D.T @ y_noisy
theta_l2 = np.linalg.solve(KtK + LAM_L2 * np.eye(N_H * N_W), Kty).reshape(N_H, N_W)
err_l2 = np.linalg.norm(theta_l2 - theta_true) / np.linalg.norm(theta_true)

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
im0 = axes[0].imshow(theta_true, **IMK)
axes[0].set_title("Ground truth  θ*", fontsize=11)
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(theta_l2, **IMK)
axes[1].set_title(f"L2 Tikhonov  (λ={LAM_L2})\nerror = {err_l2:.3f}", fontsize=11, color="#e08010", fontweight="bold")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046)
plt.suptitle("L2 Tikhonov on Smooth GT", fontsize=11, y=1.03)
plt.tight_layout()
plt.savefig("smooth_l2solve.png", dpi=150, bbox_inches="tight")
plt.close()

# 4. TV Solve
LAM_TV = 0.3
v = cp.Variable(N_H * N_W)
Tm = cp.reshape(v, (N_H, N_W), order="C")
tv2d = cp.sum(cp.abs(Tm[:, 1:] - Tm[:, :-1])) + cp.sum(cp.abs(Tm[1:, :] - Tm[:-1, :]))
cp.Problem(
    cp.Minimize(cp.sum_squares(K2D @ v - y_noisy) + LAM_TV * tv2d), [v >= 0.0]
).solve(solver=cp.CLARABEL, verbose=False)
theta_tv = v.value.reshape(N_H, N_W)
err_tv = np.linalg.norm(theta_tv - theta_true) / np.linalg.norm(theta_true)

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
im0 = axes[0].imshow(theta_true, **IMK)
axes[0].set_title("Ground truth  θ*", fontsize=11)
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(theta_tv, **IMK)
axes[1].set_title(f"TV convex optimisation  (λ={LAM_TV})\nerror = {err_tv:.3f}", fontsize=11, color="#1a9641", fontweight="bold")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046)
plt.suptitle("TV Regularization on Smooth GT", fontsize=11, y=1.03)
plt.tight_layout()
plt.savefig("smooth_tvsolve.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Errors: Direct {err_ds:.4f}, L2 {err_l2:.4f}, TV {err_tv:.4f}")
