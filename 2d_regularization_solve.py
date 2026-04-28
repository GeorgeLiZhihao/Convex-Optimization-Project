import numpy as np, matplotlib.pyplot as plt, cvxpy as cp
from setup_2d import build, IMK, N, SIGMA, NOISE

np.random.seed(42)
theta_true, K2D, y_noisy, cond = build()


LAM_L2 = 0.5
KtK = K2D.T @ K2D
Kty = K2D.T @ y_noisy
theta_l2 = np.linalg.solve(KtK + LAM_L2 * np.eye(N * N), Kty).reshape(N, N)
err_l2 = np.linalg.norm(theta_l2 - theta_true) / np.linalg.norm(theta_true)


LAM_TV = 0.3
v = cp.Variable(N * N)
Tm = cp.reshape(v, (N, N), order="C")
tv2d = cp.sum(cp.abs(Tm[:, 1:] - Tm[:, :-1])) + cp.sum(cp.abs(Tm[1:, :] - Tm[:-1, :]))
cp.Problem(
    cp.Minimize(cp.sum_squares(K2D @ v - y_noisy) + LAM_TV * tv2d), [v >= 0.0]
).solve(solver=cp.CLARABEL, verbose=False)
theta_tv = v.value.reshape(N, N)
err_tv = np.linalg.norm(theta_tv - theta_true) / np.linalg.norm(theta_true)
print(f"TV error: {err_tv:.4f}  ({(err_l2-err_tv)/err_l2*100:.0f}% better than L2)")

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

im0 = axes[0].imshow(theta_true, **IMK)
axes[0].set_title("Ground truth  θ*", fontsize=11)
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(theta_tv, **IMK)
axes[1].set_title(
    f"TV convex optimisation  (λ={LAM_TV})\nerror = {err_tv:.3f}",
    fontsize=11,
    color="#1a9641",
    fontweight="bold",
)
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

axes[1].text(
    0.5,
    -0.05,
    f"Sharp edges recovered — {(err_l2-err_tv)/err_l2*100:.0f}% better than L2 Tikhonov",
    transform=axes[1].transAxes,
    ha="center",
    color="#1a9641",
    fontsize=10,
    fontweight="bold",
)

plt.suptitle(
    f"TV convex opt.: min ||K·θ - y||²  +  λ·TV(θ),  θ≥0   (λ={LAM_TV})\n"
    "TV prior = piecewise-constant → matches ground truth exactly",
    fontsize=11,
    y=1.03,
)
plt.tight_layout()
plt.savefig("2d_regularization_solve.png", dpi=150, bbox_inches="tight")
plt.close()
