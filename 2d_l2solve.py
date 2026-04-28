import numpy as np, matplotlib.pyplot as plt
from setup_2d import build, IMK, N, SIGMA, NOISE

np.random.seed(42)

theta_true, K2D, y_noisy, cond = build()

LAM = 0.5
KtK = K2D.T @ K2D
Kty = K2D.T @ y_noisy
theta_l2 = np.linalg.solve(KtK + LAM * np.eye(N * N), Kty).reshape(N, N)
err = np.linalg.norm(theta_l2 - theta_true) / np.linalg.norm(theta_true)
print(f"L2 Tikhonov error: {err:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

im0 = axes[0].imshow(theta_true, **IMK)
axes[0].set_title("Ground truth  θ*", fontsize=11)
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(theta_l2, **IMK)
axes[1].set_title(
    f"L2 Tikhonov  (λ={LAM})\nerror = {err:.3f}",
    fontsize=11,
    color="#e08010",
    fontweight="bold",
)
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

axes[1].text(
    0.5,
    -0.05,
    "Stable but over-smoothed — hot blocks smeared into blurry blobs",
    transform=axes[1].transAxes,
    ha="center",
    color="#e08010",
    fontsize=10,
    fontweight="bold",
)

plt.suptitle(
    f"L2 Tikhonov: min ||K·θ - y||²  +  λ||θ||²   (λ={LAM})\n"
    "L2 prior = small energy → no preference for sharp boundaries",
    fontsize=11,
    y=1.03,
)
plt.tight_layout()
plt.savefig("2d_l2solve.png", dpi=150, bbox_inches="tight")
plt.close()
