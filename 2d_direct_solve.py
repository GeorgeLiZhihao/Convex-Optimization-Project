import numpy as np, matplotlib.pyplot as plt
from setup_2d import build, IMK, N, SIGMA, NOISE

np.random.seed(42)
theta_true, K2D, y_noisy, cond = build()

result, _, _, _ = np.linalg.lstsq(K2D, y_noisy, rcond=None)
theta_ls = result.reshape(N, N)
err = np.linalg.norm(theta_ls - theta_true) / np.linalg.norm(theta_true)
print(f"Direct LS error: {err:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

im0 = axes[0].imshow(theta_true, **IMK)
axes[0].set_title("Ground truth  θ*", fontsize=11)
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(theta_ls, cmap="inferno", origin="lower", interpolation="nearest")
axes[1].set_title(
    f"Direct LS deconvolution\nerror = {err:.3f}",
    fontsize=11,
    color="#c0392b",
    fontweight="bold",
)
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

axes[1].text(
    0.5,
    -0.05,
    f"Noise amplified by x{cond:.0e} — structure completely destroyed",
    transform=axes[1].transAxes,
    ha="center",
    color="#c0392b",
    fontsize=10,
    fontweight="bold",
)

plt.suptitle(
    "Direct LS: min ||K·θ - y||²   (no regularisation)\n"
    "Minimum-norm solution — no sparsity prior — catastrophic failure",
    fontsize=11,
    y=1.03,
)
plt.tight_layout()
plt.savefig("2d_direct_solve.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved out_2d_2_direct_ls.png")
