import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from setup_3d import (
    build,
    draw_voxels,
    style_ax,
    N,
    SIGMA,
    NOISE,
    CMAP,
    NORM,
    THETA_MAX,
)


np.random.seed(42)
theta_true, K3D, y_noisy, cond = build()

LAM = 0.05
KtK = K3D.T @ K3D
Kty = K3D.T @ y_noisy
theta_l2 = np.linalg.solve(KtK + LAM * np.eye(N**3), Kty)
theta_l2 = np.clip(theta_l2.reshape(N, N, N), 0.0, THETA_MAX)
err = np.linalg.norm(theta_l2 - theta_true) / np.linalg.norm(theta_true)
print(f"L2 Tikhonov error: {err:.4f}")

fig = plt.figure(figsize=(11, 5.5))
fig.patch.set_facecolor("white")

ax0 = fig.add_subplot(1, 2, 1, projection="3d")
draw_voxels(ax0, theta_true)
style_ax(ax0, "Ground truth  θ*", "black")

ax1 = fig.add_subplot(1, 2, 2, projection="3d")
draw_voxels(ax1, theta_l2)
style_ax(ax1, f"L2 Tikhonov  (λ={LAM})\nerror = {err:.3f}", "#e08010")

cax = fig.add_axes([0.92, 0.2, 0.012, 0.6])
cb = fig.colorbar(cm.ScalarMappable(cmap=CMAP, norm=NORM), cax=cax)
cb.set_label("Temperature θ", fontsize=9)
cb.ax.tick_params(labelsize=7)

fig.suptitle(
    f"L2 Tikhonov: min ||K·θ - y||²  +  λ||θ||²   (λ={LAM})\n"
    "L2 prior = small energy — no preference for sharp cube boundaries → over-smoothed",
    fontsize=11,
    y=1.02,
)
plt.subplots_adjust(left=0.02, right=0.90, wspace=0.05)
plt.savefig("3d_l2solve.png", dpi=150, bbox_inches="tight")
plt.close()
