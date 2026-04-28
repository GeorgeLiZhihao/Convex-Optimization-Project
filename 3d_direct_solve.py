import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from setup_3d import build, draw_voxels, style_ax, N, SIGMA, NOISE, CMAP, NORM


np.random.seed(42)
theta_true, K3D, y_noisy, cond = build()

result, _, _, _ = np.linalg.lstsq(K3D, y_noisy, rcond=None)
theta_ls = result.reshape(N, N, N)
err = np.linalg.norm(theta_ls - theta_true) / np.linalg.norm(theta_true)
print(f"Direct LS error: {err:.4f}")

fig = plt.figure(figsize=(11, 5.5))
fig.patch.set_facecolor("white")

ax0 = fig.add_subplot(1, 2, 1, projection="3d")
draw_voxels(ax0, theta_true)
style_ax(ax0, "Ground truth  θ*", "black")

ax1 = fig.add_subplot(1, 2, 2, projection="3d")
draw_voxels(ax1, np.clip(theta_ls, 0.0, 5.5))
style_ax(ax1, f"Direct LS deconvolution\nerror = {err:.2e}", "#c0392b")

cax = fig.add_axes([0.92, 0.2, 0.012, 0.6])
cb = fig.colorbar(cm.ScalarMappable(cmap=CMAP, norm=NORM), cax=cax)
cb.set_label("Temperature θ", fontsize=9)
cb.ax.tick_params(labelsize=7)

fig.suptitle(
    f"Direct LS: min ||K·θ - y||²   (no regularisation)\n"
    f"Noise x{cond:.0e} — structure completely destroyed (values clipped for display)",
    fontsize=11,
    y=1.02,
)
plt.subplots_adjust(left=0.02, right=0.90, wspace=0.05)
plt.savefig("3d_direct_solve.png", dpi=150, bbox_inches="tight")
plt.close()
