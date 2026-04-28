"""
3-D Heat Equation Inverse Problem
File 1 of 4 — Problem Description
===================================
Shows the 3-D setup: ground truth hot cubes vs blurred observation.
"""

import numpy as np, matplotlib.pyplot as plt


np.random.seed(42)

from setup_3d import build, draw_voxels, style_ax, N, SIGMA, NOISE, CMAP, NORM
import matplotlib.cm as cm

theta_true, K3D, y_noisy, cond = build()
theta_obs = y_noisy.reshape(N, N, N)

fig = plt.figure(figsize=(11, 5.5))
fig.patch.set_facecolor("white")

ax0 = fig.add_subplot(1, 2, 1, projection="3d")
draw_voxels(ax0, theta_true)
style_ax(ax0, "Ground truth  θ*(x,y,z,0)\n3 piecewise-constant hot cubes", "black")

ax1 = fig.add_subplot(1, 2, 2, projection="3d")
draw_voxels(ax1, theta_obs, threshold=0.8)
style_ax(
    ax1,
    f"Observation  y = K·θ + noise\nσ={SIGMA}, noise={int(NOISE*100)}%  (edges lost)",
    "#555555",
)

cax = fig.add_axes([0.92, 0.2, 0.012, 0.6])
cb = fig.colorbar(cm.ScalarMappable(cmap=CMAP, norm=NORM), cax=cax)
cb.set_label("Temperature θ", fontsize=9)
cb.ax.tick_params(labelsize=7)

fig.suptitle(
    f"Problem: recover sharp hot cubes from blurry noisy 3-D observation\n"
    f"cond(K) ≈ {cond:.0e} — direct inversion catastrophic",
    fontsize=11,
    y=1.02,
)
plt.subplots_adjust(left=0.02, right=0.90, wspace=0.05)
plt.savefig("3d_baseline.png", dpi=150, bbox_inches="tight")
plt.close()
