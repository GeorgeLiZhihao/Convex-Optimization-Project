import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cvxpy as cp
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

# L2 error for comparison
KtK = K3D.T @ K3D
Kty = K3D.T @ y_noisy
theta_l2 = np.clip(
    np.linalg.solve(KtK + 0.05 * np.eye(N**3), Kty).reshape(N, N, N), 0.0, THETA_MAX
)
err_l2 = np.linalg.norm(theta_l2 - theta_true) / np.linalg.norm(theta_true)

# TV solve
LAM = 0.01
v = cp.Variable(N**3)
T3 = cp.reshape(v, (N, N, N), order="C")
tv3d = (
    cp.sum(cp.abs(T3[1:, :, :] - T3[:-1, :, :]))
    + cp.sum(cp.abs(T3[:, 1:, :] - T3[:, :-1, :]))
    + cp.sum(cp.abs(T3[:, :, 1:] - T3[:, :, :-1]))
)
cp.Problem(
    cp.Minimize(cp.sum_squares(K3D @ v - y_noisy) + LAM * tv3d),
    [v >= 0.0, v <= THETA_MAX],
).solve(solver=cp.CLARABEL, verbose=False)
theta_tv = v.value.reshape(N, N, N)
err_tv = np.linalg.norm(theta_tv - theta_true) / np.linalg.norm(theta_true)
print(f"TV error: {err_tv:.4f}  ({(err_l2-err_tv)/err_l2*100:.0f}% better than L2)")

fig = plt.figure(figsize=(11, 5.5))
fig.patch.set_facecolor("white")

ax0 = fig.add_subplot(1, 2, 1, projection="3d")
draw_voxels(ax0, theta_true)
style_ax(ax0, "Ground truth  θ*", "black")

ax1 = fig.add_subplot(1, 2, 2, projection="3d")
draw_voxels(ax1, theta_tv)
style_ax(ax1, f"TV convex opt.  (λ={LAM})\nerror = {err_tv:.3f}", "#1a9641")

cax = fig.add_axes([0.92, 0.2, 0.012, 0.6])
cb = fig.colorbar(cm.ScalarMappable(cmap=CMAP, norm=NORM), cax=cax)
cb.set_label("Temperature θ", fontsize=9)
cb.ax.tick_params(labelsize=7)

fig.suptitle(
    f"TV convex opt.: min ||K·θ - y||²  +  λ·TV₃(θ),  0≤θ≤{THETA_MAX}   (λ={LAM})\n"
    f"TV prior = piecewise-constant → sharp cube boundaries recovered  "
    f"({(err_l2-err_tv)/err_l2*100:.0f}% better than L2)",
    fontsize=11,
    y=1.02,
)
plt.subplots_adjust(left=0.02, right=0.90, wspace=0.05)
plt.savefig("3d_regularization.png", dpi=150, bbox_inches="tight")
plt.close()
