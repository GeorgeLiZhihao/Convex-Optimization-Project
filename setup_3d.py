import numpy as np
from scipy.linalg import toeplitz
import matplotlib.cm as cm, matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

N, SIGMA, NOISE = 12, 2.0, 0.04
THETA_MAX = 5.0
VMIN, VMAX, THRESHOLD = 0.0, 5.5, 1.5
CMAP = cm.inferno
NORM = mcolors.Normalize(VMIN, VMAX)


def build():
    theta_true = np.ones((N, N, N)) * 0.5
    theta_true[1:6, 1:6, 1:6] = 5.0
    theta_true[7:11, 7:11, 7:11] = 4.0
    theta_true[1:5, 7:11, 1:5] = 3.0
    t = np.arange(N)
    row = np.exp(-(t**2) / (2 * SIGMA**2))
    row /= row.sum()
    K1 = toeplitz(row, row)
    K3D = np.kron(np.kron(K1, K1), K1)
    cond = np.linalg.cond(K3D)
    y_clean = K3D @ theta_true.ravel()
    y_noisy = y_clean + NOISE * y_clean.std() * np.random.randn(N**3)
    return theta_true, K3D, y_noisy, cond


def _cube_faces(i, j, k):
    x, y, z = i, j, k
    return [
        [[x, y, z], [x + 1, y, z], [x + 1, y + 1, z], [x, y + 1, z]],
        [[x, y, z + 1], [x + 1, y, z + 1], [x + 1, y + 1, z + 1], [x, y + 1, z + 1]],
        [[x, y, z], [x + 1, y, z], [x + 1, y, z + 1], [x, y, z + 1]],
        [[x, y + 1, z], [x + 1, y + 1, z], [x + 1, y + 1, z + 1], [x, y + 1, z + 1]],
        [[x, y, z], [x, y + 1, z], [x, y + 1, z + 1], [x, y, z + 1]],
        [[x + 1, y, z], [x + 1, y + 1, z], [x + 1, y + 1, z + 1], [x + 1, y, z + 1]],
    ]


def draw_voxels(ax, vol, threshold=THRESHOLD, alpha=0.82):
    faces, colors = [], []
    clipped = np.clip(vol, VMIN, VMAX)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if vol[i, j, k] > threshold:
                    rgba = list(CMAP(NORM(clipped[i, j, k])))
                    rgba[3] = alpha
                    for f in _cube_faces(i, j, k):
                        faces.append(f)
                        colors.append(rgba)
    if faces:
        ax.add_collection3d(
            Poly3DCollection(
                faces, facecolors=colors, linewidths=0.0, edgecolors="none"
            )
        )


def style_ax(ax, title, bc):
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_zlim(0, N)
    ax.set_xlabel("x", fontsize=8, labelpad=-3)
    ax.set_ylabel("y", fontsize=8, labelpad=-3)
    ax.set_zlabel("z", fontsize=8, labelpad=-3)
    ax.tick_params(labelsize=6, pad=0)
    ax.view_init(elev=22, azim=40)
    ax.set_box_aspect([1, 1, 1])
    ax.set_facecolor("#f8f8f8")
    ax.grid(False)
    for p in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        p.fill = False
        p.set_edgecolor("#cccccc")
    ax.set_title(title, fontsize=12, fontweight="bold", color=bc, pad=8)
