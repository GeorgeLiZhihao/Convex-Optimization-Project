import numpy as np, matplotlib.pyplot as plt
from setup_2d import build, IMK, N, SIGMA, NOISE


np.random.seed(42)
theta_true, K2D, y_noisy, cond = build()

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

im0 = axes[0].imshow(theta_true, **IMK)
axes[0].set_title(
    "Ground truth  θ*(x,y,0)\n" "Piecewise-constant: 3 hot blocks on cool background",
    fontsize=11,
)
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(y_noisy.reshape(N, N), **IMK)
axes[1].set_title(
    f"Observation  y = K_T · θ* + noise\n"
    f"Heat diffusion σ={SIGMA}, noise={int(NOISE*100)}%  "
    f"edges completely lost",
    fontsize=11,
)
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

plt.suptitle(
    "Problem: recover the sharp initial temperature from a blurry noisy observation\n"
    f"Forward operator K is a Gaussian blur — condition number = {cond:.0e}",
    fontsize=11,
    y=1.03,
)
plt.tight_layout()
plt.savefig("2d_baseline", dpi=150, bbox_inches="tight")
plt.close()
print("saved out_2d_1_problem.png")
