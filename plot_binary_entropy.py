import numpy as np
import matplotlib.pyplot as plt

def binary_entropy(p: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

p = np.linspace(0.0, 1.0, 1001)
H = binary_entropy(p)

fig, ax = plt.subplots()
ax.plot(p, H, linewidth=2, color="navy", zorder=1)

# Updated labels/titles
ax.set_title(r"Binary Entropy $H(p)$")
ax.set_xlabel(r"$p = \Pr(X=1)$")
ax.set_ylabel("H(p) [bits]")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05 * H.max())
ax.grid(True, alpha=0.3)

# Max marker in red
ax.scatter([0.5], [1.0], color="red", zorder=3, s=60)

# Shorter arrow from below, avoiding intersection with the curve
ax.annotate(
    "Maximum at p = 0.5 (1 bit)",
    xy=(0.5, 1.0),
    xytext=(0.5, 0.75),          # closer -> shorter arrow
    ha="center",
    va="top",
    arrowprops=dict(arrowstyle="->", lw=1.5, shrinkA=0, shrinkB=6),
    zorder=2,
)

plt.show()
