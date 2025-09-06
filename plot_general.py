import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

# ---- Functions (as requested) ----
def h2(p: float) -> float:
    """
    Binary entropy function in bits per *bit*:
        h2(p) = -p*log2(p) - (1-p)*log2(1-p), with the conventions 0*log2(0)=0.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))

def ascii01_bpb(p: float) -> float:
    """
    Bits per *byte* for ASCII '0'/'1' representation (one byte per bit).
    Since the byte alphabet has only two symbols here, bpb = h2(p).
    """
    return h2(p)

def bitpack_bpb(p: float) -> float:
    """
    Bits per *byte* for bit-packed representation of IID Bernoulli(p) bits.
    Each byte carries 8 iid bits, so bpb = 8*h2(p).
    """
    return 8.0 * h2(p)

# Vectorized versions for plotting
h2_vec = np.vectorize(h2)
ascii01_bpb_vec = np.vectorize(ascii01_bpb)
bitpack_bpb_vec = np.vectorize(bitpack_bpb)

# ---- Generate plot of h2(p) ----
p = np.linspace(0.0, 1.0, 1001)
H = h2_vec(p)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(p, H, label="h\u2082(p)  [bits/bit]")  # default matplotlib styling, no explicit colors
ax.set_xlabel("p = P(bit=1)")
ax.set_ylabel("Entropy (bits per bit)")
ax.set_title("Binary Entropy h\u2082(p)")
ax.grid(True, which="both", linestyle="--", alpha=0.3)
# Mark the maximum at p=0.5
ax.scatter([0.5], [h2(0.5)], marker="o")
ax.annotate("max = 1 bit @ p=0.5", xy=(0.5, h2(0.5)), xytext=(0.62, 0.9),
            arrowprops=dict(arrowstyle="->"), fontsize=9)

# Save to results/
out_dir = Path("results/plots")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "binary_entropy_h2_p.png"
fig.tight_layout()
fig.savefig(out_path, dpi=150)

# Show the plot in the notebook
plt.show()

# Provide a couple example conversions for reference
sample_ps = [0.1, 0.3, 0.5, 0.9]
table = [
    (pval, h2(pval), ascii01_bpb(pval), bitpack_bpb(pval))
    for pval in sample_ps
]
table
