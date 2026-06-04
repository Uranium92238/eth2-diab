import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- User settings ---
MATRIX_FILE = "S_MO.npy"
ROW_START, ROW_END = 10, 30
COL_START, COL_END = 10, 30
# ---------------------

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

CMAP   = "RdBu_r"
OUTDIR = Path("hplots")
OUTDIR.mkdir(exist_ok=True)

mat = np.load(MATRIX_FILE)
mat = mat[ROW_START:ROW_END, COL_START:COL_END]
N_rows, N_cols = mat.shape

vmax = np.max(np.abs(mat))

fig, ax = plt.subplots(figsize=(8, 7))

im = ax.imshow(mat, cmap=CMAP, vmin=-vmax, vmax=vmax, origin="upper", aspect="equal")

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Overlap", fontsize=13)
cbar.ax.tick_params(labelsize=12)

ax.set_title(r'$\mathbf{S}_{\mathrm{MO}}$', fontsize=18, fontweight='bold')
ax.set_xlabel(r'$|\varphi_j^{\mathrm{stack}}\rangle$', fontsize=16)
ax.set_ylabel(r'$\langle\varphi_i^{\mathrm{ref}}|$', fontsize=16)
ax.tick_params(labelsize=11)

ax.set_xticks(range(N_cols))
ax.set_yticks(range(N_rows))
ax.set_xticklabels(range(COL_START, COL_START + N_cols), fontsize=9)
ax.set_yticklabels(range(ROW_START, ROW_START + N_rows), fontsize=9)

for i in range(N_rows):
    for j in range(N_cols):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                     fill=False, edgecolor="black", linewidth=0.4))

plt.tight_layout()

stem = Path(MATRIX_FILE).stem
for ext in ("pdf", "png"):
    outpath = OUTDIR / f"{stem}.{ext}"
    plt.savefig(outpath, dpi=500, bbox_inches="tight")
    print(f"  Saved: {outpath}")

plt.close()
print("Done.")
