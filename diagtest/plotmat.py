import numpy as np
import matplotlib.pyplot as plt


# --- User settings ---
MATRIX_FILE = "SMO.dat"
N = 96
CMAP = "RdBu_r"
ROW_START, ROW_END = 10, 30
COL_START, COL_END = 10, 30
# ---------------------

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

mat = np.loadtxt(MATRIX_FILE, comments="#")
mat = mat[ROW_START:ROW_END, COL_START:COL_END]

fig, ax = plt.subplots(figsize=(8, 7))
vmax = np.max(np.abs(mat))
im = ax.imshow(mat, cmap=CMAP, vmin=-vmax, vmax=vmax, origin="upper", aspect="equal")

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Overlap", fontsize=14)
cbar.ax.tick_params(labelsize=12)

ax.set_title(r'$\mathbf{S}_{\mathrm{MO}}$', fontsize=18, fontweight='bold')
ax.set_xlabel(r'$|\varphi_j^{\mathrm{stack}}\rangle$', fontsize=16)
ax.set_ylabel(r'$\langle\varphi_i^{\mathrm{ref}}|$', fontsize=16)
ax.tick_params(labelsize=12)

nrows, ncols = mat.shape
ax.set_xticks(range(ncols))
ax.set_yticks(range(nrows))
ax.set_xticklabels(range(COL_START, COL_START + ncols))
ax.set_yticklabels(range(ROW_START, ROW_START + nrows))

plt.tight_layout()
plt.savefig(MATRIX_FILE.replace(".dat", ".png"), dpi=150, bbox_inches='tight')

