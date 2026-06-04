import numpy as np
import matplotlib.pyplot as plt

# --- User settings ---
MATRIX_FILE = "SMO.dat"
ROW_START, COL_START = 10, 10
ROW_END,   COL_END   = 30, 30
CMAP = "RdBu_r"
# ---------------------

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

mat_full = np.loadtxt(MATRIX_FILE, comments="#")
mat = mat_full[ROW_START:ROW_END, COL_START:COL_END]
nrows, ncols = mat.shape

vmax = np.max(np.abs(mat))

fig, ax = plt.subplots(figsize=(8, 7))

im = ax.imshow(mat, cmap=CMAP, vmin=-vmax, vmax=vmax, origin="upper", aspect="equal",
               interpolation="nearest")

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"$\langle\varphi_i^{\mathrm{ref}}\vert\varphi_j^{\mathrm{stack}}\rangle$", fontsize=13)
cbar.ax.tick_params(labelsize=12)

ax.set_title(r'$\mathbf{S}_{\mathrm{MO}}$', fontsize=16, fontweight='bold')
ax.set_xlabel(r'$\vert\varphi_j^{\mathrm{stack}}\rangle$', fontsize=14)
ax.set_ylabel(r'$\langle\varphi_i^{\mathrm{ref}}\vert$', fontsize=14)
ax.tick_params(labelsize=11)

ax.set_xticks(range(ncols))
ax.set_yticks(range(nrows))
ax.set_xticklabels(range(COL_START, COL_START + ncols), fontsize=9)
ax.set_yticklabels(range(ROW_START, ROW_START + nrows), fontsize=9)

for i in range(nrows):
    for j in range(ncols):
        val = mat[i, j]
        text_color = "white" if abs(val) >= 0.9 * vmax else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=5.5, color=text_color)

plt.tight_layout()
#plt.savefig(MATRIX_FILE.replace(".dat", ".png"), dpi=500, bbox_inches="tight")
#plt.savefig(MATRIX_FILE.replace(".dat", ".pdf"), dpi=500, bbox_inches="tight")
plt.savefig(MATRIX_FILE.replace(".dat", ".svg"), dpi=500, bbox_inches="tight")
plt.close()

#print(f"Saved: {MATRIX_FILE.replace('.dat', '.png')}")
#print(f"Saved: {MATRIX_FILE.replace('.dat', '.pdf')}")
print(f"Saved: {MATRIX_FILE.replace('.dat', '.svg')}")
