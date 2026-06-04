"""
Plot U_raw.dat and U_orth.dat as matrix heatmaps, following the style of
plot_hmat.py.  Saved to hplots/ at 500 dpi.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

CMAP   = "RdBu_r"
OUTDIR = Path("hplots")
OUTDIR.mkdir(exist_ok=True)

MATRICES = [
    (
        "U_raw.dat",
        r"$U_{\mathrm{raw}}\ \ \langle\Phi_m|\Psi_k\rangle$",
        r"Adiabatic state $k$ (rstack)",
        r"Diabatic state $m$ (rinf)",
        r"$\langle\Phi_m|\Psi_k\rangle$",
    ),
    (
        "U_orth.dat",
        r"$\tilde{U}$ (Löwdin-orthogonalized)",
        r"Adiabatic state $k$ (rstack)",
        r"Diabatic state $m$ (rinf)",
        r"$\tilde{U}_{mk}$",
    ),
]

for fname, title, xlabel, ylabel, cbar_label in MATRICES:
    mat = np.loadtxt(fname, comments="#")
    N   = mat.shape[0]

    vmax = np.max(np.abs(mat))

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(mat, cmap=CMAP, vmin=-vmax, vmax=vmax,
                   origin="upper", aspect="equal")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=11)

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(range(N), fontsize=9)
    ax.set_yticklabels(range(N), fontsize=9)

    for i in range(N):
        for j in range(N):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                         fill=False, edgecolor="black", linewidth=0.4))

    plt.tight_layout()

    stem = Path(fname).stem
    outpath = OUTDIR / f"{stem}.pdf"
    plt.savefig(outpath, dpi=500, bbox_inches="tight")
    print(f"  Saved: {outpath}")

    plt.close()

print("Done.")
