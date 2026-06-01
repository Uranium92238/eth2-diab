"""
Plot H_diab_eV.dat, H_adiab_stack_eV.dat, and H_adiab_inf_eV.dat as
matrix heatmaps, following the style of plotmat.py.

Output: hplots/H_diab_eV.{png,pdf}
        hplots/H_adiab_stack_eV.{png,pdf}
        hplots/H_adiab_inf_eV.{png,pdf}
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Plot settings — match plotmat.py style
# ---------------------------------------------------------------------------

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

CMAP = "RdBu_r"
OUTDIR = Path("hplots")
OUTDIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Matrix metadata: file, title, axis labels
# Each entry: (filename, title_latex, xlabel_latex, ylabel_latex)
# ---------------------------------------------------------------------------

MATRICES = [
    (
        "H_diab_eV.dat",
        r"$H^{\mathrm{diab}}$",
        r"Diabatic state $n$",
        r"Diabatic state $m$",
    ),
    (
        "H_adiab_stack_eV.dat",
        r"$H^{\mathrm{adiab}}_{\mathrm{stack}}$",
        r"Adiabatic state $k$ (rstack)",
        r"Adiabatic state $k$ (rstack)",
    ),
    (
        "H_adiab_inf_eV.dat",
        r"$H^{\mathrm{adiab}}_{\mathrm{inf}}$",
        r"Adiabatic state $m$ (rinf)",
        r"Adiabatic state $m$ (rinf)",
    ),
]

# ---------------------------------------------------------------------------
# Plotting loop
# ---------------------------------------------------------------------------

for fname, title, xlabel, ylabel in MATRICES:
    mat = np.loadtxt(fname, comments="#")   # shape (16, 16)
    N = mat.shape[0]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Symmetric colour scale centred on zero so red=positive, blue=negative
    vmax = np.max(np.abs(mat))
    im = ax.imshow(mat, cmap=CMAP, vmin=-vmax, vmax=vmax,
                   origin="upper", aspect="equal")

    # Colourbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Energy (eV)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Labels and title
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(labelsize=11)

    # Tick marks at each matrix element (0-based index labels)
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(range(N), fontsize=9)
    ax.set_yticklabels(range(N), fontsize=9)

    # Annotate each cell with its value if the matrix is small enough.
    # For 16x16 the values are readable at the saved resolution.
    for i in range(N):
        for j in range(N):
            val = mat[i, j]
            # Use white text on saturated cells, black on pale cells
            text_color = "white" if abs(val) > 0.6 * vmax else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5.5, color=text_color)

    plt.tight_layout()

    stem = Path(fname).stem   # e.g. "H_diab_eV"
    for ext in ("png", "pdf"):
        outpath = OUTDIR / f"{stem}.{ext}"
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"  Saved: {outpath}")

    plt.close()

print("Done.")
