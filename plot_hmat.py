"""
Plot H_diab_eV.dat, H_adiab_target_eV.dat, and H_adiab_ref_eV.dat as
matrix heatmaps, following the style of plotmat.py.

For H_diab the colour scale is clipped to the largest OFF-DIAGONAL element
so that couplings are visible. Diagonal cells that exceed this limit saturate
to the colour extreme (marked with an asterisk in the annotation). The
adiabatic matrices are plotted with the full scale since they are diagonal
and no off-diagonal structure needs to be resolved.

Output: hplots/H_diab_eV.{png,pdf}
        hplots/H_adiab_target_eV.{png,pdf}
        hplots/H_adiab_ref_eV.{png,pdf}
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
# Matrix metadata: file, title, axis labels, clip_to_offdiag flag
# Each entry: (filename, title_latex, xlabel_latex, ylabel_latex, clip_offdiag)
#   clip_offdiag=True  → vmax set to largest |off-diagonal| element so
#                         couplings use the full colour range; saturated
#                         diagonal cells are annotated with '*'
#   clip_offdiag=False → vmax set to global |max|, standard full-range scale
# ---------------------------------------------------------------------------

MATRICES = [
    (
        "H_diab_eV.dat",
        r"$H^{\mathrm{diab}}$",
        r"Diabatic state $n$",
        r"Diabatic state $m$",
        True,    # clip to off-diagonal range so couplings are visible
    ),
    (
        "H_adiab_target_eV.dat",
        r"$H^{\mathrm{adiab}}_{\mathrm{stack}}$",
        r"Adiabatic state $k$ (rstack)",
        r"Adiabatic state $k$ (rstack)",
        False,   # purely diagonal — no clipping needed
    ),
    (
        "H_adiab_ref_eV.dat",
        r"$H^{\mathrm{adiab}}_{\mathrm{inf}}$",
        r"Adiabatic state $m$ (rinf)",
        r"Adiabatic state $m$ (rinf)",
        False,   # purely diagonal — no clipping needed
    ),
]

# ---------------------------------------------------------------------------
# Plotting loop
# ---------------------------------------------------------------------------

for fname, title, xlabel, ylabel, clip_offdiag in MATRICES:
    mat = np.loadtxt(fname, comments="#")
    N = mat.shape[0]

    fig, ax = plt.subplots(figsize=(8, 7))

    if clip_offdiag:
        # Build a version of the matrix with the diagonal zeroed out, then
        # take the maximum absolute off-diagonal element as the colour limit.
        # This maps the full RdBu_r range onto the coupling scale, making
        # even small off-diagonal elements visible. Diagonal cells that exceed
        # vmax are clipped by imshow to the colour extreme (fully saturated).
        off_diag = mat.copy()
        np.fill_diagonal(off_diag, 0.0)
        vmax = np.max(np.abs(off_diag))
        cbar_label = r"Value of Matrix Element (eV)"  # asterisk indicates diagonal saturation
    else:
        # Full scale: vmax = global maximum absolute value
        vmax = np.max(np.abs(mat))
        cbar_label = "Value of Matrix Element (eV)"

    im = ax.imshow(mat, cmap=CMAP, vmin=-vmax, vmax=vmax,
                   origin="upper", aspect="equal")

    # Colourbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=13)
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

    # Cell annotations: show numeric value.
    # Text colour: white on saturated cells (|val| >= vmax), black otherwise.
    for i in range(N):
        for j in range(N):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                         fill=False, edgecolor="black", linewidth=0.4))

    plt.tight_layout()

    stem = Path(fname).stem
    for ext in ("pdf", "png"):
        outpath = OUTDIR / f"{stem}.{ext}"
        plt.savefig(outpath, dpi=500, bbox_inches="tight")
        print(f"  Saved: {outpath}")

    plt.close()

print("Done.")
