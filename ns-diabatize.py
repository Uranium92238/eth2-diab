"""
Non-orthogonal quasi-diabatization pipeline.

Identical to diabatize.py through Step 4 (computing U[m,k] = <Phi_m|Psi_k>).
Step 5 is replaced: instead of Löwdin orthonormalization, U is used as-is and
the diabatic Hamiltonian is computed via a proper similarity transform:

    H_diab = U @ diag(E_tgt) @ U^{-1}

This preserves eigenvalues exactly but breaks symmetry when U is non-orthogonal.
The result is NOT symmetric in general — that is the point: inspecting the
asymmetry reveals how far the raw overlaps depart from orthogonality and how
much the Löwdin step affects the final H_diab.

All output files carry the prefix 'ns_' to avoid overwriting diabatize.py outputs.

All indices are 0-based throughout.

USER CONFIGURATION
------------------
Edit the block below to match your system. All other code is general.
"""

import numpy as np

# ===========================================================================
# USER CONFIGURATION — edit these values for your system
# ===========================================================================

# Input files (must exist in the working directory)
SMO_FILE    = 'S_MO.npy'            # cross-geometry MO overlap, shape (N_BAS, N_BAS)
REF_FILE    = 'rinf_vectorsX.dat'   # TDA X amplitudes at reference geometry
TARGET_FILE = 'rstack_vectorsX.dat' # TDA X amplitudes at target geometry

# Basis dimensions
N_BAS   = 96   # total number of MOs in the basis (must match S_MO shape)
N_FRZ   = 4    # frozen core MOs (excluded from CI); global indices 0..N_FRZ-1
N_OCC   = 12   # active occupied MOs; local indices 0..N_OCC-1,
               #   global indices N_FRZ..N_FRZ+N_OCC-1
N_VIRT  = 80   # virtual MOs; local indices 0..N_VIRT-1,
               #   global indices N_FRZ+N_OCC..N_BAS-1
N_ROOTS = 16   # number of TDDFT roots at each geometry

# Derived offsets (do not edit — follow from the values above)
OCC_OFFSET  = N_FRZ           # global S_MO index of local active occ 0
VIRT_OFFSET = N_FRZ + N_OCC   # global S_MO index of local virt 0

# Unit conversion
HA_TO_EV = 27.2114

# ===========================================================================
# End of user configuration
# ===========================================================================


# ---------------------------------------------------------------------------
# Step 1: Load and parse all input files
# ---------------------------------------------------------------------------

def parse_vectors(path, n_frz, n_occ, n_virt, n_roots):
    """
    Parse a plain-text TDA X amplitude file.

    Expected file layout:
      - Comment lines (starting with '#') are skipped, EXCEPT lines of the form
          # Root N  E=X.XXXXXXXXXX Ha  Y.YYYYYY eV
        which mark the start of a new root block and carry the excitation energy.
      - Each root block contains exactly (n_frz + n_occ) data rows,
        each row holding n_virt space-separated floats.
          rows 0..n_frz-1     : frozen core MOs — identically zero
          rows n_frz..n_frz+n_occ-1 : active occupied MOs — real amplitudes
      - A blank line separates consecutive root blocks.

    The amplitude array C[m, p, a] is the TDA X amplitude for reference state m
    to excite from MO p (global) into virtual MO a (local). Active amplitudes
    are in rows p = n_frz..n_frz+n_occ-1; frozen rows are zero.

    Parameters
    ----------
    path    : str, path to the amplitude file
    n_frz   : int, number of frozen core MOs (rows that are zero)
    n_occ   : int, number of active occupied MOs
    n_virt  : int, number of virtual MOs (columns per row)
    n_roots : int, expected number of root blocks

    Returns
    -------
    energies : ndarray, shape (n_roots,), float64
        Excitation energies in Hartree, extracted from Root header lines.
    C        : ndarray, shape (n_roots, n_frz+n_occ, n_virt), float64
        Full amplitude array. Rows 0..n_frz-1 are zero; active amplitudes
        are in rows n_frz..n_frz+n_occ-1.
    """
    n_occ_full = n_frz + n_occ

    energies = np.zeros(n_roots)
    C = np.zeros((n_roots, n_occ_full, n_virt))

    root_idx = -1
    row_idx  = 0
    in_block = False

    with open(path) as f:
        for line in f:
            stripped = line.strip()

            if not stripped:
                in_block = False
                continue

            if stripped.startswith('#'):
                if 'Root' in stripped and 'E=' in stripped:
                    root_idx += 1
                    row_idx  = 0
                    in_block = True
                    e_str = stripped.split('E=')[1].split('Ha')[0].strip()
                    energies[root_idx] = float(e_str)
                continue

            if in_block and root_idx >= 0 and row_idx < n_occ_full:
                C[root_idx, row_idx, :] = np.fromstring(stripped, sep=' ', count=n_virt)
                row_idx += 1

    assert root_idx + 1 == n_roots, \
        f"{path}: expected {n_roots} roots, found {root_idx + 1}"
    return energies, C


def load_inputs(smo_file, ref_file, target_file,
                n_bas, n_frz, n_occ, n_virt, n_roots):
    """
    Load and validate all three input files.

    Returns
    -------
    S_MO    : (n_bas, n_bas) float64
    E_ref   : (n_roots,) float64, reference excitation energies in Hartree
    C_ref   : (n_roots, n_frz+n_occ, n_virt) float64, reference amplitudes
    E_tgt   : (n_roots,) float64, target excitation energies in Hartree
    X_tgt   : (n_roots, n_frz+n_occ, n_virt) float64, target amplitudes
    """
    S_MO = np.load(smo_file)
    assert S_MO.shape == (n_bas, n_bas), \
        f"S_MO shape {S_MO.shape} != ({n_bas},{n_bas})"

    E_ref, C_ref = parse_vectors(ref_file,    n_frz, n_occ, n_virt, n_roots)
    E_tgt, X_tgt = parse_vectors(target_file, n_frz, n_occ, n_virt, n_roots)

    n_occ_full = n_frz + n_occ
    assert C_ref.shape == (n_roots, n_occ_full, n_virt)
    assert X_tgt.shape == (n_roots, n_occ_full, n_virt)

    return S_MO, E_ref, C_ref, E_tgt, X_tgt


def validate_inputs(C_ref, X_tgt, n_frz):
    """
    Sanity-check the parsed amplitude arrays.
    """
    assert np.allclose(C_ref[:, :n_frz, :], 0), \
        "Frozen core rows of C_ref are not zero — check parser or input file"
    assert np.allclose(X_tgt[:, :n_frz, :], 0), \
        "Frozen core rows of X_tgt are not zero — check parser or input file"

    norms_C = np.sum(C_ref[:, n_frz:, :] ** 2, axis=(1, 2))
    norms_X = np.sum(X_tgt[:, n_frz:, :] ** 2, axis=(1, 2))
    print(f"  C_ref norms   ||C_m||^2 (expect ~1.0): "
          f"min={norms_C.min():.6f}  max={norms_C.max():.6f}")
    print(f"  X_tgt  norms  ||X_k||^2 (expect ~1.0): "
          f"min={norms_X.min():.6f}  max={norms_X.max():.6f}")


# ---------------------------------------------------------------------------
# Step 2: Build global index arrays r and s
# ---------------------------------------------------------------------------

def build_index_arrays(n_occ, n_virt, occ_offset, virt_offset):
    """
    Precompute index arrays r[i, a, p] and s[j, b, q].

    r[i, a, p] = p + occ_offset     for p != i
    r[i, a, i] = a + virt_offset

    Returns
    -------
    r : (n_occ, n_virt, n_occ) int32
    s : (n_occ, n_virt, n_occ) int32
    """
    p_base = np.arange(n_occ, dtype=np.int32) + occ_offset
    r = np.broadcast_to(p_base, (n_occ, n_virt, n_occ)).copy()
    s = np.broadcast_to(p_base, (n_occ, n_virt, n_occ)).copy()

    virt_global = np.arange(n_virt, dtype=np.int32) + virt_offset
    for i in range(n_occ):
        r[i, :, i] = virt_global
    for j in range(n_occ):
        s[j, :, j] = virt_global

    return r, s


# ---------------------------------------------------------------------------
# Step 3: Precompute the determinant tensor Delta[i, a, j, b]
# ---------------------------------------------------------------------------

def build_delta_tensor(S_MO, r, s, n_occ, n_virt):
    """
    Precompute Delta[i, a, j, b] = det( S_MO[ r[i,a,:], s[j,b,:] ] )

    All (n_occ * n_virt)^2 determinants evaluated in a single batched numpy call.

    Returns
    -------
    Delta : (n_occ, n_virt, n_occ, n_virt) float64
    """
    r_bcast = r.reshape(n_occ, n_virt,    1,      1, n_occ,     1)
    s_bcast = s.reshape(    1,     1, n_occ, n_virt,     1, n_occ)

    n_pairs = n_occ * n_virt
    mem_GB = n_pairs**2 * n_occ**2 * 8 / 1e9
    print(f"  Allocating batch submatrix array "
          f"({n_occ},{n_virt},{n_occ},{n_virt},{n_occ},{n_occ}) "
          f"≈ {mem_GB:.2f} GB ...")
    M = S_MO[r_bcast, s_bcast]

    n_dets = n_pairs**2
    print(f"  Computing {n_dets:,} determinants via np.linalg.det ...")
    Delta = np.linalg.det(M)

    return Delta


# ---------------------------------------------------------------------------
# Step 4: Compute the many-body overlap matrix U[m, k]
# ---------------------------------------------------------------------------

def compute_U(C_act, Delta, X_act):
    """
    U[m, k] = sum_{i,a,j,b} C[m,i,a] * Delta[i,a,j,b]^2 * X[k,j,b]

    Returns
    -------
    U : (n_roots, n_roots) float64
    """
    U = np.einsum('mia,iajb,kjb->mk', C_act, Delta**2, X_act)
    return U


# ---------------------------------------------------------------------------
# Step 5 (NS version): Compute U^{-1} directly — no orthonormalization
# ---------------------------------------------------------------------------

def invert_U(U):
    """
    Compute the matrix inverse of U using numpy.linalg.inv.

    If U is nearly singular (condition number >> 1), the inverse will amplify
    numerical noise. The condition number is printed as a diagnostic.

    Parameters
    ----------
    U : (n_roots, n_roots) float64, raw many-body overlap matrix

    Returns
    -------
    U_inv : (n_roots, n_roots) float64, inverse of U
    cond  : float, condition number of U (ratio of largest to smallest singular value)
    """
    cond = np.linalg.cond(U)
    U_inv = np.linalg.inv(U)
    return U_inv, cond


# ---------------------------------------------------------------------------
# Step 6 (NS version): Non-orthogonal similarity transform
# ---------------------------------------------------------------------------

def compute_H_diab_ns(U, U_inv, E_tgt):
    """
    Construct the diabatic Hamiltonian via a similarity transform (not congruence):

        H_diab = U @ diag(E_tgt) @ U^{-1}

    This is a proper similarity transform: it preserves eigenvalues exactly.
    However, the result is NOT symmetric when U is non-orthogonal, because
    U^{-1} != U.T in that case.

    The asymmetry ||H_diab - H_diab.T|| quantifies how far U departs from
    orthogonality and serves as a diagnostic for the quality of the raw overlaps.

    Parameters
    ----------
    U     : (n_roots, n_roots) float64, raw many-body overlap matrix
    U_inv : (n_roots, n_roots) float64, inverse of U
    E_tgt : (n_roots,) float64, target adiabatic energies in Hartree

    Returns
    -------
    H_diab : (n_roots, n_roots) float64, diabatic Hamiltonian in Hartree.
        NOT symmetric in general.
    """
    H_diab = U @ np.diag(E_tgt) @ U_inv
    return H_diab


# ---------------------------------------------------------------------------
# Output helper
# ---------------------------------------------------------------------------

def write_matrix_dat(path, mat, header):
    """
    Write a square matrix to a plain-text .dat file (SMO.dat format):
    one '#' comment header line, then one row of space-separated floats per
    matrix row at 18.11f precision.

    Parameters
    ----------
    path   : str, output file path
    mat    : (N, N) ndarray, matrix to write
    header : str, header text (written after '# ')
    """
    with open(path, 'w') as f:
        f.write(f"# {header}\n")
        for row in mat:
            f.write("  ".join(f"{v:18.11f}" for v in row) + "\n")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time

    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Load and validate inputs")
    print("=" * 60)

    S_MO, E_ref, C_ref, E_tgt, X_tgt = load_inputs(
        SMO_FILE, REF_FILE, TARGET_FILE,
        N_BAS, N_FRZ, N_OCC, N_VIRT, N_ROOTS
    )

    print(f"  S_MO shape    : {S_MO.shape}")
    print(f"  C_ref shape   : {C_ref.shape}  (n_roots, n_frz+n_occ, n_virt)")
    print(f"  X_tgt  shape  : {X_tgt.shape}  (n_roots, n_frz+n_occ, n_virt)")
    print(f"  E_ref  (Ha)   : {np.array2string(E_ref, precision=6)}")
    print(f"  E_tgt  (Ha)   : {np.array2string(E_tgt, precision=6)}")

    print("\nValidation:")
    validate_inputs(C_ref, X_tgt, N_FRZ)

    C_act = C_ref[:, N_FRZ:, :]
    X_act = X_tgt[:, N_FRZ:, :]
    print(f"\n  C_act shape   : {C_act.shape}  (n_roots, n_occ, n_virt)")
    print(f"  X_act shape   : {X_act.shape}  (n_roots, n_occ, n_virt)")
    print("\nStep 1 complete.")

    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 2: Build global index arrays r and s")
    print("=" * 60)

    r, s = build_index_arrays(N_OCC, N_VIRT, OCC_OFFSET, VIRT_OFFSET)

    assert r.shape == (N_OCC, N_VIRT, N_OCC)
    assert s.shape == (N_OCC, N_VIRT, N_OCC)
    print(f"  r shape: {r.shape}  (i, a, p) -> global S_MO row index")
    print(f"  s shape: {s.shape}  (j, b, q) -> global S_MO col index")

    expected_r00 = np.arange(N_OCC, dtype=np.int32) + OCC_OFFSET
    expected_r00[0] = 0 + VIRT_OFFSET
    assert np.array_equal(r[0, 0, :], expected_r00)
    print(f"\n  Spot-check r[i=0, a=0, :] = {r[0, 0, :]}  ✓")

    expected_r35 = np.arange(N_OCC, dtype=np.int32) + OCC_OFFSET
    expected_r35[3] = 5 + VIRT_OFFSET
    assert np.array_equal(r[3, 5, :], expected_r35)
    print(f"  Spot-check r[i=3, a=5, :] = {r[3, 5, :]}  ✓")

    assert np.array_equal(r, s)
    print("  r and s are identical  ✓")
    assert r.min() >= OCC_OFFSET and r.max() < N_BAS
    print(f"  All indices in [{r.min()}, {r.max()}] ⊂ [{OCC_OFFSET}, {N_BAS-1}]  ✓")

    print("\nStep 2 complete.")

    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3: Precompute determinant tensor Delta[i,a,j,b]")
    print("=" * 60)

    t0 = time.time()
    Delta = build_delta_tensor(S_MO, r, s, N_OCC, N_VIRT)
    t1 = time.time()

    assert Delta.shape == (N_OCC, N_VIRT, N_OCC, N_VIRT)
    print(f"  Delta shape: {Delta.shape}  (i, a, j, b)")
    print(f"  Time: {t1-t0:.1f} s")

    D_ref = S_MO[np.ix_(r[0, 0, :], s[0, 0, :])]
    det_ref = np.linalg.det(D_ref)
    assert np.isclose(Delta[0, 0, 0, 0], det_ref, rtol=1e-12)
    print(f"\n  Spot-check Delta[0,0,0,0] = {Delta[0,0,0,0]:.15f}  ✓")

    ic, ac, jc, bc = 2, 7, 5, 11
    D_ref2 = S_MO[np.ix_(r[ic, ac, :], s[jc, bc, :])]
    assert np.isclose(Delta[ic, ac, jc, bc], np.linalg.det(D_ref2), rtol=1e-12)
    print(f"  Spot-check Delta[{ic},{ac},{jc},{bc}] = {Delta[ic,ac,jc,bc]:.15f}  ✓")

    print("\nStep 3 complete.")

    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 4: Compute many-body overlap matrix U[m,k]")
    print("=" * 60)

    U = compute_U(C_act, Delta, X_act)

    assert U.shape == (N_ROOTS, N_ROOTS)
    print(f"  U shape: {U.shape}  (reference m rows × target k cols)")

    print("\n  U matrix (rows=reference m, cols=target k):")
    for row in U:
        print("  " + "  ".join(f"{v:8.4f}" for v in row))

    UUt = U @ U.T
    diag_vals = np.diag(UUt)
    print(f"\n  Diagonal of U @ U.T (expect < 1.0, not identity):")
    print(f"  {np.array2string(diag_vals, precision=4)}")

    sv = np.linalg.svd(U, compute_uv=False)
    print(f"\n  Singular values of U:")
    print(f"  {np.array2string(sv, precision=6)}")
    print(f"  Range: [{sv.min():.6f}, {sv.max():.6f}]")

    U_row_norms = np.linalg.norm(U, axis=1)
    print(f"\n  ||U[m,:]|| row norms:")
    for m, n in enumerate(U_row_norms):
        print(f"    m={m:2d}  {n:.6f}")

    np.save('ns_U_raw.npy', U)
    write_matrix_dat(
        'ns_U_raw.dat',
        U,
        f"Raw many-body overlap U[m,k] = <Phi_m|Psi_k>  shape=({N_ROOTS},{N_ROOTS})  "
        "rows=ref diabatic m, cols=target adiabatic k (0-based)"
    )
    print("\n  Saved: ns_U_raw.npy, ns_U_raw.dat")

    print("\nStep 4 complete.")

    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 5 (NS): Invert U directly — no Löwdin orthonormalization")
    print("=" * 60)

    U_inv, cond = invert_U(U)

    print(f"  Condition number of U: {cond:.6e}")
    if cond > 1e12:
        print("  WARNING: U is very ill-conditioned — U_inv will be dominated by")
        print("           numerical noise in the near-zero singular value directions.")
    elif cond > 1e6:
        print("  WARNING: U is moderately ill-conditioned.")
    else:
        print("  U is well-conditioned.")

    # Verify inversion: U @ U_inv should be identity
    inv_residual = U @ U_inv - np.eye(N_ROOTS)
    inv_err = np.abs(inv_residual).max()
    print(f"\n  ||U @ U_inv - I||_max = {inv_err:.3e}")

    np.save('ns_U_inv.npy', U_inv)
    write_matrix_dat(
        'ns_U_inv.dat',
        U_inv,
        f"Inverse of raw U: U_inv = U^{{-1}}  shape=({N_ROOTS},{N_ROOTS})  "
        "rows=ref diabatic m, cols=target adiabatic k (0-based)"
    )
    print("  Saved: ns_U_inv.npy, ns_U_inv.dat")

    print("\nStep 5 complete.")

    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 6 (NS): H_diab = U @ diag(E_tgt) @ U^{-1}")
    print("=" * 60)

    H_diab = compute_H_diab_ns(U, U_inv, E_tgt)

    assert H_diab.shape == (N_ROOTS, N_ROOTS)

    # Eigenvalues must still recover E_tgt (similarity transform preserves spectrum)
    eigvals      = np.sort(np.linalg.eigvals(H_diab).real)
    E_tgt_sorted = np.sort(E_tgt)
    eig_err = np.abs(eigvals - E_tgt_sorted).max()
    print(f"  Eigenvalue recovery (similarity transform must preserve spectrum):")
    print(f"  E_tgt sorted (Ha)   : {np.array2string(E_tgt_sorted, precision=6)}")
    print(f"  eigvals(H_diab).real: {np.array2string(eigvals,      precision=6)}")
    print(f"  max |diff|          : {eig_err:.3e}  (expect ~1e-10 or better)")
    if eig_err < 1e-8:
        print("  Eigenvalues match E_tgt  ✓")
    else:
        print("  WARNING: eigenvalue recovery poor — U may be too ill-conditioned")

    # Symmetry: H_diab is NOT expected to be symmetric here
    sym_err = np.abs(H_diab - H_diab.T).max()
    print(f"\n  ||H_diab - H_diab.T||_max = {sym_err:.6e}")
    print(f"  (non-zero asymmetry is expected — U is not orthogonal)")

    # Symmetrised version for comparison: (H + H.T) / 2
    H_sym = (H_diab + H_diab.T) / 2.0
    sym_eig_err = np.abs(np.sort(np.linalg.eigvalsh(H_sym)) - E_tgt_sorted).max()
    print(f"\n  Symmetrised H_diab = (H + H.T)/2:")
    print(f"  max |eigvalsh - E_tgt_sorted| = {sym_eig_err:.3e}")

    # Print H_diab in eV
    H_eV = H_diab * HA_TO_EV
    H_sym_eV = H_sym * HA_TO_EV
    print(f"\n  H_diab (eV), full {N_ROOTS}×{N_ROOTS} matrix (non-symmetric):")
    header_row = "      " + "".join(f"  {k:7d}" for k in range(N_ROOTS))
    print(header_row)
    for m, row in enumerate(H_eV):
        print(f"  {m:2d} |" + "".join(f"  {v:7.3f}" for v in row))

    print(f"\n  Symmetrised H_diab (eV), (H + H.T)/2:")
    print(header_row)
    for m, row in enumerate(H_sym_eV):
        print(f"  {m:2d} |" + "".join(f"  {v:7.3f}" for v in row))

    print(f"\n  On-site energies H_diab[m,m] (eV)  [same for H and H_sym]:")
    for m in range(N_ROOTS):
        print(f"    m={m:2d}  {H_eV[m,m]:8.4f} eV")

    # Largest asymmetry entries
    asym = np.abs(H_diab - H_diab.T) * HA_TO_EV
    idx = np.argwhere(asym > 0.01 * asym.max())
    print(f"\n  Largest asymmetries |H[m,n] - H[n,m]| (eV), threshold=1% of max:")
    print(f"  {'m':>4} {'n':>4}  {'|H[m,n]-H[n,m]|':>18}  {'H[m,n]':>10}  {'H[n,m]':>10}")
    shown = set()
    for m, n in idx:
        if m >= n:
            continue
        key = (m, n)
        if key not in shown:
            shown.add(key)
            print(f"  {m:4d} {n:4d}  {asym[m,n]:18.6f}  {H_eV[m,n]:10.4f}  {H_eV[n,m]:10.4f}")

    # Save outputs
    np.save('ns_H_diab.npy', H_diab)
    np.save('ns_H_diab_sym.npy', H_sym)

    with open('ns_H_diab.dat', 'w') as f:
        f.write("# Non-orthogonal diabatic Hamiltonian ns_H_diab (eV)\n")
        f.write("# H_diab = U @ diag(E_tgt) @ U^{-1}  (similarity transform, NOT congruence)\n")
        f.write("# NOT symmetric — asymmetry measures departure of U from orthogonality\n")
        f.write(f"# n_roots={N_ROOTS}  n_occ={N_OCC}  n_virt={N_VIRT}  n_frz={N_FRZ}\n")
        for row in H_eV:
            f.write("  ".join(f"{v:18.10f}" for v in row) + "\n")

    with open('ns_H_diab_sym.dat', 'w') as f:
        f.write("# Symmetrised non-orthogonal diabatic Hamiltonian ns_H_diab_sym (eV)\n")
        f.write("# H_sym = (H_diab + H_diab.T) / 2  where H_diab = U @ diag(E_tgt) @ U^{-1}\n")
        f.write(f"# n_roots={N_ROOTS}  n_occ={N_OCC}  n_virt={N_VIRT}  n_frz={N_FRZ}\n")
        for row in H_sym_eV:
            f.write("  ".join(f"{v:18.10f}" for v in row) + "\n")

    print("\n  Saved: ns_H_diab.npy, ns_H_diab.dat")
    print("  Saved: ns_H_diab_sym.npy, ns_H_diab_sym.dat")

    print("\nStep 6 complete.")

    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Export: all matrices as .dat files (SMO.dat style)")
    print("=" * 60)

    write_matrix_dat(
        'ns_H_diab_eV.dat',
        H_diab * HA_TO_EV,
        f"Non-orthogonal H_diab (eV)  shape=({N_ROOTS},{N_ROOTS})  "
        "H_diab = U @ diag(E_tgt) @ U^{{-1}}  NOT symmetric"
    )
    write_matrix_dat(
        'ns_H_diab_sym_eV.dat',
        H_sym * HA_TO_EV,
        f"Symmetrised H_diab (eV)  shape=({N_ROOTS},{N_ROOTS})  "
        "H_sym = (H_diab + H_diab.T)/2"
    )
    write_matrix_dat(
        'ns_H_adiab_target_eV.dat',
        np.diag(E_tgt) * HA_TO_EV,
        f"Adiabatic Hamiltonian H_adiab_target (eV)  shape=({N_ROOTS},{N_ROOTS})  "
        "diagonal: E_tgt eigenvalues"
    )
    write_matrix_dat(
        'ns_H_adiab_ref_eV.dat',
        np.diag(E_ref) * HA_TO_EV,
        f"Adiabatic Hamiltonian H_adiab_ref (eV)  shape=({N_ROOTS},{N_ROOTS})  "
        "diagonal: E_ref eigenvalues"
    )

    print("  Saved: ns_H_diab_eV.dat")
    print("  Saved: ns_H_diab_sym_eV.dat")
    print("  Saved: ns_H_adiab_target_eV.dat")
    print("  Saved: ns_H_adiab_ref_eV.dat")
    print("  Format: one '#' header line, then rows of 18.11f floats (SMO.dat style)")

    print("\nAll steps complete.")
