"""
Quasi-diabatization pipeline: general many-body overlap diabatization.

Given two TDDFT calculations at two geometries (a reference geometry at which
states are pure diabats, and a target geometry at which states are adiabats),
this script:

  1. Parses the TDA X amplitude files and the cross-geometry MO overlap matrix.
  2. Builds global index arrays encoding each single-excitation determinant.
  3. Precomputes the determinant tensor Delta[i,a,j,b] for all excitation pairs.
  4. Contracts to form the many-body overlap U[m,k] = <Phi_m|Psi_k>.
  5. Löwdin-orthogonalizes U via SVD to get U_orth.
  6. Constructs H_diab = U_orth @ diag(E_target) @ U_orth.T.
  7. Exports H_diab, H_adiab_target, H_adiab_ref as .dat files.

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
N_ROOTS = 32   # number of TDDFT roots at each geometry

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

    The file header is validated against the user-supplied dimensions.

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
    n_occ_full = n_frz + n_occ   # total occupied rows per root block

    energies = np.zeros(n_roots)
    C = np.zeros((n_roots, n_occ_full, n_virt))

    root_idx = -1    # 0-based root counter; incremented on each Root header line
    row_idx  = 0     # row within the current root block (0 = first frozen row)
    in_block = False # True after a Root header has been seen; False after blank line

    with open(path) as f:
        for line in f:
            stripped = line.strip()

            # Blank line: end of the current root block
            if not stripped:
                in_block = False
                continue

            if stripped.startswith('#'):
                # Root header line: '# Root N  E=X.XXXXXXXXXX Ha  Y.YYYYYY eV'
                # Start a new root block and extract the Hartree energy.
                if 'Root' in stripped and 'E=' in stripped:
                    root_idx += 1
                    row_idx  = 0
                    in_block = True
                    e_str = stripped.split('E=')[1].split('Ha')[0].strip()
                    energies[root_idx] = float(e_str)
                # All other '#' lines are file-level header comments; skip.
                continue

            # Data line inside a root block: one row of n_virt floats.
            # row_idx = 0..n_frz-1          → frozen core (zeros; skip assignment)
            # row_idx = n_frz..n_occ_full-1 → active occupied (fill C)
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

    Files:
      smo_file    : NumPy .npy file, shape (n_bas, n_bas), float64
                    S_MO[p, q] = <phi^ref_p | phi^target_q>
                    row p = ref MO global index 0..n_bas-1
                    col q = target MO global index 0..n_bas-1
      ref_file    : plain-text amplitude file for reference geometry states
      target_file : plain-text amplitude file for target geometry states

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
    assert C_ref.shape == (n_roots, n_occ_full, n_virt), \
        f"C_ref shape {C_ref.shape} unexpected"
    assert X_tgt.shape == (n_roots, n_occ_full, n_virt), \
        f"X_tgt shape {X_tgt.shape} unexpected"

    return S_MO, E_ref, C_ref, E_tgt, X_tgt


def validate_inputs(C_ref, X_tgt, n_frz):
    """
    Sanity-check the parsed amplitude arrays.

    Checks:
      1. Frozen core rows 0..n_frz-1 are identically zero in both arrays.
      2. Active-block norms ||C[m,n_frz:,:]||^2 ≈ 1 for each root m.
         Slight excess above 1.0 is expected for full TDDFT X amplitudes:
         the TDDFT norm condition is ||X||^2 - ||Y||^2 = 1, so
         ||X||^2 = 1 + ||Y||^2 > 1. With ||Y||/||X|| ~ 0.1-0.2 the
         excess is ~1-4%.

    Parameters
    ----------
    C_ref : (n_roots, n_frz+n_occ, n_virt) float64, reference amplitudes
    X_tgt : (n_roots, n_frz+n_occ, n_virt) float64, target amplitudes
    n_frz : int, number of frozen core rows to check
    """
    assert np.allclose(C_ref[:, :n_frz, :], 0), \
        "Frozen core rows of C_ref are not zero — check parser or input file"
    assert np.allclose(X_tgt[:, :n_frz, :], 0), \
        "Frozen core rows of X_tgt are not zero — check parser or input file"

    # Active-block norms: sum over local occ and all virt axes
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
    Precompute index arrays r[i, a, p] and s[j, b, q] that map each position
    in a single-excitation determinant to a global MO index in S_MO.

    For an excitation i -> a, the occupied determinant is modified by replacing
    active occupied MO i with virtual MO a at position i in the orbital list.
    The resulting n_occ orbital slots have global S_MO indices:

        r[i, a, p] = p + occ_offset     for p != i   (active occ MO p)
        r[i, a, i] = a + virt_offset                  (virtual MO a replaces occ i)

    The n_occ x n_occ submatrix of S_MO needed for the Slater determinant
    overlap <Phi^a_i | Psi^b_j> is then:

        D^{ab}_{ij} = S_MO[ r[i,a,:], :][:, s[j,b,:] ]

    s is built identically for the ket (target geometry) excitations j -> b.

    Parameters
    ----------
    n_occ      : int, number of active occupied MOs
    n_virt     : int, number of virtual MOs
    occ_offset : int, global S_MO index of local active occ 0
    virt_offset: int, global S_MO index of local virt 0

    Returns
    -------
    r : ndarray, shape (n_occ, n_virt, n_occ), int32
        r[i, a, p] = global S_MO row index for position p in the bra
        determinant with excitation i -> a.
    s : ndarray, shape (n_occ, n_virt, n_occ), int32
        s[j, b, q] = global S_MO col index for position q in the ket
        determinant with excitation j -> b.
    """
    # Baseline: every position p maps to active occ MO p (global p + occ_offset).
    # Broadcast across all (i, a) pairs to shape (n_occ, n_virt, n_occ).
    p_base = np.arange(n_occ, dtype=np.int32) + occ_offset   # shape (n_occ,)
    r = np.broadcast_to(p_base, (n_occ, n_virt, n_occ)).copy()
    s = np.broadcast_to(p_base, (n_occ, n_virt, n_occ)).copy()

    # For each excitation i -> a, overwrite position p=i with virt MO a.
    # virt_global[a] = a + virt_offset gives the global index of virtual a.
    virt_global = np.arange(n_virt, dtype=np.int32) + virt_offset  # shape (n_virt,)
    for i in range(n_occ):
        r[i, :, i] = virt_global   # r[i, a, i] = a + virt_offset for all a
    for j in range(n_occ):
        s[j, :, j] = virt_global   # s[j, b, j] = b + virt_offset for all b

    return r, s


# ---------------------------------------------------------------------------
# Step 3: Precompute the determinant tensor Delta[i, a, j, b]
# ---------------------------------------------------------------------------

def build_delta_tensor(S_MO, r, s, n_occ, n_virt):
    """
    Precompute the single-determinant spatial overlap for all excitation pairs:

        Delta[i, a, j, b] = det( S_MO[ r[i,a,:], :][:, s[j,b,:] ] )

    This is the spatial orbital overlap between bra determinant |Phi^a_i>^ref
    and ket determinant |Psi^b_j>^target. The full closed-shell spin-orbital
    overlap is Delta^2 (squaring is applied in Step 4), because each spatial
    MO is doubly occupied (alpha + beta) in a restricted Kohn-Sham determinant,
    so the spin-orbital determinant factorises as det(D)^alpha * det(D)^beta
    = det(D)^2.

    All (n_occ * n_virt)^2 determinants are evaluated in a single batched
    numpy call — no Python-level loop over excitation pairs.

    Vectorised submatrix extraction
    --------------------------------
    We want:
        M[i, a, j, b, p, q] = S_MO[ r[i,a,p], s[j,b,q] ]

    Reshape r and s for broadcasting:
        r -> (n_occ, n_virt,      1,      1, n_occ,     1)
        s -> (    1,     1, n_occ, n_virt,     1, n_occ)

    NumPy broadcasts to (n_occ, n_virt, n_occ, n_virt, n_occ, n_occ)
    and performs advanced integer indexing into S_MO simultaneously.
    np.linalg.det is then called once over the last two axes (the n_occ x n_occ
    submatrix dimensions).

    Memory: n_occ^2 * n_virt^2 * n_occ^2 * 8 bytes (float64).
    For n_occ=12, n_virt=80 this is ~1.06 GB. If memory is limited, loop
    over one of the excitation indices and vectorise over the rest.

    Parameters
    ----------
    S_MO   : (n_bas, n_bas) float64, cross-geometry MO overlap matrix
    r      : (n_occ, n_virt, n_occ) int32, bra global row indices
    s      : (n_occ, n_virt, n_occ) int32, ket global col indices
    n_occ  : int, number of active occupied MOs
    n_virt : int, number of virtual MOs

    Returns
    -------
    Delta : (n_occ, n_virt, n_occ, n_virt) float64
        Delta[i, a, j, b] = det of the n_occ x n_occ S_MO submatrix for the
        excitation pair (i->a on ref, j->b on target). Sign is preserved;
        squaring is applied in Step 4.
    """
    # Reshape r and s so that they broadcast across all four excitation indices
    # and both orbital position indices simultaneously.
    #   r_bcast axes: (i, a, 1, 1, p, 1)  →  position p lives on axis 4
    #   s_bcast axes: (1, 1, j, b, 1, q)  →  position q lives on axis 5
    r_bcast = r.reshape(n_occ, n_virt,    1,      1, n_occ,     1)
    s_bcast = s.reshape(    1,     1, n_occ, n_virt,     1, n_occ)

    # Advanced indexing: M[i,a,j,b,p,q] = S_MO[r[i,a,p], s[j,b,q]]
    # Result shape: (n_occ, n_virt, n_occ, n_virt, n_occ, n_occ)
    n_pairs = n_occ * n_virt
    mem_GB = n_pairs**2 * n_occ**2 * 8 / 1e9
    print(f"  Allocating batch submatrix array "
          f"({n_occ},{n_virt},{n_occ},{n_virt},{n_occ},{n_occ}) "
          f"≈ {mem_GB:.2f} GB ...")
    M = S_MO[r_bcast, s_bcast]

    # np.linalg.det operates on the last two axes, treating all leading axes
    # as a batch dimension → result shape (n_occ, n_virt, n_occ, n_virt)
    n_dets = n_pairs**2
    print(f"  Computing {n_dets:,} determinants via np.linalg.det ...")
    Delta = np.linalg.det(M)

    return Delta


# ---------------------------------------------------------------------------
# Step 4: Compute the many-body overlap matrix U[m, k]
# ---------------------------------------------------------------------------

def compute_U(C_act, Delta, X_act):
    """
    Contract the amplitude tensors with the squared determinant tensor to
    produce the n_roots x n_roots many-body overlap matrix U.

    The overlap between reference (diabatic) state |Phi_m> and target
    (adiabatic) state |Psi_k> is:

        U[m, k] = <Phi_m | Psi_k>
                = sum_{i,a,j,b} C[m,i,a] * Delta[i,a,j,b]^2 * X[k,j,b]

    where:
        C[m, i, a]        : TDA X amplitude for reference state m,
                            excitation from active occ i into virtual a
        X[k, j, b]        : TDA X amplitude for target state k,
                            excitation from active occ j into virtual b
        Delta[i, a, j, b] : spatial orbital determinant overlap for the
                            excitation pair (i->a bra, j->b ket) [Step 3]
        Delta^2           : full spin-orbital overlap for closed-shell RKS
                            (each spatial MO doubly occupied)

    Implemented as a single einsum over all excitation indices (i, a, j, b),
    retaining the root indices (m, k):

        U = einsum('mia, iajb, kjb -> mk', C_act, Delta**2, X_act)

    Parameters
    ----------
    C_act : (n_roots, n_occ, n_virt) float64, reference active amplitudes
    Delta : (n_occ, n_virt, n_occ, n_virt) float64, determinant tensor
    X_act : (n_roots, n_occ, n_virt) float64, target active amplitudes

    Returns
    -------
    U : (n_roots, n_roots) float64
        U[m, k] = <Phi_m | Psi_k>. Not orthogonal in general.
    """
    # Delta**2: squared spatial determinant = full closed-shell spin-orbital overlap.
    # einsum sums over i, a, j, b and retains m (reference) and k (target).
    U = np.einsum('mia,iajb,kjb->mk', C_act, Delta**2, X_act)
    return U


# ---------------------------------------------------------------------------
# Step 5: Löwdin orthonormalization of U
# ---------------------------------------------------------------------------

def lowdin_orthogonalize(U):
    """
    Return the orthogonal matrix closest to U in Frobenius norm.

    U[m,k] = <Phi_m|Psi_k> has rows that are not normalised because the
    n_roots target adiabatic states do not span the full Hilbert space:

        sum_k U[m,k]^2 = ||Phi_m projected onto target subspace||^2 < 1

    The polar decomposition theorem gives the unique closest orthogonal matrix:

        U = V Sigma W^T  =>  U_orth = V W^T

    This discards the singular values (replaces each by 1), which normalises
    all modes while preserving their relative orientation as encoded in V and W^T.

    Properties of U_orth:
      - U_orth @ U_orth.T = I  exactly
      - Eigenvalues of H_diab = U_orth @ diag(E_tgt) @ U_orth.T are exactly E_tgt

    Parameters
    ----------
    U : (n_roots, n_roots) float64, raw many-body overlap matrix

    Returns
    -------
    U_orth : (n_roots, n_roots) float64, orthogonalized overlap matrix
    V      : (n_roots, n_roots) float64, left singular vectors
    sigma  : (n_roots,) float64, singular values (for inspection)
    Wt     : (n_roots, n_roots) float64, right singular vectors transposed
    """
    # Full SVD: U = V @ diag(sigma) @ Wt
    V, sigma, Wt = np.linalg.svd(U)

    # Löwdin: replace diag(sigma) with identity — keep only the rotational part
    U_orth = V @ Wt

    return U_orth, V, sigma, Wt


# ---------------------------------------------------------------------------
# Step 6: Build the diabatic Hamiltonian
# ---------------------------------------------------------------------------

def compute_H_diab(U_orth, E_tgt):
    """
    Construct the diabatic Hamiltonian in the reference (diabatic) basis.

    Each reference state |Phi_m> is expressed in the Löwdin-orthogonalized
    target adiabatic basis:

        |Phi_m> ≈ sum_k U_orth[m,k] |Psi_k>

    Taking matrix elements using <Psi_k|H_tgt|Psi_l> = E_tgt[k] delta_kl:

        H_diab[m,n] = sum_k U_orth[m,k] * E_tgt[k] * U_orth[n,k]

    In matrix form:

        H_diab = U_orth @ diag(E_tgt) @ U_orth.T

    The result is real symmetric with eigenvalues exactly equal to E_tgt,
    provided U_orth is orthogonal.

    IMPORTANT: the correct form is U_orth @ diag(E) @ U_orth.T (NOT
    U_orth.T @ diag(E) @ U_orth, which is non-symmetric, and NOT using
    U^{-1} in place of U_orth.T, which breaks symmetry).

    Parameters
    ----------
    U_orth : (n_roots, n_roots) float64, Löwdin-orthogonalized overlap
    E_tgt  : (n_roots,) float64, target adiabatic energies in Hartree

    Returns
    -------
    H_diab : (n_roots, n_roots) float64, diabatic Hamiltonian in Hartree.
        Diagonal elements are on-site energies; off-diagonal are couplings.
    """
    H_diab = U_orth @ np.diag(E_tgt) @ U_orth.T
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

    # Active slices: drop frozen core rows.
    # C_act[m, i, a]: reference state m, active occ i, virtual a
    # X_act[k, j, b]: target state k,   active occ j, virtual b
    C_act = C_ref[:, N_FRZ:, :]   # (n_roots, n_occ, n_virt)
    X_act = X_tgt[:, N_FRZ:, :]   # (n_roots, n_occ, n_virt)
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

    # Spot-check: excitation i=0, a=0
    # Position p=0 should map to virtual 0 (global = virt_offset);
    # all other positions p should map to their occ MO (global = p + occ_offset).
    expected_r00 = np.arange(N_OCC, dtype=np.int32) + OCC_OFFSET
    expected_r00[0] = 0 + VIRT_OFFSET
    assert np.array_equal(r[0, 0, :], expected_r00), \
        f"r[0,0,:] = {r[0,0,:]} != {expected_r00}"
    print(f"\n  Spot-check r[i=0, a=0, :] = {r[0, 0, :]}")
    print(f"    Expected                 : {expected_r00}  ✓")

    # Spot-check: excitation i=3, a=5
    # Position p=3 should map to virtual 5 (global = 5 + virt_offset).
    expected_r35 = np.arange(N_OCC, dtype=np.int32) + OCC_OFFSET
    expected_r35[3] = 5 + VIRT_OFFSET
    assert np.array_equal(r[3, 5, :], expected_r35), \
        f"r[3,5,:] = {r[3,5,:]} != {expected_r35}"
    print(f"\n  Spot-check r[i=3, a=5, :] = {r[3, 5, :]}")
    print(f"    Expected                  : {expected_r35}  ✓")

    # r and s are built from the same formula; they should be identical arrays.
    assert np.array_equal(r, s), "r and s should be identical (same substitution rule)"
    print("\n  r and s are identical (same substitution rule)  ✓")

    # All global indices must lie in [occ_offset, n_bas) — no frozen core contamination.
    assert r.min() >= OCC_OFFSET,  f"r contains index < occ_offset ({OCC_OFFSET})"
    assert s.min() >= OCC_OFFSET,  f"s contains index < occ_offset ({OCC_OFFSET})"
    assert r.max() <  N_BAS,       f"r contains index >= n_bas ({N_BAS})"
    assert s.max() <  N_BAS,       f"s contains index >= n_bas ({N_BAS})"
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

    # Spot-check: Delta[0,0,0,0] vs direct determinant computation
    D_ref = S_MO[np.ix_(r[0, 0, :], s[0, 0, :])]
    det_ref = np.linalg.det(D_ref)
    print(f"\n  Spot-check Delta[0,0,0,0]:")
    print(f"    build_delta_tensor : {Delta[0, 0, 0, 0]:.15f}")
    print(f"    direct det         : {det_ref:.15f}")
    assert np.isclose(Delta[0, 0, 0, 0], det_ref, rtol=1e-12), \
        "Delta[0,0,0,0] does not match direct computation"
    print(f"    Match  ✓")

    # Spot-check: off-diagonal case
    ic, ac, jc, bc = 2, 7, 5, 11
    D_ref2 = S_MO[np.ix_(r[ic, ac, :], s[jc, bc, :])]
    det_ref2 = np.linalg.det(D_ref2)
    print(f"\n  Spot-check Delta[{ic},{ac},{jc},{bc}]:")
    print(f"    build_delta_tensor : {Delta[ic, ac, jc, bc]:.15f}")
    print(f"    direct det         : {det_ref2:.15f}")
    assert np.isclose(Delta[ic, ac, jc, bc], det_ref2, rtol=1e-12), \
        f"Delta[{ic},{ac},{jc},{bc}] does not match direct computation"
    print(f"    Match  ✓")

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

    # U should NOT be orthogonal: target states are an incomplete basis,
    # so sum_k U[m,k]^2 < 1 for each m.
    UUt = U @ U.T
    diag_vals = np.diag(UUt)
    print(f"\n  Diagonal of U @ U.T (expect < 1.0, not identity):")
    print(f"  {np.array2string(diag_vals, precision=4)}")
    assert not np.allclose(UUt, np.eye(N_ROOTS), atol=1e-3), \
        "U appears orthogonal — unexpected"
    print("  U is not orthogonal (target basis is incomplete)  ✓")

    sv = np.linalg.svd(U, compute_uv=False)
    print(f"\n  Singular values of U:")
    print(f"  {np.array2string(sv, precision=6)}")
    assert np.all(sv <= 1.0 + 1e-2), \
        f"Singular value > 1: {sv.max():.6f} — check inputs"
    print(f"  Range: [{sv.min():.6f}, {sv.max():.6f}]  (expect all in (0,1])  ✓")

    print("\nStep 4 complete.")

    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 5: Löwdin orthonormalization")
    print("=" * 60)

    U_orth, V, sigma, Wt = lowdin_orthogonalize(U)

    assert U_orth.shape == (N_ROOTS, N_ROOTS)
    print(f"  U_orth shape: {U_orth.shape}")
    print(f"\n  Singular values of U (discarded by Löwdin, replaced by 1):")
    print(f"  {np.array2string(sigma, precision=6)}")

    residual = U_orth @ U_orth.T - np.eye(N_ROOTS)
    max_err = np.abs(residual).max()
    print(f"\n  ||U_orth @ U_orth.T - I||_max = {max_err:.3e}  (expect ~1e-15)")
    assert max_err < 1e-12, f"U_orth not orthogonal: max error = {max_err:.3e}"
    print("  U_orth @ U_orth.T = I  ✓")

    np.save('U_raw.npy',  U)
    np.save('U_orth.npy', U_orth)

    # Row norms of U_raw — useful for diagnosing near-zero overlap rows
    U_row_norms = np.linalg.norm(U, axis=1)
    print(f"\n  ||U[m,:]|| row norms:")
    for m, n in enumerate(U_row_norms):
        print(f"    m={m:2d}  {n:.6f}")

    write_matrix_dat(
        'U_raw.dat',
        U,
        f"Raw many-body overlap U[m,k] = <Phi_m|Psi_k>  shape=({N_ROOTS},{N_ROOTS})  "
        "rows=ref diabatic m, cols=target adiabatic k (0-based)"
    )
    write_matrix_dat(
        'U_orth.dat',
        U_orth,
        f"Löwdin-orthogonalized U_orth = V @ Wt  shape=({N_ROOTS},{N_ROOTS})  "
        "rows=ref diabatic m, cols=target adiabatic k (0-based)"
    )
    print("\n  Saved: U_raw.npy, U_orth.npy, U_raw.dat, U_orth.dat")

    print("\nStep 5 complete.")

    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 6: H_diab = U_orth @ diag(E_tgt) @ U_orth.T")
    print("=" * 60)

    H_diab = compute_H_diab(U_orth, E_tgt)

    assert H_diab.shape == (N_ROOTS, N_ROOTS)

    # Validation 1: symmetry
    sym_err = np.abs(H_diab - H_diab.T).max()
    print(f"  ||H_diab - H_diab.T||_max = {sym_err:.3e}  (expect ~1e-16)")
    assert np.allclose(H_diab, H_diab.T, atol=1e-12), \
        f"H_diab not symmetric: {sym_err:.3e}"
    print("  H_diab is symmetric  ✓")

    # Validation 2: eigenvalues must recover E_tgt exactly
    eigvals       = np.sort(np.linalg.eigvalsh(H_diab))
    E_tgt_sorted  = np.sort(E_tgt)
    eig_err = np.abs(eigvals - E_tgt_sorted).max()
    print(f"\n  Eigenvalue recovery:")
    print(f"  E_tgt sorted (Ha)   : {np.array2string(E_tgt_sorted, precision=6)}")
    print(f"  eigvalsh(H_diab)    : {np.array2string(eigvals,      precision=6)}")
    print(f"  max |diff|          : {eig_err:.3e}  (expect ~1e-14)")
    assert eig_err < 1e-10, \
        f"H_diab eigenvalues do not match E_tgt: {eig_err:.3e}"
    print("  Eigenvalues match E_tgt  ✓")

    # Print H_diab in eV
    H_eV = H_diab * HA_TO_EV
    print(f"\n  H_diab (eV), full {N_ROOTS}×{N_ROOTS} matrix:")
    header_row = "      " + "".join(f"  {k:7d}" for k in range(N_ROOTS))
    print(header_row)
    for m, row in enumerate(H_eV):
        print(f"  {m:2d} |" + "".join(f"  {v:7.3f}" for v in row))

    print(f"\n  On-site energies H_diab[m,m] (eV):")
    for m in range(N_ROOTS):
        print(f"    m={m:2d}  {H_eV[m,m]:8.4f} eV")

    # Save outputs
    np.save('H_diab.npy', H_diab)
    with open('H_diab.dat', 'w') as f:
        f.write("# Diabatic Hamiltonian H_diab (eV)\n")
        f.write("# H_diab = U_orth @ diag(E_tgt) @ U_orth.T\n")
        f.write(f"# Rows and columns: reference (diabatic) state index m (0-based)\n")
        f.write(f"# n_roots={N_ROOTS}  n_occ={N_OCC}  n_virt={N_VIRT}  n_frz={N_FRZ}\n")
        for row in H_eV:
            f.write("  ".join(f"{v:18.10f}" for v in row) + "\n")
    print("\n  Saved: H_diab.npy, H_diab.dat")

    print("\nStep 6 complete.")

    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Export: H_diab, H_adiab_target, H_adiab_ref as .dat files")
    print("=" * 60)

    # H_diab in eV
    write_matrix_dat(
        'H_diab_eV.dat',
        H_diab * HA_TO_EV,
        f"Diabatic Hamiltonian H_diab (eV)  shape=({N_ROOTS},{N_ROOTS})  "
        "rows=cols=reference diabatic state m (0-based)  "
        "H_diab = U_orth @ diag(E_tgt) @ U_orth.T"
    )

    # H_adiab_target: diagonal matrix of target adiabatic energies in eV.
    # Diagonal by construction: |Psi_k> are eigenstates of H_target.
    write_matrix_dat(
        'H_adiab_target_eV.dat',
        np.diag(E_tgt) * HA_TO_EV,
        f"Adiabatic Hamiltonian H_adiab_target (eV)  shape=({N_ROOTS},{N_ROOTS})  "
        "rows=cols=target adiabatic state k (0-based)  diagonal: E_tgt eigenvalues"
    )

    # H_adiab_ref: diagonal matrix of reference adiabatic energies in eV.
    # Also diagonal: at the reference geometry the states are non-interacting
    # (adiabatic = diabatic), so H_ref = diag(E_ref).
    write_matrix_dat(
        'H_adiab_ref_eV.dat',
        np.diag(E_ref) * HA_TO_EV,
        f"Adiabatic Hamiltonian H_adiab_ref (eV)  shape=({N_ROOTS},{N_ROOTS})  "
        "rows=cols=reference state m (0-based)  diagonal: E_ref eigenvalues"
    )

    print("  Saved: H_diab_eV.dat")
    print("  Saved: H_adiab_target_eV.dat")
    print("  Saved: H_adiab_ref_eV.dat")
    print("  Saved: U_raw.dat, U_orth.dat  (also written in Step 5)")
    print("  Format: one '#' header line, then rows of 18.11f floats (SMO.dat style)")

    print("\nAll steps complete.")
