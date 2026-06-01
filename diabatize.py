"""
Quasi-diabatization pipeline for the eclipsed cofacial ethylene dimer.

System: CAM-B3LYP/def2-SVP, full TDDFT (no TDA), ORCA 6.0.1, D2h symmetry.
Two geometries:
  rinf   (20 Å separation) : non-interacting monomers → pure XT/CT states
                              used as the DIABATIC reference basis
  rstack (3.5 Å stacking)  : strongly coupled monomers → mixed adiabatic states

Goal: construct H_diab (16×16, eV) in the diabatic basis {|Phi_m>} by
projecting the adiabatic eigenstates {|Psi_k>} onto the diabatic basis via
the many-body overlap matrix U_mk = <Phi_m|Psi_k>, then Löwdin-orthogonalizing
and forming H_diab = U_orth @ diag(E_stack) @ U_orth.T

All indices are 0-based throughout (matches CLAUDE.md convention).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants and basis dimensions (see CLAUDE.md §Constants)
# ---------------------------------------------------------------------------

N_BAS       = 96   # total MOs in basis (48 per monomer)
N_FRZ       = 4    # frozen core MOs (C 1s), global indices 0–3; excluded from CI
N_OCC       = 12   # active occupied MOs, local indices 0–11, global 4–15
N_VIRT      = 80   # virtual MOs, local indices 0–79, global 16–95
N_ROOTS     = 16   # TDDFT roots computed at each geometry
OCC_OFFSET  = 4    # global S_MO index of local active occ 0  (i_global = i + 4)
VIRT_OFFSET = 16   # global S_MO index of local virt 0        (a_global = a + 16)
HA_TO_EV    = 27.2114


# ---------------------------------------------------------------------------
# Step 1: Load and parse all input files
# ---------------------------------------------------------------------------

def parse_vectors(path):
    """
    Parse a plain-text amplitude file (rinf_vectorsX.dat or rstack_vectorsX.dat).

    File layout (see CLAUDE.md §Input Files):
      - Lines starting with '#' are comments and are skipped, EXCEPT lines of
        the form '# Root N  E=X.XXXXXXXXXX Ha  Y.YYYYYY eV' which mark the
        start of a new root block and carry the excitation energy.
      - Each root block contains exactly (N_FRZ + N_OCC) = 16 data rows,
        each row holding N_VIRT = 80 space-separated floats.
        Row index p (0-based) within the block:
          p = 0..3   : frozen core MOs — always zero, never enter computation
          p = 4..15  : active occupied MO (p-4) → N_OCC rows of real amplitudes
        Column index a (0-based): virtual MO a, global index a + VIRT_OFFSET
      - A blank line separates consecutive root blocks.

    The amplitude C[m, p, a] = C^(m)_{(p-4), a} is the TDA X amplitude for
    rinf state m to excite from active occ MO (p-4) into virtual MO a.
    (Or X^(k)_{(p-4), b} for rstack.)

    Returns
    -------
    energies : ndarray, shape (N_ROOTS,), float64
        Excitation energies in Hartree, one per root, extracted from the
        '# Root N  E=X Ha' comment lines.
    C : ndarray, shape (N_ROOTS, N_FRZ+N_OCC, N_VIRT) = (16, 16, 80), float64
        Full amplitude array including the four frozen-core zero rows.
        Rows 0–3 are identically zero.
        Active amplitudes are in rows 4–15 (local occ index = row - 4).
    """
    energies = np.zeros(N_ROOTS)
    # Allocate full array including frozen rows so indexing matches file layout.
    # Rows 0–3 stay zero; rows 4–15 are filled from the data lines.
    C = np.zeros((N_ROOTS, N_FRZ + N_OCC, N_VIRT))

    root_idx = -1   # 0-based root counter; incremented each time a Root header is seen
    row_idx  = 0    # row within the current root block (0 = first frozen row)
    in_block = False  # True once we have seen a Root header and are reading data rows

    with open(path) as f:
        for line in f:
            stripped = line.strip()

            # Blank line signals end of the current root block
            if not stripped:
                in_block = False
                continue

            if stripped.startswith('#'):
                # Check if this is a Root energy header line, e.g.:
                #   # Root 1  E=0.2992998635 Ha  8.144368 eV
                # If so, start a new root block and parse the energy.
                if 'Root' in stripped and 'E=' in stripped:
                    root_idx += 1
                    row_idx  = 0
                    in_block = True
                    # Extract the Hartree energy: text between 'E=' and 'Ha'
                    e_str = stripped.split('E=')[1].split('Ha')[0].strip()
                    energies[root_idx] = float(e_str)
                # All other '#' lines are file-level header comments; skip them.
                continue

            # Non-blank, non-comment line inside a root block = one data row.
            # row_idx runs 0..15: first 4 are frozen (zeros), rest are active.
            if in_block and root_idx >= 0 and row_idx < (N_FRZ + N_OCC):
                # np.fromstring is faster than split+float for long rows
                C[root_idx, row_idx, :] = np.fromstring(stripped, sep=' ', count=N_VIRT)
                row_idx += 1

    return energies, C


def load_inputs():
    """
    Load all three input files needed for the diabatization.

    Files (must be present in the working directory):
      S_MO.npy          : 96×96 cross-geometry MO overlap matrix (float64)
                          S_MO[p, q] = <phi^ref_p | phi^stack_q>
                          row p = ref (rinf) MO global index 0..95
                          col q = stack (rstack) MO global index 0..95
      rinf_vectorsX.dat : TDA X amplitudes for 16 rinf states (plain text)
      rstack_vectorsX.dat: TDA X amplitudes for 16 rstack states (plain text)

    Returns
    -------
    S_MO    : (96, 96) float64
    E_ref   : (16,) float64, rinf excitation energies in Hartree
    C_ref   : (16, 16, 80) float64, rinf TDA X amplitudes (full, with frozen rows)
    E_stack : (16,) float64, rstack excitation energies in Hartree
    X_stack : (16, 16, 80) float64, rstack TDA X amplitudes (full, with frozen rows)
    """
    S_MO = np.load('S_MO.npy')
    assert S_MO.shape == (N_BAS, N_BAS), \
        f"S_MO shape {S_MO.shape} != ({N_BAS},{N_BAS})"

    E_ref,   C_ref   = parse_vectors('rinf_vectorsX.dat')
    E_stack, X_stack = parse_vectors('rstack_vectorsX.dat')

    assert C_ref.shape   == (N_ROOTS, N_FRZ + N_OCC, N_VIRT), \
        f"C_ref shape {C_ref.shape} unexpected"
    assert X_stack.shape == (N_ROOTS, N_FRZ + N_OCC, N_VIRT), \
        f"X_stack shape {X_stack.shape} unexpected"

    return S_MO, E_ref, C_ref, E_stack, X_stack


def validate_inputs(C_ref, X_stack):
    """
    Run sanity checks on the parsed amplitude arrays (CLAUDE.md §Validation Checklist).

    Checks:
      1. Frozen core rows (0–3) are identically zero in both arrays.
      2. Active-block norms sum(C[m,4:,:]**2) ≈ 1 for each root m.
         (Slight excess above 1.0 is expected: these are X amplitudes from full
         TDDFT, not pure TDA. In full TDDFT, ||X||^2 - ||Y||^2 = 1, so
         ||X||^2 = 1 + ||Y||^2 > 1. With ||Y||/||X|| ~ 0.1–0.2 the excess
         is ~1–4%, consistent with observed max ≈ 1.03.)
    """
    # Check 1: frozen core rows must be exactly zero (parser writes zeros by default,
    # but we confirm the file itself contains zeros as expected)
    assert np.allclose(C_ref[:, :N_FRZ, :], 0), \
        "Frozen core rows of C_ref are not zero — check parser or input file"
    assert np.allclose(X_stack[:, :N_FRZ, :], 0), \
        "Frozen core rows of X_stack are not zero — check parser or input file"

    # Check 2: active-block norms
    # sum over local occ (rows 4–15) and all virtuals
    norms_C = np.sum(C_ref[:, N_FRZ:, :] ** 2, axis=(1, 2))   # shape (16,)
    norms_X = np.sum(X_stack[:, N_FRZ:, :] ** 2, axis=(1, 2)) # shape (16,)
    print(f"  C_ref norms   ||C_m||^2 (expect ~1.0): "
          f"min={norms_C.min():.6f}  max={norms_C.max():.6f}")
    print(f"  X_stack norms ||X_k||^2 (expect ~1.0): "
          f"min={norms_X.min():.6f}  max={norms_X.max():.6f}")


# ---------------------------------------------------------------------------
# Step 2: Build global index arrays r and s
# ---------------------------------------------------------------------------

def build_index_arrays():
    """
    Precompute the global S_MO row-index array r[i, a, p] and column-index
    array s[j, b, q] for all excitation pairs (i, a) and (j, b).

    Physical meaning
    ----------------
    The bra single-excitation determinant |Phi^a_i>^ref occupies N_OCC = 12
    spatial orbitals drawn from the rinf MO basis:
      - all active occupied MOs EXCEPT i  (global index p + OCC_OFFSET for p != i)
      - virtual MO a in place of occ i    (global index a + VIRT_OFFSET at position i)

    The ket single-excitation determinant |Psi^b_j>^stack occupies N_OCC = 12
    spatial orbitals drawn from the rstack MO basis, constructed identically
    for the (j, b) excitation.

    The 12x12 orbital overlap submatrix D^{ab}_{ij} needed for det(<Phi^a_i|Psi^b_j>)
    is then simply:
        D^{ab}_{ij} = S_MO[ r[i,a,:], : ][ :, s[j,b,:] ]
    i.e. S_MO with rows selected by r[i,a,:] and cols selected by s[j,b,:].

    Index construction (0-based, see CLAUDE.md §Step 2):
        r[i, a, p] = p + OCC_OFFSET   for p != i   (ref occ MO p, global index p+4)
        r[i, a, i] = a + VIRT_OFFSET              (ref virt MO a, global index a+16)

        s[j, b, q] = q + OCC_OFFSET   for q != j   (stack occ MO q, global index q+4)
        s[j, b, j] = b + VIRT_OFFSET              (stack virt MO b, global index b+16)

    The replacement of row/col i (or j) encodes the single excitation: the
    orbital that was occupied in the ground state (MO i, global i+4) is
    replaced by the virtual MO a (global a+16) in the excited determinant.

    Returns
    -------
    r : ndarray, shape (N_OCC, N_VIRT, N_OCC) = (12, 80, 12), int32
        r[i, a, p] is the global S_MO row index for position p in the bra
        determinant when the excitation is occ i -> virt a.
    s : ndarray, shape (N_OCC, N_VIRT, N_OCC) = (12, 80, 12), int32
        s[j, b, q] is the global S_MO col index for position q in the ket
        determinant when the excitation is occ j -> virt b.
    """
    # Start from the "all occupied" baseline: every position p maps to
    # the corresponding active occ MO p (global index p + OCC_OFFSET).
    # Shape: (N_OCC, N_VIRT, N_OCC) — same index vector broadcast over all a.
    # p_base[p] = p + OCC_OFFSET for p in 0..11
    p_base = np.arange(N_OCC, dtype=np.int32) + OCC_OFFSET  # shape (12,)

    # Broadcast to (N_OCC, N_VIRT, N_OCC):
    #   axis 0 = i (which occ is excited out)
    #   axis 1 = a (which virt is excited into)
    #   axis 2 = p (position within the 12-orbital determinant)
    # Initially all positions hold their ground-state occ global index.
    r = np.broadcast_to(p_base, (N_OCC, N_VIRT, N_OCC)).copy()
    s = np.broadcast_to(p_base, (N_OCC, N_VIRT, N_OCC)).copy()

    # Now overwrite position p=i with the virtual global index a + VIRT_OFFSET.
    # For each i in 0..11, r[i, :, i] must become a + VIRT_OFFSET for each a.
    # np.arange(N_VIRT) + VIRT_OFFSET gives [16, 17, ..., 95], shape (80,).
    virt_global = np.arange(N_VIRT, dtype=np.int32) + VIRT_OFFSET  # shape (80,)

    for i in range(N_OCC):
        # r[i, a, i] = a + VIRT_OFFSET  for all a in 0..79
        # This replaces the occ MO i with virt MO a at position i in the det.
        r[i, :, i] = virt_global

    for j in range(N_OCC):
        # s[j, b, j] = b + VIRT_OFFSET  for all b in 0..79
        s[j, :, j] = virt_global

    return r, s


# ---------------------------------------------------------------------------
# Step 3: Precompute the determinant tensor Delta[i, a, j, b]
# ---------------------------------------------------------------------------

def build_delta_tensor(S_MO, r, s):
    """
    Precompute Delta[i, a, j, b] = det(S_MO[r[i,a,:], :][:, s[j,b,:]]) for
    all (i, a, j, b), where i,j in 0..N_OCC-1 and a,b in 0..N_VIRT-1.

    Delta is the single-determinant spatial orbital overlap between bra
    determinant |Phi^a_i>^ref and ket determinant |Psi^b_j>^stack:

        <Phi^a_i | Psi^b_j> = [det(D^{ab}_{ij})]^2

    where D^{ab}_{ij} = S_MO[r[i,a,:], :][:, s[j,b,:]] is the 12x12 submatrix
    of S_MO obtained by selecting rows r[i,a,:] and cols s[j,b,:].
    The squaring (applied in Step 4) accounts for RKS closed-shell spin: each
    spatial MO is doubly occupied (alpha + beta), so the full spin-orbital
    determinant factorises as det(D)^alpha * det(D)^beta = det(D)^2.

    We do NOT loop over (i, a, j, b) in Python. Instead we use numpy advanced
    indexing to build the entire batch of 921,600 submatrices at once as a
    single 6D array of shape (N_OCC, N_VIRT, N_OCC, N_VIRT, N_OCC, N_OCC)
    = (12, 80, 12, 80, 12, 12), then call np.linalg.det once over the last
    two axes.

    Vectorised submatrix extraction via broadcasting
    ------------------------------------------------
    We want:
        M[i, a, j, b, p, q] = S_MO[ r[i,a,p], s[j,b,q] ]

    r has shape (12, 80, 12): r[i, a, p] = global row index for position p
                               in the bra det with excitation i->a
    s has shape (12, 80, 12): s[j, b, q] = global col index for position q
                               in the ket det with excitation j->b

    Reshape for broadcasting:
        r_bcast : (12, 80,  1,  1, 12,  1)   axes: i, a, _, _, p, _
        s_bcast : ( 1,  1, 12, 80,  1, 12)   axes: _, _, j, b, _, q

    NumPy broadcasts these to (12, 80, 12, 80, 12, 12) and uses them as
    two index arrays into S_MO, which is exactly advanced integer indexing:
        M = S_MO[ r_bcast, s_bcast ]   shape (12, 80, 12, 80, 12, 12)

    Memory: 12*80*12*80*12*12 * 8 bytes = 921,600 * 144 * 8 ≈ 1.06 GB.
    If this is too large, set chunked=True to loop over i (12 iterations of
    ~88 MB each) and build Delta row-by-row.

    Parameters
    ----------
    S_MO : ndarray, shape (96, 96), float64
        Cross-geometry MO overlap matrix.
    r : ndarray, shape (12, 80, 12), int32
        Global S_MO row indices for each bra excitation (i, a).
    s : ndarray, shape (12, 80, 12), int32
        Global S_MO col indices for each ket excitation (j, b).
    chunked : bool
        If True, loop over i to reduce peak memory from ~1.06 GB to ~88 MB.

    Returns
    -------
    Delta : ndarray, shape (12, 80, 12, 80), float64
        Delta[i, a, j, b] = det of the 12x12 S_MO submatrix for excitation
        pair (i->a on ref, j->b on stack). Sign matters; squaring is in Step 4.
    """
    # Reshape r and s for broadcasting into the 6D batch array M.
    #   r: (i, a, p)   -> (i, a, 1, 1, p, 1)  so p lives on axis 4
    #   s: (j, b, q)   -> (1, 1, j, b, 1, q)  so q lives on axis 5
    r_bcast = r.reshape(N_OCC, N_VIRT, 1, 1, N_OCC, 1)   # (12, 80,  1,  1, 12,  1)
    s_bcast = s.reshape(1, 1, N_OCC, N_VIRT, 1, N_OCC)   # ( 1,  1, 12, 80,  1, 12)

    # Advanced indexing: S_MO[r_bcast, s_bcast] selects element
    # S_MO[r[i,a,p], s[j,b,q]] for every combination of (i,a,j,b,p,q).
    # NumPy broadcasts r_bcast and s_bcast to (12,80,12,80,12,12) before indexing.
    # Result M[i,a,j,b,p,q] = S_MO[r[i,a,p], s[j,b,q]]
    # This is the (p,q) element of the 12x12 submatrix D^{ab}_{ij}.
    print("  Allocating batch submatrix array (12,80,12,80,12,12) ≈ 1.06 GB ...")
    M = S_MO[r_bcast, s_bcast]   # shape (12, 80, 12, 80, 12, 12)
    print("  Computing 921,600 determinants via np.linalg.det ...")

    # np.linalg.det operates on the last two axes by default, treating the
    # leading axes as a batch dimension. So det is taken over axes (-2,-1),
    # i.e. the 12x12 (p,q) matrix for each (i,a,j,b).
    Delta = np.linalg.det(M)   # shape (12, 80, 12, 80)

    return Delta


# ---------------------------------------------------------------------------
# Step 4: Compute the many-body overlap matrix U[m, k]
# ---------------------------------------------------------------------------

def compute_U(C_act, Delta, X_act):
    """
    Contract the amplitude tensors with the squared determinant tensor to
    produce the 16x16 many-body overlap matrix U.

    Theory (CLAUDE.md §The Exact Many-Body Overlap U_mk)
    -----------------------------------------------------
    The overlap between diabatic state |Phi_m> (rinf) and adiabatic state
    |Psi_k> (rstack) is:

        U[m, k] = <Phi_m | Psi_k>
                = sum_{i,a,j,b} C[m,i,a] * Delta[i,a,j,b]^2 * X[k,j,b]

    where:
        C[m, i, a]        : TDA X amplitude for rinf state m, excitation i->a
        X[k, j, b]        : TDA X amplitude for rstack state k, excitation j->b
        Delta[i, a, j, b] : det of the 12x12 S_MO submatrix for excitation
                            pair (i->a on ref, j->b on stack)  [from Step 3]
        Delta^2           : squaring accounts for RKS closed-shell spin
                            (alpha and beta spin blocks give identical dets,
                             so the full spin-orbital det = det_spatial^2)

    The sum runs over all active occupied indices i,j in [0, N_OCC) and all
    virtual indices a,b in [0, N_VIRT), giving a single (16,16) result.

    Implementation
    --------------
    This is a single np.einsum call:

        U = einsum('mia, iajb, kjb -> mk', C_act, Delta**2, X_act)

    Axis labels:
        m : rinf root index  0..15  (diabatic state)
        k : rstack root index 0..15 (adiabatic state)
        i : bra active occ   0..11
        a : bra virtual      0..79
        j : ket active occ   0..11
        b : ket virtual      0..79

    The einsum sums over i, a, j, b and retains m, k — exactly the double
    sum over all excitation pairs in the formula above.

    Parameters
    ----------
    C_act  : ndarray, shape (16, 12, 80), float64
        Active-slice TDA X amplitudes for rinf states. C_act[m, i, a].
    Delta  : ndarray, shape (12, 80, 12, 80), float64
        Determinant tensor from Step 3. Delta[i, a, j, b].
    X_act  : ndarray, shape (16, 12, 80), float64
        Active-slice TDA X amplitudes for rstack states. X_act[k, j, b].

    Returns
    -------
    U : ndarray, shape (16, 16), float64
        U[m, k] = <Phi_m | Psi_k>, the many-body overlap matrix.
        Not orthogonal in general (see Step 5 for Löwdin orthonormalization).
    """
    # Square the determinant tensor first: Delta^2[i,a,j,b] = det(D)^2
    # gives the full closed-shell spin-orbital overlap for each det pair.
    # Then contract: sum over all (i,a,j,b) excitation pairs, keep (m,k).
    U = np.einsum('mia,iajb,kjb->mk', C_act, Delta**2, X_act)
    return U


# ---------------------------------------------------------------------------
# Step 5: Löwdin orthonormalization of U
# ---------------------------------------------------------------------------

def lowdin_orthogonalize(U):
    """
    Return the orthogonal matrix U_orth that is closest to U in Frobenius norm.

    Why U is not orthogonal (CLAUDE.md §Löwdin Orthonormalization)
    --------------------------------------------------------------
    U[m,k] = <Phi_m|Psi_k> is the overlap of diabatic state m onto adiabatic
    state k. Because the 16 adiabatic states do not span the full Hilbert space,
    the rows of U are not normalised:

        sum_k U[m,k]^2 = ||Phi_m projected onto 16 adiabats||^2  < 1

    Therefore U U^T != I and U is not orthogonal.

    We cannot use U^T in place of U^{-1} (wrong because U U^T != I), and we
    cannot use U^{-1} in place of U^T (breaks symmetry of H_diab). Both are
    wrong (see CLAUDE.md §What NOT To Do).

    Löwdin orthonormalization via SVD
    ---------------------------------
    The polar decomposition theorem guarantees that the orthogonal matrix
    closest to any matrix A in Frobenius norm is obtained from its SVD:

        A = V Sigma W^T  =>  A_orth = V W^T

    The singular values Sigma are simply discarded (replaced by 1), which
    is equivalent to projecting each diabatic state fully onto the adiabatic
    subspace and then renormalising, preserving the relative orientation
    (the V and W^T factors) as faithfully as possible.

    Properties of U_orth = V W^T:
      - U_orth @ U_orth.T = I  exactly  (16x16 identity)
      - U_orth is the closest orthogonal matrix to U  (minimum Frobenius error)
      - Eigenvalues of H_diab = U_orth @ diag(E_stack) @ U_orth.T are exactly
        the rstack adiabatic energies E_stack (non-negotiable validation criterion)

    Parameters
    ----------
    U : ndarray, shape (16, 16), float64
        Raw many-body overlap matrix from Step 4.

    Returns
    -------
    U_orth : ndarray, shape (16, 16), float64
        Löwdin-orthogonalized overlap matrix. Satisfies U_orth @ U_orth.T = I.
    V      : ndarray, shape (16, 16), float64  [left singular vectors]
    sigma  : ndarray, shape (16,), float64     [singular values, for inspection]
    Wt     : ndarray, shape (16, 16), float64  [right singular vectors transposed]
    """
    # Full SVD: U = V @ diag(sigma) @ Wt
    # V   : (16,16) unitary, left singular vectors  (columns are left  sv vectors)
    # sigma: (16,)  singular values in descending order
    # Wt  : (16,16) unitary, right singular vectors transposed (rows are right sv vectors)
    V, sigma, Wt = np.linalg.svd(U)

    # Löwdin: drop the singular values, keep the rotational part.
    # U_orth = V @ I @ Wt  where I replaces diag(sigma).
    # This is the unique closest orthogonal matrix to U (von Neumann trace inequality).
    U_orth = V @ Wt

    return U_orth, V, sigma, Wt


# ---------------------------------------------------------------------------
# Step 6: Build the diabatic Hamiltonian
# ---------------------------------------------------------------------------

def compute_H_diab(U_orth, E_stack):
    """
    Construct the diabatic Hamiltonian H_diab in the rinf diabatic basis.

    Theory (CLAUDE.md §The Diabatic Hamiltonian)
    ---------------------------------------------
    Each diabatic state |Phi_m> is expanded in the Löwdin-orthogonalized
    adiabatic basis:

        |Phi_m> ≈ sum_k U_orth[m,k] |Psi_k>

    Taking matrix elements of H_stack (the rstack electronic Hamiltonian)
    and using <Psi_k|H_stack|Psi_l> = E_stack[k] * delta_kl (the adiabatic
    states are eigenstates of H_stack with energies E_stack):

        H_diab[m,n] = <Phi_m|H_stack|Phi_n>
                    = sum_k U_orth[m,k] * E_stack[k] * U_orth[n,k]

    In matrix form:

        H_diab = U_orth @ diag(E_stack) @ U_orth.T

    Properties guaranteed by construction:
      - Real symmetric: H_diab = H_diab.T  (U_orth is orthogonal, diag(E) is symmetric)
      - Eigenvalues of H_diab are exactly E_stack (the rstack adiabatic energies)
        because U_orth is orthogonal: U_orth.T @ H_diab @ U_orth = diag(E_stack)

    IMPORTANT sign conventions (CLAUDE.md §What NOT To Do):
      - Use U_orth @ diag(E) @ U_orth.T  NOT  U_orth.T @ diag(E) @ U_orth
        (the latter gives a non-symmetric result when rows != cols convention differs)
      - Do NOT use U^{-1} in place of U_orth.T — breaks symmetry
      - U_orth[m,k] is row=diabatic, col=adiabatic, so the sandwich is
        (diabatic x adiabatic) @ (adiabatic x adiabatic) @ (adiabatic x diabatic)
        = (diabatic x diabatic)  ✓

    Parameters
    ----------
    U_orth  : ndarray, shape (16, 16), float64
        Löwdin-orthogonalized overlap matrix from Step 5. U_orth[m,k].
    E_stack : ndarray, shape (16,), float64
        Rstack adiabatic energies in Hartree.

    Returns
    -------
    H_diab : ndarray, shape (16, 16), float64
        Diabatic Hamiltonian in Hartree. Real symmetric.
        Diagonal elements H_diab[m,m] are the on-site diabatic energies.
        Off-diagonal elements H_diab[m,n] are the diabatic couplings.
    """
    # Sandwich the diagonal adiabatic energy matrix between U_orth and U_orth.T.
    # np.diag(E_stack) is (16,16); the full product is (16,16) @ (16,16) @ (16,16).
    H_diab = U_orth @ np.diag(E_stack) @ U_orth.T
    return H_diab


# ---------------------------------------------------------------------------
# Main: Step 1 + Step 2 driver
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Step 1: Load and validate inputs")
    print("=" * 60)

    S_MO, E_ref, C_ref, E_stack, X_stack = load_inputs()

    print(f"  S_MO shape    : {S_MO.shape}   (ref MO rows × stack MO cols)")
    print(f"  C_ref shape   : {C_ref.shape}  (roots × nocc_full × nvirt)")
    print(f"  X_stack shape : {X_stack.shape}  (roots × nocc_full × nvirt)")
    print(f"  E_ref   (Ha)  : {np.array2string(E_ref,  precision=6)}")
    print(f"  E_stack (Ha)  : {np.array2string(E_stack, precision=6)}")

    print("\nValidation:")
    validate_inputs(C_ref, X_stack)

    # Active slices: drop the four frozen-core zero rows.
    # C_act[m, i, a] = C^(m)_{ia}, shape (16, 12, 80)
    #   m : root index 0..15 (rinf diabatic state)
    #   i : local active occ index 0..11  (global S_MO row = i + OCC_OFFSET)
    #   a : local virtual index 0..79     (global S_MO col = a + VIRT_OFFSET)
    C_act = C_ref[:, N_FRZ:, :]    # (16, 12, 80)
    X_act = X_stack[:, N_FRZ:, :]  # (16, 12, 80)
    print(f"\n  C_act shape   : {C_act.shape}  (roots × nocc_active × nvirt)")
    print(f"  X_act shape   : {X_act.shape}  (roots × nocc_active × nvirt)")

    print("\nStep 1 complete.")

    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 2: Build global index arrays r and s")
    print("=" * 60)

    r, s = build_index_arrays()

    # Shape checks
    assert r.shape == (N_OCC, N_VIRT, N_OCC), f"r shape {r.shape} unexpected"
    assert s.shape == (N_OCC, N_VIRT, N_OCC), f"s shape {s.shape} unexpected"
    print(f"  r shape: {r.shape}  (i, a, p) -> global S_MO row index")
    print(f"  s shape: {s.shape}  (j, b, q) -> global S_MO col index")

    # Spot-check: for excitation i=0, a=0
    #   positions p != 0 should be p + 4 (i.e. 4,5,6,...,15)
    #   position  p == 0 should be 0 + 16 = 16
    expected_r00 = np.array([16, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int32)
    assert np.array_equal(r[0, 0, :], expected_r00), \
        f"r[0,0,:] = {r[0,0,:]} != expected {expected_r00}"
    print(f"\n  Spot-check r[i=0, a=0, :] = {r[0, 0, :]}")
    print(f"    Expected:                  {expected_r00}  ✓")

    # Spot-check: for excitation i=3, a=5
    #   position p=3 should be 5 + 16 = 21
    #   all other positions p should be p + 4
    expected_r35 = np.arange(N_OCC, dtype=np.int32) + OCC_OFFSET
    expected_r35[3] = 5 + VIRT_OFFSET   # = 21
    assert np.array_equal(r[3, 5, :], expected_r35), \
        f"r[3,5,:] = {r[3,5,:]} != expected {expected_r35}"
    print(f"\n  Spot-check r[i=3, a=5, :] = {r[3, 5, :]}")
    print(f"    Expected:                  {expected_r35}  ✓")

    # Verify s is independent of r (same structure, different variable)
    assert np.array_equal(r, s), "r and s should be identical arrays (same formula)"
    print("\n  r and s are identical (both encode the same substitution rule)  ✓")

    # Validate all global indices are within [OCC_OFFSET, N_BAS)
    # i.e. no index falls in the frozen core block 0..3
    assert r.min() >= OCC_OFFSET, f"r contains index < {OCC_OFFSET} (frozen core!)"
    assert s.min() >= OCC_OFFSET, f"s contains index < {OCC_OFFSET} (frozen core!)"
    assert r.max() < N_BAS,       f"r contains index >= {N_BAS} (out of basis!)"
    assert s.max() < N_BAS,       f"s contains index >= {N_BAS} (out of basis!)"
    print(f"  All indices in [{r.min()}, {r.max()}] ⊂ [{OCC_OFFSET}, {N_BAS-1}]  ✓")

    print("\nStep 2 complete.")

    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3: Precompute determinant tensor Delta[i,a,j,b]")
    print("=" * 60)

    import time
    t0 = time.time()
    Delta = build_delta_tensor(S_MO, r, s)
    t1 = time.time()

    assert Delta.shape == (N_OCC, N_VIRT, N_OCC, N_VIRT), \
        f"Delta shape {Delta.shape} unexpected"
    print(f"  Delta shape: {Delta.shape}  (i, a, j, b)")
    print(f"  Time: {t1-t0:.1f} s")

    # --- Manual spot-check: Delta[0,0,0,0] vs direct computation ---
    # For i=0, a=0, j=0, b=0: both bra and ket replace occ 0 with virt 0.
    # Manually extract the 12x12 submatrix and compute its determinant.
    D_manual = S_MO[np.ix_(r[0, 0, :], s[0, 0, :])]   # shape (12, 12)
    det_manual = np.linalg.det(D_manual)
    print(f"\n  Spot-check Delta[0,0,0,0]:")
    print(f"    build_delta_tensor : {Delta[0, 0, 0, 0]:.15f}")
    print(f"    direct det         : {det_manual:.15f}")
    assert np.isclose(Delta[0, 0, 0, 0], det_manual, rtol=1e-12), \
        "Delta[0,0,0,0] does not match direct determinant computation!"
    print(f"    Match: ✓")

    # Spot-check a non-trivial off-diagonal case: i=2, a=7, j=5, b=11
    i_, a_, j_, b_ = 2, 7, 5, 11
    D_manual2 = S_MO[np.ix_(r[i_, a_, :], s[j_, b_, :])]
    det_manual2 = np.linalg.det(D_manual2)
    print(f"\n  Spot-check Delta[{i_},{a_},{j_},{b_}]:")
    print(f"    build_delta_tensor : {Delta[i_, a_, j_, b_]:.15f}")
    print(f"    direct det         : {det_manual2:.15f}")
    assert np.isclose(Delta[i_, a_, j_, b_], det_manual2, rtol=1e-12), \
        f"Delta[{i_},{a_},{j_},{b_}] does not match direct det!"
    print(f"    Match: ✓")

    print("\nStep 3 complete.")

    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 4: Compute many-body overlap matrix U[m,k]")
    print("=" * 60)

    U = compute_U(C_act, Delta, X_act)

    assert U.shape == (N_ROOTS, N_ROOTS), f"U shape {U.shape} unexpected"
    print(f"  U shape: {U.shape}  (diabatic m rows × adiabatic k cols)")

    # Print U rounded to 4 decimal places for visual inspection
    print("\n  U matrix (rows=diabatic m, cols=adiabatic k):")
    for row in U:
        print("  " + "  ".join(f"{v:8.4f}" for v in row))

    # Sanity check 1: U should NOT be exactly orthogonal.
    # The 16 adiabatic states do not span the full Hilbert space, so
    # sum_k U[m,k]*U[n,k] != delta_mn. This is expected and is why we need
    # Löwdin orthonormalization in Step 5.
    UUt = U @ U.T
    diag_vals = np.diag(UUt)
    print(f"\n  Diagonal of U @ U.T (expect < 1.0, not identity):")
    print(f"  {np.array2string(diag_vals, precision=4)}")
    assert not np.allclose(UUt, np.eye(N_ROOTS), atol=1e-3), \
        "U appears orthogonal — unexpected; Löwdin step would be trivial"
    print("  U is not orthogonal (as expected — adiabatic basis is incomplete)  ✓")

    # Sanity check 2: singular values of U.
    # All should lie in (0, 1]. Values near 0 mean a diabatic state has
    # negligible projection onto the 16 adiabatic states (bad, would indicate
    # the adiabatic window is too small). Values near 1 are ideal.
    sv = np.linalg.svd(U, compute_uv=False)
    print(f"\n  Singular values of U:")
    print(f"  {np.array2string(sv, precision=6)}")
    assert np.all(sv <= 1.0 + 1e-10), \
        f"Singular value > 1 found: {sv.max():.6f} — check input data"
    print(f"  Range: [{sv.min():.6f}, {sv.max():.6f}]  (expect all in (0,1])  ✓")

    print("\nStep 4 complete.")

    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 5: Löwdin orthonormalization")
    print("=" * 60)

    U_orth, V, sigma, Wt = lowdin_orthogonalize(U)

    assert U_orth.shape == (N_ROOTS, N_ROOTS), \
        f"U_orth shape {U_orth.shape} unexpected"
    print(f"  U_orth shape: {U_orth.shape}")

    # Print singular values: these show how much each mode of U was rescaled.
    # Values near 1 mean U was already close to orthogonal in that direction.
    # Very small singular values (near 0) mean those modes had tiny overlap
    # with the adiabatic subspace — Löwdin inflates them to 1, which is the
    # largest correction and the least physically reliable part of U_orth.
    print(f"\n  Singular values of U (discarded by Löwdin, replaced by 1):")
    print(f"  {np.array2string(sigma, precision=6)}")

    # Primary validation: U_orth must satisfy U_orth @ U_orth.T = I_{16}
    # to machine precision. Any deviation here is a bug.
    residual = U_orth @ U_orth.T - np.eye(N_ROOTS)
    max_err = np.abs(residual).max()
    print(f"\n  ||U_orth @ U_orth.T - I||_max = {max_err:.3e}  (expect ~1e-15)")
    assert max_err < 1e-12, \
        f"U_orth is not orthogonal: max error = {max_err:.3e}"
    print("  U_orth @ U_orth.T = I  ✓")

    # Save raw and orthogonalized overlap matrices for downstream use / inspection
    np.save('U_raw.npy',  U)
    np.save('U_orth.npy', U_orth)
    print("\n  Saved: U_raw.npy, U_orth.npy")

    print("\nStep 5 complete.")

    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 6: Diabatic Hamiltonian H_diab = U_orth @ diag(E_stack) @ U_orth.T")
    print("=" * 60)

    H_diab = compute_H_diab(U_orth, E_stack)

    assert H_diab.shape == (N_ROOTS, N_ROOTS), \
        f"H_diab shape {H_diab.shape} unexpected"

    # --- Validation 1: symmetry ---
    # H_diab must be exactly symmetric (real symmetric matrix).
    sym_err = np.abs(H_diab - H_diab.T).max()
    print(f"  ||H_diab - H_diab.T||_max = {sym_err:.3e}  (expect ~1e-16)")
    assert np.allclose(H_diab, H_diab.T, atol=1e-12), \
        f"H_diab is not symmetric: max error = {sym_err:.3e}"
    print("  H_diab is symmetric  ✓")

    # --- Validation 2: eigenvalues must recover E_stack exactly ---
    # This is the non-negotiable check: the congruence transformation
    # U_orth @ diag(E) @ U_orth.T preserves eigenvalues iff U_orth is orthogonal.
    eigvals = np.sort(np.linalg.eigvalsh(H_diab))
    E_stack_sorted = np.sort(E_stack)
    eig_err = np.abs(eigvals - E_stack_sorted).max()
    print(f"\n  Eigenvalue recovery:")
    print(f"  E_stack sorted (Ha) : {np.array2string(E_stack_sorted, precision=6)}")
    print(f"  eigvalsh(H_diab)    : {np.array2string(eigvals,        precision=6)}")
    print(f"  max |diff|          : {eig_err:.3e}  (expect ~1e-14)")
    assert eig_err < 1e-10, \
        f"H_diab eigenvalues do not match E_stack: max error = {eig_err:.3e}"
    print("  Eigenvalues match E_stack  ✓")

    # --- Print H_diab in eV ---
    H_eV = H_diab * HA_TO_EV
    print(f"\n  H_diab (eV), full 16×16 matrix:")
    print(f"  (rows/cols = diabatic state index m, 0-based)")
    header = "      " + "".join(f"  {k:7d}" for k in range(N_ROOTS))
    print(header)
    for m, row in enumerate(H_eV):
        row_str = "".join(f"  {v:7.3f}" for v in row)
        print(f"  {m:2d} |{row_str}")

    # --- Diagonal: on-site diabatic energies ---
    print(f"\n  On-site energies H_diab[m,m] (eV):")
    for m in range(N_ROOTS):
        print(f"    m={m:2d}  {H_eV[m,m]:8.4f} eV")

    # --- Save outputs ---
    np.save('H_diab.npy', H_diab)
    # Write human-readable .dat file in eV
    with open('H_diab.dat', 'w') as f:
        f.write("# Diabatic Hamiltonian H_diab (eV)\n")
        f.write("# H_diab = U_orth @ diag(E_stack) @ U_orth.T\n")
        f.write("# Rows and columns: diabatic state index m (0-based)\n")
        f.write("# Method: CAM-B3LYP/def2-SVP TDDFT, ORCA 6.0.1\n")
        f.write("#\n")
        for row in H_eV:
            f.write("  ".join(f"{v:18.10f}" for v in row) + "\n")
    print("\n  Saved: H_diab.npy, H_diab.dat")

    print("\nStep 6 complete.")

    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Export: H_diab, H_adiab_stack, H_adiab_inf as .dat files")
    print("=" * 60)

    def write_matrix_dat(path, mat, header):
        """
        Write a square matrix to a plain-text .dat file matching the SMO.dat
        format: one comment header line, then one row of space-separated floats
        per matrix row, 11 decimal places.

        Parameters
        ----------
        path   : str, output file path
        mat    : ndarray (N, N), matrix to write
        header : str, content of the single comment line (without the leading #)
        """
        with open(path, 'w') as f:
            f.write(f"# {header}\n")
            for row in mat:
                f.write("  ".join(f"{v:18.11f}" for v in row) + "\n")

    # --- H_diab in eV ---
    # Already computed above as H_diab (Ha); convert here for the .dat export.
    write_matrix_dat(
        'H_diab_eV.dat',
        H_diab * HA_TO_EV,
        "Diabatic Hamiltonian H_diab (eV)  shape=(16,16)  "
        "rows=cols=diabatic state m (0-based)  "
        "H_diab = U_orth @ diag(E_stack) @ U_orth.T"
    )

    # --- H_adiab_stack: diagonal matrix of rstack adiabatic energies in eV ---
    # The rstack Hamiltonian in the adiabatic basis is simply diag(E_stack).
    # It is diagonal by construction (|Psi_k> are eigenstates of H_stack).
    H_adiab_stack = np.diag(E_stack) * HA_TO_EV   # (16, 16), eV
    write_matrix_dat(
        'H_adiab_stack_eV.dat',
        H_adiab_stack,
        "Adiabatic Hamiltonian H_adiab_stack (eV)  shape=(16,16)  "
        "rows=cols=adiabatic state k (0-based)  diagonal: E_stack eigenvalues"
    )

    # --- H_adiab_inf: diagonal matrix of rinf adiabatic energies in eV ---
    # The rinf Hamiltonian in its own adiabatic (and here also diabatic) basis
    # is diag(E_ref), since at rinf the monomers are non-interacting and the
    # states are already pure XT/CT — no mixing, so adiabatic = diabatic.
    H_adiab_inf = np.diag(E_ref) * HA_TO_EV   # (16, 16), eV
    write_matrix_dat(
        'H_adiab_inf_eV.dat',
        H_adiab_inf,
        "Adiabatic Hamiltonian H_adiab_inf (eV)  shape=(16,16)  "
        "rows=cols=rinf state m (0-based)  diagonal: E_ref eigenvalues (adiabatic=diabatic at rinf)"
    )

    print("  Saved: H_diab_eV.dat")
    print("  Saved: H_adiab_stack_eV.dat")
    print("  Saved: H_adiab_inf_eV.dat")
    print("  Format: SMO.dat-style — one comment header, then rows of floats (18.11f)")

    print("\nAll steps complete.")
