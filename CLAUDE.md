# CLAUDE.md — Quasi-Diabatization Pipeline for Ethylene Dimer

## Purpose

Authoritative theory and implementation guide for computing the diabatic Hamiltonian
of an eclipsed cofacial ethylene dimer from TDDFT data. Read completely before
writing any code.

---

## Physical System

- **System**: eclipsed cofacial ethylene dimer (H-aggregate, D2h symmetry)
- **Method**: CAM-B3LYP/def2-SVP, full TDDFT (no TDA), NoUseSym, ORCA 6.0.1
- **AO basis**: 96 functions total (48 per monomer)
- **Orbital window** (0-based throughout):
  - indices 0–3: frozen core (C 1s), excluded from CI
  - indices 4–15: active occupied (12 MOs)
  - indices 16–95: virtual (80 MOs)
- **Two geometries**:
  - **rinf** (20 Å separation): monomers non-interacting. TDDFT states are
    pure excitonic (XT) or charge-transfer (CT) by construction.
    Used as the diabatic reference basis.
  - **rstack** (3.5 Å π-stacking): states are heavily mixed adiabats.

---

## Input Files

| File | Contents |
|---|---|
| `S_MO.npy` | 96×96 float64 numpy array, cross-geometry MO overlap (precomputed, validated) |
| `rinf_vectorsX.dat` | TDA X amplitudes for all 16 rinf states, plain text |
| `rstack_vectorsX.dat` | TDA X amplitudes for all 16 rstack states, plain text |

### Amplitude File Format (`rinf_vectorsX.dat`, `rstack_vectorsX.dat`)

Plain text. Lines beginning with `#` are comments and must be skipped.
Blank lines separate root blocks.

File-level header (comment lines, skip):
```
# rinf: amplitudes  shape=(nroots=16, nocc_full=16, nvirt=80)
# nfrozen=4 (rows 0..3 are zero)  nocc_active=12
# Each block: root N (1-based), energy, then nocc_full rows of nvirt values
```

Each root block:
```
# Root N  E=X.XXXXXXXXXX Ha  Y.YYYYYY eV
<row 0:   80 space-separated floats>   ← frozen core, all zero
<row 1:   80 floats>                   ← frozen core, all zero
<row 2:   80 floats>                   ← frozen core, all zero
<row 3:   80 floats>                   ← frozen core, all zero
<row 4:   80 floats>                   ← active occ index 0 (global index 4)
<row 5:   80 floats>                   ← active occ index 1 (global index 5)
...
<row 15:  80 floats>                   ← active occ index 11 (global index 15)
<blank line>
```

Parse by:
1. Skipping all `#` lines
2. Collecting the energy from the comment `# Root N  E=X Ha  Y eV` (field after `E=`, before `Ha`)
3. Reading the next 16 non-comment, non-blank lines as float rows for that root
4. Repeating for all 16 roots

After parsing, the amplitude array has shape `(16, 16, 80)` = `(nroots, nocc_full, nvirt)`.
Rows 0–3 are identically zero (frozen core). Active amplitudes are in rows 4–15.
Energies are extracted from the comment lines (in Hartree).

The rstack file has the same format and the same shape.

---

## Notation

| Symbol | Meaning |
|---|---|
| $N_f = 4$ | Frozen core MOs |
| $N_o = 12$ | Active occupied MOs |
| $N_v = 80$ | Virtual MOs |
| $N_r = 16$ | Number of TDDFT roots at each geometry |
| $m$ | Diabatic ref (rinf) state index, $0,\ldots,15$ |
| $k$ | Adiabatic stack (rstack) state index, $0,\ldots,15$ |
| $i, j$ | Active occupied indices, local $0,\ldots,11$, global $i+4$ |
| $a, b$ | Virtual indices, local $0,\ldots,79$, global $a+16$ |
| $C^{(m)}_{ia}$ | TDA amplitude for rinf state $m$: excitation occ $i$ → virt $a$ |
| $X^{(k)}_{jb}$ | TDA amplitude for rstack state $k$: excitation occ $j$ → virt $b$ |
| $S^{\text{MO}}_{pq}$ | MO overlap $\langle\varphi^{\text{ref}}_p\|\varphi^{\text{stack}}_q\rangle$, row = ref, col = stack |

All indices are **0-based** throughout.

---

## Theory

### The MO Overlap Matrix $S^{\text{MO}}$

$$S^{\text{MO}}_{pq} = \langle\varphi^{\text{ref}}_p | \varphi^{\text{stack}}_q\rangle$$

- Row index $p$ (0-based, 0..95): **ref** (rinf) MO
- Col index $q$ (0-based, 0..95): **stack** (rstack) MO
- Shape: $96\times96$, loaded from `S_MO.npy`
- **Not symmetric**: cross-geometry overlaps have no reason to be symmetric
- Validated: frontier MO subblock shows expected ±1/√2 Hadamard structure

---

### The TDA Amplitudes

$C^{(m)}_{ia}$ from `rinf_vectorsX.dat`: amplitude for rinf state $m$ to excite
from active occupied MO $i$ to virtual MO $a$. These are the X amplitudes
extracted from full TDDFT (Y amplitudes neglected; justified since $\|Y\|/\|X\|
\approx 0.1$–$0.2$ for these XT and CT states).

$X^{(k)}_{jb}$ from `rstack_vectorsX.dat`: same meaning for rstack state $k$.

In code, after parsing:
- `C_ref` has shape `(16, 16, 80)` — full array including frozen zero rows
- `X_stack` has shape `(16, 16, 80)` — same
- Active slices used in computation: `C_ref[:, 4:16, :]` and `X_stack[:, 4:16, :]`,
  both shape `(16, 12, 80)`

---

### The Many-Body States

The diabatic ref states are CIS-like expansions over the rinf MO basis:

$$|\Phi_m\rangle = \sum_{i=0}^{11}\sum_{a=0}^{79} C^{(m)}_{ia}\,|\Phi^a_i\rangle^{\text{ref}}$$

where $|\Phi^a_i\rangle^{\text{ref}}$ is the closed-shell Slater determinant formed
from the rinf reference ground state by replacing active occupied spatial MO $i$
with virtual MO $a$.

The adiabatic stack states:

$$|\Psi_k\rangle = \sum_{j=0}^{11}\sum_{b=0}^{79} X^{(k)}_{jb}\,|\Psi^b_j\rangle^{\text{stack}}$$

These satisfy $H^{\text{stack}}|\Psi_k\rangle = E^{\text{stack}}_k|\Psi_k\rangle$
and $\langle\Psi_k|\Psi_l\rangle = \delta_{kl}$.

---

### Why the Slater Determinant Overlap is a Determinant

The overlap between any two Slater determinants built from $N$ orbitals each is
the determinant of the $N\times N$ matrix of pairwise orbital overlaps.

Derivation sketch: write both determinants via the Leibniz expansion. The
many-electron overlap integral factorizes over electrons (each electron coordinate
appears in exactly one factor). Relabelling the double sum over permutations
$(\sigma, \tau)$ via $\pi = \tau\circ\sigma^{-1}$ eliminates $\sigma$ entirely
(it contributes $N!$ identical copies), leaving the Leibniz definition of the
determinant. Crucially, this does **not** reduce to a product of individual overlaps
unless the orbital overlap matrix is diagonal — which it is not across two different
geometries.

---

### The Orbital Overlap Submatrix $D^{ab}_{ij}$

$|\Phi^a_i\rangle^{\text{ref}}$ occupies $N_o = 12$ ref spatial orbitals:
all active occupied except $i$, with virtual $a$ placed at position $i$.

$|\Psi^b_j\rangle^{\text{stack}}$ occupies $N_o = 12$ stack spatial orbitals:
all active occupied except $j$, with virtual $b$ placed at position $j$.

Their overlap is:

$$\langle\Phi^a_i|\Psi^b_j\rangle = \left[\det(D^{ab}_{ij})\right]^2$$

The squaring comes from RKS: each spatial orbital is doubly occupied (alpha + beta),
the spin-orbital overlap matrix is block-diagonal in spin, and
$\det(\text{block-diag}(A,A)) = [\det(A)]^2$.

$D^{ab}_{ij}$ is the $12\times12$ matrix of spatial orbital overlaps:

$$D^{ab}_{ij}[p,q] = S^{\text{MO}}_{r_p,\,s_q}$$

where the **global** index maps (0-based) are:

$$r_p = \begin{cases} p + 4 & p \neq i \\ a + 16 & p = i \end{cases}
\qquad
s_q = \begin{cases} q + 4 & q \neq j \\ b + 16 & q = j \end{cases}
\qquad p,q\in\{0,\ldots,11\}$$

**$D^{ab}_{ij}$ is exactly the submatrix of $S^{\text{MO}}$ with rows
$\mathbf{r}^{(ia)}$ and columns $\mathbf{s}^{(jb)}$.** No approximation.
All four blocks of $S^{\text{MO}}$ (occ-occ, occ-virt, virt-occ, virt-virt)
contribute through the determinant. The four blocks are:

| Position in $D^{ab}_{ij}$ | Block of $S^{\text{MO}}$ | Physical meaning |
|---|---|---|
| $p\neq i,\; q\neq j$ | $S_{\text{oo}}[p,q]$ | ref occ vs stack occ |
| $p=i,\; q\neq j$ | $S_{\text{vo}}[a,q]$ | ref virt vs stack occ |
| $p\neq i,\; q=j$ | $S_{\text{ov}}[p,b]$ | ref occ vs stack virt |
| $p=i,\; q=j$ | $S_{\text{vv}}[a,b]$ | ref virt vs stack virt |

---

### The Exact Many-Body Overlap $U_{mk}$

$$U_{mk} = \langle\Phi_m|\Psi_k\rangle
= \sum_{i,a}\sum_{j,b} C^{(m)}_{ia}\,X^{(k)}_{jb}\,
\left[\det\!\left(S^{\text{MO}}\!\left[\mathbf{r}^{(ia)},\,\mathbf{s}^{(jb)}\right]\right)\right]^2$$

This is exact. No approximation about any block of $S^{\text{MO}}$ being small.

---

### Computational Strategy: Precompute the Determinant Tensor

The key observation: $\det(S^{\text{MO}}[\mathbf{r}^{(ia)}, \mathbf{s}^{(jb)}])$
depends only on the orbital pair $(i,a,j,b)$ — **not** on which roots $m,k$ are
being computed. Therefore:

**Step A**: Precompute once:

$$\Delta_{iajb} = \det\!\left(S^{\text{MO}}\!\left[\mathbf{r}^{(ia)},\,\mathbf{s}^{(jb)}\right]\right)$$

Shape of $\Delta$: $(N_o, N_v, N_o, N_v) = (12, 80, 12, 80)$.
Number of $12\times12$ determinants to evaluate: $12^2\times80^2 = 921{,}600$.

**Step B**: Contract over all root pairs simultaneously:

$$U_{mk} = \sum_{i,a,j,b} C^{(m)}_{ia}\cdot\Delta^2_{iajb}\cdot X^{(k)}_{jb}$$

This is a single einsum over the precomputed tensor.

The bottleneck is Step A. Vectorize the determinant evaluations using numpy
advanced indexing to batch submatrix extraction, then apply `np.linalg.det`
over the batch axis. Do not write a Python-level quadruple loop.

---

### Löwdin Orthonormalization

The raw $U$ is not orthogonal. The 16 adiabatic states do not span the full
Hilbert space — completeness $\sum_k|\Psi_k\rangle\langle\Psi_k| = \mathbf{1}$
requires infinitely many states. Therefore:

$$\sum_{k=1}^{16} U_{mk}U_{nk} \neq \delta_{mn}$$

Using non-orthogonal $U$ directly in $H^{\text{diab}} = U\Lambda U^T$ gives a
congruence transformation: eigenvalues are not preserved and the result is not
symmetric. Using $U^{-1}$ in place of $U^T$ preserves eigenvalues but breaks
symmetry. Both are wrong.

Löwdin orthonormalization gives the orthogonal matrix **closest to $U$ in
Frobenius norm**. Compute via SVD:

$$U = V\Sigma W^T \implies \tilde{U} = VW^T$$

This satisfies $\tilde{U}\tilde{U}^T = I$ exactly, regardless of the conditioning
of $\Sigma$.

---

### The Diabatic Hamiltonian

Expand each diabatic state in the orthonormalized adiabatic basis:

$$|\Phi_m\rangle \approx \sum_k \tilde{U}_{mk}|\Psi_k\rangle$$

Take matrix elements using $\langle\Psi_k|H^{\text{stack}}|\Psi_l\rangle
= E^{\text{stack}}_k\delta_{kl}$:

$$H^{\text{diab}}_{mn} = \langle\Phi_m|H|\Phi_n\rangle
= \sum_k \tilde{U}_{mk}\,E^{\text{stack}}_k\,\tilde{U}_{nk}$$

$$\boxed{H^{\text{diab}} = \tilde{U}\,\mathrm{diag}(E^{\text{stack}})\,\tilde{U}^T}$$

$H^{\text{diab}}$ is $16\times16$, real symmetric, and has eigenvalues
$\{E^{\text{stack}}_k\}$ — these are non-negotiable validation criteria.

Truncation to the $4\times4$ block $\{|XT_A\rangle,|XT_B\rangle,|CT_1\rangle,|CT_2\rangle\}$
is a post-processing step done after inspecting the dominant $C^{(m)}_{ia}$
character of each rinf state.

---

## Implementation Plan

### Step 1 — Load inputs ✓ DONE

- Load `S_MO.npy` → array of shape `(96, 96)`
- Parse `rinf_vectorsX.dat`:
  - Skip `#` lines; collect energy from `# Root N  E=X Ha` lines
  - Read 16 data rows per root block
  - Result: `C_ref` shape `(16, 16, 80)`, `E_ref` length 16 (Ha)
- Parse `rstack_vectorsX.dat` → `X_stack` shape `(16, 16, 80)`, `E_stack` length 16 (Ha)
- Extract active slices: `C_act = C_ref[:, 4:16, :]`, `X_act = X_stack[:, 4:16, :]`,
  both shape `(16, 12, 80)`

Implemented in `diabatize.py`: `parse_vectors()`, `load_inputs()`, `validate_inputs()`.
All functions carry detailed docstrings and inline comments explaining the file
layout, index conventions, and physical meaning of each quantity.
Validation output: shapes correct, frozen core rows zero, norms ≈ 1.00–1.03
(slight excess expected: X amplitudes from full TDDFT, ‖X‖² = 1 + ‖Y‖² > 1 when Y non-negligible).

### Step 2 — Build global index arrays ✓ DONE

For every $(i, a)$ pair build the 12-element row index vector $\mathbf{r}^{(ia)}$:

```
r[i, a, p] = p + 4   if p != i
r[i, a, i] = a + 16
```

Shape of `r`: `(12, 80, 12)` — for each $(i,a)$, a length-12 vector of global
S_MO row indices.

Similarly build `s[j, b, q]` for column indices, same shape `(12, 80, 12)`.

Implemented in `diabatize.py`: `build_index_arrays()`.
Strategy: broadcast the baseline occupied index vector `p + OCC_OFFSET` to
shape `(12, 80, 12)`, then overwrite position `p=i` with `a + VIRT_OFFSET`
via a loop over i (12 iterations, each a vectorised slice over all 80 virtuals).
Verified:
- `r[0,0,:]` = `[16, 5, 6, ..., 15]` ✓ (virt 0 replaces occ 0 at position 0)
- `r[3,5,:]` = `[4, 5, 6, 21, 8, ..., 15]` ✓ (virt 5 → global 21 replaces occ 3)
- All indices in `[4, 95]` — frozen core block `[0,3]` never appears ✓

### Step 3 — Precompute determinant tensor $\Delta$ ✓ DONE

For each of the $12\times80\times12\times80 = 921{,}600$ pairs $(i,a,j,b)$:
- Select the $12\times12$ submatrix `S_MO[r[i,a,:], :][:, s[j,b,:]]`
- Compute its determinant

Store as `Delta` of shape `(12, 80, 12, 80)`.

Vectorize: construct all submatrices as a batched array of shape
`(12, 80, 12, 80, 12, 12)` using numpy advanced indexing, then call
`np.linalg.det` once over the batch dimensions.

Memory note: `(12*80*12*80, 12, 12)` float64 = ~$921{,}600 \times 144 \times 8$
bytes ≈ 1.06 GB. If memory is a constraint, loop over one index pair and
vectorize over the other.

Implemented in `diabatize.py`: `build_delta_tensor()`.
Strategy: reshape `r` to `(12,80,1,1,12,1)` and `s` to `(1,1,12,80,1,12)`,
broadcast-index into S_MO to get M of shape `(12,80,12,80,12,12)`, then call
`np.linalg.det(M)` over the last two axes. Runtime: ~6 s on a single CPU core.
Verified:
- `Delta[0,0,0,0]` = `-0.515001672168122` matches direct `np.linalg.det` ✓
- `Delta[2,7,5,11]` = `-2.486e-9` matches direct `np.linalg.det` ✓

### Step 4 — Compute $U$ ✓ DONE

```python
U = np.einsum('mia,iajb,kjb->mk', C_act, Delta**2, X_act)
```

Shape: `(16, 16)`.

Implemented in `diabatize.py`: `compute_U()`.
`Delta**2` is formed first (squaring for RKS closed-shell spin), then a single
einsum contracts over all (i,a,j,b) excitation pairs to yield U[m,k].
Verified: U is not orthogonal (U@U.T diagonal entries range from ~1e-15 to ~0.49),
and all singular values lie in [0, 0.988] as expected. Rows 14–15 near zero
indicate those rinf states have negligible overlap with the 16 rstack adiabats.

### Step 5 — Löwdin orthonormalization ✓ DONE

```python
V, s, Wt = np.linalg.svd(U)
U_orth = V @ Wt
```

Implemented in `diabatize.py`: `lowdin_orthogonalize()`.
SVD decomposes U = V Σ Wt; Löwdin drops Σ (replaces each singular value with 1)
and returns U_orth = V @ Wt — the orthogonal matrix closest to U in Frobenius norm.
Verified:
- `||U_orth @ U_orth.T - I||_max = 1.9e-15`  (machine precision)  ✓
- Singular values of U range from 1.3e-9 to 0.988 — the near-zero modes
  (indices 6–15) correspond to diabatic/adiabatic states with negligible
  mutual overlap; Löwdin inflates these to 1, which is the least reliable
  part of U_orth and worth keeping in mind during physical interpretation.
- Saved `U_raw.npy` and `U_orth.npy`.

### Step 6 — Diabatic Hamiltonian ✓ DONE

```python
H_diab = U_orth @ np.diag(E_stack) @ U_orth.T
```

Implemented in `diabatize.py`: `compute_H_diab()`.
Verified:
- `||H_diab - H_diab.T||_max = 3.8e-17`  (exact symmetry)  ✓
- `max|eigvalsh(H_diab) - E_stack_sorted| = 6.7e-16`  (machine precision)  ✓
- Saved `H_diab.npy` and `H_diab.dat` (eV, human-readable)

Selected on-site energies (eV): m=0: 9.04, m=1: 8.89, m=8: 8.49, m=10: 8.40.
Notable off-diagonal couplings (eV): H[1,8]=-0.833, H[1,10]=-0.615,
H[4,15]=0.556, H[9,10]=0.439, H[6,7]=-0.523.

### Step 7 — Output

Save `U_raw.npy`, `U_orth.npy`, `H_diab.npy`.
Write `H_diab.dat` (eV, human-readable).
Print diagonal (on-site energies) and key off-diagonal elements.

---

## Validation Checklist

| Check | Expected |
|---|---|
| `C_ref.shape` | `(16, 16, 80)` |
| `X_stack.shape` | `(16, 16, 80)` |
| `C_ref[:, :4, :]` all zero | True (frozen core rows) |
| `np.sum(C_ref[m,4:,:]**2)` for each $m$ | $\approx 1.0$ |
| `np.sum(X_stack[k,4:,:]**2)` for each $k$ | $\approx 1.0$ |
| Manual check: `np.linalg.det(S_MO[r[0,0,:],:][:,s[0,0,:]])` vs `Delta[0,0,0,0]` | Equal |
| `U_orth @ U_orth.T` | $\approx I_{16}$ |
| `np.sort(np.linalg.eigvalsh(H_diab))` | $\approx$ `np.sort(E_stack)` |
| `np.allclose(H_diab, H_diab.T)` | True |

---

## Constants

```python
N_BAS        = 96   # total MOs in basis
N_FRZ        = 4    # frozen core MOs
N_OCC        = 12   # active occupied MOs
N_VIRT       = 80   # virtual MOs
N_ROOTS      = 16   # TDDFT roots per geometry
OCC_OFFSET   = 4    # global index of local active occ 0
VIRT_OFFSET  = 16   # global index of local virt 0
```

---

## What NOT To Do

- **Do not** use $S_{\text{oo}}^{-T}$ in any formula — $S_{\text{oo}}$ is not symmetric
- **Do not** use $\tilde{U}^T\Lambda\tilde{U}$ — gives wrong (non-symmetric) $H^{\text{diab}}$
- **Do not** use $U^{-1}$ in place of $\tilde{U}^T$ — breaks symmetry of $H^{\text{diab}}$
- **Do not** forget the squaring $[\det(\cdot)]^2$ — comes from RKS closed-shell spin
- **Do not** confuse local active occ index $i\in[0,12)$ with global S_MO index $i+4\in[4,16)$
- **Do not** confuse local virtual index $a\in[0,80)$ with global S_MO index $a+16\in[16,96)$
- **Do not** include frozen core rows in the submatrix — only active occ and virt MOs
  enter $D^{ab}_{ij}$
- **Do not** approximate any block of $S^{\text{MO}}$ as zero — the exact formula
  uses the full submatrix determinant

---

## Current Status — Pipeline Complete

All 7 steps from the implementation plan are fully implemented, validated, and
generalised. The working directory contains the following outputs:

### Output Files

| File | Description |
|---|---|
| `diabatize.py` | Full pipeline, general (user edits `N_*` config block at top) |
| `plot_hmat.py` | Heatmap plotter for all three H matrices |
| `U_raw.npy` | Raw many-body overlap matrix U, shape (16,16) |
| `U_orth.npy` | Löwdin-orthogonalized U_orth, shape (16,16) |
| `H_diab.npy` | Diabatic Hamiltonian in Hartree, shape (16,16) |
| `H_diab.dat` | H_diab in eV, human-readable, multi-comment header |
| `H_diab_eV.dat` | H_diab in eV, SMO.dat-style (single header line) |
| `H_adiab_target_eV.dat` | diag(E_stack) in eV, SMO.dat-style |
| `H_adiab_ref_eV.dat` | diag(E_ref) in eV, SMO.dat-style |
| `hplots/H_diab_eV.{png,pdf}` | Heatmap of H_diab, colour scale clipped to off-diagonal range |
| `hplots/H_adiab_target_eV.{png,pdf}` | Heatmap of H_adiab_stack (diagonal) |
| `hplots/H_adiab_ref_eV.{png,pdf}` | Heatmap of H_adiab_inf (diagonal) |

### Validated Results

All non-negotiable validation criteria pass:
- `||H_diab - H_diab.T||_max` ≈ 1e-17 (exact symmetry)
- `max|eigvalsh(H_diab) - E_stack_sorted|` ≈ 7e-16 (machine precision eigenvalue recovery)
- `||U_orth @ U_orth.T - I||_max` ≈ 2e-15 (exact orthogonality)

Key numerical results (eV):

**On-site diabatic energies** H_diab[m,m]:
m=0: 9.04, m=1: 8.89, m=2: 10.31, m=3: 9.09, m=4: 9.91, m=5: 9.46,
m=6: 9.77, m=7: 9.51, m=8: 8.49, m=9: 9.42, m=10: 8.40, m=11: 9.22,
m=12: 9.43, m=13: 9.20, m=14: 8.67, m=15: 9.23

**Largest off-diagonal couplings**:
H[1,8] = −0.833 eV, H[1,10] = −0.615 eV, H[4,15] = +0.556 eV,
H[6,7] = −0.523 eV, H[9,10] = +0.439 eV

### Open Issues for Next Investigation

The following were observed but not yet resolved:

1. **Row/column norms of U are far below 1 for many states.**
   `||U[m,:]||` ranges from ~0 (rows 14–15) to ~0.70. This means several
   rinf diabatic states have very little overlap with the 16 rstack adiabats.
   Physical question: are rows 14–15 symmetry-forbidden transitions, very
   high-energy states outside the rstack window, or an artefact?

2. **Singular values of U drop sharply after index 5.**
   The first 6 singular values are 0.99–0.26; the remaining 10 drop to
   1.3e-9. Löwdin inflates all of these to 1, meaning the last 10 modes of
   U_orth are dominated by the SVD frame choice rather than physical overlap.
   The H_diab elements coupling states m=14,15 to others are therefore
   unreliable and should be treated with caution.

3. **State assignment not yet done.**
   The 16 rinf states have not been assigned to XT_A, XT_B, CT_1, CT_2
   characters by inspecting the dominant $C^{(m)}_{ia}$ amplitudes. The
   4×4 sub-block of H_diab in the {XT_A, XT_B, CT_1, CT_2} basis has
   not yet been extracted. This is the physically most important quantity
   (electronic coupling $J$ and charge-transfer coupling $t$).

4. **Individual H_diab matrix elements are not run-to-run reproducible**
   for states in degenerate SVD subspaces (e.g. H[9,11] differed between
   runs). This is a known SVD sign/frame ambiguity for degenerate singular
   values — the full matrix H_diab is invariant, but individual elements
   within a degenerate block can rotate. Affects interpretation of couplings
   involving states with near-identical rinf energies (e.g. the pairs
   m=4/5, m=6/7, m=9/10, m=12/13).

### Code Architecture (`diabatize.py`)

The pipeline is fully general: set `N_BAS`, `N_FRZ`, `N_OCC`, `N_VIRT`,
`N_ROOTS` and the three input filenames in the `USER CONFIGURATION` block at
the top, and the rest runs unchanged for any TDDFT/TDA system in ORCA format.

Functions (in order of the pipeline):
- `parse_vectors(path, n_frz, n_occ, n_virt, n_roots)` — parse amplitude .dat file
- `load_inputs(...)` — load S_MO.npy + both amplitude files, validate shapes
- `validate_inputs(C_ref, X_tgt, n_frz)` — frozen core check + norm check
- `build_index_arrays(n_occ, n_virt, occ_offset, virt_offset)` — build r, s
- `build_delta_tensor(S_MO, r, s, n_occ, n_virt)` — vectorised det tensor
- `compute_U(C_act, Delta, X_act)` — einsum contraction → U[m,k]
- `lowdin_orthogonalize(U)` — SVD → U_orth = V @ Wt
- `compute_H_diab(U_orth, E_tgt)` — U_orth @ diag(E) @ U_orth.T
- `write_matrix_dat(path, mat, header)` — SMO.dat-style export