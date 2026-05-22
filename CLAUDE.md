# CLAUDE.md — Quasi-Diabatization Pipeline for Ethylene Dimer

## Project Goal

Compute the diabatic Hamiltonian $H^{\text{diab}} = U^\dagger E U$ for an eclipsed
cofacial ethylene dimer (H-aggregate), extracting the physical couplings $J$, $\lambda$,
$t$, and $\Delta E_{\text{offset}}$ from CAM-B3LYP/def2-SVP full TDDFT (16 roots).

---

## Physical Context

- **System**: eclipsed cofacial ethylene dimer (H-aggregate, D2h symmetry)
- **Method**: CAM-B3LYP/def2-SVP full TDDFT (16 roots, no TDA, NoUseSym)
- **rinf**: monomers at 20 Å separation — states are pure XT or pure CT by construction,
  used as the diabatic reference basis $\{|\Phi_m\rangle\}$
- **rstack**: monomers at 3.5 Å π-stacking — states are heavily mixed adiabats
  $\{|\Psi_k\rangle\}$
- **AO basis**: 96 functions total — AOs 1–48 on monomer A, AOs 49–96 on monomer B
- **Orbital window**: nfrozen=4 (C 1s core, MOs 1–4), nocc_active=12 (MOs 5–16), nvirt=80 (MOs 17–96) — all 1-based
- **In S_MO indexing (0-based)**: frozen occ = rows 0–3, active occ = rows 4–15, virt = rows 16–95
- **Frontier MOs** (1-based, rinf at 20 Å):
  - MO 15: π_B (HOMO on monomer B)
  - MO 16: π_A (HOMO on monomer A)
  - MO 17: π*_B (LUMO on monomer B)
  - MO 18: π*_A (LUMO on monomer A)

---

## Input Files

| File | Description |
|---|---|
| `displaced_rinf_at_rstack.molden` | rinf MO coefficients, monomer B positions moved to z=3.5 Å. MO coefficients frozen at rinf values. |
| `orca.molden` | True rstack MO coefficients (monomer B at z=3.5 Å). |
| `rinf.cis` | ORCA binary CIS file for rinf geometry (16 roots, 32 vectors = X+Y and X-Y pairs). |
| `rstack.cis` | ORCA binary CIS file for rstack geometry (same structure). |

---

## Theory

### MO overlap matrix

$$\mathbf{S}^{\text{MO}} = \mathbf{C}^{\text{rinf}} \, (\mathbf{C}^{\text{rstack}})^{-1}$$

where $C_{p\nu}$ is the MO coefficient matrix (MO index $p$, AO index $\nu$).

### Many-body state overlap

For single-excitation states in the CIS/TDDFT approximation:

$$U_{mk} = \langle \Phi_m^{\text{rinf}} | \Psi_k^{\text{rstack}} \rangle = \det(S_{\text{oo}}) \sum_{i,a,j,b} X^{(m)}_{ia} [S_{\text{oo}}^{-1}]_{ji} \, S_{\text{vv},ab} \, X^{(k)}_{jb}$$

where the blocks are taken over **active** orbitals only (frozen core excluded):

$$S_{\text{oo}} = S^{\text{MO}}[4{:}16,\, 4{:}16] \quad (12\times12)$$
$$S_{\text{vv}} = S^{\text{MO}}[16{:}96,\, 16{:}96] \quad (80\times80)$$

and $X^{(m)}_{ia}$ runs over active occ $i \in [0,12)$ and virt $a \in [0,80)$.

Compactly: $U_{mk} = \det(S_{\text{oo}}) \cdot \text{Tr}(X^{(m)} S_{\text{vv}} X^{(k)\top} S_{\text{oo}}^{-\top})$

### Diabatic Hamiltonian

After Löwdin orthonormalization $U \to U_{\text{orth}} = U (U^\top U)^{-1/2}$:

$$H^{\text{diab}} = U_{\text{orth}}^\dagger \, \mathbf{E} \, U_{\text{orth}}$$

where $\mathbf{E} = \text{diag}(E_1^{\text{rstack}}, \ldots, E_{16}^{\text{rstack}})$.

---

## Pipeline Status

### ✅ Step 1 — MO overlap matrix `S_MO.npy`

**Script**: `compute_smo.py`  
**Inputs**: `displaced_rinf_at_rstack.molden`, `orca.molden`  
**Outputs**: `S_MO.npy` (96×96), `SMO.dat` (human-readable)

Validated: identity test, row norms ≤ 1, frontier MO overlaps as expected.

### ✅ Step 2 — Parse CIS vectors

**Script**: `parse_cis.py`  
**Inputs**: `rinf.cis`, `rstack.cis`  
**Outputs**:
- `rinf_energies.dat`, `rstack_energies.dat` — excitation energies (Ha and eV)
- `rinf_vectorsX.dat`, `rstack_vectorsX.dat` — X amplitudes, one block per root
- `rinf_vectorsY.dat`, `rstack_vectorsY.dat` — Y amplitudes, one block per root

**Array shape**: `(nroots=16, nocc_full=16, nvirt=80)` — rows 0–3 are zero (frozen core),
active amplitudes in rows 4–15.

**CIS binary format** (little-endian):
- File header: `int32 nvec` + `int32[8] orbital_window` = 36 bytes
- Per vector: `int32[5]` (n, sym, mult, iblock, iroot) + 4-byte pad + `float64 en` +
  `bool8 is_transition` + 7-byte pad = 40 bytes header, then `float64[n]` amplitudes
- TDDFT: vectors come in pairs (X+Y, X-Y); X = ((X+Y)+(X-Y))/2, Y = ((X+Y)-(X-Y))/2
- Result: nvec=32, nroots=16, nocc_active=12, nfrozen=4, nvirt=80, n_amp=960

User-settable variables at top of script: `INPUT_FILES`, `NFROZEN`.

Validated: X norms ≈ 1.0, Y norms ≈ 0.1–0.2 (small as expected for full TDDFT), for all 16 roots at both geometries.

### 🔲 Step 3 — Many-body state overlap U and diabatic Hamiltonian

**In progress — experimental Fortran path being tested.**  
**Inputs**: `S_MO.npy`, `rinf_vectorsX.dat` / `rstack_vectorsX.dat`, `rstack_energies.dat`.  
**Outputs**: `U_raw.npy`, `U_orth.npy`, `H_diab.npy`, `H_diab.dat`.

**Key implementation notes for this step**:
- Load X amplitudes from `*_parsed.npz` (keys: `X`, `nocc_full`, `nvirt`, `nfrozen`)
- Extract active blocks from S_MO: `S_oo = S_MO[4:16, 4:16]`, `S_vv = S_MO[16:96, 16:96]`
- Use active X only: `X_active = X[:, 4:16, :]` (shape `(16, 12, 80)`)
- Compute U: `U[m,k] = det(S_oo) * trace(X_active[m] @ S_vv @ X_active[k].T @ S_oo_inv.T)`
- Löwdin orthonormalize: `U_orth = U @ (U.T @ U)^{-1/2}`
- Diabatic Hamiltonian: `H_diab = U_orth.T @ diag(E_rstack) @ U_orth`
- H_diab is **16×16**; truncation to 4×4 {XT_A, XT_B, CT₁, CT₂} is a post-processing step

#### Experimental: `diabatize.f90` (top-level)

A standalone Fortran 90 program (no dependencies, no LAPACK) implementing the
**exact** CI overlap formula via the matrix determinant lemma. **This is experimental
and has not yet been validated against reference data.** If it produces wrong results
we will fall back to implementing the exact Python formula directly.

**What it does:**
- Reads `smo.dat` (nbas×nbas plain text S_MO) directly — no AO/MO machinery
- Reads `ci_bra.dat` / `ci_ket.dat` as dense matrices: nroot rows × ndet cols
  (one row per state, one column per Slater determinant)
- Reads `slater.dat`: ndet rows × nbas cols of signed orbital indices
  (positive = alpha spin, negative = beta spin, 0 = empty; 1-based)
- Classifies each determinant relative to ground state (which occ → which virt)
- Evaluates each Slater determinant pair overlap via the matrix determinant lemma
  (Sherman-Morrison rank-1 update on the ground-state S_oo block and its inverse),
  which is the **exact** formula for arbitrary geometry displacement
- Accumulates `U_mk = sum_{i,j} C_bra(m,i) * S_det(i,j) * C_ket(k,j)`
- Outputs the nroot×nroot U matrix to stdout

**Input file formats:**
- `diabatize.input`: single line `nbas ncore ndiscarded nelec nroot`
- `smo.dat`: nbas rows, each with nbas space-separated doubles
- `ci_bra.dat` / `ci_ket.dat`: nroot rows × ndet cols, free format
- `slater.dat`: ndet rows × nbas cols, signed integer orbital indices

**Compile:**
```
gfortran -O2 -o diabatize diabatize.f90
```

**Known issues / things to verify before trusting output:**
- `classify_det` parity calculation assumes a specific canonical ordering of
  orbital indices in the Slater file — must match how `slater.dat` is generated
- `smo_col_col` helper currently indexes `smo_cols` incorrectly — needs to be
  `smo(occ(i), virt)` not `smo_cols(i, virt)`; fix before running
- The `col_d` vector in the excited|excited branch uses `smo_cols(ket_virt,:)`
  as a proxy for `base(:,q)` which may be wrong depending on storage layout

---

## Implementation Notes

- **Language**: Python 3, numpy only.
- **MO indexing**: 0-based throughout. Molden files use 1-based AO indices (converted on parse).
- **X amplitudes**: `X[root, i, a]` where `i` = occ index (0-based over nocc_full=16),
  `a` = virt index (0-based over nvirt=80). Flat binary ordering is occ-outer, virt-inner.
  Frozen core rows (i=0..3) are zero; active amplitudes in rows i=4..15.
- **S_MO indexing**: row/col 0–3 = frozen occ, 4–15 = active occ, 16–95 = virt (all 0-based).
  Active blocks: `S_oo = S_MO[4:16, 4:16]`, `S_vv = S_MO[16:96, 16:96]`.
- **The Hamiltonian is 16×16**. Truncation to 4×4 is a post-processing step.
- **npz files** (`rinf_parsed.npz`, `rstack_parsed.npz`) store: `X`, `Y`, `energies`,
  `nocc_active`, `nocc_full`, `nvirt`, `nfrozen`, `orb_window`.
