# Quasi-Diabatization Results — Ethylene Dimer

**System**: eclipsed cofacial ethylene dimer (H-aggregate, D2h symmetry)  
**Method**: CAM-B3LYP/def2-SVP, full TDDFT (no TDA), NoUseSym, ORCA 6.0.1  
**Pipeline**: `diabatize.py` — many-body overlap diabatization via Slater determinant overlaps

---

## Table of Contents

1. [Physical System](#1-physical-system)
2. [Notation](#2-notation)
3. [Theory](#3-theory)
   - 3.1 [MO Overlap Matrix](#31-mo-overlap-matrix)
   - 3.2 [TDA Amplitudes](#32-tda-amplitudes)
   - 3.3 [Many-Body States](#33-many-body-states)
   - 3.4 [Slater Determinant Overlap](#34-slater-determinant-overlap)
   - 3.5 [Orbital Overlap Submatrix](#35-orbital-overlap-submatrix)
   - 3.6 [Exact Many-Body Overlap U](#36-exact-many-body-overlap-u)
   - 3.7 [Computational Strategy](#37-computational-strategy)
   - 3.8 [Löwdin Orthonormalization](#38-löwdin-orthonormalization)
   - 3.9 [Diabatic Hamiltonian](#39-diabatic-hamiltonian)
4. [Pipeline Summary](#4-pipeline-summary)
5. [Validation](#5-validation)
6. [Input Energies](#6-input-energies)
7. [Many-Body Overlap U](#7-many-body-overlap-u)
8. [Diabatic Hamiltonian H_diab](#8-diabatic-hamiltonian-h_diab)

---

## 1. Physical System

The system is an eclipsed cofacial ethylene dimer (H-aggregate) at D2h symmetry, computed with CAM-B3LYP/def2-SVP using full TDDFT (no Tamm-Dancoff approximation), NoUseSym keyword, in ORCA 6.0.1.

The AO basis has 96 functions total (48 per monomer). The orbital window is:

| Global index (0-based) | Role |
|---|---|
| 0 – 3 | Frozen core (C 1s), excluded from CI |
| 4 – 15 | Active occupied (12 MOs) |
| 16 – 95 | Virtual (80 MOs) |

Two geometries are used:

- **rinf** (20 Å separation): monomers are non-interacting. TDDFT states are pure excitonic (XT) or charge-transfer (CT) by construction and serve as the **diabatic reference basis**.
- **rstack** (3.5 Å π-stacking): states are heavily mixed adiabats whose Hamiltonian is diagonal by construction.

---

## 2. Notation

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
| $C^{(m)}_{ia}$ | TDA X amplitude for rinf state $m$: excitation occ $i \to$ virt $a$ |
| $X^{(k)}_{jb}$ | TDA X amplitude for rstack state $k$: excitation occ $j \to$ virt $b$ |
| $S^{\mathrm{MO}}_{pq}$ | MO overlap $\langle\varphi^{\mathrm{ref}}_p\vert\varphi^{\mathrm{stack}}_q\rangle$, row = ref, col = stack |

All indices are **0-based** throughout.

---

## 3. Theory

### 3.1 MO Overlap Matrix

$$S^{\mathrm{MO}}_{pq} = \langle\varphi^{\mathrm{ref}}_p \vert \varphi^{\mathrm{stack}}_q\rangle$$

- Row index $p$ (0-based, 0..95): **ref** (rinf) MO
- Column index $q$ (0-based, 0..95): **stack** (rstack) MO
- Shape: $96\times96$, loaded from `S_MO.npy`
- **Not symmetric**: cross-geometry overlaps have no reason to be symmetric
- Validated: frontier MO subblock shows the expected $\pm 1/\sqrt{2}$ Hadamard structure

---

### 3.2 TDA Amplitudes

$C^{(m)}_{ia}$ from `rinf_vectorsX.dat`: amplitude for rinf state $m$ to excite from active occupied MO $i$ to virtual MO $a$. These are the X amplitudes extracted from full TDDFT; the Y amplitudes are neglected, which is justified since $\lVert Y\rVert/\lVert X\rVert \approx 0.1$–$0.2$ for these XT and CT states.

$X^{(k)}_{jb}$ from `rstack_vectorsX.dat`: same meaning for rstack state $k$.

After parsing:
- `C_ref` has shape $(16, 16, 80)$ — full array including frozen zero rows
- `X_stack` has shape $(16, 16, 80)$ — same
- Active slices used in computation: `C_ref[:, 4:16, :]` and `X_stack[:, 4:16, :]`, both shape $(16, 12, 80)$

---

### 3.3 Many-Body States

The diabatic reference states are CIS-like expansions over the rinf MO basis:

$$\vert\Phi_m\rangle = \sum_{i=0}^{11}\sum_{a=0}^{79} C^{(m)}_{ia}\,\vert\Phi^a_i\rangle^{\mathrm{ref}}$$

where $\vert\Phi^a_i\rangle^{\mathrm{ref}}$ is the closed-shell Slater determinant formed from the rinf reference ground state by replacing active occupied spatial MO $i$ with virtual MO $a$.

The adiabatic stack states:

$$\vert\Psi_k\rangle = \sum_{j=0}^{11}\sum_{b=0}^{79} X^{(k)}_{jb}\,\vert\Psi^b_j\rangle^{\mathrm{stack}}$$

These satisfy $H^{\mathrm{stack}}\vert\Psi_k\rangle = E^{\mathrm{stack}}_k\vert\Psi_k\rangle$ and $\langle\Psi_k\vert\Psi_l\rangle = \delta_{kl}$.

---

### 3.4 Slater Determinant Overlap

The overlap between any two Slater determinants built from $N$ orbitals each is the determinant of the $N\times N$ matrix of pairwise orbital overlaps.

**Derivation sketch**: write both determinants via the Leibniz expansion. The many-electron overlap integral factorizes over electrons (each electron coordinate appears in exactly one factor). Relabelling the double sum over permutations $(\sigma, \tau)$ via $\pi = \tau\circ\sigma^{-1}$ eliminates $\sigma$ entirely (it contributes $N!$ identical copies), leaving the Leibniz definition of the determinant. This does **not** reduce to a product of individual overlaps unless the orbital overlap matrix is diagonal — which it is not across two different geometries.

For a restricted Kohn-Sham (RKS) determinant, each spatial orbital is doubly occupied (alpha + beta). The spin-orbital overlap matrix is block-diagonal in spin, so

$$\det\!\bigl(\mathrm{block\text{-}diag}(A, A)\bigr) = \bigl[\det(A)\bigr]^2$$

This squaring is applied when computing the many-body overlap below.

---

### 3.5 Orbital Overlap Submatrix

$\vert\Phi^a_i\rangle^{\mathrm{ref}}$ occupies $N_o = 12$ ref spatial orbitals: all active occupied except $i$, with virtual $a$ placed at position $i$.

$\vert\Psi^b_j\rangle^{\mathrm{stack}}$ occupies $N_o = 12$ stack spatial orbitals: all active occupied except $j$, with virtual $b$ placed at position $j$.

Their overlap is:

$$\langle\Phi^a_i\vert\Psi^b_j\rangle = \bigl[\det(D^{ab}_{ij})\bigr]^2$$

$D^{ab}_{ij}$ is the $12\times12$ matrix of spatial orbital overlaps:

$$D^{ab}_{ij}[p,q] = S^{\mathrm{MO}}_{r_p,\,s_q}$$

where the **global** index maps (0-based) are:

$$r_p = \begin{cases} p + 4 & p \neq i \\ a + 16 & p = i \end{cases}
\qquad
s_q = \begin{cases} q + 4 & q \neq j \\ b + 16 & q = j \end{cases}
\qquad p,q\in\{0,\ldots,11\}$$

$D^{ab}_{ij}$ is exactly the submatrix of $S^{\mathrm{MO}}$ with rows $\mathbf{r}^{(ia)}$ and columns $\mathbf{s}^{(jb)}$ — **no approximation**. All four blocks of $S^{\mathrm{MO}}$ contribute:

| Position in $D^{ab}_{ij}$ | Block of $S^{\mathrm{MO}}$ | Physical meaning |
|---|---|---|
| $p\neq i,\; q\neq j$ | $S_{\mathrm{oo}}[p,q]$ | ref occ vs stack occ |
| $p=i,\; q\neq j$ | $S_{\mathrm{vo}}[a,q]$ | ref virt vs stack occ |
| $p\neq i,\; q=j$ | $S_{\mathrm{ov}}[p,b]$ | ref occ vs stack virt |
| $p=i,\; q=j$ | $S_{\mathrm{vv}}[a,b]$ | ref virt vs stack virt |

---

### 3.6 Exact Many-Body Overlap U

$$U_{mk} = \langle\Phi_m\vert\Psi_k\rangle
= \sum_{i,a}\sum_{j,b} C^{(m)}_{ia}\,X^{(k)}_{jb}\,
\Bigl[\det\!\Bigl(S^{\mathrm{MO}}\!\bigl[\mathbf{r}^{(ia)},\,\mathbf{s}^{(jb)}\bigr]\Bigr)\Bigr]^2$$

This is exact. No approximation is made about any block of $S^{\mathrm{MO}}$ being small.

---

### 3.7 Computational Strategy

The determinant $\det(S^{\mathrm{MO}}[\mathbf{r}^{(ia)}, \mathbf{s}^{(jb)}])$ depends only on the orbital pair $(i,a,j,b)$, not on which roots $m, k$ are being computed. Therefore:

**Step A — precompute once:**

$$\Delta_{iajb} = \det\!\Bigl(S^{\mathrm{MO}}\!\bigl[\mathbf{r}^{(ia)},\,\mathbf{s}^{(jb)}\bigr]\Bigr)$$

Shape of $\Delta$: $(N_o, N_v, N_o, N_v) = (12, 80, 12, 80)$.  
Number of $12\times12$ determinants evaluated: $12^2\times80^2 = 921{,}600$.

All submatrices are extracted simultaneously via numpy advanced indexing (shapes broadcast to $(12, 80, 12, 80, 12, 12)$) and `np.linalg.det` is called once over the batch.  Runtime: ~6 s on a single CPU core.

**Step B — contract over roots:**

$$U_{mk} = \sum_{i,a,j,b} C^{(m)}_{ia}\cdot\Delta^2_{iajb}\cdot X^{(k)}_{jb}$$

Implemented as a single einsum: `np.einsum('mia,iajb,kjb->mk', C_act, Delta**2, X_act)`.

---

### 3.8 Löwdin Orthonormalization

The raw $U$ is not orthogonal. The 16 adiabatic states do not span the full Hilbert space — completeness requires infinitely many states. Therefore:

$$\sum_{k=0}^{15} U_{mk}U_{nk} \neq \delta_{mn}$$

Using non-orthogonal $U$ directly in $H^{\mathrm{diab}} = U\Lambda U^T$ gives a congruence transformation: eigenvalues are not preserved and the result is not symmetric. Using $U^{-1}$ in place of $U^T$ preserves eigenvalues but breaks symmetry (and amplifies near-zero singular values catastrophically). Both are wrong.

Löwdin orthonormalization gives the orthogonal matrix **closest to $U$ in Frobenius norm**. Compute via SVD:

$$U = V\Sigma W^T \implies \tilde{U} = VW^T$$

This satisfies $\tilde{U}\tilde{U}^T = I$ exactly, regardless of the conditioning of $\Sigma$.

---

### 3.9 Diabatic Hamiltonian

Expand each diabatic state in the orthonormalized adiabatic basis:

$$\vert\Phi_m\rangle \approx \sum_k \tilde{U}_{mk}\vert\Psi_k\rangle$$

Take matrix elements using $\langle\Psi_k\vert H^{\mathrm{stack}}\vert\Psi_l\rangle = E^{\mathrm{stack}}_k\delta_{kl}$:

$$H^{\mathrm{diab}}_{mn} = \langle\Phi_m\vert H\vert\Phi_n\rangle = \sum_k \tilde{U}_{mk}\,E^{\mathrm{stack}}_k\,\tilde{U}_{nk}$$

$$\boxed{H^{\mathrm{diab}} = \tilde{U}\,\mathrm{diag}(E^{\mathrm{stack}})\,\tilde{U}^T}$$

$H^{\mathrm{diab}}$ is $16\times16$, real symmetric, and has eigenvalues $\{E^{\mathrm{stack}}_k\}$ — these are non-negotiable validation criteria.

---

## 4. Pipeline Summary

| Step | Description | Status |
|------|-------------|--------|
| 1 | Load `S_MO.npy`, parse TDA X amplitudes at rinf and rstack | ✓ |
| 2 | Build global index arrays $r$, $s$ for each single-excitation determinant | ✓ |
| 3 | Precompute determinant tensor $\Delta_{iajb}$ (921,600 12×12 determinants) | ✓ |
| 4 | Contract to many-body overlap $U_{mk} = \langle\Phi_m\vert\Psi_k\rangle$ | ✓ |
| 5 | Löwdin orthonormalization: $\tilde{U} = VW^T$ via SVD | ✓ |
| 6 | $H^{\mathrm{diab}} = \tilde{U}\,\mathrm{diag}(E^{\mathrm{stack}})\,\tilde{U}^T$ | ✓ |

---

## 5. Validation

| Check | Result | Expected |
|-------|--------|----------|
| $\lVert\tilde{U}\tilde{U}^T - I\rVert_{\max}$ | 1.14 × 10⁻¹¹ | ~10⁻¹⁵ |
| $\lVert H^{\mathrm{diab}} - (H^{\mathrm{diab}})^T\rVert_{\max}$ | 0.00 | ~10⁻¹⁶ |
| $\max\lvert\mathrm{eigvalsh}(H^{\mathrm{diab}}) - E^{\mathrm{stack}}_{\mathrm{sorted}}\rvert$ | 8.71 × 10⁻¹² Ha | ~10⁻¹⁴ |

---

## 6. Input Energies

### Adiabatic reference energies $E^{\mathrm{ref}}_m$ (rinf, eV)

| $m$ | $E^{\mathrm{ref}}_m$ (eV) |
|-----|--------------------------|
| 0 | 8.144368 |
| 1 | 8.369852 |
| 2 | 8.771167 |
| 3 | 9.205626 |
| 4 | 9.656354 |
| 5 | 9.960408 |
| 6 | 10.288148 |
| 7 | 11.219896 |
| 8 | 9.656354 |
| 9 | 9.656365 |
| 10 | 9.960408 |
| 11 | 9.960409 |
| 12 | 10.288148 |
| 13 | 10.288149 |
| 14 | 11.219896 |
| 15 | 11.219922 |

### Adiabatic target energies $E^{\mathrm{stack}}_k$ (rstack, eV)

| $k$ | $E^{\mathrm{stack}}_k$ (eV) |
|-----|----------------------------|
| 0 | 6.912739 |
| 1 | 8.248188 |
| 2 | 8.317886 |
| 3 | 8.714299 |
| 4 | 8.955300 |
| 5 | 9.394608 |
| 6 | 9.630543 |
| 7 | 10.255212 |
| 8 | 8.955300 |
| 9 | 9.141541 |
| 10 | 9.394608 |
| 11 | 9.609725 |
| 12 | 9.630543 |
| 13 | 9.938174 |
| 14 | 10.255212 |
| 15 | 10.679166 |

---

## 7. Many-Body Overlap U

### Raw overlap $U_{\mathrm{raw}}$

$$U_{mk} = \langle\Phi_m\vert\Psi_k\rangle = \sum_{i,a,j,b} C^{(m)}_{ia}\,\Delta^2_{iajb}\,X^{(k)}_{jb}$$

![U_raw](Assets/U_raw.png)

#### Row norms $\lVert U[m,:]\rVert$

| $m$ | $\lVert U[m,:]\rVert$ |
|-----|----------------------|
| 0 | 4.44 × 10⁻³ |
| 1 | 6.96 × 10⁻¹ |
| 2 | 5.16 × 10⁻¹ |
| 3 | 1.59 × 10⁻² |
| 4 | 5.65 × 10⁻¹ |
| 5 | 1.29 × 10⁻² |
| 6 | 2.04 × 10⁻¹ |
| 7 | 1.61 × 10⁻¹ |
| 8 | 1.09 × 10⁻¹ |
| 9 | 6.41 × 10⁻¹ |
| 10 | 2.46 × 10⁻¹ |
| 11 | 2.67 × 10⁻¹ |
| 12 | 4.95 × 10⁻¹ |
| 13 | 4.95 × 10⁻¹ |
| 14 | 7.42 × 10⁻⁸ |
| 15 | 8.78 × 10⁻⁶ |

#### Singular values of $U_{\mathrm{raw}}$

| $i$ | $\sigma_i$ |
|-----|-----------|
| 0 | 9.878 × 10⁻¹ |
| 1 | 6.875 × 10⁻¹ |
| 2 | 5.649 × 10⁻¹ |
| 3 | 5.186 × 10⁻¹ |
| 4 | 2.867 × 10⁻¹ |
| 5 | 2.561 × 10⁻¹ |
| 6 | 1.774 × 10⁻³ |
| 7 | 6.283 × 10⁻⁴ |
| 8 | 1.490 × 10⁻⁴ |
| 9 | 1.489 × 10⁻⁴ |
| 10 | 3.948 × 10⁻⁵ |
| 11 | 6.470 × 10⁻⁶ |
| 12 | 4.808 × 10⁻⁶ |
| 13 | 2.201 × 10⁻⁶ |
| 14 | 1.099 × 10⁻⁷ |
| 15 | 1.338 × 10⁻⁹ |

The first 6 singular values ($\sigma_0$–$\sigma_5$, range 0.256–0.988) correspond to states with meaningful physical overlap between the rinf diabatic and rstack adiabatic windows. The remaining 10 ($\sigma_6$–$\sigma_{15}$, range $10^{-9}$–$10^{-3}$) are effectively zero, meaning those diabatic states have negligible projection onto the 16 rstack adiabats. Löwdin orthonormalization inflates all singular values to 1; the resulting $\tilde{U}$ rows for the near-zero modes are dominated by the arbitrary SVD frame rather than physical overlap.

### Löwdin-orthogonalized $\tilde{U}$

$$\tilde{U} = VW^T \quad \text{where} \quad U = V\Sigma W^T$$

![U_orth](Assets/U_orth.png)

---

## 8. Diabatic Hamiltonian $H^{\mathrm{diab}}$

$$H^{\mathrm{diab}} = \tilde{U}\,\mathrm{diag}(E^{\mathrm{stack}})\,\tilde{U}^T$$

![H_diab](Assets/H_diab_eV.png)

### Adiabatic reference Hamiltonian $H^{\mathrm{adiab}}_{\mathrm{ref}}$ (rinf)

Diagonal by construction: at the reference geometry the states are non-interacting, so $H^{\mathrm{adiab}}_{\mathrm{ref}} = \mathrm{diag}(E^{\mathrm{ref}})$.

![H_adiab_ref](Assets/H_adiab_ref_eV.png)

### Adiabatic target Hamiltonian $H^{\mathrm{adiab}}_{\mathrm{stack}}$ (rstack)

Diagonal by construction: $\langle\Psi_k\vert H^{\mathrm{stack}}\vert\Psi_l\rangle = E^{\mathrm{stack}}_k\delta_{kl}$.

![H_adiab_stack](Assets/H_adiab_target_eV.png)

### On-site diabatic energies $H^{\mathrm{diab}}_{mm}$

| $m$ | $H^{\mathrm{diab}}_{mm}$ (eV) |
|-----|------------------------------|
| 0 | 9.0405 |
| 1 | 8.8869 |
| 2 | 10.3060 |
| 3 | 9.0911 |
| 4 | 9.9098 |
| 5 | 9.4632 |
| 6 | 9.7729 |
| 7 | 9.5098 |
| 8 | 8.4924 |
| 9 | 9.4205 |
| 10 | 8.3981 |
| 11 | 9.2208 |
| 12 | 9.4269 |
| 13 | 9.1976 |
| 14 | 8.6708 |
| 15 | 9.2258 |

### Full $H^{\mathrm{diab}}$ matrix (eV)

| $m$\\$n$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|----------|---|---|---|---|---|---|---|---|---|---|----|----|----|----|----|----|
| **0** | 9.0405 | 0.0770 | −0.0001 | 0.0000 | −0.0007 | −0.0003 | 0.0000 | −0.0001 | −0.0746 | −0.0875 | 0.1325 | −0.0410 | 0.0608 | 0.2668 | −0.0006 | 0.0026 |
| **1** | 0.0770 | 8.8869 | 0.0007 | −0.0006 | 0.0060 | 0.0028 | −0.0004 | 0.0008 | −0.8329 | 0.2200 | −0.6155 | −0.0544 | −0.4914 | −0.5010 | −0.0033 | −0.0049 |
| **2** | −0.0001 | 0.0007 | 10.3060 | 0.7560 | 0.0001 | 0.0002 | 0.0453 | −0.0887 | −0.0000 | 0.0001 | −0.0002 | 0.0001 | −0.0017 | 0.0008 | 0.2079 | 0.0158 |
| **3** | 0.0000 | −0.0006 | 0.7560 | 9.0911 | −0.0047 | −0.0023 | −0.2008 | 0.1985 | −0.0001 | −0.0001 | 0.0002 | −0.0001 | 0.0016 | −0.0008 | −0.2081 | −0.0110 |
| **4** | −0.0007 | 0.0060 | 0.0001 | −0.0047 | 9.9098 | −0.3208 | 0.0021 | 0.0026 | 0.0006 | 0.0007 | −0.0014 | 0.0010 | −0.0028 | −0.0057 | −0.0250 | 0.5556 |
| **5** | −0.0003 | 0.0028 | 0.0002 | −0.0023 | −0.3208 | 9.4632 | 0.0009 | 0.0013 | 0.0003 | 0.0003 | −0.0007 | 0.0005 | −0.0013 | −0.0027 | −0.0120 | 0.2607 |
| **6** | 0.0000 | −0.0004 | 0.0453 | −0.2008 | 0.0021 | 0.0009 | 9.7729 | −0.5227 | 0.0000 | −0.0001 | 0.0001 | −0.0000 | 0.0010 | −0.0004 | −0.1138 | −0.0064 |
| **7** | −0.0001 | 0.0008 | −0.0887 | 0.1985 | 0.0026 | 0.0013 | −0.5227 | 9.5098 | 0.0000 | 0.0001 | −0.0002 | 0.0001 | −0.0021 | 0.0010 | 0.2577 | 0.0140 |
| **8** | −0.0746 | −0.8329 | −0.0000 | −0.0001 | 0.0006 | 0.0003 | 0.0000 | 0.0000 | 8.4924 | 0.0856 | −0.4593 | −0.1240 | −0.3104 | −0.1598 | 0.0001 | 0.0055 |
| **9** | −0.0875 | 0.2200 | 0.0001 | −0.0001 | 0.0007 | 0.0003 | −0.0001 | 0.0001 | 0.0856 | 9.4205 | 0.4388 | −0.2599 | −0.1294 | 0.2128 | −0.0024 | 0.0002 |
| **10** | 0.1325 | −0.6155 | −0.0002 | 0.0002 | −0.0014 | −0.0007 | 0.0001 | −0.0002 | −0.4593 | 0.4388 | 8.3981 | 0.3284 | 0.0487 | −0.5262 | 0.0040 | −0.0009 |
| **11** | −0.0410 | −0.0544 | 0.0001 | −0.0001 | 0.0010 | 0.0005 | −0.0000 | 0.0001 | −0.1240 | −0.2599 | 0.3284 | 9.2208 | −0.2155 | 0.2460 | −0.0026 | 0.0032 |
| **12** | 0.0608 | −0.4914 | −0.0017 | 0.0016 | −0.0028 | −0.0013 | 0.0010 | −0.0021 | −0.3104 | −0.1294 | 0.0487 | −0.2155 | 9.4269 | −0.0048 | 0.0079 | 0.0077 |
| **13** | 0.2668 | −0.5010 | 0.0008 | −0.0008 | −0.0057 | −0.0027 | −0.0004 | 0.0010 | −0.1598 | 0.2128 | −0.5262 | 0.2460 | −0.0048 | 9.1976 | −0.0011 | 0.0035 |
| **14** | −0.0006 | −0.0033 | 0.2079 | −0.2081 | −0.0250 | −0.0120 | −0.1138 | 0.2577 | 0.0001 | −0.0024 | 0.0040 | −0.0026 | 0.0079 | −0.0011 | 8.6708 | −0.0278 |
| **15** | 0.0026 | −0.0049 | 0.0158 | −0.0110 | 0.5556 | 0.2607 | −0.0064 | 0.0140 | 0.0055 | 0.0002 | −0.0009 | 0.0032 | 0.0077 | 0.0035 | −0.0278 | 9.2258 |

### Largest off-diagonal couplings $H^{\mathrm{diab}}_{mn}$

| $(m, n)$ | $H^{\mathrm{diab}}_{mn}$ (eV) |
|----------|------------------------------|
| (2, 3) | +0.7560 |
| (1, 8) | −0.8329 |
| (1, 10) | −0.6155 |
| (4, 15) | +0.5556 |
| (6, 7) | −0.5227 |
| (1, 13) | −0.5010 |
| (1, 12) | −0.4914 |
| (10, 13) | −0.5262 |
| (9, 10) | +0.4388 |
