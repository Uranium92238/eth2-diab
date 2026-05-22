# Handout: Quasi-Diabatization of Ethylene Dimer

## System

Eclipsed cofacial ethylene dimer (H-aggregate, D2h). CAM-B3LYP/def2-SVP, 16 TDDFT
roots, full TDDFT (no TDA), NoUseSym. Two geometries:
- **rinf**: 20 Å separation — states are pure XT or CT by construction
- **rstack**: 3.5 Å π-stacking — states are heavily mixed adiabats

Goal: extract the 4×4 diabatic Hamiltonian in the basis
{|XT_A⟩, |XT_B⟩, |CT₁⟩, |CT₂⟩} with parameters J, λ, t, ΔE_offset.

---

## What Has Been Done

**S_MO has been computed** (`S_MO.npy`, 96×96).

It is the MO overlap matrix between the rinf diabatic reference MOs (displaced to
rstack geometry) and the rstack adiabatic MOs:

$$S^{\text{MO}}_{pq} = \langle \tilde{\varphi}_p^{\text{rinf}} | \varphi_q^{\text{rstack}} \rangle$$

Validated. The frontier subblock shows the expected ±1/√2 Hadamard rotation between
monomer-localized rinf MOs and delocalized rstack dimer MOs.

---

## What Comes Next

**Step 1 — State overlaps → U**

Contract S_MO with the CI vectors (X amplitudes from `orca.cis`) at both geometries
to get the many-body state overlap matrix:

$$U_{mk} = \langle \Phi_m^{\text{rinf}} | \Psi_k^{\text{rstack}} \rangle = \sum_{ia,jb} X^{(m)}_{ia} X^{(k)}_{jb} \det(\mathbf{D}^{ab}_{ij})$$

where $\mathbf{D}^{ab}_{ij}$ is the occ-occ block of S_MO with row $i$ replaced by
row $a$ and column $j$ replaced by column $b$. Then Löwdin orthonormalize U.

**Step 2 — Diabatic Hamiltonian**

$$H^{\text{diab}} = \mathbf{U}^\dagger \mathbf{E} \mathbf{U}$$

where E is diagonal in the rstack adiabatic excitation energies. The off-diagonal
elements give λ (XT-CT coupling), J (excitonic coupling), and t (CT hopping).
