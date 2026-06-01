import numpy as np

# --- User settings ---
RINF_MOLDEN  = "orca.molden"
RSTACK_MOLDEN = "orca.molden"
N_MO = 96          # number of MOs (= number of AOs for a square C matrix)
# ---------------------

N_AO = N_MO


def parse_molden(path):
    """Parse [MO] block from an ORCA molden file into a (N_AO x N_AO) C matrix.

    C[p, nu] = coefficient of AO nu (0-based) in MO p (0-based).
    """
    with open(path) as f:
        lines = f.readlines()

    # Find start of [MO] section
    mo_start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "[mo]":
            mo_start = i + 1
            break
    if mo_start is None:
        raise ValueError(f"No [MO] section found in {path}")

    C = np.zeros((N_AO, N_AO))
    mo_idx = -1
    in_mo = False

    for line in lines[mo_start:]:
        stripped = line.strip()
        if stripped.startswith("Sym="):
            mo_idx += 1
            in_mo = True
            if mo_idx >= N_AO:
                raise ValueError(f"More than {N_AO} MO blocks found in {path}")
        elif stripped.startswith(("Ene=", "Spin=", "Occup=")):
            continue
        elif in_mo and stripped:
            parts = stripped.split()
            if len(parts) == 2:
                try:
                    ao_idx = int(parts[0]) - 1  # convert to 0-based
                    coeff = float(parts[1])
                    C[mo_idx, ao_idx] = coeff
                except ValueError:
                    pass

    n_mo = mo_idx + 1
    if n_mo != N_AO:
        raise ValueError(f"Expected {N_AO} MO blocks, found {n_mo} in {path}")

    return C


def validate(C, label):
    print(f"\n--- {label} ---")
    print(f"  Shape: {C.shape}")
    print(f"  Non-zero elements: {np.count_nonzero(C)}")

    # Check orthonormality: C @ C.T should be close to identity if S_AO ~ (C.T @ C)^-1
    # Instead check row norms as a basic sanity check
    row_norms = np.linalg.norm(C, axis=1)
    print(f"  Row norms: min={row_norms.min():.4f}, max={row_norms.max():.4f}, mean={row_norms.mean():.4f}")

    # Check condition number
    cond = np.linalg.cond(C)
    print(f"  Condition number: {cond:.3e}")

    # Self-overlap: C @ inv(C) should be identity
    S_self = C @ np.linalg.inv(C)
    identity_err = np.max(np.abs(S_self - np.eye(N_AO)))
    print(f"  Self-overlap max deviation from identity: {identity_err:.3e}")


if __name__ == "__main__":
    rinf_path = RINF_MOLDEN
    rstack_path = RSTACK_MOLDEN

    print("Parsing molden files...")
    C_rinf = parse_molden(rinf_path)
    print(f"  C_rinf parsed: {C_rinf.shape}, from {rinf_path}")

    C_rstack = parse_molden(rstack_path)
    print(f"  C_rstack parsed: {C_rstack.shape}, from {rstack_path}")

    validate(C_rinf, "C_rinf (displaced_rinf_at_rstack)")
    validate(C_rstack, "C_rstack (orca / rstack)")

    # --- Step 2: compute S_MO ---
    print("\n--- Step 2: S_MO = C_rinf @ inv(C_rstack) ---")
    S_MO = C_rinf @ np.linalg.inv(C_rstack)
    np.save("S_MO.npy", S_MO)
    print(f"  Saved S_MO.npy, shape {S_MO.shape}")

    with open("SMO.dat", "w") as f:
        f.write(f"# S_MO matrix ({N_MO}x{N_MO})  rows=rinf MOs, cols=rstack MOs  (0-based indexing)\n")
        for row in S_MO:
            f.write("  ".join(f"{v:12.8f}" for v in row) + "\n")
    print("  Saved SMO.dat")

    # Validation 1: identity test (using rstack against itself)
    S_self = C_rstack @ np.linalg.inv(C_rstack)
    id_err = np.max(np.abs(S_self - np.eye(N_MO)))
    print(f"  Identity test (C_rstack @ inv(C_rstack)): max |err| = {id_err:.3e}")

    # Validation 2: row norms of S_MO should be <= 1
    row_norms = np.linalg.norm(S_MO, axis=1)
    print(f"  S_MO row norms: min={row_norms.min():.4f}, max={row_norms.max():.4f}")
    n_violated = np.sum(row_norms > 1.0)
    if n_violated:
        print(f"  WARNING: {n_violated} rows have norm > 1")
    else:
        print(f"  All row norms <= 1 — OK")

    # Validation 3: CT rows should be near zero
    print("\n--- Frontier MO overlaps (S_MO rows, max |S_MO[p,:]|) ---")
    for p, label in [(12, "MO13 CT"), (13, "MO14 CT"),
                     (14, "MO15 pi_B HOMO"), (15, "MO16 pi_A HOMO"),
                     (16, "MO17 pi*_B LUMO"), (17, "MO18 pi*_A LUMO")]:
        row = S_MO[p]
        print(f"  Row {p:2d} ({label}): max|S|={np.max(np.abs(row)):.4f}  norm={np.linalg.norm(row):.4f}")
