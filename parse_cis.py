"""
Parse ORCA .cis binary files and write CI vectors to human-readable text.

Binary layout (little-endian):
  File header: int32 nvec, int32[8] orbital_window  (36 bytes total)
  Per vector:  int32 n, int32 sym, int32 mult, int32 iblock, int32 iroot,
               [4 bytes padding], float64 en, bool8 is_transition,
               [7 bytes padding]  -> 40 bytes header
               float64[n] amplitudes

For TDDFT (non-TDA) vectors come in pairs: X+Y then X-Y.
X = ((X+Y) + (X-Y)) / 2,  Y = ((X+Y) - (X-Y)) / 2.

The CIS window covers only active (non-frozen) orbitals. With NFROZEN frozen
core orbitals, the amplitude arrays are padded with zeros so that occ indices
run over all NOCC_FULL = NFROZEN + nocc_active occupied MOs.

Output files:
  <stem>_energies.dat   -- root index, energy in Ha and eV
  <stem>_vectorsX.dat   -- X amplitudes, shape (nroots, NOCC_FULL, nvirt)
  <stem>_vectorsY.dat   -- Y amplitudes, same shape
  Rows = all occ MOs (frozen rows are zero); cols = virt MOs.
"""

import struct
import numpy as np
from pathlib import Path

# --- User settings ---
INPUT_FILES = [
    'rinf.cis',
    'rstack.cis',
]
NFROZEN = 4   # number of frozen core orbitals excluded from the CIS window
# ---------------------

HEADER_SIZE = 40   # sizeof(Orca_Cis_Vec_Header) with alignment padding
HA_TO_EV = 27.2114


def parse_cis(path):
    path = Path(path)
    data = path.read_bytes()

    offset = 0
    nvec = struct.unpack_from('<i', data, offset)[0]; offset += 4
    orb_win = struct.unpack_from('<8i', data, offset); offset += 32

    nocc_a  = orb_win[1] - orb_win[0] + 1
    nvirt_a = orb_win[3] - orb_win[2] + 1
    n_amp   = nocc_a * nvirt_a

    assert nvec % 2 == 0, f"Expected even nvec for TDDFT, got {nvec}"
    nroots = nvec // 2

    print(f"{path.name}: nvec={nvec} ({nroots} roots)  nocc={nocc_a} nvirt={nvirt_a}  n_amp={n_amp}")

    nocc_full = NFROZEN + nocc_a   # total occ MOs including frozen

    energies = np.zeros(nroots)
    XpY = np.zeros((nroots, nocc_a, nvirt_a))   # X+Y
    XmY = np.zeros((nroots, nocc_a, nvirt_a))   # X-Y

    for ivec in range(nvec):
        n_elem, sym, mult, iblock, iroot = struct.unpack_from('<5i', data, offset)
        en = struct.unpack_from('<d', data, offset + 24)[0]
        offset += HEADER_SIZE

        assert n_elem == n_amp, f"vec {ivec}: expected {n_amp} amplitudes, got {n_elem}"

        amps = np.frombuffer(data, dtype='<f8', count=n_elem, offset=offset).copy()
        offset += n_elem * 8

        root = ivec // 2
        if ivec % 2 == 0:
            energies[root] = en
            XpY[root] = amps.reshape(nocc_a, nvirt_a)
        else:
            XmY[root] = amps.reshape(nocc_a, nvirt_a)

    # Recover X and Y from the two vectors
    X_active = (XpY + XmY) / 2.0
    Y_active = (XpY - XmY) / 2.0

    # Pad with zero rows for frozen core orbitals
    X = np.zeros((nroots, nocc_full, nvirt_a))
    Y = np.zeros((nroots, nocc_full, nvirt_a))
    X[:, NFROZEN:, :] = X_active
    Y[:, NFROZEN:, :] = Y_active

    return {
        'energies': energies,
        'X': X,
        'Y': Y,
        'nocc_active': nocc_a,
        'nocc_full': nocc_full,
        'nvirt': nvirt_a,
        'nfrozen': NFROZEN,
        'orb_window': np.array(orb_win, dtype=np.int32),
    }


def write_outputs(result, stem):
    energies = result['energies']
    X = result['X']
    Y = result['Y']
    nocc_full = result['nocc_full']
    nocc_active = result['nocc_active']
    nvirt = result['nvirt']
    nfrozen = result['nfrozen']
    nroots = len(energies)

    # --- energies file ---
    efile = Path(stem + '_energies.dat')
    with open(efile, 'w') as f:
        f.write(f"# {stem}: excitation energies\n")
        f.write(f"# {'Root':>6}  {'Energy(Ha)':>18}  {'Energy(eV)':>14}\n")
        for i, e in enumerate(energies):
            f.write(f"  {i+1:6d}  {e:18.10f}  {e*HA_TO_EV:14.8f}\n")
    print(f"  Saved {efile}")

    # --- vector files ---
    header_common = (
        f"# {stem}: amplitudes  shape=(nroots={nroots}, nocc_full={nocc_full}, nvirt={nvirt})\n"
        f"# nfrozen={nfrozen} (rows 0..{nfrozen-1} are zero)  nocc_active={nocc_active}\n"
        f"# Each block: root N (1-based), energy, then nocc_full rows of nvirt values\n"
    )
    for label, arr in [('X', X), ('Y', Y)]:
        vfile = Path(stem + f'_vectors{label}.dat')
        with open(vfile, 'w') as f:
            f.write(header_common)
            for root in range(nroots):
                f.write(f"\n# Root {root+1}  E={energies[root]:.10f} Ha  {energies[root]*HA_TO_EV:.6f} eV\n")
                for i in range(nocc_full):
                    f.write("  ".join(f"{arr[root, i, a]:14.10f}" for a in range(nvirt)) + "\n")
        print(f"  Saved {vfile}")


if __name__ == '__main__':
    for fname in INPUT_FILES:
        result = parse_cis(fname)
        stem = Path(fname).stem
        write_outputs(result, stem)

        # Validation: X norms should be ~1 (computed on active block only)
        X = result['X']
        nf = result['nfrozen']
        norms_X = np.linalg.norm(X[:, nf:, :].reshape(len(X), -1), axis=1)
        Y = result['Y']
        norms_Y = np.linalg.norm(Y[:, nf:, :].reshape(len(Y), -1), axis=1)
        print(f"  X norms: {norms_X.round(4)}")
        print(f"  Y norms: {norms_Y.round(4)}")
        print()
