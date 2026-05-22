"""
Validate parse_cis.py against the C parser logic from orca_parse_cis.c.

The C code reads the binary directly via fread using the compiled struct layout.
We reproduce that here using ctypes (same struct, same compiler alignment rules)
and compare against the parse_cis.py output element by element.

Pass: all energies and all amplitudes match to machine precision.
"""

import ctypes
import struct
import numpy as np
from pathlib import Path
from parse_cis import parse_cis, NFROZEN

HA_TO_EV = 27.2114
RTOL = 0.0
ATOL = 1e-15   # double precision, expect exact bit-for-bit match


# ---------------------------------------------------------------------------
# C struct reproduced via ctypes — same layout as orca_parse_cis.c
# ---------------------------------------------------------------------------

class OrcaCisVecHeader(ctypes.Structure):
    _fields_ = [
        ('n',             ctypes.c_int),
        ('sym',           ctypes.c_int),
        ('mult',          ctypes.c_int),
        ('iblock',        ctypes.c_int),
        ('iroot',         ctypes.c_int),
        ('en',            ctypes.c_double),
        ('is_transition', ctypes.c_bool),
    ]


def read_cis_c_style(path):
    """Read a .cis file exactly as the C code does: raw fread into the struct."""
    data = Path(path).read_bytes()
    offset = 0

    nvec = struct.unpack_from('<i', data, offset)[0]; offset += 4
    orb_win = struct.unpack_from('<8i', data, offset); offset += 32

    nocc_a  = orb_win[1] - orb_win[0] + 1
    nvirt_a = orb_win[3] - orb_win[2] + 1
    n_amp   = nocc_a * nvirt_a
    hsize   = ctypes.sizeof(OrcaCisVecHeader)

    energies_raw = []   # one per raw vector (32 total for TDDFT)
    amps_raw     = []   # one array per raw vector

    for _ in range(nvec):
        hdr = OrcaCisVecHeader.from_buffer_copy(data[offset: offset + hsize])
        offset += hsize
        amps = np.frombuffer(data, dtype='<f8', count=hdr.n, offset=offset).copy()
        offset += hdr.n * 8
        energies_raw.append(hdr.en)
        amps_raw.append(amps)

    return energies_raw, amps_raw, nocc_a, nvirt_a, n_amp


def c_reconstruct_X(energies_raw, amps_raw, nocc_a, nvirt_a, nroots):
    """Reconstruct X as C-derived: X = (XpY + XmY)/2, energy from even vectors."""
    energies = np.zeros(nroots)
    X = np.zeros((nroots, nocc_a, nvirt_a))
    for ivec in range(nroots * 2):
        root = ivec // 2
        if ivec % 2 == 0:
            energies[root] = energies_raw[ivec]
            X[root] += amps_raw[ivec].reshape(nocc_a, nvirt_a)
        else:
            X[root] += amps_raw[ivec].reshape(nocc_a, nvirt_a)
    X /= 2.0
    return energies, X


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def validate(cis_file):
    print(f"\n{'='*60}")
    print(f"Validating: {cis_file}")
    print(f"{'='*60}")

    # --- C-style read ---
    energies_raw, amps_raw, nocc_a, nvirt_a, n_amp = read_cis_c_style(cis_file)
    nroots = len(energies_raw) // 2
    e_c, X_c = c_reconstruct_X(energies_raw, amps_raw, nocc_a, nvirt_a, nroots)

    # --- Python parser ---
    result = parse_cis(cis_file)
    e_py = result['energies']
    X_py = result['X'][:, NFROZEN:, :]   # strip frozen padding for comparison

    print(f"\nstruct sizeof = {ctypes.sizeof(OrcaCisVecHeader)} bytes  "
          f"(expected 40)")
    print(f"nvec={nroots*2}  nroots={nroots}  nocc_active={nocc_a}  nvirt={nvirt_a}")

    # --- Energy comparison ---
    e_diff = np.abs(e_c - e_py)
    print(f"\nEnergy comparison (Ha):")
    print(f"  max |C - Python| = {e_diff.max():.3e}")
    all_ok = True
    for i in range(nroots):
        flag = "" if e_diff[i] <= ATOL else "  *** MISMATCH ***"
        print(f"  root {i+1:2d}: C={e_c[i]:.10f}  Py={e_py[i]:.10f}  diff={e_diff[i]:.3e}{flag}")
        if e_diff[i] > ATOL:
            all_ok = False

    # --- Amplitude comparison ---
    amp_diff = np.abs(X_c - X_py)
    print(f"\nAmplitude comparison (X, active block):")
    print(f"  shape: C={X_c.shape}  Python={X_py.shape}")
    print(f"  max |C - Python| = {amp_diff.max():.3e}")
    print(f"  mean|C - Python| = {amp_diff.mean():.3e}")
    if amp_diff.max() > ATOL:
        all_ok = False
        idx = np.unravel_index(amp_diff.argmax(), amp_diff.shape)
        print(f"  worst mismatch at root={idx[0]+1} occ={idx[1]} virt={idx[2]}:")
        print(f"    C={X_c[idx]:.15f}  Python={X_py[idx]:.15f}")

    # --- Spot-check printout ---
    print(f"\nSpot-check: root 1, first active occ row, first 10 virts:")
    print(f"  {'idx':>4}  {'C':>18}  {'Python':>18}  {'diff':>12}")
    for a in range(10):
        diff = abs(X_c[0, 0, a] - X_py[0, 0, a])
        print(f"  {a:4d}  {X_c[0,0,a]:18.12f}  {X_py[0,0,a]:18.12f}  {diff:.3e}")

    print(f"\n{'PASS' if all_ok else 'FAIL'}: {cis_file}")
    return all_ok


if __name__ == '__main__':
    results = []
    for fname in ['rinf.cis', 'rstack.cis']:
        results.append(validate(fname))
    print(f"\n{'='*60}")
    print(f"Overall: {'ALL PASS' if all(results) else 'SOME FAILURES'}")
