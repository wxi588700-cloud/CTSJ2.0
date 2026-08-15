"""Shared test fixtures: a synthetic mini-target with an R|T cleavage site
and a C-C disulfide spanning the two future fragments.

Topology: helix A (res 1-45) --loop-- helix B (res 50-110), packed at 9.6 A.
Cleavage site R60|T61; disulfide C20 (helix A) -- C75 (helix B); the C20 SG
is positioned so the pair is a proper 2.05 A SS bond.
"""
from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np

HELIX_RISE = 1.5
HELIX_TWIST = np.deg2rad(100.0)
CA_R = 2.3

SEQ = (
    "AAEAAKAAEAAKAAEAAKAAECAAKAAEAAKAAEAAKAAEAAKAAEAAK"  # 1-50 (C at 20)
    "GGS"                                              # 51-53 (short loop)
    "AAEAAKAAEAAKAAERTAAEAAKAAECAAKAAEAAKAAEAAKAAEAAKAAEAAKAAEAAK"  # 54-113
)
# enforce key residues: R60, T61, C20, C75
SEQ = list(SEQ)
SEQ[19] = "C"   # 20
SEQ[59] = "R"   # 60
SEQ[60] = "T"   # 61
SEQ[74] = "C"   # 75
SEQ = "".join(SEQ)
assert SEQ[19] == "C" and SEQ[59] == "R" and SEQ[60] == "T" and SEQ[74] == "C"

THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}


def _helix_ca(i: int, origin: np.ndarray, axis_sign: float, phase: float) -> np.ndarray:
    t = phase + i * HELIX_TWIST
    return origin + np.array([CA_R * np.cos(t), CA_R * np.sin(t),
                              axis_sign * HELIX_RISE])


def build_mini_target(path: Path) -> Path:
    """Write the synthetic mini target as mmCIF; returns the path."""
    n = len(SEQ)
    # helix A: residues 1..45 along +z at origin, phase set so res20 faces +x
    phase_a = -20 * HELIX_TWIST  # t_20 = 0
    # helix B: residues 50..n along -z at x=9.6, phase set so res75 faces -x
    phase_b = np.pi - (75 - 50) * HELIX_TWIST
    ca = {}
    for i in range(1, 46):
        ca[i] = _helix_ca(i, np.array([0.0, 0.0, 0.0]), +1.0, phase_a)
    z_b0 = 30.0 + 1.5 * (75 - 50)  # align z of residues 20 and 75
    for i in range(50, n + 1):
        ca[i] = _helix_ca(i, np.array([9.6, 0.0, z_b0]), -1.0, phase_b)
    # loop residues 46-49: linear interpolation
    for j, i in enumerate(range(46, 50)):
        frac = (j + 1) / 5.0
        ca[i] = ca[45] + (ca[50] - ca[45]) * frac + np.array([0, 0.5 * np.sin(np.pi * frac), 0])

    st = gemmi.Structure()
    st.name = "MINITARGET"
    st.spacegroup_hm = "P 1"
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    keys = sorted(ca)
    for idx, i in enumerate(keys):
        res = gemmi.Residue()
        res.name = THREE[SEQ[i - 1]]
        res.seqid = gemmi.SeqId(i, " ")
        res.het_flag = "A"
        res.subchain = f"A{i}"

        def add(name: str, element: str, pos: np.ndarray):
            a = gemmi.Atom()
            a.name = name
            a.element = gemmi.Element(element)
            a.pos = gemmi.Position(*pos)
            a.occ = 1.0
            a.b_iso = 30.0
            res.add_atom(a)

        p = ca[i]
        prev = ca[keys[idx - 1]] if idx > 0 else p + np.array([0, 0, -3.8])
        nxt = ca[keys[idx + 1]] if idx < len(keys) - 1 else p + np.array([0, 0, 3.8])
        u_prev = (prev - p) / (np.linalg.norm(prev - p) + 1e-9)
        u_next = (nxt - p) / (np.linalg.norm(nxt - p) + 1e-9)
        add("N", "N", p + 1.27 * u_prev)
        add("CA", "C", p)
        add("C", "C", p + 1.27 * u_next)
        add("O", "O", p + 1.27 * u_next + np.array([0, 1.24, 0]))
        if res.name == "CYS":
            add("CB", "C", p + 2.5 * (np.array([1, 0, 0]) if i < 50 else np.array([-1, 0, 0])))
        chain.add_residue(res)
    # disulfide SGs: place manually at midpoint-ish positions 2.05 A apart
    c20 = next(r for r in chain if r.seqid.num == 20)
    c75 = next(r for r in chain if r.seqid.num == 75)
    mid = (ca[20] + ca[75]) / 2.0
    sg20 = mid + np.array([1.025, 0, 0])   # toward helix A side
    sg75 = mid - np.array([1.025, 0, 0])
    a = gemmi.Atom(); a.name = "SG"; a.element = gemmi.Element("S")
    a.pos = gemmi.Position(*sg20); a.occ = 1.0; a.b_iso = 30.0
    c20.add_atom(a)
    b = gemmi.Atom(); b.name = "SG"; b.element = gemmi.Element("S")
    b.pos = gemmi.Position(*sg75); b.occ = 1.0; b.b_iso = 30.0
    c75.add_atom(b)

    model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = st.make_mmcif_document()
    doc.write_file(str(path), gemmi.cif.Style.Aligned)
    return path


MINI_FASTA = ">MINI synthetic target\n" + SEQ + "\n"
