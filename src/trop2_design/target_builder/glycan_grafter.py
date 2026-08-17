"""Deterministic glycan grafting (M02.5 hybrid strategy).

Architecture decision (empirically grounded, 2026-08-17):
  - Boltz 2.0.3 bond constraints work perfectly for protein-protein
    (all six native disulfides, incl. cross-fragment C73-C108, land at
    1.4-1.9 A) but FAIL silently when multiple ligand CCD chains carry
    glycan constraints (single-tree smoke: 1.30 A N-bond OK; four trees:
    24-50 A).  We therefore split the job:
      * protein conformations: Boltz with the six SS bond constraints
      * glycan conformations: THIS deterministic grafter - atomic
        templates (Boltz-generated per glycoform, shipped under
        glycan_templates/) rigidly installed at each Asn ND2 with seeded
        orientation sampling + clash filtering.
    Topology stays 100% under deterministic control (PRD: AI models only
    provide conformational hypotheses; topology authority is external).
"""
from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np

from ..io import polymer_residues, read_structure, residue_one_letter, write_cif
from ..io.geometry import kabsch, min_pair_distance, rotation_matrix

SUGARS = {"NAG", "BMA", "MAN", "GAL", "FUC", "SIA"}
TEMPLATE_DIR = Path(__file__).parent / "glycan_templates"
N_BOND_LEN = 1.43          # Asn ND2 - GlcNAc C1
PROBE_CLEARANCE = 2.9      # min glycan-to-protein heavy atom distance

TEMPLATE_BY_PROFILE = {
    "high_mannose_man5": "tpl_high_mannose_man5_model_0.cif",
    "complex_biantennary": "tpl_complex_biantennary_model_0.cif",
    "core_fucosylated_sialylated": "tpl_core_fucosylated_sialylated_model_0.cif",
}


def load_template(profile_id: str) -> list[dict]:
    """Glycan residues from the atomic template: local coords relative to
    the root GlcNAc C1 with element/atom names preserved."""
    cif = TEMPLATE_DIR / TEMPLATE_BY_PROFILE[profile_id]
    st = read_structure(cif)
    residues = []
    for ch in st[0]:
        for res in ch:
            if res.name in SUGARS:
                atoms = [{"name": a.name,
                          "element": a.element.name,
                          "xyz": np.array([a.pos.x, a.pos.y, a.pos.z])}
                         for a in res]
                residues.append({"name": res.name, "seqid": res.seqid.num,
                                 "atoms": atoms})
    residues.sort(key=lambda r: r["seqid"])
    if not residues:
        raise ValueError(f"no sugar residues in template {cif}")
    root_c1 = next(a["xyz"] for a in residues[0]["atoms"] if a["name"] == "C1")
    for r in residues:
        for a in r["atoms"]:
            a["xyz"] = a["xyz"] - root_c1      # template frame: root C1 at 0
    return residues


def _protein_heavy_atoms(model, exclude_resnum: int | None = None,
                         exclude_chain: str | None = None) -> np.ndarray:
    pts = []
    for ch in model:
        for res in polymer_residues(ch):
            if exclude_resnum is not None and ch.name == exclude_chain \
                    and res.seqid.num == exclude_resnum:
                continue          # own Asn side chain may come close
            for a in res:
                if a.element.name != "H":
                    pts.append([a.pos.x, a.pos.y, a.pos.z])
    return np.asarray(pts)


def graft_one(protein_model, chain: str, local_idx: int,
              template: list[dict], rng: np.random.Generator,
              n_orientations: int = 64) -> tuple[list[dict] | None, str]:
    """Install one glycan template at an Asn.

    Returns (placed_residues | None, status) where status is
    'clean' (clash-free), 'soft' (best-effort: no clash-free orientation
    existed, kept the largest-clearance one - recorded for audit), or
    'failed'.
    """
    asn = None
    for res in polymer_residues(protein_model[chain]):
        if res.seqid.num == local_idx:
            asn = res
            break
    if asn is None or asn.name != "ASN":
        return None, "failed"
    nd2 = asn.find_atom("ND2", "*")
    cg = asn.find_atom("CG", "*")
    if nd2 is None or cg is None:
        return None, "failed"
    nd2_pos = np.array([nd2.pos.x, nd2.pos.y, nd2.pos.z])
    ref = nd2_pos - np.array([cg.pos.x, cg.pos.y, cg.pos.z])
    ref /= np.linalg.norm(ref) + 1e-9

    protein_pts = _protein_heavy_atoms(protein_model, local_idx, chain)
    template_pts = np.vstack([[a["xyz"] for a in r["atoms"]] for r in template])

    def _materialise(R):
        placed = template_pts @ R.T + nd2_pos + ref * N_BOND_LEN
        out, k = [], 0
        for r in template:
            n = len(r["atoms"])
            out.append({"name": r["name"], "seqid": r["seqid"],
                        "atoms": [{"name": a["name"], "element": a["element"],
                                   "xyz": placed[k + i]}
                                  for i, a in enumerate(r["atoms"])], })
            k += n
        return placed, out

    x_axis = np.array([1.0, 0.0, 0.0])
    v = np.cross(x_axis, ref)
    s = float(np.linalg.norm(v))
    Ralign = np.eye(3)
    if s > 1e-9:
        Ralign = rotation_matrix(v / s, float(np.arcsin(np.clip(s, -1, 1))))

    best_R, best_clear = None, -1.0
    for _ in range(n_orientations):
        R = Ralign @ rotation_matrix(rng.normal(size=3) /
                                     (np.linalg.norm(rng.normal(size=3)) + 1e-12),
                                     rng.uniform(0, 2 * np.pi)) \
            @ rotation_matrix(x_axis, rng.uniform(0, 2 * np.pi))
        placed = template_pts @ R.T + nd2_pos + ref * N_BOND_LEN
        clear = min_pair_distance(placed, protein_pts) if len(protein_pts) else 9.9
        if clear >= PROBE_CLEARANCE:
            _, out = _materialise(R)
            return out, "clean"
        if clear > best_clear:
            best_clear, best_R = clear, R
    if best_R is not None:
        _, out = _materialise(best_R)
        return out, f"soft(clearance={best_clear:.2f}A)"
    return None, "failed"


def graft_state(protein_cif: Path, profile_sites: dict, seed: int,
                out_cif: Path, chain_of_site: dict) -> dict:
    """Graft all four glycan trees onto a protein conformation.

    profile_sites: {site: profile_id}; chain_of_site: {site: (chain, local)}
    Writes the glycosylated mmCIF; returns per-site report.
    """
    st = read_structure(protein_cif)
    model = st[0]
    rng = np.random.default_rng(seed)
    report = {}
    templates = {}
    glycan_chain_counter = 0
    placed_sites: list[tuple[str, list[dict]]] = []   # (chain_name, residues)
    for site in sorted(profile_sites):
        profile_id = profile_sites[site]
        if profile_id not in templates:
            templates[profile_id] = load_template(profile_id)
        chain, local = chain_of_site[site]
        placed, status = graft_one(model, chain, local, templates[profile_id], rng)
        if placed is None:
            report[str(site)] = {"ok": False, "reason": "graft failed"}
            continue
        gname = GLYCAN_CHAIN_NAMES[glycan_chain_counter]
        glycan_chain_counter += 1
        placed_sites.append((gname, placed))
        c1 = next(a["xyz"] for a in placed[0]["atoms"] if a["name"] == "C1")
        asn_nd2 = None
        for res in polymer_residues(model[chain]):
            if res.seqid.num == local:
                nd2 = res.find_atom("ND2", "*")
                asn_nd2 = np.array([nd2.pos.x, nd2.pos.y, nd2.pos.z])
        report[str(site)] = {
            "ok": True, "chain": gname, "status": status,
            "n_residues": len(placed),
            "n_bond_distance": round(float(np.linalg.norm(c1 - asn_nd2)), 2)
            if asn_nd2 is not None else None,
        }

    # ---- plain-text PDB writer: no gemmi entity black boxes, byte-exact
    # control over what we emit (protein ATOM + glycan HETATM records)
    out_cif = Path(out_cif)
    out_cif.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    serial = 1
    for ch in model:
        # protein view: amino-acid residues only (the input conformation may
        # carry unconstrained free-sugar chains from earlier experiments -
        # glycan placement is exclusively OUR grafter's job here)
        prot_residues = [r for r in ch if r.name not in SUGARS]
        if not prot_residues:
            continue
        for res in prot_residues:
            for a in res:
                el = a.element.name if hasattr(a.element, "name") else str(a.element)
                lines.append(_pdb_atom_line(serial, "ATOM  ", a.name,
                                            res.name, ch.name,
                                            res.seqid.num, a.pos.x, a.pos.y,
                                            a.pos.z, el, a.occ, a.b_iso))
                serial += 1
        lines.append(f"TER   {serial:>5}      {'':>3} {'':>3} {ch.name}"
                     f"{'':>2}{'':>4}{'':>1}")
        serial += 1
    for gname, placed in placed_sites:
        for r in placed:
            for a in r["atoms"]:
                lines.append(_pdb_atom_line(serial, "HETATM", a["name"],
                                            r["name"], gname, r["seqid"],
                                            a["xyz"][0], a["xyz"][1],
                                            a["xyz"][2], a["element"], 1.0, 30.0))
                serial += 1
        lines.append(f"TER   {serial:>5}")
        serial += 1
    lines.append("END")
    out_cif.write_text("\n".join(lines) + "\n")
    return report


def _pdb_atom_line(serial: int, record: str, name: str, resname: str,
                   chain: str, resnum: int, x: float, y: float, z: float,
                   element: str, occ: float, b: float) -> str:
    aname = f" {name:<3s}" if len(name) < 4 else name
    return (f"{record}{serial:>5} {aname:>4s}{'':>1}{resname:>3s} {chain:1s}"
            f"{resnum:>4}{'':>1}   {x:>8.3f}{y:>8.3f}{z:>8.3f}"
            f"{occ:>6.2f}{b:>6.2f}          {element:>2s}{'':>2}")


GLYCAN_CHAIN_NAMES = ["F", "G", "H", "I", "J", "K"]
