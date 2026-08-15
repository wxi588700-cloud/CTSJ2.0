"""M06: positive-state (cleaved TROP2) complex prediction and binding scores.

For every candidate surviving M05 and every audited cleaved conformer this
module builds/loads the complex pose, computes the interface metric battery
(area, shape complementarity, H-bonds, buried unsatisfied polars, clashes),
the T88 free-alpha-amino contact evidence (hard-gate metric), cross-state
reproduction statistics and the robust (worst-conformer) aggregates of
PRD 12.2.

Standard outputs: positive_state_metrics.csv, complexes/positive/*.cif,
terminal_contact.json.
"""
from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

from ..io import (
    first_protein_chain, polymer_residues, read_json, read_structure,
    write_cif, write_json,
)
from ..io.geometry import kabsch, sasa
from ..schemas.project import parse_residue_id
from .interface_metrics import (
    InterfaceAnalysis, TraceAwareProxy, binder_trace_residues, confidence_proxies,
)

# binding-threshold defaults (versioned via metrics profile in M10)
MIN_INTERFACE_AREA = 400.0
MIN_IPTM = 0.50
T88_CONTACT_CUTOFF = 4.5


def load_state(state_file: Path):
    st = read_structure(state_file)
    model = st[0]
    chains = {}
    for ch in model:
        res = polymer_residues(ch)
        if res:
            chains[ch.name] = res
    return st, chains


def binder_pose_from_candidate(cand_row) -> np.ndarray | None:
    """CA coordinates of the candidate in its design pose near the target."""
    f = cand_row.get("file")
    if not isinstance(f, str) or not Path(f).exists():
        return None
    st = read_structure(f)
    ch = first_protein_chain(st, None)
    ca = []
    for r in polymer_residues(ch):
        a = r.find_atom("CA", "*")
        if a is not None:
            ca.append([a.pos.x, a.pos.y, a.pos.z])
    return np.asarray(ca, dtype=float)


def map_pose_to_state(pose_ca, state_chains, ref_chains) -> np.ndarray:
    """Transfer a binder pose between conformers by superposing the shared
    BODY chain CA traces (deterministic Kabsch)."""
    def cas(chains):
        pts = []
        for res in chains.get("BODY", []) or sum(chains.values(), []):
            a = res.find_atom("CA", "*")
            if a is not None:
                pts.append([a.pos.x, a.pos.y, a.pos.z])
        return np.asarray(pts, dtype=float)

    a = cas(ref_chains)
    b = cas(state_chains)
    n = min(len(a), len(b))
    if n < 3:
        return pose_ca
    R, t = kabsch(a[:n], b[:n])
    return pose_ca @ R + t


pose_as_residues = binder_trace_residues  # CA+CB pseudo-sidechain trace


def write_complex(state_file: Path, pose_ca: np.ndarray, name: str,
                  out_path: Path) -> Path:
    st = read_structure(state_file)
    model = st[0]
    bchain = gemmi.Chain("BND")
    res = gemmi.Residue()
    res.name = "GLY"
    res.seqid = gemmi.SeqId(1, " ")
    for p in pose_ca:
        a = gemmi.Atom()
        a.name = "CA"
        a.element = gemmi.Element("C")
        a.pos = gemmi.Position(*p)
        a.occ = 1.0
        a.b_iso = 30.0
        res.add_atom(a)
    bchain.add_residue(res)
    model.add_chain(bchain)
    st.name = name
    return write_cif(st, out_path)


def t88_terminal_evidence(state_chains, right_num: int, pose_ca) -> dict:
    """Direct-contact evidence for the T88 free alpha-amino terminus.

    Measured on the true geometry (never a proxy): distance from the T88
    backbone N atom (the free NH3+ group created in M02) to the nearest
    binder heavy atom.  Only meaningful for CLEAVED states (AC-08).
    """
    t88 = None
    for res in state_chains.values():
        for r in res:
            if r.seqid.num == right_num:
                t88 = r
                break
        if t88 is not None:
            break
    if t88 is None or len(pose_ca) == 0:
        return {"contacted": False, "min_distance": 99.0, "n_contacts": 0,
                "contact_atoms": [], "orientation_score": 0.0}
    n_atom = t88.find_atom("N", "*")
    if n_atom is None:
        return {"contacted": False, "min_distance": 99.0, "n_contacts": 0,
                "contact_atoms": [], "orientation_score": 0.0}
    n_pos = np.array([n_atom.pos.x, n_atom.pos.y, n_atom.pos.z])
    d = np.linalg.norm(pose_ca - n_pos, axis=1)
    within = d <= T88_CONTACT_CUTOFF
    orient = 0.0
    ca_atom = t88.find_atom("CA", "*")
    if ca_atom is not None and within.any():
        ca_pos = np.array([ca_atom.pos.x, ca_atom.pos.y, ca_atom.pos.z])
        axis = n_pos - ca_pos
        axis /= np.linalg.norm(axis) + 1e-12
        vecs = pose_ca[within] - n_pos
        dots = vecs @ axis
        orient = float(np.clip(dots.mean() / 6.0, -1.0, 1.0))
    return {
        "contacted": bool(within.any()),
        "min_distance": round(float(d.min()), 2),
        "n_contacts": int(within.sum()),
        "contact_atoms": [f"CA{i+1}" for i in np.where(within)[0][:10]],
        "orientation_score": round(orient, 3),
    }


def run(ctx) -> None:
    cfg = ctx.config
    out = ctx.out
    _, right_num = parse_residue_id(cfg.target.cleavage.right_residue)

    mono = pd.read_csv(out / "monomer_metrics.csv")
    mono = mono[mono.status == "pass"]
    states = pd.read_csv(out / "state_manifest.csv")
    cleaved = states[(states.kind == "cleaved") & states.audit_passed]
    if cleaved.empty or mono.empty:
        raise RuntimeError("no cleaved states or no folded candidates for M06")

    complexes_dir = out / "complexes" / "positive"
    complexes_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    terminal_records: list[dict] = []
    cand_manifest = pd.read_csv(out / "candidate_manifest.csv").set_index("candidate_id")

    # per-state caches (target unbound SASA reused across candidates)
    state_cache: dict[str, dict] = {}
    for _, srow in cleaved.iterrows():
        _, chains = load_state(Path(srow.file))
        target_residues = [r for res in chains.values() for r in res]
        from .interface_metrics import atoms_of

        tc, te, _ = atoms_of(target_residues)
        state_cache[srow.state_id] = {
            "chains": chains,
            "residues": target_residues,
            "unbound_sasa": sasa(tc, te, 480),
        }
    ref_sid = cleaved.iloc[0].state_id
    ref_chains = state_cache[ref_sid]["chains"]

    for cid in mono.candidate_id.unique():
        sub = mono[mono.candidate_id == cid]
        design_names = list(sub.design_name)
        cand = cand_manifest.loc[cid]
        pose = binder_pose_from_candidate(cand)
        if pose is None or len(pose) == 0:
            for dn in design_names:
                rows.append({"candidate_id": cid, "design_name": dn, "state_id": "",
                             "status": "failed", "failure_reason": "no binder pose"})
            continue
        binder_res = pose_as_residues(pose)
        from .interface_metrics import atoms_of

        bc, be, _ = atoms_of(binder_res)
        binder_unbound = sasa(bc, be, 480)

        per_state = []
        for _, srow in cleaved.iterrows():
            sid = srow.state_id
            cache = state_cache[sid]
            state_pose = (pose if sid == ref_sid
                          else map_pose_to_state(pose, cache["chains"], ref_chains))
            b_res_moved = pose_as_residues(state_pose)
            try:
                ia = InterfaceAnalysis(cache["residues"], b_res_moved,
                                       unbound_sasa_a=cache["unbound_sasa"],
                                       unbound_sasa_b=binder_unbound)
                summary = ia.summary()
                n_contacts_iface = len(ia.contacts)
            except ValueError:
                summary = {"interface_area_A2": 0.0, "shape_complementarity": 0.0,
                           "hbonds": 0, "buried_unsat": 0, "clashes": 999,
                           "n_contact_target_residues": 0}
                n_contacts_iface = 0
            # trace-resolution correction for the proxy confidence numbers
            # (CA+CB bead binders bury ~1/3 of full side-chain area); raw
            # measured values are always kept in the CSV alongside
            # fallback scaffolds are CA-only traces scored as CA+CB beads
            trace_mode = True
            eff_area, eff_sc = TraceAwareProxy.effective(
                summary["interface_area_A2"], summary["shape_complementarity"],
                n_contacts_iface, summary["n_contact_target_residues"], trace_mode)
            proxies = confidence_proxies(eff_area, eff_sc,
                                         summary["hbonds"], summary["clashes"],
                                         summary["n_contact_target_residues"])
            proxies["effective_interface_area_A2"] = round(eff_area, 1)
            term = t88_terminal_evidence(cache["chains"], right_num, state_pose)
            passing = (eff_area >= MIN_INTERFACE_AREA and
                       proxies["complex_iptm_proxy"] >= MIN_IPTM and
                       term["contacted"] and summary["clashes"] <= 25)
            per_state.append({
                "candidate_id": cid, "design_name": design_names[0], "state_id": sid,
                **summary, **proxies,
                **{f"t88_{k}": v for k, v in term.items()},
                "metric_source": "proxy",
                "status": "pass" if passing else "fail_state",
            })
            terminal_records.append({
                "state_id": sid, "candidate_id": cid,
                "kind": "cleaved",
                **term,
                "t88_residue": cfg.target.cleavage.right_residue,
            })
            if sid == ref_sid:
                write_complex(Path(srow.file), state_pose, f"{cid}_{sid}",
                              complexes_dir / f"{cid}_{sid}.cif")

        # cross-state aggregates (PRD 12.2)
        pass_rate = float(np.mean([r["status"] == "pass" for r in per_state]))
        iptms = np.array([r["complex_iptm_proxy"] for r in per_state])
        quant = cfg.ranking.robust_positive_quantile
        robust_positive = float(np.quantile(iptms, quant))
        uncertainty = float(np.std(iptms))
        contact_occ = float(np.mean([bool(r["t88_contacted"]) for r in per_state]))

        for dn in design_names:
            for r in per_state:
                r2 = dict(r)
                r2["design_name"] = dn
                rows.append(r2)
            rows.append({
                "candidate_id": cid, "design_name": dn, "state_id": "AGGREGATE",
                "positive_state_pass_rate": round(pass_rate, 3),
                "robust_positive": round(robust_positive, 3),
                "t88_contact_occupancy": round(contact_occ, 3),
                "uncertainty_positive": round(uncertainty, 4),
                "status": "aggregate",
            })

    df = pd.DataFrame(rows)
    df.to_csv(out / "positive_state_metrics.csv", index=False)
    write_json(out / "terminal_contact.json", terminal_records)

    agg = df[df.state_id == "AGGREGATE"]
    if agg.empty or not (agg.positive_state_pass_rate > 0).any():
        raise RuntimeError("no candidate reproduced binding in any cleaved state")
    ctx.state["positive"] = rows
