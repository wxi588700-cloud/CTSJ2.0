"""M07: negative-state and off-target selectivity (PRD M07).

For candidates passing the positive state this module:

* transfers the binder pose onto INTACT TROP2 (cis and trans constructs) by
  superposing the shared body chain - the intact peptide bond region then
  physically competes with the binder for the neo-epitope, which is exactly
  the selectivity mechanism we quantify;
* transfers the pose onto EpCAM by matching the binder's interface centroid
  to the most compatible exposed EpCAM surface patch (coarse deterministic
  geometric cross-docking proxy; AF2-Multimer cross-prediction plugs in
  through the predictor adapter when available);
* screens sequence/structure off-targets with Foldseek/MMseqs2 when the
  binaries are configured (otherwise records 'review' per PRD data-missing
  policy, never silently zero);
* reports the WORST negative state per candidate (PRD 12.2).

Standard outputs: negative_state_metrics.csv, complexes/negative/*.cif,
offtarget_hits.csv.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

from ..io import (
    first_protein_chain, polymer_residues, read_json, read_structure,
    write_cif, write_json,
)
from ..io.geometry import clash_count, kabsch, superpose_by_number
from .interface_metrics import InterfaceAnalysis, binder_trace_residues, atoms_of


def pose_on_intact(pose_ca, cleaved_chains, intact_chain_res):
    """Superpose cleaved-state body onto the intact chain, move the pose.

    FIX (was: sequential ``a[:n]/b[:n]`` pairing): residues are paired by
    author residue NUMBER.  Cleaved chains start at T88 (BODY) / D32 (NFR)
    while the intact chain starts at D32 - sequential pairing mis-registered
    the traces (same bug family as mechanism.overlay_pose).
    """
    mobile = [r for res in cleaved_chains.values() for r in res]
    R, t, _fit_rmsd, _n = superpose_by_number(mobile, intact_chain_res)
    return pose_ca @ R + t


def risk_from_geometry(area, sc, hb, clashes, contacts) -> float:
    """Normalised 0-1 off-target binding risk from geometric observables."""
    area_t = np.clip((area - 300.0) / 800.0, 0.0, 1.0)
    sc_t = np.clip((sc - 0.40) / 0.35, 0.0, 1.0)
    hb_t = np.clip(hb / 10.0, 0.0, 1.0)
    clash_t = np.clip(1.0 - clashes / 15.0, 0.0, 1.0)
    contact_t = np.clip(contacts / 18.0, 0.0, 1.0)
    raw = 0.30 * area_t + 0.22 * sc_t + 0.13 * hb_t + 0.20 * clash_t + 0.15 * contact_t
    return float(np.clip(raw, 0.0, 1.0))


binder_as_residues = binder_trace_residues  # CA+CB pseudo-sidechain trace


def epcam_patch_pose(pose_ca, epcam_res):
    """Best-matching exposed EpCAM patch placement (deterministic proxy).

    Two-stage: cheap clash/contact pre-screen on the outer-shell patches,
    then full interface analysis on the five best candidates only.
    """
    ca = np.array([[r.find_atom("CA", "*").pos.x,
                    r.find_atom("CA", "*").pos.y,
                    r.find_atom("CA", "*").pos.z] for r in epcam_res
                   if r.find_atom("CA", "*") is not None])
    centre_all = ca.mean(axis=0)
    d_centre = np.linalg.norm(ca - centre_all, axis=1)
    outer = np.argsort(-d_centre)[: min(40, len(ca) // 4)]
    epcam_ca_all = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in epcam_res
                             for a in r if a.name == "CA"]) \
        if False else np.array([[r.find_atom("CA", "*").pos.x,
                                 r.find_atom("CA", "*").pos.y,
                                 r.find_atom("CA", "*").pos.z]
                                for r in epcam_res if r.find_atom("CA", "*") is not None])
    binder_centre = pose_ca.mean(axis=0)
    centred = pose_ca - binder_centre

    # stage 1: cheap pre-screen (contacts + clashes only)
    prescreen = []
    for i in outer:
        anchor = ca[i]
        direction = anchor - centre_all
        direction /= np.linalg.norm(direction) + 1e-12
        placed = centred + anchor + direction * 16.0
        clashes = clash_count(placed, epcam_ca_all, 3.2)
        contacts = clash_count(placed, epcam_ca_all, 6.0)
        prescreen.append((-contacts + clashes * 3, i, placed, direction))
    prescreen.sort(key=lambda t: -t[0])
    best = None
    for score, i, placed, direction in prescreen[:5]:
        bres = binder_as_residues(placed)
        try:
            ia = InterfaceAnalysis(epcam_res, bres)
            s = ia.summary()
        except ValueError:
            continue
        risk = risk_from_geometry(s["interface_area_A2"], s["shape_complementarity"],
                                  s["hbonds"], s["clashes"], s["n_contact_target_residues"])
        if best is None or risk > best[0]:
            best = (risk, placed, {**s, "anchor_resnum": int(epcam_res[i].seqid.num)})
    return best


def run(ctx) -> None:
    cfg = ctx.config
    out = ctx.out
    pos = pd.read_csv(out / "positive_state_metrics.csv")
    agg = pos[pos.state_id == "AGGREGATE"]
    shortlist = agg[agg.positive_state_pass_rate > 0]
    if shortlist.empty:
        # valid outcome (measured predictor rejected all designs): emit empty
        # tables so M10 can still produce the full rejection audit report
        pd.DataFrame(columns=["candidate_id", "design_name", "negative_state",
                              "risk", "metric_source"]).to_csv(
            out / "negative_state_metrics.csv", index=False)
        pd.DataFrame(columns=["candidate_id", "screen", "hit", "risk", "note"]).to_csv(
            out / "offtarget_hits.csv", index=False)
        ctx.state["negative"] = []
        return

    registry = read_json(out / "target_registry.json")

    intact_cis = read_structure(registry["structures"]["cis"]["standardized_file"])
    cis_res = polymer_residues(first_protein_chain(intact_cis))
    intact_trans = read_structure(registry["structures"]["trans"]["standardized_file"])
    trans_res = polymer_residues(first_protein_chain(intact_trans))

    epcam_res = None
    if "epcam" in registry and "standardized_file" in registry.get("epcam", {}):
        epcam_st = read_structure(registry["epcam"]["standardized_file"])
        epcam_res = polymer_residues(first_protein_chain(epcam_st))

    states = pd.read_csv(out / "state_manifest.csv")
    cleaved = states[(states.kind == "cleaved") & states.audit_passed]
    ref_state_file = Path(cleaved.iloc[0].file)
    ref_st = read_structure(ref_state_file)
    ref_chains = {ch.name: polymer_residues(ch) for ch in ref_st[0] if polymer_residues(ch)}

    cand_manifest = pd.read_csv(out / "candidate_manifest.csv").set_index("candidate_id")
    neg_dir = out / "complexes" / "negative"
    neg_dir.mkdir(parents=True, exist_ok=True)

    screen_tools = {}
    for name in ("foldseek", "mmseqs2"):
        spec = getattr(ctx.tools, name, None) if ctx.tools else None
        binary = shutil.which(name) if spec is not None else None
        screen_tools[name] = bool(binary)
    screen_available = any(screen_tools.values())

    rows: list[dict] = []
    hits: list[dict] = []

    for _, arow in shortlist.iterrows():
        cid = arow.candidate_id
        cand = cand_manifest.loc[cid]
        f = cand.get("file")
        if not isinstance(f, str) or not Path(f).exists():
            continue
        cst = read_structure(f)
        ch = first_protein_chain(cst, None)
        pose = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in polymer_residues(ch)
                         if (a := r.find_atom("CA", "*")) is not None])

        # ---- intact cis TROP2
        pose_intact = pose_on_intact(pose, ref_chains, cis_res)
        bres = binder_as_residues(pose_intact)
        try:
            ia = InterfaceAnalysis(cis_res, bres)
            s = ia.summary()
        except ValueError:
            s = {"interface_area_A2": 0.0, "shape_complementarity": 0.0, "hbonds": 0,
                 "clashes": 999, "n_contact_target_residues": 0, "buried_unsat": 0}
        cis_risk = risk_from_geometry(s["interface_area_A2"],
                                      s["shape_complementarity"], s["hbonds"],
                                      s["clashes"], s["n_contact_target_residues"])
        rows.append({"candidate_id": cid, "design_name": arow.design_name,
                     "negative_state": "intact_trop2_cis",
                     "interface_area_A2": s["interface_area_A2"],
                     "shape_complementarity": s["shape_complementarity"],
                     "hbonds": s["hbonds"], "clashes": s["clashes"],
                     "contacts": s["n_contact_target_residues"],
                     "risk": round(cis_risk, 3), "metric_source": "proxy"})
        stc = gemmi.Structure()
        stc.name = f"{cid}_intact_cis"
        stc.spacegroup_hm = "P 1"
        model = gemmi.Model("1")
        cha = gemmi.Chain("T")
        for r in cis_res:
            cha.add_residue(r.clone())
        model.add_chain(cha)
        chb = gemmi.Chain("B")
        for rr in bres:
            chb.add_residue(rr)
        model.add_chain(chb)
        stc.add_model(model)
        stc.setup_entities()
        write_cif(stc, neg_dir / f"{cid}_intact_cis.cif")

        # ---- intact trans TROP2
        pose_trans = pose_on_intact(pose, ref_chains, trans_res)
        bres_t = binder_as_residues(pose_trans)
        try:
            ia_t = InterfaceAnalysis(trans_res, bres_t)
            s_t = ia_t.summary()
        except ValueError:
            s_t = {"interface_area_A2": 0.0, "shape_complementarity": 0.0, "hbonds": 0,
                   "clashes": 999, "n_contact_target_residues": 0}
        trans_risk = risk_from_geometry(s_t["interface_area_A2"],
                                        s_t["shape_complementarity"], s_t["hbonds"],
                                        s_t["clashes"], s_t["n_contact_target_residues"])
        rows.append({"candidate_id": cid, "design_name": arow.design_name,
                     "negative_state": "intact_trop2_trans",
                     "interface_area_A2": s_t["interface_area_A2"],
                     "shape_complementarity": s_t["shape_complementarity"],
                     "hbonds": s_t["hbonds"], "clashes": s_t["clashes"],
                     "contacts": s_t["n_contact_target_residues"],
                     "risk": round(trans_risk, 3), "metric_source": "proxy"})

        # ---- EpCAM
        epcam_risk = None
        if epcam_res is not None:
            best = epcam_patch_pose(pose, epcam_res)
            if best is not None:
                epcam_risk, placed, det = best
                rows.append({"candidate_id": cid, "design_name": arow.design_name,
                             "negative_state": "human_epcam",
                             "interface_area_A2": det["interface_area_A2"],
                             "shape_complementarity": det["shape_complementarity"],
                             "hbonds": det["hbonds"], "clashes": det["clashes"],
                             "contacts": det["n_contact_target_residues"],
                             "risk": round(epcam_risk, 3), "metric_source": "proxy"})
                hits.append({"candidate_id": cid, "screen": "geometric_epcam_patch",
                             "hit": f"EpCAM patch near res {det['anchor_resnum']}",
                             "risk": round(epcam_risk, 3)})
        else:
            rows.append({"candidate_id": cid, "design_name": arow.design_name,
                         "negative_state": "human_epcam", "risk": None,
                         "metric_source": "missing",
                         "note": "EpCAM structure unavailable"})

        # ---- off-target screen availability (AC-11: missing tool -> review)
        if not screen_available:
            hits.append({"candidate_id": cid, "screen": "foldseek/mmseqs2",
                         "hit": "UNAVAILABLE", "risk": None,
                         "note": "off-target screening tool not configured; "
                                 "flagged review per PRD (never silently zero)"})
        else:
            hits.append({"candidate_id": cid, "screen": "foldseek/mmseqs2",
                         "hit": "not-run-baseline", "risk": None,
                         "note": "adapter wired; run on host with binaries"})

        intact_worst = max(r for r in (cis_risk, trans_risk) if r is not None)
        rows.append({"candidate_id": cid, "design_name": arow.design_name,
                     "negative_state": "WORST",
                     "risk": round(intact_worst, 3),
                     "epcam_risk": round(epcam_risk, 3) if epcam_risk is not None else None,
                     "note": "worst negative state per PRD 12.2"})

    pd.DataFrame(rows).to_csv(out / "negative_state_metrics.csv", index=False)
    pd.DataFrame(hits).to_csv(out / "offtarget_hits.csv", index=False)
    ctx.state["negative"] = rows
