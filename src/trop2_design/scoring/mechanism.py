"""M08: cis/trans mechanism and cell-surface geometry (PRD M08).

Superposes each positive complex onto the cis (7E5N) and trans (7E5M)
assemblies, quantifies binder coverage of the cis interface, occlusion of
the trans interface, and membrane/glycan collisions using the M03 exclusion
mask.

Standard outputs: mechanism_metrics.csv, assembly_overlays/*.cif,
clash_report.json.
"""
from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

from ..io import (
    first_protein_chain, iter_protein_chains, polymer_residues, read_json,
    read_structure, write_cif, write_json,
)
from ..io.geometry import clash_count, kabsch


def assembly_interface_residues(assembly_st, chain_a: str, chain_b: str,
                                cutoff: float = 5.0) -> set[int]:
    """Residue numbers of chain A contacting chain B in an assembly."""
    model = assembly_st[0]
    a = model.find_chain(chain_a)
    b = model.find_chain(chain_b)
    if a is None or b is None:
        return set()
    pa = np.array([[at.pos.x, at.pos.y, at.pos.z] for r in polymer_residues(a) for at in r])
    pb = np.array([[at.pos.x, at.pos.y, at.pos.z] for r in polymer_residues(b) for at in r])
    nums = [r.seqid.num for r in polymer_residues(a)]
    owner = []
    for i, r in enumerate(polymer_residues(a)):
        owner.extend([i] * len(r))
    if len(pa) == 0 or len(pb) == 0:
        return set()
    from scipy.spatial import cKDTree

    tree = cKDTree(pb)
    near = tree.query_ball_point(pa, r=cutoff)
    hit_idx = {owner[i] for i, js in enumerate(near) if js}
    return {nums[i] for i in hit_idx}


def overlay_pose(pose_ca, cleaved_body_res, assembly_chain_res):
    """Superpose cleaved body CA trace onto an assembly chain, move pose."""
    def cas(res_list):
        return np.array([[r.find_atom("CA", "*").pos.x,
                          r.find_atom("CA", "*").pos.y,
                          r.find_atom("CA", "*").pos.z] for r in res_list
                         if r.find_atom("CA", "*") is not None])

    a = cas(cleaved_body_res)
    b = cas(assembly_chain_res)
    n = min(len(a), len(b), 60)
    R, t = kabsch(a[:n], b[:n])
    return pose_ca @ R + t


def run(ctx) -> None:
    cfg = ctx.config
    out = ctx.out
    pos = pd.read_csv(out / "positive_state_metrics.csv")
    agg = pos[pos.state_id == "AGGREGATE"]
    shortlist = agg[agg.positive_state_pass_rate > 0]
    if shortlist.empty:
        pd.DataFrame(columns=["candidate_id", "design_name", "cis_block",
                              "trans_occlusion", "glycan_membrane_clash"]).to_csv(
            out / "mechanism_metrics.csv", index=False)
        write_json(out / "clash_report.json", [])
        ctx.state["mechanism"] = []
        return

    registry = read_json(out / "target_registry.json")
    exclusion = read_json(out / "exclusion_mask.json")

    cis_asm = read_structure(Path(registry["structures"]["cis"]["source_file"]))
    cis_a_name = cfg.target.cis_structure.chain or first_protein_chain(cis_asm, None).name
    cis_chain_names = [c.name for c in cis_asm[0]]
    cis_b_name = next((n for n in cis_chain_names if n != cis_a_name), None)

    trans_asm = read_structure(Path(registry["structures"]["trans"]["source_file"]))
    trans_a_name = cfg.target.trans_structure.chain or first_protein_chain(trans_asm, None).name
    trans_chain_names = [c.name for c in trans_asm[0]]
    trans_b_name = next((n for n in trans_chain_names if n != trans_a_name), None)

    cis_iface = assembly_interface_residues(cis_asm, cis_a_name, cis_b_name)
    trans_iface = assembly_interface_residues(trans_asm, trans_a_name, trans_b_name)

    cis_a_res = polymer_residues(cis_asm[0].find_chain(cis_a_name))
    cis_b_res = polymer_residues(cis_asm[0].find_chain(cis_b_name)) if cis_b_name else []
    trans_a_res = polymer_residues(trans_asm[0].find_chain(trans_a_name))
    trans_b_res = polymer_residues(trans_asm[0].find_chain(trans_b_name)) if trans_b_name else []

    # cleaved-state body residues for pose transfer
    states = pd.read_csv(out / "state_manifest.csv")
    cleaved = states[(states.kind == "cleaved") & states.audit_passed]
    ref_st = read_structure(Path(cleaved.iloc[0].file))
    body_chain = None
    for ch in ref_st[0]:
        res = polymer_residues(ch)
        if res and ch.name == "BODY":
            body_chain = res
    if body_chain is None:
        for ch in ref_st[0]:
            body_chain = polymer_residues(ch)
            if body_chain:
                break

    cand_manifest = pd.read_csv(out / "candidate_manifest.csv").set_index("candidate_id")
    overlay_dir = out / "assembly_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    mem_normal = np.array(exclusion["membrane"]["normal"])
    mem_cutoff = exclusion["membrane"]["cutoff_offset"]
    glycan_spheres = exclusion.get("glycans", [])

    rows: list[dict] = []
    clash_report: list[dict] = []

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

        def overlay_assembly(asm_st, a_res, b_res, a_name, b_name, iface, kind):
            pose_asm = overlay_pose(pose, body_chain, a_res)
            b_ca = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in b_res
                             if (a := r.find_atom("CA", "*")) is not None]) if b_res else np.zeros((0, 3))
            # interface coverage: iface residues within 8 A of binder
            iface_ca = {r.seqid.num: np.array([r.find_atom("CA", "*").pos.x,
                                               r.find_atom("CA", "*").pos.y,
                                               r.find_atom("CA", "*").pos.z])
                        for r in a_res if r.seqid.num in iface and r.find_atom("CA", "*")}
            covered = 0
            for num, p in iface_ca.items():
                if np.linalg.norm(pose_asm - p, axis=1).min() <= 8.0:
                    covered += 1
            coverage = covered / max(len(iface), 1)
            # occlusion of partner chain: contacts + clashes
            n_contacts_b = 0
            clashes_b = 0
            if len(b_ca):
                from scipy.spatial import cKDTree

                tree = cKDTree(b_ca)
                near = tree.query_ball_point(pose_asm, r=8.0)
                n_contacts_b = sum(len(x) for x in near)
                cl = tree.query_ball_point(pose_asm, r=3.2)
                clashes_b = sum(len(x) for x in cl)
            # persist overlay structure
            st = gemmi.Structure()
            st.name = f"{cid}_{kind}"
            st.spacegroup_hm = "P 1"
            model = gemmi.Model("1")
            for chn in asm_st[0]:
                if chn.name in (a_name, b_name):
                    model.add_chain(chn.clone())
            bch = gemmi.Chain("BND")
            res = gemmi.Residue()
            res.name = "GLY"
            res.seqid = gemmi.SeqId(1, " ")
            for p in pose_asm:
                at = gemmi.Atom()
                at.name = "CA"
                at.element = gemmi.Element("C")
                at.pos = gemmi.Position(*p)
                res.add_atom(at)
            bch.add_residue(res)
            model.add_chain(bch)
            st.add_model(model)
            st.setup_entities()
            write_cif(st, overlay_dir / f"{cid}_{kind}.cif")
            return coverage, n_contacts_b, clashes_b

        cis_cov, cis_contacts_b, cis_clashes_b = overlay_assembly(
            cis_asm, cis_a_res, cis_b_res, cis_a_name, cis_b_name, cis_iface, "cis")
        trans_cov, trans_contacts_b, trans_clashes_b = overlay_assembly(
            trans_asm, trans_a_res, trans_b_res, trans_a_name, trans_b_name, trans_iface, "trans")

        # membrane + glycan collisions on the cleaved-state pose
        pose_state = pose  # already in cleaved-state frame (design pose)
        mem_dists = pose_state @ mem_normal - mem_cutoff
        membrane_violations = int((mem_dists > 0).sum())
        glycan_violations = 0
        for sph in glycan_spheres:
            centre = np.array(sph["center"])
            d = np.linalg.norm(pose_state - centre, axis=1)
            # 4.0 A around the Asn anchor = the glycan stem/core region
            # (large flexible antennae are soft-excluded only in M03 design)
            glycan_violations += int((d < 4.0).sum())
        glycan_membrane_clash = membrane_violations + glycan_violations

        rows.append({
            "candidate_id": cid, "design_name": arow.design_name,
            "cis_block": round(min(cis_cov, 1.0), 3),
            "cis_interface_size": len(cis_iface),
            "trans_occlusion": round(min(trans_cov, 1.0), 3),
            "trans_interface_size": len(trans_iface),
            "trans_partner_contacts": trans_contacts_b,
            "trans_partner_clashes": trans_clashes_b,
            "glycan_membrane_clash": glycan_membrane_clash,
            "membrane_violations": membrane_violations,
            "glycan_violations": glycan_violations,
        })
        clash_report.append({
            "candidate_id": cid,
            "membrane_violations": membrane_violations,
            "glycan_violations": glycan_violations,
            "trans_partner_clashes": trans_clashes_b,
            "cis_partner_clashes": cis_clashes_b,
            "verdict": "clash" if glycan_membrane_clash > 0 or trans_clashes_b > 12 else "ok",
        })

    pd.DataFrame(rows).to_csv(out / "mechanism_metrics.csv", index=False)
    write_json(out / "clash_report.json", clash_report)
    ctx.state["mechanism"] = rows
