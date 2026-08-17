"""M03: T88-neo-terminus epitope, membrane orientation and glycan accessibility.

Computes T88-neighbourhood SASA / polarity / curvature / conformational
variability across the cleaved conformer ensemble, marks space excluded by
glycans and the membrane, and proposes hotspot residues for binder design.

Standard outputs: epitope_patch.json, hotspots.txt, exclusion_mask.json,
accessibility_metrics.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io import (
    POLAR_ELEMENTS, atom_coords, first_protein_chain, find_residue,
    polymer_residues, read_json, read_structure, residue_sasa,
    write_json,
)
from ..schemas.project import parse_residue_id

NEIGHBOUR_RADIUS = 10.0  # A around T88 defining the patch


def residue_centroid(res) -> np.ndarray:
    pts = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in res])
    return pts.mean(axis=0)


def membrane_plane(chain, residues) -> tuple[np.ndarray, float]:
    """Estimate a membrane plane from the stalk direction of the ECD.

    The ECD C-terminus is closest to the membrane.  Plane normal points from
    the domain centroid toward the last resolved residues' centroid; the
    plane offset sits slightly below them.
    """
    ca = np.array([[res.find_atom("CA", "*").pos.x,
                    res.find_atom("CA", "*").pos.y,
                    res.find_atom("CA", "*").pos.z] for res in residues
                   if res.find_atom("CA", "*") is not None])
    centroid = ca.mean(axis=0)
    tail = ca[-5:].mean(axis=0)
    normal = tail - centroid
    normal /= np.linalg.norm(normal) + 1e-12
    offset = float(np.dot(normal, tail))
    return normal, offset


def run(ctx) -> None:
    cfg = ctx.config
    out = ctx.out
    right_aa, right_num = parse_residue_id(cfg.target.cleavage.right_residue)

    states = read_json(out / "state_manifest.json") if (out / "state_manifest.json").exists() else None
    import pandas as pd
    state_df = pd.read_csv(out / "state_manifest.csv")
    cleaved = state_df[(state_df.kind == "cleaved") & (state_df.audit_passed)]
    if cleaved.empty:
        raise RuntimeError("no audited cleaved states available for epitope analysis")

    per_state_patches = []
    patch_residues: dict[int, dict] = {}

    for _, row in cleaved.iterrows():
        st = read_structure(row.file)
        model = st[0]
        # body chain contains T88 (the C-terminal fragment after cleavage)
        body = None
        frag = None
        for ch in model:
            names = {r.seqid.num for r in polymer_residues(ch)}
            if right_num in names:
                body = ch
            else:
                frag = ch
        if body is None:
            raise RuntimeError(f"T88 ({right_num}) not found in {row.file}")
        body_res = polymer_residues(body)
        t88 = find_residue(body, right_num)
        t88_c = residue_centroid(t88)

        # neighbourhood across BOTH chains (the free N-terminus sits next to
        # the disulfide-tethered fragment)
        all_res = polymer_residues(body) + (polymer_residues(frag) if frag else [])
        neigh = [r for r in all_res
                 if np.linalg.norm(residue_centroid(r) - t88_c) <= NEIGHBOUR_RADIUS]

        # SASA for the full state
        sasa_map = residue_sasa(all_res)

        normal, moffset = membrane_plane(body, body_res)

        for r in neigh:
            c = residue_centroid(r)
            polar_frac = 0.0
            n_atoms = 0
            for a in r:
                el = a.element.name if hasattr(a.element, "name") else str(a.element)
                polar_frac += 1.0 if el in POLAR_ELEMENTS else 0.0
                n_atoms += 1
            polar_frac = polar_frac / max(n_atoms, 1)
            mem_dist = float(np.dot(normal, c) - moffset)
            rec = patch_residues.setdefault(r.seqid.num, {
                "residue": r.name, "resnum": r.seqid.num,
                "chain": body.name if r in body_res else (frag.name if frag else "?"),
                "sasa": [], "polar_frac": round(polar_frac, 3),
                "centroid": c.tolist(), "membrane_dist": round(mem_dist, 2),
                "states": 0,
            })
            rec["sasa"].append(round(sasa_map.get(r.seqid.num, 0.0), 1))
            rec["states"] += 1

        per_state_patches.append({
            "state_id": row.state_id,
            "n_neighbours": len(neigh),
            "t88_centroid": t88_c.tolist(),
            "membrane_normal": normal.tolist(),
            "membrane_offset": moffset,
        })

    # aggregate: conformational variability, mean SASA
    for num, rec in patch_residues.items():
        s = rec.pop("sasa")
        rec["mean_sasa"] = round(float(np.mean(s)), 1) if s else 0.0
        rec["sasa_std"] = round(float(np.std(s)), 1) if s else 0.0
        rec["observed_states"] = len(s)

    # patch curvature: ratio of patch surface to convex-hull-ish estimate
    centroids = np.array([rec["centroid"] for rec in patch_residues.values()])
    curvature = 0.0
    if len(centroids) >= 4:
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(centroids)
            curvature = round(float(hull.volume), 1)
        except Exception:
            curvature = 0.0

    # --- hotspots: near T88, decent exposure, low variability, above membrane
    scored = []
    for num, rec in patch_residues.items():
        d_t88 = float(np.linalg.norm(np.array(rec["centroid"]) -
                                     np.array(per_state_patches[0]["t88_centroid"])))
        score = (1.0 - min(d_t88 / NEIGHBOUR_RADIUS, 1.0))
        score *= (0.4 + 0.6 * min(rec["mean_sasa"] / 120.0, 1.0))
        score *= (1.0 - min(rec["sasa_std"] / 60.0, 0.8))
        if rec["membrane_dist"] < 4.0:
            score *= 0.3  # buried against membrane
        scored.append((num, rec, round(score, 3), round(d_t88, 1)))
    scored.sort(key=lambda t: -t[2])
    hotspots = [(f"{rec['residue']}{num}", sc) for num, rec, sc, _ in scored[:12]
                if sc > 0.15]

    # --- exclusion mask: glycan spheres + membrane half-space
    # PRD v1.1: when a published target bundle exists, use its REAL grafted
    # glycan spheres (per-representative masks) instead of the 12 A
    # heuristic anchors; both stay honest for the legacy path
    glycan_sites = cfg.target.glycosylation_sites
    glycan_spheres = []
    bundle_masks_dir = out / "target_bundles" / "glycan_masks"
    if bundle_masks_dir.is_dir():
        import json as _json
        n_bundle = 0
        for mf in sorted(bundle_masks_dir.glob("*.json")):
            for entry in _json.loads(mf.read_text()):
                glycan_spheres.append({
                    "site": f"{entry.get('ccd', 'GLY')}@{entry.get('chain', '?')}",
                    "center": entry["center"],
                    "radius": entry["radius"],
                })
                n_bundle += 1
        if n_bundle:
            exclusion_note = (f"glycans: {n_bundle} spheres from grafted "
                              "target bundle states (real coordinates)")
    st0 = read_structure(cleaved.iloc[0].file)
    model0 = st0[0]
    if not glycan_spheres:   # legacy: no bundle masks -> heuristic anchors
        for ch in model0:
            for res in polymer_residues(ch):
                if res.seqid.num in glycan_sites:
                    nd2 = res.find_atom("ND2", "*")
                    anchor = (np.array([nd2.pos.x, nd2.pos.y, nd2.pos.z])
                              if nd2 is not None else residue_centroid(res))
                    glycan_spheres.append({
                        "site": f"N{res.seqid.num}",
                        "center": anchor.tolist(),
                        "radius": cfg.target.glycan_exclusion_radius,
                    })

    # normal points from the ECD centroid toward the C-terminal stalk /
    # membrane anchor; the bilayer starts BEYOND the last resolved residues
    # plus a clearance margin.  Binder atoms must stay on the domain side:
    # dot(normal, x) < cutoff_offset.
    normal = np.array(per_state_patches[0]["membrane_normal"])
    moffset = per_state_patches[0]["membrane_offset"] + cfg.target.membrane_clearance
    exclusion = {
        "membrane": {"normal": normal.tolist(), "cutoff_offset": round(moffset, 2),
                     "rule": "binder atoms must satisfy dot(normal, x) < cutoff_offset "
                             "(domain side; membrane lies along +normal)"},
        "glycan_source": locals().get(
            "exclusion_note", "glycans: heuristic 12 A spheres at sequon anchors "
            "(no target bundle in this run - legacy v1.0 path)"),
        "glycans": glycan_spheres,
        "cis_interface_residues": [],   # filled by M08 overlay analysis
        "trans_interface_residues": [],
    }

    # --- write outputs
    epitope = {
        "t88_residue": cfg.target.cleavage.right_residue,
        "patch_radius_A": NEIGHBOUR_RADIUS,
        "n_states": len(per_state_patches),
        "residues": list(patch_residues.values()),
        "patch_convex_volume_A3": curvature,
        "variability_note": "sasa_std summarises cross-conformer variability",
    }
    write_json(out / "epitope_patch.json", epitope)
    write_json(out / "exclusion_mask.json", exclusion)

    with open(out / "hotspots.txt", "w") as fh:
        fh.write("# M03 suggested hotspot residues for binder design\n")
        fh.write("# ranked by proximity x exposure x consistency; T88 always included\n")
        fh.write(f"{cfg.target.cleavage.right_residue}\n")
        _, right_num = parse_residue_id(cfg.target.cleavage.right_residue)
        for name, sc in hotspots:
            # hotspots entries are 3-letter+number (e.g. THR88); skip the
            # duplicate of the mandatory neo-N-terminus entry above
            try:
                num = int("".join(ch for ch in name if ch.isdigit()))
            except ValueError:
                num = None
            if num == right_num:
                continue
            fh.write(f"{name}\t{sc}\n")

    import pandas as pd
    rows = []
    for rec in patch_residues.values():
        rows.append({
            "residue": rec["residue"], "resnum": rec["resnum"], "chain": rec["chain"],
            "mean_sasa_A2": rec["mean_sasa"], "sasa_std_A2": rec["sasa_std"],
            "polar_fraction": rec["polar_frac"], "membrane_dist_A": rec["membrane_dist"],
            "observed_states": rec["observed_states"],
        })
    pd.DataFrame(rows).sort_values("resnum").to_csv(out / "accessibility_metrics.csv",
                                                    index=False)
    ctx.state["hotspots"] = [h[0] for h in hotspots]
    ctx.state["t88_centroid"] = per_state_patches[0]["t88_centroid"]
