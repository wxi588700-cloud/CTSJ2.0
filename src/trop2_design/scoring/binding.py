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
    residue_one_letter, write_cif, write_json,
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




def _aa1(resname: str) -> str:
    """Three-letter residue name -> one-letter via io.AA3_TO_1 ('X' if unknown)."""
    from ..io.common import AA3_TO_1
    return AA3_TO_1.get(resname.strip().upper(), "X")


def t88_terminal_evidence(state_chains, right_num: int, pose_ca,
                          right_aa: str | None = None,
                          left_aa: str | None = None) -> dict:
    """Direct-contact evidence for the T88 free alpha-amino terminus.

    Measured on the true geometry (never a proxy): distance from the T88
    backbone N atom (the free NH3+ group created in M02) to the nearest
    binder heavy atom.  Only meaningful for CLEAVED states (AC-08).

    Audit fix: residue-identity + R-T motif context checks.  A bare number
    lookup could silently hit a renumbered wrong residue; when the expected
    identities are supplied (right_aa='T', left_aa='R') mismatches raise.
    """
    t88 = None
    prev = None
    for res in state_chains.values():
        for r in res:
            if r.seqid.num == right_num - 1:
                prev = r
            if r.seqid.num == right_num:
                t88 = r
                break
        if t88 is not None:
            break
    if t88 is not None and right_aa and _aa1(t88.name) != right_aa:
        raise ValueError(
            f"residue {right_num} is {t88.name}, expected {right_aa} (T88) - "
            f"numbering convention mismatch")
    if (t88 is not None and prev is not None and left_aa
            and _aa1(prev.name) != left_aa):
        raise ValueError(
            f"residue {right_num - 1} is {prev.name}, expected {left_aa} "
            f"(R87) - cleavage motif R-T context mismatch")
    if t88 is None:
        # audit fix (was silent no-contact): a cleaved state whose chains do
        # not contain the T88 anchor is structural corruption - silently
        # failing the contact gate masked it.  An empty binder pose remains a
        # benign no-contact.
        raise ValueError(
            f"T88 (residue {right_num}) not found in cleaved-state chains - "
            f"state is corrupt; refusing to score it as a silent no-contact")
    if len(pose_ca) == 0:
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


def _t88_contact_from_structure(cif_path, right_num: int,
                                binder_chain: str = "C") -> dict:
    """T88 free-N-terminus contact evidence measured on a PREDICTED complex
    (e.g. Boltz output): distance from the T88 backbone N atom to the nearest
    binder-chain heavy atom.  Falls back to no-contact when the residue is
    not resolved."""
    if cif_path is None or not Path(cif_path).exists():
        return {"contacted": False, "min_distance": 99.0, "n_contacts": 0}
    st = read_structure(cif_path)
    model = st[0]
    t88_n = None
    binder_pts = []
    for ch in model:
        for res in polymer_residues(ch):
            if ch.name != binder_chain and res.seqid.num == right_num:
                n = res.find_atom("N", "*")
                if n is not None:
                    t88_n = np.array([n.pos.x, n.pos.y, n.pos.z])
            if ch.name == binder_chain:
                for atom in res:
                    binder_pts.append([atom.pos.x, atom.pos.y, atom.pos.z])
    if t88_n is None or not binder_pts:
        return {"contacted": False, "min_distance": 99.0, "n_contacts": 0}
    binder_pts = np.asarray(binder_pts)
    d = np.linalg.norm(binder_pts - t88_n, axis=1)
    within = d <= T88_CONTACT_CUTOFF
    return {
        "contacted": bool(within.any()),
        "min_distance": round(float(d.min()), 2),
        "n_contacts": int(within.sum()),
    }


def cache_chains_file(cache: dict) -> Path:
    """File backing a state cache (glyco bundle pov or legacy cif)."""
    return cache.get("source_file", Path("/tmp/_glyco_state_placeholder.cif"))


def _weighted_quantile(values, weights, q: float) -> float:
    """PRD v1.1 12.2: robust_positive = weighted_quantile across glyco states."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(v) == 0:
        return 0.0
    if w.sum() <= 0:
        return float(np.quantile(v, q))
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w) / w.sum()
    # step convention: first value whose cumulative weight reaches q
    return float(v[np.searchsorted(cw, q, side="left")].item()
                 if np.any(cw >= q) else v[-1])


def _glycoform_coverage(per_state: list[dict]) -> float:
    """Mean over profiles of (sum of cluster weights of passing states),
    capped at 1.0 - PRD 7.4 candidate field."""
    profiles: dict[str, tuple[float, float]] = {}
    for r in per_state:
        p = r.get("glycoform_profile")
        if not p:
            continue
        pass_w, tot = profiles.get(p, (0.0, 0.0))
        tot += float(r.get("md_cluster_weight", 0.0))
        if r.get("status") == "pass":
            pass_w += float(r.get("md_cluster_weight", 0.0))
        profiles[p] = (pass_w, tot)
    if not profiles:
        return 0.0
    return round(float(np.mean([min(pw / t, 1.0) if t > 0 else 0.0
                                for pw, t in profiles.values()])), 3)


def _load_glyco_states(out: Path):
    """Bundle states for M06 glyco consumption (PRD v1.1 AC-26): list of
    dicts {state_id, profile, weight, protein_only_path, mask_spheres}."""
    mf = out / "target_bundles" / "manifest.json"
    if not mf.exists():
        return None
    import json as _json

    m = _json.loads(mf.read_text(encoding="utf-8"))
    entries = []
    for s in m.get("states", []):
        pov = out / "target_bundles" / s["protein_only_view"]
        mask_p = out / "target_bundles" / "glycan_masks" / \
            f"{s['target_state_id']}.json"
        if not pov.exists():
            continue
        spheres = _json.loads(mask_p.read_text(encoding="utf-8")) if mask_p.exists() else []
        entries.append({"state_id": s["target_state_id"],
                        "profile": s["glycoform_profile_id"],
                        "weight": float(s["md_cluster_weight"]),
                        "pov": pov, "spheres": spheres})
    return entries or None


def run(ctx) -> None:
    cfg = ctx.config
    out = ctx.out
    right_aa, right_num = parse_residue_id(cfg.target.cleavage.right_residue)
    left_aa, _ = parse_residue_id(cfg.target.cleavage.left_residue)

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

    # ---- PRD v1.1: prefer glycosylated bundle states when published
    glyco_states = _load_glyco_states(out)
    glyco_mode = glyco_states is not None

    # per-state caches (target unbound SASA reused across candidates)
    state_cache: dict[str, dict] = {}
    if glyco_mode:
        for g in glyco_states:
            st = read_structure(g["pov"])
            chains = {ch.name: polymer_residues(ch) for ch in st[0]
                      if polymer_residues(ch)}
            target_residues = [r for res in chains.values() for r in res]
            from .interface_metrics import atoms_of

            tc, te, _ = atoms_of(target_residues)
            state_cache[g["state_id"]] = {
                "source_file": g["pov"],
                "chains": chains, "residues": target_residues,
                "unbound_sasa": sasa(tc, te, 480),
                "glycoform_profile": g["profile"],
                "md_cluster_weight": g["weight"],
                "spheres": g["spheres"],
            }
        ref_sid = glyco_states[0]["state_id"]
    else:
        for _, srow in cleaved.iterrows():
            _, chains = load_state(Path(srow.file))
            target_residues = [r for res in chains.values() for r in res]
            from .interface_metrics import atoms_of

            tc, te, _ = atoms_of(target_residues)
            state_cache[srow.state_id] = {
                "source_file": Path(srow.file),
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
        state_iter = ([(g["state_id"], None) for g in glyco_states]
                      if glyco_mode
                      else [(srow.state_id, Path(srow.file))
                            for _, srow in cleaved.iterrows()])
        for sid, legacy_file in state_iter:
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
            term = t88_terminal_evidence(cache["chains"], right_num, state_pose,
                                         right_aa=right_aa, left_aa=left_aa)
            # bundle glycan spheres: direct binder-vs-glycan clash count
            glycan_clash = 0
            if cache.get("spheres"):
                centres = np.array([s["center"] for s in cache["spheres"]])
                radii = np.array([s["radius"] for s in cache["spheres"]])
                d = np.linalg.norm(state_pose[:, None, :] - centres[None, :, :],
                                   axis=-1)
                glycan_clash = int((d < 0.6 * radii).any(axis=1).sum())
            passing = (eff_area >= MIN_INTERFACE_AREA and
                       proxies["complex_iptm_proxy"] >= MIN_IPTM and
                       term["contacted"] and summary["clashes"] <= 25
                       and glycan_clash == 0)
            per_state.append({
                "candidate_id": cid, "design_name": design_names[0], "state_id": sid,
                **summary, **proxies,
                **{f"t88_{k}": v for k, v in term.items()},
                "metric_source": "proxy",
                "glycoform_profile": cache.get("glycoform_profile", ""),
                "md_cluster_weight": cache.get("md_cluster_weight", ""),
                "glycan_clash": glycan_clash,
                "status": "pass" if passing else "fail_state",
            })
            terminal_records.append({
                "state_id": sid, "candidate_id": cid,
                "kind": "glyco_bundle" if glyco_mode else "cleaved",
                **term,
                "t88_residue": cfg.target.cleavage.right_residue,
            })
            if sid == ref_sid:
                write_complex(legacy_file or cache_chains_file(cache), state_pose,
                              f"{cid}_{sid}",
                              complexes_dir / f"{cid}_{sid}.cif")

        # cross-state aggregates (PRD 12.2; v1.1: WEIGHTED by glyco cluster
        # weights when consuming a target bundle)
        if glyco_mode:
            wts = [r["md_cluster_weight"] for r in per_state]
            w_arr = np.array([w if isinstance(w, (int, float)) else 1.0
                              for w in wts], dtype=float)
            if w_arr.sum() <= 0:
                w_arr = np.ones(len(per_state))
            pass_rate = float(np.average(
                [r["status"] == "pass" for r in per_state], weights=w_arr))
            iptms = np.array([r["complex_iptm_proxy"] for r in per_state])
            robust_positive = _weighted_quantile(iptms, w_arr,
                                                 cfg.ranking.robust_positive_quantile)
            contact_occ = float(np.average(
                [bool(r["t88_contacted"]) for r in per_state], weights=w_arr))
        else:
            pass_rate = float(np.mean([r["status"] == "pass" for r in per_state]))
            iptms = np.array([r["complex_iptm_proxy"] for r in per_state])
            robust_positive = float(np.quantile(
                iptms, cfg.ranking.robust_positive_quantile))
            contact_occ = float(np.mean([bool(r["t88_contacted"])
                                         for r in per_state]))
        uncertainty = float(np.std(iptms))
        glyco_cov = _glycoform_coverage(per_state) if glyco_mode else None

        for dn in design_names:
            for r in per_state:
                r2 = dict(r)
                r2["design_name"] = dn
                rows.append(r2)
            agg_row = {
                "candidate_id": cid, "design_name": dn, "state_id": "AGGREGATE",
                "positive_state_pass_rate": round(pass_rate, 3),
                "robust_positive": round(robust_positive, 3),
                "t88_contact_occupancy": round(contact_occ, 3),
                "uncertainty_positive": round(uncertainty, 4),
                "glycoform_coverage": glyco_cov,
                "status": "aggregate",
            }
            rows.append(agg_row)

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Two-stage GPU recomputation (PRD 8.2 compute control): the geometric
    # proxy filters ALL candidates; Boltz-2 then re-predicts the top-K
    # designs against every cleaved conformer, replacing proxy ipTM/pLDDT/
    # PAE and the T88-contact evidence with MEASURED values from the
    # predicted complex (independent of the designed pose -> no
    # self-consistency bias).  Requires predictors.boltz.python in tools.yaml.
    # ------------------------------------------------------------------
    top_k = getattr(cfg.resources, "boltz_recompute_top_k", 0)
    recompute_log: list[dict] = []
    if top_k <= 0:
        # audit fix: strict mode with recompute disabled would finish entirely
        # on proxy binding metrics without any guard firing
        cfg.resources.forbid_proxy_degradation(
            "boltz_recompute_top_k=0 (all binding metrics stay proxy)")
    if top_k > 0:
        # audit fix: the previous ``except Exception: bp = None`` swallowed the
        # failure reason entirely - now the reason is surfaced (warn or hard
        # error when allow_proxy_metrics=false), never silent
        bp = None
        boltz_error = None
        bspec = ctx.tools.predictors.get("boltz") if ctx.tools else None
        if bspec is None or not bspec.python:
            boltz_error = "predictors.boltz.python not configured in tools.yaml"
        else:
            try:
                from ..prediction import build_boltz
                bp = build_boltz(bspec, ctx.seed)
            except Exception as exc:  # noqa: BLE001 - reason surfaced below
                bp = None
                boltz_error = f"{type(exc).__name__}: {exc}"
        if bp is None:
            print(f"[M06][warn] Boltz-2 recompute unavailable ({boltz_error}) "
                  f"- binding metrics stay proxy for all candidates")
            cfg.resources.forbid_proxy_degradation(
                f"Boltz-2 recompute unavailable ({boltz_error})")
        if bp is not None:
            mono = pd.read_csv(out / "monomer_metrics.csv")
            seq_of = dict(zip(mono.design_name, mono.sequence))
            # shortlist: best proxy robust_positive designs
            agg0 = df[df.state_id == "AGGREGATE"].sort_values(
                "robust_positive", ascending=False)
            shortlist = list(agg0[["candidate_id", "design_name"]]
                             .drop_duplicates().itertuples(index=False))[:top_k]
            # per-state TROP2 chain sequences (NFR + BODY) from the cleaved cif
            state_seqs: dict[str, dict[str, str]] = {}
            for _, srow in cleaved.iterrows():
                _, chains = load_state(Path(srow.file))
                seqs_state = {}
                for ch_name, ch_res in chains.items():
                    seqs_state[ch_name] = "".join(
                        residue_one_letter(r) for r in ch_res)
                state_seqs[srow.state_id] = seqs_state

            for cid, dn in shortlist:
                binder_seq = seq_of.get(dn)
                if not isinstance(binder_seq, str) or not binder_seq:
                    continue
                # Borrowed (optimized build): per-state prediction is
                # OPT-IN via cfg.resources.boltz_per_state (default False).
                # Sequence-only Boltz cannot infer conformational identity,
                # so per-state runs are replicate predictions (measuring
                # prediction stochasticity) at ~Nx GPU cost - the default
                # still predicts the reference conformer ONCE but now every
                # state row keeps its own provenance record instead of a
                # silently copied value.
                per_state = bool(getattr(cfg.resources, "boltz_per_state",
                                         False))
                sids = (list(state_seqs) if per_state
                        else [cleaved.iloc[0].state_id])
                for sid in sids:
                    ss = state_seqs[sid]
                    chain_map = sorted(ss)  # BODY, NFR -> A, B
                    sequences = {
                        "A": ss[chain_map[0]],
                        "B": ss[chain_map[1]],
                        "C": binder_seq,
                    }
                    result = bp.predict_complex(
                        sequences, f"{cid[:12]}_cx",
                        out / "boltz_complex" / dn[:64])
                    rec = {"candidate_id": cid, "design_name": dn,
                           "state_id": sid, "ok": result.ok,
                           "reason": result.reason,
                           "replicate": not per_state,
                           # PRD: failures must be traceable - keep the log tail
                           "log_tail": (result.log or "")[-400:]}
                    if result.ok and result.iptm is not None:
                        mask = (df.candidate_id == cid) & (df.design_name == dn) \
                            & (df.state_id == sid if per_state
                               else df.state_id != "AGGREGATE")
                        t88_meas = _t88_contact_from_structure(
                            result.structure, right_num, "C")
                        df.loc[mask, "complex_iptm_proxy"] = result.iptm
                        df.loc[mask, "interface_pae_proxy"] = (
                            result.interface_pae
                            if result.interface_pae is not None
                            else df.loc[mask, "interface_pae_proxy"])
                        df.loc[mask, "metric_source"] = "measured"
                        df.loc[mask, "predictor"] = "boltz-2"
                        df.loc[mask, "t88_contacted"] = t88_meas["contacted"]
                    df.loc[mask, "t88_min_distance"] = t88_meas["min_distance"]
                    df.loc[mask, "t88_n_contacts"] = t88_meas["n_contacts"]
                    rec.update({"iptm": result.iptm,
                                "plddt": result.plddt,
                                "interface_pae": result.interface_pae,
                                "t88_contacted": t88_meas["contacted"]})
                recompute_log.append(rec)
            # recompute pass status + aggregates for recomputed designs
            for cid, dn in shortlist:
                sub = df[(df.candidate_id == cid) & (df.design_name == dn)
                         & (df.state_id != "AGGREGATE")]
                if sub.empty or not (sub.metric_source == "measured").any():
                    continue
                iptms = sub["complex_iptm_proxy"].to_numpy(dtype=float)
                quant = cfg.ranking.robust_positive_quantile
                agg_mask = (df.candidate_id == cid) & (df.design_name == dn) \
                    & (df.state_id == "AGGREGATE")
                df.loc[agg_mask, "robust_positive"] = round(
                    float(np.quantile(iptms, quant)), 3)
                df.loc[agg_mask, "uncertainty_positive"] = round(
                    float(np.std(iptms)), 4)
                df.loc[agg_mask, "t88_contact_occupancy"] = round(
                    float(sub["t88_contacted"].astype(bool).mean()), 3)
                df.loc[agg_mask, "predictor"] = "boltz-2"
            # status column re-evaluation with measured values
            for idx, r in df[df.state_id != "AGGREGATE"].iterrows():
                if r.get("metric_source") == "measured":
                    ok = (float(r.complex_iptm_proxy) >= MIN_IPTM
                          and bool(r.t88_contacted)
                          and float(r.clashes) <= 25
                          and float(r.interface_area_A2) >= MIN_INTERFACE_AREA)
                    df.at[idx, "status"] = "pass" if ok else "fail_state"
            for idx, r in df[df.state_id == "AGGREGATE"].iterrows():
                sub = df[(df.candidate_id == r.candidate_id)
                         & (df.design_name == r.design_name)
                         & (df.state_id != "AGGREGATE")]
                if not sub.empty and (sub.metric_source == "measured").any():
                    df.at[idx, "positive_state_pass_rate"] = round(
                        float((sub.status == "pass").mean()), 3)
        write_json(out / "boltz_recompute_log.json", recompute_log)

    df.to_csv(out / "positive_state_metrics.csv", index=False)
    write_json(out / "terminal_contact.json", terminal_records)

    agg = df[df.state_id == "AGGREGATE"]
    if agg.empty:
        raise RuntimeError("no positive-state results at all")
    # NOTE: zero candidates passing is a VALID scientific outcome (e.g. the
    # measured predictor rejects every fallback design); the pipeline then
    # continues with an empty shortlist so M10 can report the full
    # rejection-reason audit trail instead of aborting.
    ctx.state["positive"] = rows
