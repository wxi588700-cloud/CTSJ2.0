"""M10 stage: assemble candidate metrics, run gates + Pareto + diversity
clustering, export shortlist and the HTML report.

Standard outputs: candidate_metrics.csv, pareto_front.csv,
rejection_reasons.csv, top_candidates/, report.html.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ..io import write_json
from ..schemas.metrics import (
    HardFilterProfile, MetricsProfile, default_metrics_profile, v1_strict_profile,
)
from ..reporting.html import render_report
from .pareto import (
    apply_gates, greedy_cluster, non_dominated_sort, normalise,
    pairwise_identity, robust_selectivity,
)


def _collect(out: Path) -> pd.DataFrame:
    """One row per (candidate_id, design_name) with all module metrics."""
    pos = pd.read_csv(out / "positive_state_metrics.csv")
    agg = pos[pos.state_id == "AGGREGATE"]
    neg = pd.read_csv(out / "negative_state_metrics.csv")
    mech = pd.read_csv(out / "mechanism_metrics.csv")
    dev = pd.read_csv(out / "developability_metrics.csv")
    mono = pd.read_csv(out / "monomer_metrics.csv")
    try:
        cand_man = pd.read_csv(out / "candidate_manifest.csv")
    except Exception:
        cand_man = pd.DataFrame()

    rows = []
    # PRD v1.1 AC-26: outputs carry the target bundle id when present
    bundle_id = ""
    mf = out / "target_bundles" / "manifest.json"
    if mf.exists():
        import json as _json
        try:
            bundle_id = _json.loads(mf.read_text(encoding="utf-8")).get("target_bundle_id", "")
        except Exception:
            bundle_id = ""
    neg_worst = neg[neg.negative_state == "WORST"].set_index(["candidate_id", "design_name"])
    for _, r in agg.iterrows():
        key = (r.candidate_id, r.design_name)
        row = {
            "candidate_id": r.candidate_id,
            "design_name": r.design_name,
            "target_bundle_id": bundle_id,
            "positive_state_pass_rate": r.get("positive_state_pass_rate"),
            "t88_contact": bool(r.get("t88_contact_occupancy", 0) > 0),
            "t88_contact_occupancy": r.get("t88_contact_occupancy"),
            "complex_iptm": r.get("robust_positive"),
            "glycoform_coverage": r.get("glycoform_coverage"),
            "uncertainty": r.get("uncertainty_positive"),
        }
        if key in neg_worst.index:
            nw = neg_worst.loc[key]
            row["intact_risk"] = nw.get("risk")
            row["epcam_risk"] = nw.get("epcam_risk")
        m = mech[(mech.candidate_id == r.candidate_id) & (mech.design_name == r.design_name)]
        if not m.empty:
            row.update({
                "cis_block": m.iloc[0].cis_block,
                "trans_occlusion": m.iloc[0].trans_occlusion,
                "glycan_membrane_clash": float(m.iloc[0].glycan_membrane_clash),
            })
        d = dev[(dev.candidate_id == r.candidate_id) & (dev.design_name == r.design_name)]
        if not d.empty:
            row.update({
                "aggregation_risk": d.iloc[0].aggregation_risk,
                "solubility_score": d.iloc[0].solubility_score,
                "mhc2_risk": d.iloc[0].mhc2_risk,
                "liability_count": float(d.iloc[0].liability_count),
                "developability_flags": d.iloc[0].developability_flags,
                "mw_da": d.iloc[0].mw_da,
                "pI": d.iloc[0].pI,
                "net_charge": d.iloc[0]["net_charge_pH7.4"],
            })
        mm = mono[(mono.candidate_id == r.candidate_id) & (mono.design_name == r.design_name)]
        if not mm.empty:
            row["fold_plddt"] = mm.iloc[0].fold_plddt
            row["sequence"] = mm.iloc[0].sequence
        # audit fix v2 (external review P0): per-metric provenance columns -
        # a single row-level "measured" label hid that negatives/mechanism/
        # developability remain geometric proxies.
        pos_src = r.get("metric_source")
        if pos_src is None or (isinstance(pos_src, float) and pd.isna(pos_src)):
            # AGGREGATE rows may lack metric_source: infer from per-state rows
            _ps = pos[(pos.candidate_id == r.candidate_id) &
                      (pos.design_name == r.design_name)]
            pos_src = ("measured" if (_ps.metric_source == "measured").any()
                       else "proxy") if not _ps.empty else "proxy"
        pos_src = str(pos_src)
        mono_src = (str(mm.iloc[0].metric_source)
                    if (not mm.empty and "metric_source" in mm.columns) else "proxy")
        # AF2 (ColabDesign) self-reported design metrics for M04b candidates
        if not cand_man.empty:
            cm_row = cand_man[cand_man.candidate_id == r.candidate_id]
            if not cm_row.empty:
                for col in ("af2_plddt", "af2_iptm", "af2_ipae"):
                    v = cm_row.iloc[0].get(col)
                    row[col] = None if pd.isna(v) else v
        row["af2_source"] = ("colabdesign-af2(self-report)"
                             if row.get("af2_iptm") is not None else "")
        row["complex_iptm_source"] = pos_src
        row["fold_plddt_source"] = mono_src
        row["intact_risk_source"] = "proxy"     # M07 geometric pose transfer
        row["epcam_risk_source"] = "proxy"      # M07 patch proxy
        row["mechanism_source"] = "geometry"    # M08 superposition
        row["mhc2_source"] = "proxy"            # heuristic until real adapter
        row["metric_source"] = ("measured" if "measured" in (pos_src, mono_src)
                                else "proxy")  # row-level: ANY measured part
        rows.append(row)
    return pd.DataFrame(rows)


PARETO_OBJECTIVES = [
    # (column, direction) - hard-gate survivors only, PRD 12.2 objectives
    ("robust_selectivity", "maximize"),
    ("t88_contact_occupancy", "maximize"),
    ("cis_block", "maximize"),
    ("solubility_score", "maximize"),
    ("fold_plddt_norm", "maximize"),
]


def _load_profiles(ranking_cfg):
    """Load versioned decision-model profiles.

    PRD appendix A: thresholds MUST live in versioned configuration - any
    change produces a new ranking_profile_id.  We prefer the YAML profiles
    shipped under models/ (the auditable "decision model" of this project)
    and fall back to the identical in-code defaults when absent.
    """
    root = Path(__file__).resolve().parents[3]
    gate_yaml = root / "models" / f"hard_filter_{ranking_cfg.hard_filter_profile}.yaml"
    metric_yaml = root / "models" / f"{ranking_cfg.metrics_profile}.yaml"
    profile = metrics_profile = None
    try:
        if gate_yaml.exists():
            profile = HardFilterProfile.from_yaml(gate_yaml)
    except Exception:
        profile = None
    try:
        if metric_yaml.exists():
            import yaml as _yaml

            with open(metric_yaml) as fh:
                metrics_profile = MetricsProfile.model_validate(
                    _yaml.safe_load(fh))
    except Exception:
        metrics_profile = None
    if profile is None:
        profile = v1_strict_profile()
    if metrics_profile is None:
        metrics_profile = default_metrics_profile()
    return profile, metrics_profile


def run(ctx) -> None:
    cfg = ctx.config
    out = ctx.out
    df = _collect(out)
    if df.empty:
        raise RuntimeError("no candidates reached M10")

    profile, metrics_profile = _load_profiles(cfg.ranking)
    lambda_pen = cfg.ranking.uncertainty_lambda
    quantile = cfg.ranking.robust_positive_quantile

    # ---- PRD 12.2 robust aggregates per candidate
    pos = pd.read_csv(out / "positive_state_metrics.csv")
    for i, r in df.iterrows():
        per_state = pos[(pos.candidate_id == r.candidate_id) &
                        (pos.design_name == r.design_name) &
                        (pos.state_id != "AGGREGATE")]
        pos_scores = per_state["complex_iptm_proxy"].to_numpy(dtype=float) \
            if "complex_iptm_proxy" in per_state else np.array([r.get("complex_iptm") or 0.0])
        neg_scores = np.array([x for x in [r.get("intact_risk"), r.get("epcam_risk")]
                               if x is not None and not pd.isna(x)], dtype=float)
        rob = robust_selectivity(pos_scores, neg_scores, lambda_pen, quantile)
        df.at[i, "robust_positive"] = rob["robust_positive"]
        df.at[i, "worst_offtarget"] = rob["worst_offtarget"]
        df.at[i, "uncertainty_penalty"] = rob["uncertainty_penalty"]
        df.at[i, "robust_selectivity"] = rob["robust_selectivity"]

    # ---- hard gates (terminal; AC-12 irreversibility)
    statuses, reasons_all = [], []
    for _, r in df.iterrows():
        status, reasons = apply_gates(r.to_dict(), profile.gates)
        statuses.append(status)
        reasons_all.append(reasons)
    df["hard_filter_status"] = statuses
    df["rejection_reasons"] = ["; ".join(x) for x in reasons_all]

    # PRD v1.1 12.1 (glyco runs only): binding confined to a single glycoform
    # / conformation is terminal - cannot be rescued by weighted scores
    bundle_id_col = str(df["target_bundle_id"].iloc[0]) \
        if "target_bundle_id" in df.columns and len(df) else ""
    if bundle_id_col not in ("", "nan", "None"):
        for idx, r in df.iterrows():
            cov = r.get("glycoform_coverage")
            if cov is None or pd.isna(cov):
                continue
            if float(cov) < 0.34:   # below one of three panels
                if df.at[idx, "hard_filter_status"] == "pass":
                    df.at[idx, "hard_filter_status"] = "reject"
                base = str(r.rejection_reasons)
                extra = (f"cross-glycoform robustness insufficient "
                         f"(glycoform_coverage={cov} < 0.34)")
                df.at[idx, "rejection_reasons"] = (
                    base + "; " + extra if base not in ("", "nan") else extra)

    # ---- normalised objectives + Pareto among gate survivors
    # audit fix: defensive column init - with zero gate survivors (or an
    # empty monomer table) these columns must still EXIST, otherwise every
    # downstream consumer (report, predict.py) KeyErrors on them
    for _c in ("fold_plddt", "pareto_rank", "weighted_display_score",
               "robust_positive", "worst_offtarget", "uncertainty_penalty",
               "robust_selectivity"):
        if _c not in df.columns:
            df[_c] = np.nan
    df["fold_plddt_norm"] = normalise(
        df["fold_plddt"].astype(float).to_numpy(), "maximize") * 100.0
    df["pareto_rank"] = np.nan            # filled below for survivors only
    # borrowed (optimized build): formal Pareto fronts contain ONLY fully
    # passed candidates - review rows carry missing data and polluting the
    # fronts (or the normalisation) with fillna(0) values misranks them
    survivors = df[df.hard_filter_status == "pass"].copy()
    # objectives whose columns are missing (e.g. empty mechanism table after
    # the measured predictor rejected every design) are skipped, never zero-
    # filled silently
    available_objectives = [(c, d) for c, d in PARETO_OBJECTIVES
                             if c in survivors.columns]
    if not survivors.empty and available_objectives:
        pts = np.column_stack([
            survivors[col].astype(float).fillna(0.0).to_numpy()
            for col, _ in available_objectives
        ])
        fronts = non_dominated_sort(pts)
        survivors["pareto_rank"] = fronts
        df.loc[survivors.index, "pareto_rank"] = survivors["pareto_rank"]

        # crowding-distance tiebreak inside each front for determinism
        survivors = survivors.sort_values(
            ["pareto_rank", "robust_selectivity", "candidate_id"],
            ascending=[True, False, True])
        # weighted display score (only within-front ordering, PRD 12.3)
        for group_col in ["complex_iptm"]:
            pass
        w_map = {}
        for wg in profile.display_weights:
            for m in wg.metrics:
                w_map[m] = wg.weight / max(len(wg.metrics), 1)
        for idx, r in survivors.iterrows():
            total, weight = 0.0, 0.0
            for metric, w in w_map.items():
                if metric in df.columns and not pd.isna(r.get(metric)):
                    col = df[metric].astype(float)
                    direction = metrics_profile.metric(metric).direction \
                        if metrics_profile.metric(metric) else "maximize"
                    norm = normalise(col.to_numpy(), direction)
                    pos_in_col = df.columns.get_loc(metric)
                    total += w * norm[df.index.get_loc(idx)]
                    weight += w
            df.at[idx, "weighted_display_score"] = round(total / weight, 4) if weight else None

    # ---- diversity clustering (sequence identity, PRD 0.70)
    seqs = df.sequence.fillna("").tolist()
    df["family_cluster"] = greedy_cluster(seqs, cfg.ranking.diversity_cluster_identity)

    # ---- export
    df.to_csv(out / "candidate_metrics.csv", index=False)
    front = df[df.hard_filter_status == "pass"].sort_values(
        ["pareto_rank", "weighted_display_score"],
        ascending=[True, False], na_position="last")
    front.to_csv(out / "pareto_front.csv", index=False)

    rej = df[df.hard_filter_status == "reject"][
        ["candidate_id", "design_name", "rejection_reasons"]]
    rej.to_csv(out / "rejection_reasons.csv", index=False)

    # ---- top candidates with family cap
    top_dir = out / "top_candidates"
    top_dir.mkdir(parents=True, exist_ok=True)
    seen_families: dict[int, int] = {}
    top_rows = []
    for _, r in front.iterrows():
        fam = int(r.family_cluster)
        if seen_families.get(fam, 0) >= cfg.ranking.max_per_family:
            continue
        seen_families[fam] = seen_families.get(fam, 0) + 1
        top_rows.append(r)
        if len(top_rows) >= cfg.ranking.export_top_n:
            break
    top = pd.DataFrame(top_rows)
    if not top.empty:
        top.to_csv(top_dir / "top_candidates.csv", index=False)
        # copy structures + sequences
        for _, r in top.iterrows():
            cand_dir = top_dir / r.candidate_id
            cand_dir.mkdir(exist_ok=True)
            with open(cand_dir / "sequence.txt", "w") as fh:
                fh.write(f">{r.candidate_id}_{r.design_name}\n{r.sequence}\n")
            for pattern in [f"{r.candidate_id}_*.cif"]:
                for src in (out / "complexes" / "positive").glob(pattern):
                    shutil.copy2(src, cand_dir / src.name)

    # ---- HTML report (AC-17)
    report = render_report(out, df, top, profile, metrics_profile, ctx)
    (out / "report.html").write_text(report, encoding="utf-8")

    ctx.state["ranked"] = df.to_dict(orient="records")
