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
from ..schemas.metrics import default_metrics_profile, v1_strict_profile
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

    rows = []
    neg_worst = neg[neg.negative_state == "WORST"].set_index(["candidate_id", "design_name"])
    for _, r in agg.iterrows():
        key = (r.candidate_id, r.design_name)
        row = {
            "candidate_id": r.candidate_id,
            "design_name": r.design_name,
            "positive_state_pass_rate": r.get("positive_state_pass_rate"),
            "t88_contact": bool(r.get("t88_contact_occupancy", 0) > 0),
            "t88_contact_occupancy": r.get("t88_contact_occupancy"),
            "complex_iptm": r.get("robust_positive"),
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
            row["metric_source"] = "proxy"
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


def run(ctx) -> None:
    cfg = ctx.config
    out = ctx.out
    df = _collect(out)
    if df.empty:
        raise RuntimeError("no candidates reached M10")

    profile = v1_strict_profile()
    metrics_profile = default_metrics_profile()
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

    # ---- normalised objectives + Pareto among gate survivors
    df["fold_plddt_norm"] = normalise(df["fold_plddt"].to_numpy(), "maximize") * 100.0
    survivors = df[df.hard_filter_status != "reject"].copy()
    if not survivors.empty:
        pts = np.column_stack([
            survivors[col].astype(float).fillna(0.0).to_numpy()
            for col, _ in PARETO_OBJECTIVES
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
