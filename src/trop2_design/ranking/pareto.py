"""M10: multi-objective decision - hard gates, Pareto fronts, robust
selectivity, diversity clustering (PRD section 12).

Implements exactly the PRD 12.2 aggregate:
    robust_positive      = quantile(score_across_cleaved_states, 0.10)
    worst_offtarget      = max(score_across_intact_trop2_epcam_and_other)
    uncertainty_penalty  = lambda * std(score_across_models_seeds_states)
    robust_selectivity   = robust_positive - worst_offtarget - penalty

Hard gates are terminal (PRD 12.1): a rejected candidate can never be
rescued by a high weighted score.  Weighted display scores only order
candidates inside the same Pareto front.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalise(values: np.ndarray, direction: str) -> np.ndarray:
    """Min-max normalisation respecting metric direction; NaN-safe."""
    v = np.asarray(values, dtype=float)
    if np.all(np.isnan(v)):
        return v
    lo, hi = np.nanmin(v), np.nanmax(v)
    if hi - lo < 1e-12:
        return np.where(np.isnan(v), np.nan, 1.0)
    n = (v - lo) / (hi - lo)
    if direction == "minimize":
        n = 1.0 - n
    return n


def non_dominated_sort(points: np.ndarray) -> list[int]:
    """NSGA-II style front indices for NaN-free objective matrix (maximised).

    points: (n_candidates, n_objectives) all to be maximised.
    Returns list of front rank per candidate (0 = best front).
    """
    n = len(points)
    fronts = np.full(n, -1)
    dominated_by: list[set[int]] = [set() for _ in range(n)]
    dominates_count = np.zeros(n, dtype=int)
    rank = 0
    assigned = 0
    pending = set(range(n))
    while pending and assigned < n:
        current: list[int] = []
        for i in pending:
            dominated = False
            for j in pending:
                if i == j:
                    continue
                if _dominates(points[j], points[i]):
                    dominated = True
                    break
            if not dominated:
                current.append(i)
        if not current:  # numerical safety
            current = list(pending)
        for i in current:
            fronts[i] = rank
        pending -= set(current)
        assigned += len(current)
        rank += 1
    return fronts.tolist()


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Pareto dominance: a >= b everywhere and a > b somewhere."""
    return bool(np.all(a >= b) and np.any(a > b))


def greedy_cluster(sequences: list[str], identity_threshold: float) -> list[int]:
    """Greedy centroid clustering by pairwise identity; returns cluster id."""
    clusters: list[list[int]] = []
    for i, seq in enumerate(sequences):
        placed = False
        for c_id, members in enumerate(clusters):
            rep = sequences[members[0]]
            if pairwise_identity(rep, seq) >= identity_threshold:
                members.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    out = [0] * len(sequences)
    for c_id, members in enumerate(clusters):
        for m in members:
            out[m] = c_id
    return out


def pairwise_identity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / max(len(a), len(b))


def apply_gates(row: dict, gates) -> tuple[str, list[str]]:
    """Evaluate hard gates; returns (status, reasons)."""
    reasons = []
    for gate in gates:
        value = row.get(gate.metric)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "review", [f"{gate.metric}: missing -> review (never zero)"]
        if gate.op == ">=" and not float(value) >= float(gate.threshold):
            reasons.append(f"{gate.reject_message or gate.metric} "
                           f"({gate.metric}={value} < {gate.threshold})")
        elif gate.op == "<=" and not float(value) <= float(gate.threshold):
            reasons.append(f"{gate.reject_message or gate.metric} "
                           f"({gate.metric}={value} > {gate.threshold})")
        elif gate.op == "==" and bool(value) != bool(gate.threshold):
            reasons.append(f"{gate.reject_message or gate.metric} "
                           f"({gate.metric}={value} != {gate.threshold})")
        elif gate.op == "exists" and (value is None or value == "" or
                                      (isinstance(value, float) and np.isnan(value))):
            reasons.append(f"{gate.metric} missing")
    if reasons:
        return "reject", reasons
    return "pass", []


def robust_selectivity(positive_scores: np.ndarray, negative_scores: np.ndarray,
                       lambda_pen: float, quantile: float) -> dict:
    """PRD 12.2 robust aggregate."""
    robust_positive = float(np.quantile(positive_scores, quantile))
    worst_offtarget = float(np.max(negative_scores)) if len(negative_scores) else 0.0
    uncertainty = float(np.std(np.concatenate([positive_scores, negative_scores]))) \
        if len(negative_scores) else float(np.std(positive_scores))
    penalty = lambda_pen * uncertainty
    return {
        "robust_positive": round(robust_positive, 4),
        "worst_offtarget": round(worst_offtarget, 4),
        "uncertainty_penalty": round(penalty, 4),
        "robust_selectivity": round(robust_positive - worst_offtarget - penalty, 4),
    }
