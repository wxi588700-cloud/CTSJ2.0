"""AC-13 prerequisite unit tests: Pareto logic, gates, robust selectivity,
clustering and normalisation."""
from __future__ import annotations

import numpy as np
import pytest

from trop2_design.ranking.pareto import (
    apply_gates, greedy_cluster, non_dominated_sort, normalise,
    pairwise_identity, robust_selectivity,
)
from trop2_design.schemas.metrics import GateRule, default_metrics_profile, v1_strict_profile


class TestNonDominatedSort:
    def test_simple_fronts(self):
        # 0=(1,1) dominates 1=(0.5,1) which dominates 2=(0.2,0.1)
        pts = np.array([[1.0, 1.0], [0.5, 1.0], [0.2, 0.1]])
        fronts = non_dominated_sort(pts)
        assert fronts == [0, 1, 2]

    def test_tie_keeps_same_front(self):
        # equal on obj2, 0 better on obj1; 1 not dominated by 0? (1.0 > 0.5) -> it IS dominated
        pts = np.array([[1.0, 1.0], [0.5, 1.0], [1.0, 0.5]])
        fronts = non_dominated_sort(pts)
        # front 0: {0}; both 1 and 2 dominated only by 0 -> front 1
        assert fronts[0] == 0
        assert fronts[1] == 1
        assert fronts[2] == 1

    def test_dominance_chain(self):
        pts = np.array([[3.0, 3.0], [2.0, 3.0], [1.0, 1.0]])
        fronts = non_dominated_sort(pts)
        assert fronts[0] == 0
        assert fronts[1] == 1
        assert fronts[2] == 2

    def test_identical_points_same_front(self):
        pts = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        fronts = non_dominated_sort(pts)
        assert fronts[0] == fronts[1] == fronts[2] == 0


class TestGates:
    def test_missing_metric_goes_to_review_not_zero(self):
        gates = [GateRule(metric="intact_risk", op="<=", threshold=0.55)]
        status, reasons = apply_gates({"intact_risk": None}, gates)
        assert status == "review"
        assert "never zero" in reasons[0]

    def test_ac12_gate_irreversible(self):
        """AC-12: high EpCAM risk rejects even with perfect other metrics."""
        gates = v1_strict_profile().gates
        row = {
            "positive_state_pass_rate": 0.5, "t88_contact": True,
            "intact_risk": 0.1, "epcam_risk": 0.95,
            "glycan_membrane_clash": 0, "fold_plddt": 95.0,
            "aggregation_risk": 0.1, "trans_occlusion": 0.05,
        }
        status, reasons = apply_gates(row, gates)
        assert status == "reject"
        assert any("EpCAM" in r for r in reasons)

    def test_all_pass(self):
        row = {
            "positive_state_pass_rate": 0.8, "t88_contact": True,
            "intact_risk": 0.1, "epcam_risk": 0.2,
            "glycan_membrane_clash": 0, "fold_plddt": 85.0,
            "aggregation_risk": 0.3, "trans_occlusion": 0.1,
        }
        status, reasons = apply_gates(row, v1_strict_profile().gates)
        assert status == "pass", reasons


class TestRobustSelectivity:
    def test_prd_12_2_formula(self):
        pos = np.array([0.8, 0.9, 0.7, 0.85, 0.6])
        neg = np.array([0.2, 0.3])
        r = robust_selectivity(pos, neg, lambda_pen=0.5, quantile=0.10)
        assert r["robust_positive"] == pytest.approx(np.quantile(pos, 0.10), abs=1e-4)
        assert r["worst_offtarget"] == pytest.approx(0.3, abs=1e-4)
        expected = (np.quantile(pos, 0.10) - 0.3
                    - 0.5 * np.std(np.concatenate([pos, neg])))
        assert r["robust_selectivity"] == pytest.approx(expected, abs=1e-3)

    def test_worst_state_dominates(self):
        a = robust_selectivity(np.array([0.9, 0.9]), np.array([0.1, 0.1]), 0.5, 0.1)
        b = robust_selectivity(np.array([0.9, 0.9]), np.array([0.1, 0.9]), 0.5, 0.1)
        assert b["robust_selectivity"] < a["robust_selectivity"]


class TestClustering:
    def test_identity_clustering(self):
        seqs = ["AAAAAAAAAA", "AAAAAAAAAA", "TTTTTTTTTT"]
        labels = greedy_cluster(seqs, 0.7)
        assert labels[0] == labels[1]
        assert labels[2] != labels[0]

    def test_pairwise_identity(self):
        assert pairwise_identity("AAAA", "AAAA") == 1.0
        assert pairwise_identity("AAAA", "AAAT") == 0.75


class TestNormalise:
    def test_direction(self):
        v = normalise(np.array([1.0, 2.0, 3.0]), "maximize")
        assert v[2] == 1.0 and v[0] == 0.0
        v = normalise(np.array([1.0, 2.0, 3.0]), "minimize")
        assert v[0] == 1.0 and v[2] == 0.0

    def test_all_nan(self):
        assert np.all(np.isnan(normalise(np.array([np.nan, np.nan]), "maximize")))


class TestMetricProfiles:
    def test_default_profile_directions(self):
        p = default_metrics_profile()
        assert p.metric("complex_iptm").direction == "maximize"
        assert p.metric("intact_risk").direction == "minimize"
        assert p.metric("t88_contact").direction == "gate"

    def test_weights_sum_to_one(self):
        profile = v1_strict_profile()
        assert sum(w.weight for w in profile.display_weights) == pytest.approx(1.0)
