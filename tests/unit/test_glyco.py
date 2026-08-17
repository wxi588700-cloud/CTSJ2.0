"""PRD v1.1 glyco pipeline unit tests (AC-19/25/27/30 subsets)."""
from __future__ import annotations

from pathlib import Path

import pytest

from trop2_design.schemas.glyco import (
    GlycoformRegistry, TargetBundleManifest, TargetState,
)
from trop2_design.schemas.project import (
    GlycosylationConfig, ProjectConfig, TargetPredictionConfig,
)
from trop2_design.target_builder import glyco_target as gt


@pytest.fixture(scope="module")
def registry(project_root: Path) -> GlycoformRegistry:
    return GlycoformRegistry.from_yaml(project_root / "configs" / "glycoforms_v1.yaml")


class TestRegistry:  # AC-19
    def test_three_profiles_four_sites(self, registry):
        assert len(registry.profiles) == 3
        for p in registry.profiles:
            assert sorted(int(k) for k in p.sites) == [33, 120, 168, 208]
            for sg in p.sites.values():
                assert len(sg.residues) >= 7
                assert sg.occupancy == 1.0

    def test_tree_parents_ordered(self, registry):
        for p in registry.profiles:
            for sg in p.sites.values():
                for i, r in enumerate(sg.residues):
                    assert r.parent < i

    def test_bad_ccd_rejected(self):
        with pytest.raises(Exception):
            GlycoformRegistry.model_validate({
                "registry_id": "x", "profiles": [{
                    "profile_id": "p", "sites": {"33": {
                        "site": 33, "occupancy": 1.0, "iupac": "x",
                        "residues": [
                            {"ccd": "XXX", "parent": -1},
                            {"ccd": "NAG", "parent": 0, "parent_atom": "O4"},
                        ]}}}]})

    def test_glycan_template_assets_exist(self):
        for name in gt.TEMPLATE_DIR.glob("*.cif") if hasattr(gt, "TEMPLATE_DIR") else []:
            pass
        from trop2_design.target_builder import glycan_grafter as gg
        assert len(list(gg.TEMPLATE_DIR.glob("*.cif"))) == 3


class TestSiteMapping:
    def test_fragment_site_map(self, registry, project_root):
        # synthetic fragments without real NXS/T sequons must be rejected
        frags = {
            "BODY": {"sequence": "A" * 181, "start": 88, "residues": []},
            "NFR": {"sequence": "A" * 32 + "N" + "A" * 23, "start": 32,
                    "residues": []},
        }
        with pytest.raises(ValueError):
            gt.site_chain_map(frags)  # motif check rejects plain A runs

    def test_local_index_math(self):
        frag = {"sequence": "AAANXS", "start": 10}
        assert gt.local_site_index(frag, 13) == 4  # N at full-length 13
        with pytest.raises(ValueError):
            gt.local_site_index(frag, 12)  # position 12 is A


class TestManifestContract:  # AC-25
    def test_manifest_schema_roundtrip(self):
        m = TargetBundleManifest(
            target_bundle_id="TB-TEST",
            glycoform_registry_id="reg",
            profile_ids=["p1"],
            evidence_level="assumed_sensitivity_panel",
            glycan_site_occupancy={"p1": {"33": 1.0}},
            states=[TargetState(
                target_state_id="s1", glycoform_profile_id="p1",
                file="glycosylated_states/s1.pdb",
                protein_only_view="protein_only_views/s1.cif",
                md_cluster_weight=1.0)],
            cleavage_topology_pass=True, terminal_state_pass=True,
            disulfide_pass=True, glycan_topology_pass=True)
        d = m.model_dump()
        assert d["target_bundle_id"] == "TB-TEST"
        assert d["states"][0]["md_cluster_weight"] == 1.0

    def test_bundle_id_deterministic(self):
        a = gt.bundle_id("h", "r", ["a", "b"], "1.1", [1, 2])
        b = gt.bundle_id("h", "r", ["b", "a"], "1.1", [2, 1])
        assert a == b and a.startswith("TB-")


class TestLegacyCompat:  # AC-27
    def test_v1_config_parses_without_target_prediction(self, project_root):
        cfg = ProjectConfig.from_yaml(project_root / "configs" / "trop2_v1.yaml")
        assert cfg.target_prediction is None

    def test_v1_1_config_parses(self, project_root):
        cfg = ProjectConfig.from_yaml(project_root / "configs" / "trop2_v1_1.yaml")
        tp = cfg.target_prediction
        assert tp is not None and tp.mode == "glyco_ensemble"
        assert tp.glycosylation.sites == [33, 120, 168, 208]
        assert tp.glycosylation.source == "assumed_sensitivity_panel"

    def test_default_glycosylation_config(self):
        g = GlycosylationConfig()
        assert g.enabled is True and g.profile_ids  # non-empty default panel
        t = TargetPredictionConfig()
        assert t.min_representatives == 5 and t.graft_seeds >= 1


class TestWeightedQuantile:
    def test_matches_unweighted_equal_weights(self):
        import numpy as np
        from trop2_design.scoring.binding import _weighted_quantile
        v = [0.1, 0.5, 0.9]
        assert _weighted_quantile(v, [1, 1, 1], 0.5) == pytest.approx(
            float(np.quantile(v, 0.5)), abs=0.06)

    def test_weight_pulls_quantile(self):
        from trop2_design.scoring.binding import _weighted_quantile
        # heavy weight on the low value -> q=0.5 collapses onto it
        assert _weighted_quantile([0.1, 0.9], [100, 1], 0.5) == pytest.approx(0.1, abs=0.02)
        # heavy weight on the high value -> q=0.5 pulled above midpoint
        assert _weighted_quantile([0.1, 0.9], [1, 100], 0.5) > 0.4

    def test_empty(self):
        from trop2_design.scoring.binding import _weighted_quantile
        assert _weighted_quantile([], [], 0.1) == 0.0


class TestGlycoformCoverage:
    def test_single_profile_pass(self):
        from trop2_design.scoring.binding import _glycoform_coverage
        recs = [{"glycoform_profile": "p", "md_cluster_weight": 0.6,
                 "status": "pass"},
                {"glycoform_profile": "p", "md_cluster_weight": 0.4,
                 "status": "fail_state"}]
        assert _glycoform_coverage(recs) == pytest.approx(0.6, abs=0.01)

    def test_mean_over_profiles(self):
        from trop2_design.scoring.binding import _glycoform_coverage
        recs = [
            {"glycoform_profile": "a", "md_cluster_weight": 1.0, "status": "pass"},
            {"glycoform_profile": "b", "md_cluster_weight": 1.0, "status": "fail_state"},
        ]
        assert _glycoform_coverage(recs) == pytest.approx(0.5, abs=0.01)

    def test_legacy_no_profiles(self):
        from trop2_design.scoring.binding import _glycoform_coverage
        assert _glycoform_coverage([{"status": "pass"}]) == 0.0
