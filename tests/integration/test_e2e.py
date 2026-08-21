"""End-to-end integration test on the real TROP2 structures.

Runs the full M01-M10 pipeline in deterministic (CPU, proxy-metric) mode
with a reduced candidate count, then asserts the PRD acceptance outputs:
AC-01/02/03 (mapping, topology, conformers), AC-05 (generation adapter
smoke), AC-09/10 (negative states, mechanism), AC-11 (developability),
AC-13 (pareto+diversity), AC-17 (HTML report).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from trop2_design.schemas.project import ProjectConfig
from trop2_design.schemas.tools import ToolsConfig
from trop2_design.workflow import build_context, build_manifest, build_pipeline


@pytest.fixture(scope="module")
def e2e_run(project_root: Path, tmp_path_factory) -> Path:
    cfg = ProjectConfig.from_yaml(project_root / "configs" / "trop2_v1.yaml")
    # resolve paths against the real project root
    root = project_root
    cfg.target.sequence_fasta = root / cfg.target.sequence_fasta
    cfg.target.cis_structure.path = root / cfg.target.cis_structure.path
    cfg.target.trans_structure.path = root / cfg.target.trans_structure.path
    cfg.target.alternate_structure.path = root / cfg.target.alternate_structure.path
    cfg.negatives.epcam_structure = root / cfg.negatives.epcam_structure
    cfg.negatives.epcam_fasta = root / cfg.negatives.epcam_fasta
    cfg.design.max_candidates = 12  # reduced for test runtime
    cfg.design.gradient.enabled = False  # CPU e2e: no GPU gradient stage;
    # also keeps the full 12-candidate budget for M04 (the production
    # reserved-quota would cut it to 12-6)
    tools = ToolsConfig()           # no GPU tools -> deterministic proxy mode
    ctx = build_context(root, cfg, tools, run_id="pytest_e2e")
    manifest = build_manifest(root, cfg, tools, ctx)
    runner = build_pipeline(ctx, manifest)
    runner.run()
    return ctx.out


@pytest.mark.slow
def test_all_standard_outputs_exist(e2e_run: Path):
    expected = [
        "target_registry.json", "residue_mapping.csv", "input_qc.json",
        "state_manifest.csv", "topology_audit.json",
        "epitope_patch.json", "hotspots.txt", "exclusion_mask.json",
        "accessibility_metrics.csv",
        "candidates.fasta", "candidate_manifest.csv", "generation_log.json",
        "monomer_metrics.csv",
        "positive_state_metrics.csv", "terminal_contact.json",
        "negative_state_metrics.csv", "offtarget_hits.csv",
        "mechanism_metrics.csv", "clash_report.json",
        "developability_metrics.csv", "liability_flags.csv",
        "immunogenicity_hits.csv",
        "candidate_metrics.csv", "pareto_front.csv", "rejection_reasons.csv",
        "report.html", "run_manifest.json", "task_status.csv",
        "resolved_config.yaml",
    ]
    for name in expected:
        assert (e2e_run / name).exists(), f"missing {name}"


@pytest.mark.slow
def test_ac01_mapping_and_hashes(e2e_run: Path):
    reg = json.loads((e2e_run / "target_registry.json").read_text(encoding="utf-8"))
    assert reg["target"]["uniprot_id"] == "P09758"
    qc = json.loads((e2e_run / "input_qc.json").read_text(encoding="utf-8"))
    assert all(qc["ac01_required_residues_mapped"].values())
    df = pd.read_csv(e2e_run / "residue_mapping.csv")
    for num in (87, 88, 73, 108):
        assert num in set(df[df.role == "cis"].uniprot_num)
    assert reg["input_files"]["cis_structure"]  # sha256 recorded


@pytest.mark.slow
def test_ac02_ac03_topology_audit(e2e_run: Path):
    audit = json.loads((e2e_run / "topology_audit.json").read_text(encoding="utf-8"))
    cleaved = [s for s in audit["states"] if s["kind"] == "cleaved"]
    assert len(cleaved) >= 5
    for s in cleaved:
        assert s["passed"] is True
        assert s["peptide_bond_left_right"] is False
        assert s["required_disulfides_present"] is True
        assert (73, 108) in [tuple(d) for d in s["disulfides"]]
        assert s["left_terminal"].startswith("R87")
        assert s["right_terminal"].startswith("T88")
    manifest = pd.read_csv(e2e_run / "state_manifest.csv")
    assert ((manifest.kind == "intact")).any()


@pytest.mark.slow
def test_ac05_candidates_generated(e2e_run: Path):
    cm = pd.read_csv(e2e_run / "candidate_manifest.csv")
    assert len(cm) >= 10
    assert cm.candidate_id.is_unique


@pytest.mark.slow
def test_ac09_negative_states(e2e_run: Path):
    neg = pd.read_csv(e2e_run / "negative_state_metrics.csv")
    for state in ("intact_trop2_cis", "intact_trop2_trans"):
        assert state in set(neg.negative_state)
    assert "WORST" in set(neg.negative_state)
    assert (e2e_run / "complexes" / "negative").glob("*.cif").__next__()


@pytest.mark.slow
def test_ac10_mechanism_outputs(e2e_run: Path):
    mech = pd.read_csv(e2e_run / "mechanism_metrics.csv")
    assert {"cis_block", "trans_occlusion", "glycan_membrane_clash"} <= set(mech.columns)
    assert (e2e_run / "assembly_overlays").glob("*.cif").__next__()


@pytest.mark.slow
def test_ac11_developability_outputs(e2e_run: Path):
    dev = pd.read_csv(e2e_run / "developability_metrics.csv")
    assert {"mw_da", "pI", "solubility_score", "aggregation_risk"} <= set(dev.columns)


@pytest.mark.slow
def test_ac13_pareto_and_diversity(e2e_run: Path):
    df = pd.read_csv(e2e_run / "candidate_metrics.csv")
    assert set(df.hard_filter_status) <= {"pass", "reject", "review"}
    passed = df[df.hard_filter_status == "pass"]
    if not passed.empty:
        assert passed.pareto_rank.notna().all()
        assert passed.family_cluster.nunique() >= 2
    rej = pd.read_csv(e2e_run / "rejection_reasons.csv")
    if not rej.empty:
        assert rej.rejection_reasons.notna().all()


@pytest.mark.slow
def test_ac17_html_report_readable(e2e_run: Path):
    html = (e2e_run / "report.html").read_text(encoding="utf-8")
    assert "TROP2" in html
    assert "R87A/T88A" in html          # experimental controls suggested
    assert "淘汰" in html or "rejection" in html.lower()
    assert "hard" in html or "门槛" in html
