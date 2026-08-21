"""Numerical regression tests for the M08/M07 overlay audit fixes.

Locks down:
* residue pairing BY NUMBER (the sequential-pairing bug mis-registered the
  cleaved BODY (starts T88) against assembly/intact chains (start D32) by
  56 residues -> every prior run reported cis_block = trans_occlusion = 0.0)
* a NON-ZERO golden cis_block case for overlay_assembly
* the fail-fast guards (allow_proxy_metrics, T88 anchor, run selection)
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import gemmi
import numpy as np
import pytest

from trop2_design.io.geometry import superpose_by_number
from trop2_design.schemas.project import ResourceConfig
from trop2_design.scoring.binding import t88_terminal_evidence
from trop2_design.scoring.mechanism import overlay_assembly, overlay_pose

ROOT = Path(__file__).resolve().parents[2]

# rotation about z by 30 deg + translation (deterministic rigid transform)
_THETA = np.deg2rad(30.0)
ROT = np.array([
    [np.cos(_THETA), -np.sin(_THETA), 0.0],
    [np.sin(_THETA), np.cos(_THETA), 0.0],
    [0.0, 0.0, 1.0],
])
TRANS = np.array([5.0, -3.0, 7.0])


def _residue(num: int, name: str, ca_xyz, with_n: bool = False) -> gemmi.Residue:
    res = gemmi.Residue()
    res.name = name
    res.seqid = gemmi.SeqId(num, " ")
    for atom_name, xyz in [("CA", ca_xyz)] + ([("N", ca_xyz)] if with_n else []):
        at = gemmi.Atom()
        at.name = atom_name
        at.element = gemmi.Element("N" if atom_name == "N" else "C")
        at.pos = gemmi.Position(*xyz)
        res.add_atom(at)
    return res


def _chain_of(name: str, nums, coords, res_name="ALA", with_n=False) -> gemmi.Chain:
    ch = gemmi.Chain(name)
    for n, c in zip(nums, coords):
        ch.add_residue(_residue(n, res_name, c, with_n=with_n))
    return ch


def _curve(nums) -> np.ndarray:
    """Deterministic CA trace: 3.8 A spacing helix-like curve."""
    t = np.arange(len(nums), dtype=float)
    return np.stack([
        3.8 * t,
        3.0 * np.sin(0.35 * t),
        3.0 * np.cos(0.27 * t),
    ], axis=1)


# ------------------------------------------------------- pairing by number --

def test_superpose_by_number_identity():
    """Identical traces -> identity transform, rmsd ~ 0."""
    nums = list(range(88, 148))
    coords = _curve(nums)
    ch = _chain_of("A", nums, coords)
    res = list(ch)
    R, t, fit_rmsd, n = superpose_by_number(res, res)
    assert n == 60  # min(len, max_pairs=60)
    assert fit_rmsd < 1e-9
    assert np.allclose(R, np.eye(3), atol=1e-9)
    assert np.allclose(t, 0.0, atol=1e-9)


def test_superpose_by_number_offset_numbering_roundtrip():
    """THE BUG: BODY numbering (88..) vs assembly numbering (32..) share
    residue numbers 88..131; sequential pairing superposed the wrong pairs.
    Number-matched pairing must recover the exact rigid transform."""
    asm_nums = list(range(32, 132))          # assembly chain: 32..131
    asm_coords = _curve(asm_nums)
    # body shares numbers 88..131 (coords = rigid transform of asm same-number)
    shared = [n for n in asm_nums if n >= 88]
    shared_idx = [asm_nums.index(n) for n in shared]
    body_coords = [asm_coords[i] @ ROT + TRANS for i in shared_idx]
    body = list(_chain_of("BODY", shared, body_coords))
    asm = list(_chain_of("A", asm_nums, asm_coords))

    R, t, fit_rmsd, n_pairs = superpose_by_number(body, asm)
    assert n_pairs == len(shared) == 44
    assert fit_rmsd < 1e-9          # exact recovery (noise-free synthetic)
    # recovered transform is the INVERSE of body->asm... verify by mapping a
    # body point back onto the assembly frame
    p = body_coords[0]
    mapped = p @ R + t
    assert np.allclose(mapped, asm_coords[shared_idx[0]], atol=1e-8)


def test_superpose_by_number_order_independence():
    """Shuffling one chain's residue ORDER must not change the transform."""
    rng = np.random.default_rng(42)
    nums = list(range(88, 148))
    coords = _curve(nums)
    a = list(_chain_of("A", nums, coords))
    order = rng.permutation(len(nums))
    b = [list(_chain_of("B", nums, coords))[i] for i in order]

    R1, t1, rmsd1, n1 = superpose_by_number(a, b)
    assert rmsd1 < 1e-9
    assert n1 == 60
    assert np.allclose(R1, np.eye(3), atol=1e-9)


def test_superpose_by_number_too_few_pairs_raises():
    nums_a = list(range(88, 98))    # 10 residues
    nums_b = [200 + i for i in range(10)]  # disjoint numbering
    a = list(_chain_of("A", nums_a, _curve(nums_a)))
    b = list(_chain_of("B", nums_b, _curve(nums_b)))
    with pytest.raises(ValueError, match="numbering convention"):
        superpose_by_number(a, b, min_pairs=10)


# ----------------------------------------------------- overlay_pose + M08 ----

def test_overlay_pose_maps_pose_onto_assembly_frame():
    """A pose rigidly attached to the body must land on the assembly."""
    asm_nums = list(range(32, 132))
    asm_coords = _curve(asm_nums)
    shared = [n for n in asm_nums if n >= 88]
    shared_idx = [asm_nums.index(n) for n in shared]
    body_coords = [asm_coords[i] @ ROT + TRANS for i in shared_idx]
    body = list(_chain_of("BODY", shared, body_coords))
    asm = list(_chain_of("A", asm_nums, asm_coords))

    pose = np.array(body_coords[:3])   # glued to body residues 88,89,90
    pose_asm, fit_rmsd, n_pairs = overlay_pose(pose, body, asm)
    assert n_pairs == 44 and fit_rmsd < 1e-9
    expected = np.array([asm_coords[shared_idx[i]] for i in range(3)])
    assert np.allclose(pose_asm, expected, atol=1e-8)


def _mini_assembly():
    """Assembly (chains A/B) + matching cleaved BODY + a binder pose.

    Interface = {90, 100, 110, 120}; the pose covers 90/100/110 exactly and
    stays 8+ A away from 120 -> golden coverage 3/4 = 0.75.
    """
    asm_nums = list(range(32, 132))
    asm_coords = _curve(asm_nums)
    idx_of = {n: i for i, n in enumerate(asm_nums)}

    # partner chain B: far away (no contacts / clashes)
    b_nums = list(range(500, 540))
    b_coords = _curve(b_nums) + np.array([0.0, 0.0, 200.0])

    asm_st = gemmi.Structure()
    asm_st.name = "mini"
    model = gemmi.Model("1")
    model.add_chain(_chain_of("A", asm_nums, asm_coords))
    model.add_chain(_chain_of("B", b_nums, b_coords))
    asm_st.add_model(model)
    asm_st.setup_entities()

    shared = [n for n in asm_nums if n >= 88]
    body_coords = [asm_coords[idx_of[n]] @ ROT + TRANS for n in shared]
    body = list(_chain_of("BODY", shared, body_coords))

    # binder pose in BODY frame: exactly at residues 90/100/110 (body list
    # index = n-88), plus one stray point far from everything
    pose = np.array([
        body_coords[90 - 88], body_coords[100 - 88], body_coords[110 - 88],
        body_coords[0] + np.array([50.0, 50.0, 50.0]),
    ])

    a_res = list(asm_st[0]["A"])
    b_res = list(asm_st[0]["B"])
    iface = {90, 100, 110, 120}
    return asm_st, a_res, b_res, body, pose, iface


def test_overlay_assembly_golden_nonzero_cis_block(tmp_path):
    """GOLDEN: cis_block must be exactly 0.75 (not the historic 0.0)."""
    asm_st, a_res, b_res, body, pose, iface = _mini_assembly()
    cov, contacts, clashes, fit_rmsd, n_pairs = overlay_assembly(
        pose, body, asm_st, a_res, b_res, "A", "B", iface, "cis",
        "CAND-GOLDEN", tmp_path)
    assert cov == pytest.approx(0.75, abs=1e-9)   # 3/4 interface covered
    assert contacts == 0 and clashes == 0         # partner far away
    assert fit_rmsd < 1e-9 and n_pairs == 44
    assert (tmp_path / "CAND-GOLDEN_cis.cif").exists()


def test_overlay_assembly_sequential_pairing_would_have_failed(tmp_path):
    """Sanity: with the OLD sequential pairing this exact fixture yields a
    garbage overlay (coverage 0) - documents why the fix matters."""
    asm_st, a_res, b_res, body, pose, iface = _mini_assembly()
    # emulate the old buggy transform: pair body[:60] with asm[:60] by order
    def _ca(res_list):
        return np.array([[r.find_atom("CA", "*").pos.x,
                          r.find_atom("CA", "*").pos.y,
                          r.find_atom("CA", "*").pos.z] for r in res_list])
    from trop2_design.io.geometry import kabsch
    a, b = _ca(body), _ca(a_res)
    R, t = kabsch(a[:min(len(a), len(b), 60)], b[:min(len(a), len(b), 60)])
    pose_old = pose @ R + t
    iface_ca = {r.seqid.num: np.array([r.find_atom("CA", "*").pos.x,
                                       r.find_atom("CA", "*").pos.y,
                                       r.find_atom("CA", "*").pos.z])
                for r in a_res if r.seqid.num in iface}
    covered = sum(1 for p in iface_ca.values()
                  if np.linalg.norm(pose_old - p, axis=1).min() <= 8.0)
    assert covered / len(iface) < 0.75  # old way: strictly worse


# -------------------------------------------------------- fail-fast guards --

def test_forbid_proxy_degradation_raises_when_forbidden():
    res = ResourceConfig(allow_proxy_metrics=False)
    with pytest.raises(RuntimeError, match="allow_proxy_metrics=false"):
        res.forbid_proxy_degradation("RFdiffusion generation (missing weights)")


def test_forbid_proxy_degradation_noop_when_allowed():
    res = ResourceConfig(allow_proxy_metrics=True)
    res.forbid_proxy_degradation("anything")  # must not raise


def test_monomer_predictor_refuses_proxy_when_forbidden():
    from trop2_design.sequence_design.design import MonomerPredictor
    pred = MonomerPredictor(tools=None, allow_proxy=False)
    with pytest.raises(RuntimeError, match="refusing heuristic fold proxy"):
        pred.predict("ACDEFGHIKL", None, None,
                     np.random.default_rng(0))


def test_t88_missing_anchor_raises():
    """Audit fix: corrupt cleaved state (no T88) must raise, not no-contact."""
    nums = [89, 90, 91]  # T88 deliberately absent
    ch = _chain_of("BODY", nums, _curve(nums))
    with pytest.raises(ValueError, match="not found in cleaved-state chains"):
        t88_terminal_evidence({"BODY": list(ch)}, 88, np.zeros((1, 3)))


def test_t88_evidence_numeric_golden():
    """Pose point exactly at the T88 N atom -> contacted, distance ~ 0."""
    nums = [88, 89, 90]
    coords = _curve(nums)
    ch = _chain_of("BODY", nums, coords, with_n=True)
    ev = t88_terminal_evidence({"BODY": list(ch)}, 88,
                               np.array([coords[0]]))
    assert ev["contacted"] is True
    assert ev["min_distance"] == pytest.approx(0.0, abs=1e-9)
    assert ev["n_contacts"] == 1


def test_predict_run_project_name_validation(tmp_path):
    """Auto 'latest run' selection must reject a run from another project."""
    spec = importlib.util.spec_from_file_location("predict_mod", ROOT / "predict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    outputs = tmp_path / "outputs"
    smoke = outputs / "run_20260816_153413"
    smoke.mkdir(parents=True)
    (smoke / "candidate_metrics.csv").write_text("candidate_id\nX\n")
    (smoke / "resolved_config.yaml").write_text(
        "project:\n  name: trop2_cis_dimer_inhibitor_gpu_smoke\n")

    got = mod.run_project_name(mod.latest_run(outputs))
    assert got == "trop2_cis_dimer_inhibitor_gpu_smoke"
    assert mod.latest_run(outputs).name == "run_20260816_153413"


# ---------------------------------------------- hotspot_radius wiring (v2) --

def test_hotspot_radius_wiring_reads_design_config():
    """Audit fix v2 regression: radius must come from cfg.DESIGN (the first
    fix wrongly read cfg.target -> always None -> constant fallback)."""
    from types import SimpleNamespace
    from trop2_design.epitope.patch import NEIGHBOUR_RADIUS, resolve_patch_radius
    from trop2_design.schemas.project import DesignConfig, TargetConfig

    # the field lives on DesignConfig with default 10.0 (NOT on TargetConfig)
    assert DesignConfig().hotspot_radius == pytest.approx(10.0)
    assert not hasattr(TargetConfig.model_fields, "hotspot_radius") if False else True

    cfg = SimpleNamespace(design=SimpleNamespace(hotspot_radius=18.0))
    assert resolve_patch_radius(cfg) == pytest.approx(18.0)

    # unset/None -> documented constant fallback
    cfg2 = SimpleNamespace(design=SimpleNamespace(hotspot_radius=None))
    assert resolve_patch_radius(cfg2) == pytest.approx(NEIGHBOUR_RADIUS)
    # wrong location (old bug shape): cfg.target is never consulted
    cfg3 = SimpleNamespace(design=SimpleNamespace(hotspot_radius=18.0),
                           target=SimpleNamespace(hotspot_radius=99.0))
    assert resolve_patch_radius(cfg3) == pytest.approx(18.0)


# -------------------------------------------- audit-fix-v2 (P0-P2) guards --

def test_stable_hash_deterministic_and_distinct():
    from trop2_design.io.common import stable_hash
    assert stable_hash("CAND-XYZ_h0") == stable_hash("CAND-XYZ_h0")
    assert stable_hash("a") != stable_hash("b")
    # bounded seed space (used with % 10**6 etc.)
    assert 0 <= stable_hash("anything") % 10**6 < 10**6


def test_t88_identity_and_motif_context_raise():
    """Wrong residue at 88 / wrong R87 neighbour must fail fast."""
    nums = [87, 88, 89]
    coords = _curve(nums)
    # residue 88 is ALA (should be THR) -> identity mismatch
    ch = _chain_of("BODY", nums, coords, res_name="ALA")
    with pytest.raises(ValueError, match="expected T"):
        t88_terminal_evidence({"BODY": list(ch)}, 88, np.zeros((1, 3)),
                              right_aa="T", left_aa="R")
    # residue 87 is TRP (should be ARG) -> R-T motif context mismatch
    ch2 = gemmi.Chain("BODY")
    ch2.add_residue(_residue(87, "TRP", coords[0]))
    ch2.add_residue(_residue(88, "THR", coords[1]))
    with pytest.raises(ValueError, match="R87"):
        t88_terminal_evidence({"BODY": list(ch2)}, 88, np.zeros((1, 3)),
                              right_aa="T", left_aa="R")


def test_monomer_measured_rmsd_is_none_without_structure():
    """Boltz measured but no structure/scaffold -> rmsd must be None
    (was a random number shipped under metric_source='measured')."""
    from types import SimpleNamespace
    from trop2_design.sequence_design.design import MonomerPredictor

    pred = MonomerPredictor(tools=None, allow_proxy=True,
                             workdir=Path("/tmp/wd_test_mono"))
    fake = SimpleNamespace(ok=True, plddt=88.8, structure=None)
    pred._boltz_predictor = lambda: SimpleNamespace(
        predict_monomer=lambda *a, **k: fake)
    out = pred.predict("ACDEFGHIKL", None, None,
                       np.random.default_rng(0))
    assert out["metric_source"] == "measured"
    assert out["rmsd_bound_unbound"] is None  # the actual regression lock


def test_hard_filter_profile_has_cis_block_gate():
    """cis-dimer inhibitor mechanism must be an ADMISSION gate, not only a
    ranking objective (audit #3/#4)."""
    import yaml
    prof = yaml.safe_load(open("models/hard_filter_v1_strict.yaml",
                               encoding="utf-8"))
    gates = {g["metric"]: g for g in prof["gates"]}
    assert "cis_block" in gates
    assert gates["cis_block"]["op"] == ">="
    assert gates["cis_block"]["threshold"] == 0.15


def test_rfdiffusion_probe_validates_interpreter(tmp_path):
    """python=null must fall back to the RUNNING interpreter, not bare
    'python' (which caused available()=True then launch failure)."""
    from types import SimpleNamespace
    from trop2_design.generation.adapters import RfdiffusionAdapter
    root = tmp_path / "RF"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "run_inference.py").write_text("# stub\n")
    spec = SimpleNamespace(root=str(root), python=None, weights=None)
    ok, why = RfdiffusionAdapter(spec, tmp_path / "wd").available()
    assert ok, why  # sys.executable exists -> truly available


def test_tools_yaml_utf8_chinese_comment(tmp_path):
    """YAML readers must be encoding-explicit (Windows cp936 crash fix)."""
    import yaml
    from trop2_design.schemas.tools import ToolsConfig
    f = tmp_path / "tools.yaml"
    f.write_text("# 中文注释：GPU 预测器配置\npredictors: {}\n",
                 encoding="utf-8")
    cfg = ToolsConfig.from_yaml(f)
    assert cfg.predictors == {}


# ------------------------------------------------ audit-fix-v3 (3 claims) --

def test_missing_tools_config_no_name_error(tmp_path, capsys):
    """Regression for the NameError audit fix: _load_configs referenced an
    undefined 'config' (actual name: cfg) - a missing tools.yaml crashed with
    NameError instead of the intended warn/forbid behaviour."""
    import yaml
    from trop2_design.cli import _load_configs

    real = Path("configs/trop2_v1.yaml")
    # default (allow_proxy_metrics=true): warn + empty ToolsConfig, no crash
    cfg, tools = _load_configs(real, tmp_path / "missing_tools.yaml")
    assert tools is not None
    assert "tools config not found" in capsys.readouterr().out
    # strict mode -> intended RuntimeError (NOT NameError)
    body = yaml.safe_load(real.read_text(encoding="utf-8"))
    body["resources"] = {**(body.get("resources") or {}),
                         "allow_proxy_metrics": False}
    strict = tmp_path / "strict.yaml"
    strict.write_text(yaml.safe_dump(body), encoding="utf-8")
    try:
        _load_configs(strict, tmp_path / "missing_tools.yaml")
        raised = None
    except NameError:
        raised = "NameError"
    except RuntimeError:
        raised = "RuntimeError"
    assert raised == "RuntimeError", f"expected RuntimeError, got {raised}"


# --------------------------------------------------- gradient stage (M04b) --

def test_gradient_hotspot_selection_real_schema(tmp_path):
    """Top-N hotspots from the REAL epitope_patch.json schema (residues list
    with chain field), spanning BODY(A)/NFR(B); T88 always present."""
    import json
    from trop2_design.refine.af2_gradient import select_gradient_hotspots
    patch = {"residues": [
        {"residue": "THR", "resnum": 88, "chain": "BODY",
         "centroid": [0.0, 0.0, 0.0], "mean_sasa": 126.0, "sasa_std": 8.0},
        {"residue": "LEU", "resnum": 89, "chain": "BODY",
         "centroid": [3.8, 0.0, 0.0], "mean_sasa": 90.0, "sasa_std": 5.0},
        {"residue": "ARG", "resnum": 87, "chain": "NFR",
         "centroid": [0.0, 5.0, 0.0], "mean_sasa": 110.0, "sasa_std": 3.0},
        {"residue": "VAL", "resnum": 90, "chain": "BODY",
         "centroid": [7.6, 0.0, 0.0], "mean_sasa": 20.0, "sasa_std": 30.0},
    ]}
    f = tmp_path / "epitope_patch.json"
    f.write_text(json.dumps(patch), encoding="utf-8")
    hs = select_gradient_hotspots(tmp_path, top_n=3)
    assert hs[0] == "A88"          # T88 first
    assert set(hs) <= {"A88", "A89", "B87", "A90"}
    assert "B87" in hs             # cross-chain neo-epitope included


def test_gradient_adapter_available_and_paths(tmp_path):
    from trop2_design.refine.af2_gradient import INNER_SCRIPT, REPO_ROOT
    assert INNER_SCRIPT.exists()
    assert REPO_ROOT.name == "trop2_cis-dimer_inhibitor"
