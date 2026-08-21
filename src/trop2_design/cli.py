"""trop2 CLI (M00): prepare / generate / evaluate / rank / report / run.

Every subcommand shares the content-hash cache, so re-running after an
interruption resumes from the last successful stage (AC-15) and identical
inputs + seed reproduce identical discrete results (AC-14).
"""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="trop2",
    help="trop2_cis-dimer_inhibitor - TROP2 R87-T88 cleaved-state cis-dimer inhibitor design platform (PRD v1.0)",
    no_args_is_help=True,
    add_completion=False,
)


def _load_configs(project_yaml: Path, tools_yaml: Path | None):
    from .schemas.project import ProjectConfig
    from .schemas.tools import ToolsConfig

    cfg = ProjectConfig.from_yaml(project_yaml)
    if tools_yaml and not Path(tools_yaml).exists():
        # audit fix: silent empty ToolsConfig degraded EVERYTHING to proxy
        print(f"[trop2][warn] tools config not found: {tools_yaml} "
              f"- all predictors/probes unavailable")
        cfg.resources.forbid_proxy_degradation(
            f"tools config missing ({tools_yaml})")
    tools = ToolsConfig.from_yaml(tools_yaml) if tools_yaml and Path(tools_yaml).exists() else ToolsConfig()
    return cfg, tools


def _resolve_paths(cfg, root: Path):
    """Make relative config paths absolute against the project root."""
    t = cfg.target
    t.sequence_fasta = root / t.sequence_fasta
    for ref in ("cis_structure", "trans_structure"):
        getattr(t, ref).path = root / getattr(t, ref).path
    if t.alternate_structure is not None:
        t.alternate_structure.path = root / t.alternate_structure.path
    if cfg.negatives.epcam_structure is not None:
        cfg.negatives.epcam_structure = root / cfg.negatives.epcam_structure
    if cfg.negatives.epcam_fasta is not None:
        cfg.negatives.epcam_fasta = root / cfg.negatives.epcam_fasta
    if cfg.design.import_fasta is not None:
        cfg.design.import_fasta = root / cfg.design.import_fasta
    if cfg.design.import_pdb_dir is not None:
        cfg.design.import_pdb_dir = root / cfg.design.import_pdb_dir
    return cfg


def _project_root(project: Path) -> Path:
    """Repo root = nearest ancestor containing pyproject.toml."""
    for cand in [project.resolve().parent, *project.resolve().parents]:
        if (cand / "pyproject.toml").exists():
            return cand
    return project.resolve().parent


def _prepare_run(project: Path, tools: Path | None, run_id: str | None,
                 stage_names: set[str] | None, resume: bool):
    from .workflow import build_context, build_manifest, build_pipeline

    cfg, tools_cfg = _load_configs(project, tools)
    root = _project_root(project)
    cfg = _resolve_paths(cfg, root)
    ctx = build_context(root, cfg, tools_cfg, run_id)
    manifest = build_manifest(root, cfg, tools_cfg, ctx)
    runner = build_pipeline(ctx, manifest)
    if resume:
        # resume = run everything not yet cached (the cache itself decides)
        stage_names = None
    return ctx, runner


STAGE_MAP = {
    "prepare": {"M01_target_ingestion", "M02_cleaved_states", "M03_epitope"},
    "generate": {"M04_generate", "M04b_gradient_refine"},
    "evaluate": {"M05_sequence_design", "M06_positive_state",
                 "M07_negative_state", "M08_mechanism", "M09_developability"},
    "rank": {"M10_ranking"},
    "report": {"M10_ranking"},
}


@app.command()
def run(
    project: Path = typer.Option("configs/trop2_v1.yaml", help="project YAML"),
    tools: Path | None = typer.Option("configs/tools.yaml", help="tools YAML"),
    run_id: str | None = typer.Option(None, help="reuse an existing run directory"),
    stages: str | None = typer.Option(None, help="comma list or group: prepare/generate/evaluate/rank/all"),
) -> None:
    """Run the full M00-M10 pipeline (or a subset)."""
    selected = None
    if stages and stages != "all":
        selected = set()
        for item in stages.split(","):
            item = item.strip()
            if not item:
                continue
            if item in STAGE_MAP:          # group name -> expand to stage set
                selected |= STAGE_MAP[item]
            else:                          # literal stage name (e.g. M06_positive_state)
                selected.add(item)
        if not selected:
            selected = None

    # audit fix (external review P1): running a stage subset in a NEW run
    # directory must include the dependency closure, otherwise required
    # inputs (e.g. epitope_patch.json for M04) do not exist yet
    if selected and run_id is None:
        from .workflow import build_context, build_manifest, build_pipeline
        _cfg, _tools = _load_configs(project, tools)
        _root = _project_root(project)
        _ctx, _man = build_context(_root, _cfg, _tools, run_id), None
        _runner = build_pipeline(_ctx, _man) if _man else build_pipeline(_ctx, build_manifest(_root, _cfg, _tools, _ctx))
        deps = {s.name: set(s.depends_on) for s in _runner.stages}
        closure = set(selected)
        frontier = list(closure)
        while frontier:
            for dep in deps.get(frontier.pop(), ()):
                if dep not in closure:
                    closure.add(dep)
                    frontier.append(dep)
        if closure != selected:
            typer.echo(f"[trop2] dependency closure: "
                       f"{sorted(selected)} -> {sorted(closure)}")
        selected = closure

    project = project.resolve()
    ctx, runner = _prepare_run(project, tools, run_id, selected, resume=True)
    typer.echo(f"[trop2] run directory: {ctx.out}")
    try:
        runner.run(only=selected)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[trop2] FAILED: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"[trop2] done. report: {ctx.out / 'report.html'}")


@app.command()
def prepare(project: Path = typer.Option("configs/trop2_v1.yaml"),
         run_id: str | None = typer.Option(None,
         help="reuse an existing run directory (deps must be complete)")) -> None:
    """Stages M01-M03: ingestion, cleaved states, epitope."""
    run(project=project, tools=Path("configs/tools.yaml"),
        stages="prepare", run_id=run_id)


@app.command()
def generate(project: Path = typer.Option("configs/trop2_v1.yaml"),
         run_id: str | None = typer.Option(None,
         help="reuse an existing run directory (deps must be complete)")) -> None:
    """Stage M04: candidate generation + import."""
    run(project=project, tools=Path("configs/tools.yaml"),
        stages="generate", run_id=run_id)


@app.command()
def evaluate(project: Path = typer.Option("configs/trop2_v1.yaml"),
         run_id: str | None = typer.Option(None,
         help="reuse an existing run directory (deps must be complete)")) -> None:
    """Stages M05-M09: sequence design, positive/negative states, mechanism, developability."""
    run(project=project, tools=Path("configs/tools.yaml"),
        stages="evaluate", run_id=run_id)


@app.command()
def rank(project: Path = typer.Option("configs/trop2_v1.yaml"),
         run_id: str | None = typer.Option(None,
         help="reuse an existing run directory (deps must be complete)")) -> None:
    """Stage M10: hard gates, Pareto ranking, shortlist."""
    run(project=project, tools=Path("configs/tools.yaml"),
        stages="rank", run_id=run_id)


@app.command()
def report(run_dir: Path = typer.Argument(..., help="outputs/<run_id> directory")) -> None:
    """Re-render report.html from an existing run directory (weights/thresholds
    can be changed without re-running structure predictions - PRD scenario E)."""
    import json

    from .schemas.project import ProjectConfig
    from .schemas.tools import ToolsConfig
    from .ranking import rank as m10

    resolved = run_dir / "resolved_config.yaml"
    if not resolved.exists():
        typer.echo(f"resolved_config.yaml missing in {run_dir}", err=True)
        raise typer.Exit(1)
    import yaml

    cfg = ProjectConfig.model_validate(yaml.safe_load(resolved.read_text(encoding="utf-8")))
    tools = ToolsConfig()
    mf = run_dir / "run_manifest.json"
    seed = json.loads(mf.read_text(encoding="utf-8"))["seed"] if mf.exists() else cfg.resources.seed

    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.out = run_dir
    ctx.project_root = run_dir.parent.parent
    ctx.config = cfg
    ctx.tools = tools
    ctx.seed = seed
    ctx.state = {}
    m10.run(ctx)
    typer.echo(f"[trop2] report re-rendered: {run_dir / 'report.html'}")


# --------------------------------------------------------------------------
# PRD v1.1: cleaved glycosylated target bundle (M02-G)
# --------------------------------------------------------------------------

def _load_v11(project: Path, tools: Path | None):
    cfg, tools_cfg = _load_configs(project, tools)
    root = _project_root(project)
    cfg = _resolve_paths(cfg, root)
    if cfg.target_prediction is None or \
            cfg.target_prediction.mode != "glyco_ensemble":
        raise typer.BadParameter(
            "target_prediction.mode != glyco_ensemble - use trop2_v1_1.yaml")
    if cfg.target_prediction.glycosylation.registry and \
            not (root / cfg.target_prediction.glycosylation.registry).exists():
        # registry path may already be absolute
        pass
    return root, cfg, tools_cfg


@app.command("prepare-target")
def prepare_target(
    project: Path = typer.Option("configs/trop2_v1_1.yaml"),
    tools: Path | None = typer.Option("configs/tools.yaml"),
    run_id: str | None = typer.Option(None, help="reuse an existing run dir"),
) -> None:
    """PRD v1.1: M01-M02 legacy states + hybrid glycosylated target bundle."""
    root, cfg, tools_cfg = _load_v11(project, tools)
    tp = cfg.target_prediction
    from .workflow import build_context, build_manifest, build_pipeline

    ctx = build_context(root, cfg, tools_cfg, run_id)
    manifest = build_manifest(root, cfg, tools_cfg, ctx)
    runner = build_pipeline(ctx, manifest)
    typer.echo(f"[prepare-target] run: {ctx.out}")
    runner.run(only={"M01_target_ingestion", "M02_cleaved_states"})

    # hybrid bundle build on top of the legacy cleaved reference state
    try:
        from .prediction import build_boltz
        bspec = tools_cfg.predictors.get("boltz")
        boltz = build_boltz(bspec, cfg.resources.seed) if bspec else None
    except Exception:
        boltz = None
    if boltz is None:
        typer.echo("[prepare-target] ERROR: boltz predictor unavailable "
                   "(configure predictors.boltz in tools.yaml)", err=True)
        raise typer.Exit(code=1)
    from .schemas.glyco import GlycoformRegistry
    from .target_builder.glyco_target import build_target_bundle

    reg_path = cfg.target_prediction.glycosylation.registry
    reg_path = reg_path if reg_path.is_absolute() else root / reg_path
    registry = GlycoformRegistry.from_yaml(reg_path)
    import pandas as pd

    states = pd.read_csv(ctx.out / "state_manifest.csv")
    ref = states[(states.kind == "cleaved") & states.audit_passed].iloc[0]
    import json as _json

    reg = _json.loads((ctx.out / "target_registry.json").read_text(encoding="utf-8"))
    tpl_hash = reg["structures"]["cis"]["sha256"]
    m = build_target_bundle(
        Path(ref.file), registry, ctx.out, boltz,
        seeds=tp.seeds, template_hash=tpl_hash,
        software_version=tp.target_bundle_version,
        min_representatives=tp.min_representatives,
        sampling_steps=tp.sampling_steps,
        graft_seeds=tp.graft_seeds)
    if m is None:
        typer.echo("[prepare-target] bundle build produced no states", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"[prepare-target] bundle {m.target_bundle_id}: "
               f"{len(m.states)} states across {len(m.profile_ids)} profiles "
               f"(SS pass={m.disulfide_pass}, glyco pass={m.glycan_topology_pass})")
    typer.echo(f"[prepare-target] manifest: {ctx.out / 'target_bundles' / 'manifest.json'}")


@app.command("validate-target")
def validate_target(
    bundle_dir: Path = typer.Argument(..., help="outputs/<run_id>/target_bundles"),
) -> None:
    """PRD AC-30: audit a published bundle; exits non-zero on failure."""
    import json as _json

    mf = bundle_dir / "manifest.json"
    if not mf.exists():
        typer.echo("[validate-target] manifest.json missing", err=True)
        raise typer.Exit(code=1)
    m = _json.loads(mf.read_text(encoding="utf-8"))
    errors, warnings = [], []
    for key in ("cleavage_topology_pass", "terminal_state_pass",
                "disulfide_pass", "glycan_topology_pass"):
        if not m.get(key):
            errors.append(f"{key}=False")
    if m.get("evidence_level") == "assumed_sensitivity_panel":
        warnings.append("glycoforms are ASSUMED sensitivity panels "
                        "(no site-specific glycoproteomics)")
    import os

    for s in m.get("states", []):
        for field in ("file", "protein_only_view"):
            p = bundle_dir / s[field]
            if not p.exists():
                errors.append(f"{s['target_state_id']}: missing {field} {s[field]}")
    mask_dir = bundle_dir / "glycan_masks"
    n_masks = len(list(mask_dir.glob("*.json"))) if mask_dir.exists() else 0
    for s in m.get("states", []):
        if not (mask_dir / f"{s['target_state_id']}.json").exists():
            errors.append(f"{s['target_state_id']}: missing glycan mask")
    typer.echo(f"[validate-target] {m['target_bundle_id']}: "
               f"{len(m.get('states', []))} states, {n_masks} masks")
    for w in warnings:
        typer.echo(f"  WARN: {w}")
    if errors:
        for e in errors:
            typer.echo(f"  ERROR: {e}", err=True)
        raise typer.Exit(code=1)
    typer.echo("[validate-target] PASS")


@app.command("export-target-bundle")
def export_target_bundle(
    bundle_dir: Path = typer.Argument(...),
    out_dir: Path = typer.Option(None, help="copy destination (default: print)"),
) -> None:
    """PRD 7.3: export the immutable bundle contract summary/files."""
    import json as _json
    import shutil as _shutil

    m = _json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = {
        "target_bundle_id": m["target_bundle_id"],
        "target_bundle_version": m["target_bundle_version"],
        "glycoform_registry_id": m["glycoform_registry_id"],
        "evidence_level": m["evidence_level"],
        "profiles": {pid: [s["target_state_id"] for s in m["states"]
                           if s["glycoform_profile_id"] == pid]
                     for pid in m["profile_ids"]},
        "QC": {k: m[k] for k in ("cleavage_topology_pass", "terminal_state_pass",
                                 "disulfide_pass", "glycan_topology_pass")},
    }
    typer.echo(_json.dumps(summary, indent=2, ensure_ascii=False))
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("manifest.json", "glycosylated_states", "protein_only_views",
                    "glycan_masks", "topology", "provenance"):
            src = bundle_dir / sub
            if src.is_dir():
                _shutil.copytree(src, out_dir / sub, dirs_exist_ok=True)
            elif src.is_file():
                _shutil.copy2(src, out_dir / sub)
        typer.echo(f"[export-target-bundle] copied to {out_dir}")


@app.command()
def status(run_dir: Path = typer.Argument(..., help="outputs/<run_id>")) -> None:
    """Show task_status.csv of a run."""
    f = run_dir / "task_status.csv"
    if not f.exists():
        typer.echo("no task_status.csv", err=True)
        raise typer.Exit(1)
    typer.echo(f.read_text(encoding="utf-8"))


if __name__ == "__main__":
    app()
