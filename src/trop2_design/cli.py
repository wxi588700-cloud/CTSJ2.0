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
    "generate": {"M04_generate"},
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
def prepare(project: Path = typer.Option("configs/trop2_v1.yaml")) -> None:
    """Stages M01-M03: ingestion, cleaved states, epitope."""
    run(project=project, stages="prepare", run_id=None, tools=None)


@app.command()
def generate(project: Path = typer.Option("configs/trop2_v1.yaml")) -> None:
    """Stage M04: candidate generation + import."""
    run(project=project, stages="generate", run_id=None, tools=None)


@app.command()
def evaluate(project: Path = typer.Option("configs/trop2_v1.yaml")) -> None:
    """Stages M05-M09: sequence design, positive/negative states, mechanism, developability."""
    run(project=project, stages="evaluate", run_id=None, tools=None)


@app.command()
def rank(project: Path = typer.Option("configs/trop2_v1.yaml")) -> None:
    """Stage M10: hard gates, Pareto ranking, shortlist."""
    run(project=project, stages="rank", run_id=None, tools=None)


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

    cfg = ProjectConfig.model_validate(yaml.safe_load(resolved.read_text()))
    tools = ToolsConfig()
    mf = run_dir / "run_manifest.json"
    seed = json.loads(mf.read_text())["seed"] if mf.exists() else cfg.resources.seed

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


@app.command()
def status(run_dir: Path = typer.Argument(..., help="outputs/<run_id>")) -> None:
    """Show task_status.csv of a run."""
    f = run_dir / "task_status.csv"
    if not f.exists():
        typer.echo("no task_status.csv", err=True)
        raise typer.Exit(1)
    typer.echo(f.read_text())


if __name__ == "__main__":
    app()
