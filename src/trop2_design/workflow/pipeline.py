"""Pipeline construction (M00): turns a validated ProjectConfig into a DAG.

Implements the PRD 8.1 main chain:
M00 -> M01 -> M02 -> M03 -> M04 -> M05 -> M06 -> (M07, M08, M09) -> M10
with the dependency rules of PRD 8.2 (only M06-positive candidates proceed to
the expensive negative-state and developability stages).
"""
from __future__ import annotations

import platform
from pathlib import Path

from ..schemas.results import RunManifest
from ..io import content_hash, sha256_file, write_json
from .engine import CODE_VERSION, RunContext, Stage, WorkflowRunner, git_commit, python_version


def new_run_id() -> str:
    import datetime as _dt

    return _dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")


def build_context(project_root: Path, config, tools, run_id: str | None = None) -> RunContext:
    run_id = run_id or new_run_id()
    out_dir = project_root / "outputs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # resolved config copy lives next to outputs (M00 standard output)
    resolved = out_dir / "resolved_config.yaml"
    import yaml

    with open(resolved, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config.resolved_copy(), fh, sort_keys=False, allow_unicode=True)
    return RunContext(project_root=project_root, out_dir=out_dir, config=config,
                      tools=tools, seed=config.resources.seed)


def build_manifest(project_root: Path, config, tools, ctx: RunContext) -> RunManifest:
    input_hashes: dict[str, str] = {}
    t = config.target
    for label, path in [
        ("trop2_fasta", t.sequence_fasta),
        ("cis_structure", t.cis_structure.path),
        ("trans_structure", t.trans_structure.path),
        ("alternate_structure", t.alternate_structure.path if t.alternate_structure else None),
        ("epcam_structure", config.negatives.epcam_structure),
        ("epcam_fasta", config.negatives.epcam_fasta),
    ]:
        if path is not None and Path(path).exists():
            input_hashes[label] = sha256_file(path)
    tool_versions = {"python": python_version(), "package": CODE_VERSION}
    licenses: dict[str, str] = {}
    if tools:
        for name in ("rfdiffusion", "proteinmpnn", "foldseek", "mmseqs2", "netmhc2pan"):
            spec = getattr(tools, name)
            if spec is not None:
                tool_versions[name] = spec.version or "local-checkout"
                licenses[name] = spec.license
        for key, pred in tools.predictors.items():
            tool_versions[f"predictor:{key}"] = pred.kind
            licenses[f"predictor:{key}"] = pred.license
    return RunManifest(
        run_id=ctx.out.name,
        config_hash=content_hash(config.resolved_copy()),
        config_copy=config.resolved_copy(),
        seed=config.resources.seed,
        input_hashes=input_hashes,
        tool_versions=tool_versions,
        licenses=licenses,
        git_commit=git_commit(project_root),
        platform={
            "python": python_version(),
            "os": platform.platform(),
            "machine": platform.machine(),
        },
    )


def build_pipeline(ctx: RunContext, manifest: RunManifest) -> WorkflowRunner:
    """Assemble the full M00-M10 DAG with PRD 8.2 dependency rules."""
    from ..target_builder import ingest as m01
    from ..target_builder import cleave as m02
    from ..epitope import patch as m03
    from ..generation import generate as m04
    from ..sequence_design import design as m05
    from ..scoring import binding as m06
    from ..scoring import specificity as m07
    from ..scoring import mechanism as m08
    from ..scoring import developability as m09
    from ..ranking import rank as m10

    out = ctx.out
    stages = [
        Stage("M01_target_ingestion", m01.run,
              inputs=[ctx.config.target.sequence_fasta, ctx.config.target.cis_structure.path,
                      ctx.config.target.trans_structure.path],
              outputs=[out / "target_registry.json", out / "residue_mapping.csv",
                       out / "input_qc.json"],
              description="standardise structures, sequence and numbering (M01)"),
        Stage("M02_cleaved_states", m02.run,
              depends_on=["M01_target_ingestion"],
              inputs=[out / "target_registry.json"],
              outputs=[out / "state_manifest.csv", out / "topology_audit.json"],
              description="R87-T88 chain break, disulfide audit, conformer ensemble (M02)"),
        Stage("M03_epitope", m03.run,
              depends_on=["M02_cleaved_states"],
              inputs=[out / "state_manifest.csv"],
              outputs=[out / "epitope_patch.json", out / "hotspots.txt",
                       out / "exclusion_mask.json", out / "accessibility_metrics.csv"],
              description="T88 epitope, hotspots, glycan/membrane exclusion (M03)"),
        Stage("M04_generate", m04.run,
              depends_on=["M03_epitope"],
              inputs=[out / "epitope_patch.json"],
              outputs=[out / "candidates.fasta", out / "candidate_manifest.csv"],
              description="RFdiffusion adapter + FASTA/PDB import (M04)"),
        Stage("M05_sequence_design", m05.run,
              depends_on=["M04_generate"],
              inputs=[out / "candidate_manifest.csv"],
              outputs=[out / "monomer_metrics.csv"],
              description="ProteinMPNN adapter + monomer fold filter (M05)"),
        Stage("M06_positive_state", m06.run,
              depends_on=["M05_sequence_design"],
              inputs=[out / "monomer_metrics.csv", out / "state_manifest.csv"],
              outputs=[out / "positive_state_metrics.csv", out / "terminal_contact.json"],
              description="cleaved-state binding + T88 neo-terminus contact (M06)"),
        Stage("M07_negative_state", m07.run,
              depends_on=["M06_positive_state"],
              inputs=[out / "positive_state_metrics.csv"],
              outputs=[out / "negative_state_metrics.csv", out / "offtarget_hits.csv"],
              description="intact TROP2 / EpCAM negative design + off-target screen (M07)"),
        Stage("M08_mechanism", m08.run,
              depends_on=["M06_positive_state"],
              inputs=[out / "positive_state_metrics.csv"],
              outputs=[out / "mechanism_metrics.csv", out / "clash_report.json"],
              description="cis block / trans occlusion / membrane-glycan clashes (M08)"),
        Stage("M09_developability", m09.run,
              depends_on=["M06_positive_state"],
              inputs=[out / "positive_state_metrics.csv"],
              outputs=[out / "developability_metrics.csv", out / "liability_flags.csv",
                       out / "immunogenicity_hits.csv"],
              description="solubility, aggregation, MHC-II, sequence liabilities (M09)"),
        Stage("M10_ranking", m10.run,
              depends_on=["M06_positive_state", "M07_negative_state",
                          "M08_mechanism", "M09_developability"],
              inputs=[out / "positive_state_metrics.csv", out / "negative_state_metrics.csv",
                      out / "mechanism_metrics.csv", out / "developability_metrics.csv"],
              outputs=[out / "candidate_metrics.csv", out / "pareto_front.csv",
                       out / "rejection_reasons.csv", out / "report.html"],
              description="hard gates, Pareto, diversity clustering, report (M10)"),
    ]
    return WorkflowRunner(stages, ctx, manifest)
