"""Project-level configuration (PRD 7.2 项目配置示例).

The YAML consumed by the CLI is validated against ``ProjectConfig``.  Unknown
fields are rejected so mis-typed medical parameters fail before any expensive
computation starts.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_RESIDUE_RE = re.compile(r"^([A-Z])(-?\d+)([A-Z])?$")


class StrictModel(BaseModel):
    """Base model rejecting unknown fields (PRD: 未知字段必须在运行前报错)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def parse_residue_id(spec: str) -> tuple[str, int]:
    """Parse residue identifiers like ``R87`` / ``C108`` / ``T-5``.

    Returns ``(aa_one_letter, number)``.  Raises ``ValueError`` on malformed
    identifiers so invalid residue numbers fail at config time.
    """
    m = _RESIDUE_RE.match(spec.strip().upper())
    if not m:
        raise ValueError(f"Invalid residue id {spec!r}: expected e.g. 'R87', 'T88', 'C108'")
    aa, num = m.group(1), int(m.group(2))
    if aa not in "ACDEFGHIKLMNPQRSTVWY":
        raise ValueError(f"Invalid amino acid letter in residue id {spec!r}")
    return aa, num


class CleavageConfig(StrictModel):
    """Definition of the R87-T88 proteolytic cleavage (PRD 2.1 / M02)."""

    left_residue: str = Field(..., description="Residue becoming the new C-terminus, e.g. R87")
    right_residue: str = Field(..., description="Residue becoming the new N-terminus, e.g. T88")
    preserve_disulfides: list[tuple[str, str]] = Field(
        default=[("C73", "C108")],
        description="Disulfides that must survive cleavage, binding the two fragments",
    )
    left_terminal_state: Literal["COO-", "COOH", "amide"] = "COO-"
    right_terminal_state: Literal["NH3+", "NH2", "acetyl"] = "NH3+"
    # Numbering convention used by the config file.  ``author`` means the
    # numbering found in the input mmCIF files (7E5N/7E5M/7PEE use UniProt
    # numbering, verified to contain R87/T88/C73/C108 directly).
    numbering: Literal["author", "uniprot"] = "author"

    @field_validator("left_residue", "right_residue")
    @classmethod
    def _validate_residue(cls, v: str) -> str:
        parse_residue_id(v)
        return v.strip().upper()

    @field_validator("preserve_disulfides")
    @classmethod
    def _validate_disulfides(cls, v):
        for pair in v:
            if len(pair) != 2:
                raise ValueError(f"Each preserve_disulfides entry needs 2 residues, got {pair}")
            for r in pair:
                aa, _ = parse_residue_id(r)
                if aa != "C":
                    raise ValueError(f"disulfide residue {r} is not a cysteine (C...)")
        return v

    @model_validator(mode="after")
    def _adjacent_cleavage(self):
        try:
            self.site  # raises when numbering is not adjacent
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self

    @property
    def site(self) -> tuple[int, int]:
        left_aa, left_num = parse_residue_id(self.left_residue)
        right_aa, right_num = parse_residue_id(self.right_residue)
        if right_num != left_num + 1:
            raise ValueError(
                f"Cleavage residues must be adjacent in sequence numbering: "
                f"{self.left_residue} -> {self.right_residue}"
            )
        return left_num, right_num


class StructureRef(StrictModel):
    path: Path
    chain: str | None = Field(None, description="Polymer chain to extract; None = first protein chain")
    role: str = Field("template", description="semantic role recorded in the target registry")


class TargetConfig(StrictModel):
    name: str = "TROP2"
    uniprot_id: str = "P09758"
    gene: str = "TACSTD2"
    species: str = "Homo sapiens"
    sequence_fasta: Path
    cis_structure: StructureRef
    trans_structure: StructureRef
    alternate_structure: StructureRef | None = None
    cleavage: CleavageConfig
    glycosylation_sites: list[int] = Field(
        default_factory=lambda: [33, 120, 168, 184],
        description="UniProt/author numbering of N-glycosylation sites used for geometric exclusion",
    )
    glycan_exclusion_radius: float = Field(12.0, ge=4.0, le=30.0, description="Å sphere around glycan anchor")
    membrane_clearance: float = Field(8.0, ge=2.0, le=30.0, description="Å binder must stay above membrane plane")


class DesignConfig(StrictModel):
    binder_length: tuple[int, int] = (60, 120)
    generator: Literal["rfdiffusion", "import", "both"] = "rfdiffusion"
    hotspot_mode: Literal["t88_neo_n_terminus", "manual"] = "t88_neo_n_terminus"
    hotspot_radius: float = Field(10.0, ge=4.0, le=25.0)
    n_designs_per_scaffold: int = Field(3, ge=1, le=20)
    max_candidates: int = Field(24, ge=1, le=500)
    import_fasta: Path | None = None
    import_pdb_dir: Path | None = None
    fixed_interface_positions: list[int] = Field(default_factory=list)
    forbidden_aa: list[str] = Field(default_factory=lambda: ["C"], description="single letters disallowed in designed seqs")
    allowed_aa: list[str] | None = None


class NegativesConfig(StrictModel):
    targets: list[str] = Field(
        default_factory=lambda: ["intact_trop2_cis", "intact_trop2_trans", "human_epcam"],
        description="negative states evaluated in M07",
    )
    epcam_structure: Path | None = None
    epcam_uniprot: str = "P16422"
    epcam_fasta: Path | None = None
    surfaceome_screen: bool = Field(False, description="run optional Foldseek/MMseqs2 off-target screen")


class RankingConfig(StrictModel):
    hard_filter_profile: str = "v1_strict"
    metrics_profile: str = "metrics_v1"
    method: Literal["pareto", "pareto_weighted"] = "pareto"
    diversity_cluster_identity: float = Field(0.70, ge=0.3, le=1.0)
    max_per_family: int = Field(3, ge=1, le=12)
    export_top_n: int = Field(24, ge=1, le=100)
    uncertainty_lambda: float = Field(0.5, ge=0.0, le=5.0, description="λ in robust_selectivity penalty")
    robust_positive_quantile: float = Field(0.10, ge=0.0, le=0.5)


class ResourceConfig(StrictModel):
    seed: int = 20260816
    max_cpu: int = Field(8, ge=1, le=256)
    gpu_required: bool = False
    max_residues_per_predict: int = Field(1200, ge=100, le=20000, description="pre-flight OOM guard")
    allow_proxy_metrics: bool = Field(
        True,
        description="permit deterministic geometric proxies when a predictor is unavailable "
        "(always flagged metric_source='proxy' and review-required)",
    )
    boltz_recompute_top_k: int = Field(
        8, ge=0, le=100,
        description="when Boltz is configured: re-run real complex prediction for the "
                    "top-K candidates (per cleaved conformer), replacing proxy "
                    "ipTM/pLDDT/PAE with measured values; 0 disables GPU recomputation",
    )

    def forbid_proxy_degradation(self, what: str) -> None:
        """Fail-fast guard for degraded/proxy paths (audit fix).

        ``allow_proxy_metrics=false`` converts the historical *silent*
        degradations (deterministic scaffold fallback, heuristic fold scores,
        proxy binding metrics when Boltz is unavailable) into explicit
        RuntimeErrorS, so a production run can never finish on proxy data
        without the operator knowing.
        """
        if not self.allow_proxy_metrics:
            raise RuntimeError(
                f"{what} - unavailable and allow_proxy_metrics=false; refusing "
                f"to silently degrade to proxy. Configure the real tool in "
                f"tools.yaml or set allow_proxy_metrics: true")


class GlycosylationConfig(StrictModel):
    enabled: bool = True
    sites: list[int] = Field(default_factory=lambda: [33, 120, 168, 208])
    profile_ids: list[str] = Field(default_factory=lambda: [
        "high_mannose_man5", "complex_biantennary",
        "core_fucosylated_sialylated"])
    registry: Path = Field(Path("configs/glycoforms_v1.yaml"))
    source: Literal["assumed_sensitivity_panel", "literature",
                    "measured_site_specific"] = "assumed_sensitivity_panel"


class TargetPredictionConfig(StrictModel):
    """PRD v1.1 target_prediction block: glyco_ensemble mode switches M02
    to the hybrid bundle builder; absent block = legacy v1.0 path."""
    mode: Literal["glyco_ensemble", "legacy_cleaved"] = "glyco_ensemble"
    target_bundle_version: str = "1.1"
    glycosylation: GlycosylationConfig = GlycosylationConfig()
    seeds: list[int] = Field(default_factory=lambda: [20260817, 20260818, 20260819])
    graft_seeds: int = Field(2, ge=1, le=5)
    min_representatives: int = Field(5, ge=1, le=20)
    sampling_steps: int = Field(100, ge=20, le=400)


class ProjectConfig(StrictModel):
    project: dict[str, str] = Field(..., description="name/species/seed metadata block")
    target: TargetConfig
    design: DesignConfig
    negatives: NegativesConfig
    ranking: RankingConfig
    resources: ResourceConfig = ResourceConfig()
    target_prediction: TargetPredictionConfig | None = Field(
        None, description="PRD v1.1 glyco_ensemble mode; None = legacy v1.0")

    @field_validator("project")
    @classmethod
    def _project_block(cls, v: dict[str, str]) -> dict[str, str]:
        if "name" not in v:
            raise ValueError("project.name is required")
        return v

    @staticmethod
    def from_yaml(path: Path) -> "ProjectConfig":
        import yaml

        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return ProjectConfig.model_validate(raw)

    def resolved_copy(self) -> dict:
        """Serialised config with deterministic ordering for hashing."""
        import json

        return json.loads(self.model_dump_json())
