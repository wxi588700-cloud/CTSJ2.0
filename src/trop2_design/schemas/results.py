"""Result-record schemas (run manifest, topology audit, candidate metrics)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StageStatus(BaseModel):
    """One row of task_status.csv (M00 standard output)."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    status: Literal["ok", "cached", "skipped", "failed", "running"]
    cache_key: str = ""
    started: str = ""
    finished: str = ""
    duration_s: float = 0.0
    note: str = ""


class RunManifest(BaseModel):
    """run_manifest.json - full audit of a single pipeline run (M00)."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    created: str = Field(default_factory=utcnow)
    git_commit: str = "unknown"
    package_version: str = "1.0.0"
    config_hash: str
    config_copy: dict[str, Any]
    seed: int
    input_hashes: dict[str, str] = Field(default_factory=dict)
    tool_versions: dict[str, str] = Field(default_factory=dict)
    licenses: dict[str, str] = Field(default_factory=dict)
    stages: list[StageStatus] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    platform: dict[str, str] = Field(default_factory=dict)


class TopologyAudit(BaseModel):
    """Chemical-topology audit of one cleaved/intact state (M02 standard output)."""

    model_config = ConfigDict(extra="forbid")

    state_id: str
    kind: Literal["cleaved", "intact"]
    source_structure: str
    source_chain: str
    peptide_bond_left_right: bool
    left_terminal: str = Field("", description="e.g. 'R87 new C-terminus COO-'")
    right_terminal: str = Field("", description="e.g. 'T88 new N-terminus NH3+'")
    chains: list[str] = Field(default_factory=list)
    residues: int = 0
    disulfides: list[tuple[int, int]] = Field(default_factory=list)
    required_disulfides_present: bool = False
    max_clash_overlap: float = 0.0
    min_nonbonded_distance: float = 99.0
    passed: bool = False
    failures: list[str] = Field(default_factory=list)
    transformations: list[str] = Field(default_factory=list)


class StateManifestRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state_id: str
    kind: Literal["cleaved", "intact"]
    file: str
    audit_passed: bool
    audit_failures: list[str] = Field(default_factory=list)


class TerminalContact(BaseModel):
    """M06 T88 neo-N-terminus contact record (hard-gate evidence)."""

    model_config = ConfigDict(extra="forbid")

    state_id: str
    candidate_id: str
    t88_residue: str = "T88"
    contacted: bool
    min_distance: float = 99.0
    contact_atoms: list[str] = Field(default_factory=list)
    n_contacts: int = 0
    orientation_score: float = 0.0
    binder_atoms: list[str] = Field(default_factory=list)


class CandidateRecord(BaseModel):
    """Row of candidate_metrics.csv (PRD 7.3 minimum fields)."""

    model_config = ConfigDict(extra="allow")

    candidate_id: str
    sequence: str
    source: str
    backbone_family: str = ""
    positive_state_pass_rate: float | None = None
    t88_terminal_contact: bool | None = None
    t88_contact_occupancy: float | None = None
    intact_trop2_risk: float | None = None
    epcam_risk: float | None = None
    cis_block_score: float | None = None
    trans_occlusion_score: float | None = None
    glycan_membrane_clash: float | None = None
    developability_flags: list[str] = Field(default_factory=list)
    uncertainty: float | None = None
    hard_filter_status: Literal["pass", "reject", "review"] = "review"
    rejection_reasons: list[str] = Field(default_factory=list)
    pareto_rank: int | None = None
    robust_positive: float | None = None
    worst_offtarget: float | None = None
    robust_selectivity: float | None = None
    weighted_display_score: float | None = None
    metric_source: dict[str, str] = Field(default_factory=dict, description="per-metric 'measured'/'proxy' provenance")
