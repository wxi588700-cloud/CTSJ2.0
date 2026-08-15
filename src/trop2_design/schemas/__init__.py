"""Pydantic data models for the TROP2 cleaved-state binder design platform.

Module numbering follows PRD v1.0 section 5/6.  All user-facing configuration
models forbid unknown fields so that medical researchers get an explicit
error instead of a silently ignored typo (PRD 5.2 / engineering conventions).
"""
from .project import (
    ProjectConfig,
    CleavageConfig,
    TargetConfig,
    DesignConfig,
    NegativesConfig,
    RankingConfig,
    ResourceConfig,
)
from .tools import ToolsConfig, ToolSpec, PredictorSpec
from .metrics import MetricSpec, MetricsProfile, HardFilterProfile, GateRule, WeightGroup
from .results import (
    TopologyAudit,
    StateManifestRow,
    TerminalContact,
    CandidateRecord,
    RunManifest,
    StageStatus,
)

__all__ = [
    "ProjectConfig", "CleavageConfig", "TargetConfig", "DesignConfig",
    "NegativesConfig", "RankingConfig", "ResourceConfig",
    "ToolsConfig", "ToolSpec", "PredictorSpec",
    "MetricSpec", "MetricsProfile", "HardFilterProfile", "GateRule", "WeightGroup",
    "TopologyAudit", "StateManifestRow", "TerminalContact", "CandidateRecord",
    "RunManifest", "StageStatus",
]
