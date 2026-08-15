"""Metric dictionary and ranking profiles (PRD appendix A + section 12).

Thresholds live in versioned configuration: any threshold change produces a
new ``ranking_profile_id`` so results are always auditable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MetricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    direction: Literal["maximize", "minimize", "gate"]
    unit: str
    description: str
    source: str = ""
    proxy_allowed: bool = Field(True, description="may be computed by deterministic geometric proxy when predictor missing")


class GateRule(BaseModel):
    """One hard gate: a candidate violating it is rejected and the violation
    cannot be compensated by any weighted score (PRD 12.1)."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    op: Literal[">=", "<=", "==", "exists"]
    threshold: float | bool | None = None
    reject_message: str = ""


class WeightGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    weight: float = Field(ge=0, le=1)
    metrics: list[str]


class MetricsProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str
    metrics: list[MetricSpec]
    missing_value_policy: Literal["review", "reject", "zero"] = "review"

    def metric(self, name: str) -> MetricSpec | None:
        return next((m for m in self.metrics if m.name == name), None)


class HardFilterProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str
    gates: list[GateRule]
    display_weights: list[WeightGroup] = Field(default_factory=list)

    @field_validator("display_weights")
    @classmethod
    def _weights_sum(cls, v: list[WeightGroup]) -> list[WeightGroup]:
        if v:
            total = sum(w.weight for w in v)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"display weights must sum to 1.0, got {total}")
        return v

    @staticmethod
    def from_yaml(path: Path) -> "HardFilterProfile":
        import yaml

        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return HardFilterProfile.model_validate(raw)


def default_metrics_profile() -> MetricsProfile:
    """Metric dictionary from PRD appendix A."""
    specs = [
        ("fold_plddt", "maximize", "0-100", "binder monomer fold confidence", "AF2/ColabFold"),
        ("complex_iptm", "maximize", "0-1", "complex interface confidence", "AF2-Multimer"),
        ("interface_pae", "minimize", "A", "interface positional uncertainty", "AF2-Multimer"),
        ("t88_contact", "gate", "bool+occupancy", "direct recognition of the T88 free alpha-amino terminus", "atom geometry"),
        ("t88_contact_occupancy", "maximize", "0-1", "fraction of cleaved states with direct T88 contact", "atom geometry"),
        ("positive_pass_rate", "maximize", "0-1", "fraction of cleaved conformers passing binding thresholds", "cross-state statistics"),
        ("intact_risk", "minimize", "normalized", "binding risk against intact TROP2", "negative-state prediction"),
        ("epcam_risk", "minimize", "normalized", "binding risk against EpCAM", "negative-state prediction"),
        ("cis_block", "maximize", "0-1", "coverage/blocking of the cis interface", "structural superposition"),
        ("trans_occlusion", "minimize", "0-1", "occlusion of the trans interface", "structural superposition"),
        ("glycan_membrane_clash", "minimize", "count", "glycan and membrane collisions", "spatial geometry"),
        ("shape_complementarity", "maximize", "0-1", "interface shape complementarity", "interface analyzer"),
        ("buried_unsat", "minimize", "count", "buried unsatisfied polar atoms", "interface analyzer"),
        ("aggregation_risk", "minimize", "relative", "structural aggregation hotspots", "A3D/CamSol-style"),
        ("solubility_score", "maximize", "0-1", "sequence+surface solubility estimate", "CamSol-style"),
        ("mhc2_risk", "minimize", "peptides", "MHC-II presentation risk", "NetMHCIIpan"),
        ("liability_count", "minimize", "count", "sequence liability flags", "sequence rules"),
        ("uncertainty", "minimize", "sd", "cross-model/seed/state disagreement", "aggregate statistics"),
    ]
    return MetricsProfile(
        profile_id="metrics_v1",
        metrics=[MetricSpec(name=n, direction=d, unit=u, description=desc, source=src) for n, d, u, desc, src in specs],
    )


def v1_strict_profile() -> HardFilterProfile:
    """Hard gates from PRD 12.1/12.2 - every violation is terminal."""
    return HardFilterProfile(
        profile_id="v1_strict",
        gates=[
            GateRule(metric="positive_state_pass_rate", op=">=", threshold=0.2,
                     reject_message="cleaved-state complex not reproducibly predicted"),
            GateRule(metric="t88_contact", op="==", threshold=True,
                     reject_message="no direct T88 neo-N-terminus recognition"),
            GateRule(metric="intact_risk", op="<=", threshold=0.55,
                     reject_message="high-confidence binding to intact TROP2"),
            GateRule(metric="epcam_risk", op="<=", threshold=0.55,
                     reject_message="high-confidence binding to EpCAM"),
            GateRule(metric="glycan_membrane_clash", op="<=", threshold=0,
                     reject_message="unacceptable membrane/glycan collision"),
            GateRule(metric="fold_plddt", op=">=", threshold=70.0,
                     reject_message="binder monomer does not fold confidently"),
            # aggregation_risk is a RELATIVE score (appendix A); 0.70 marks the
        # "obvious self-aggregation" regime for the V1 heuristic scorer
        GateRule(metric="aggregation_risk", op="<=", threshold=0.70,
                     reject_message="obvious self-aggregation tendency"),
            GateRule(metric="trans_occlusion", op="<=", threshold=0.40,
                     reject_message="binder sterically occludes the trans interface"),
        ],
        display_weights=[
            WeightGroup(name="cleaved binding & T88 terminus", weight=0.25,
                        metrics=["positive_pass_rate", "t88_contact_occupancy", "complex_iptm"]),
            WeightGroup(name="intact/EpCAM selectivity", weight=0.25,
                        metrics=["intact_risk", "epcam_risk"]),
            WeightGroup(name="cis/trans, membrane & glycan geometry", weight=0.15,
                        metrics=["cis_block", "trans_occlusion", "glycan_membrane_clash"]),
            WeightGroup(name="fold & interface stability", weight=0.15,
                        metrics=["fold_plddt", "shape_complementarity", "buried_unsat"]),
            WeightGroup(name="solubility & manufacturability", weight=0.10,
                        metrics=["solubility_score", "aggregation_risk"]),
            WeightGroup(name="immuno/protease/chemical risk", weight=0.10,
                        metrics=["mhc2_risk", "liability_count"]),
        ],
    )
