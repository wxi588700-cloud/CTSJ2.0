"""External tool / predictor configuration (M00).

Tools are located through ``configs/tools.yaml``.  Everything is optional at
import time; each adapter performs an availability probe before use and the
run manifest records which tool actually produced each artifact.  Algorithm
code downloaded previously (RFdiffusion, ProteinMPNN) is referenced here by
path and copied in by ``scripts/setup_external_tools.sh``.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ToolSpec(BaseModel):
    """A locally installed external algorithm."""

    model_config = ConfigDict(extra="forbid")

    root: Path = Field(..., description="checkout directory of the tool")
    command: list[str] = Field(default_factory=list, description="argv prefix, e.g. [python, run_inference.py]")
    python: Path | None = Field(None, description="interpreter of the tool's own conda env")
    weights: Path | None = None
    license: str = "unknown"
    version: str | None = None

    def resolve(self, rel: str) -> Path:
        return (self.root / rel).resolve()


class PredictorSpec(BaseModel):
    """A complex/monomer structure predictor adapter."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(..., description="adapter key: af2_multimer | colabfold | boltz | heuristic")
    python: Path | None = None
    weights: Path | None = None
    license: str = "unknown"
    notes: str = ""
    version: str | None = Field(None, description="installed version tag")
    device: int | None = Field(
        None, ge=0,
        description="pin the predictor to this CUDA_VISIBLE_DEVICES index "
                    "(e.g. 6); None = auto-pick the GPU with most free VRAM")


class ToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rfdiffusion: ToolSpec | None = None
    proteinmpnn: ToolSpec | None = None
    foldseek: ToolSpec | None = None
    mmseqs2: ToolSpec | None = None
    netmhc2pan: ToolSpec | None = None
    predictors: dict[str, PredictorSpec] = Field(default_factory=dict)

    @staticmethod
    def from_yaml(path: Path) -> "ToolsConfig":
        import yaml

        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        return ToolsConfig.model_validate(raw)

    def primary_predictor(self) -> PredictorSpec:
        """Primary complex predictor; falls back to the deterministic
        geometric proxy adapter so the pipeline stays runnable on CPU-only
        machines (every proxy metric is flagged and gated behind review)."""
        for key in ("af2_multimer", "colabfold", "boltz"):
            if key in self.predictors:
                return self.predictors[key]
        return PredictorSpec(kind="heuristic", license="MIT", notes="built-in deterministic geometric proxy")
