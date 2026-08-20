"""Glycoform registry schemas (PRD v1.1 M02.3).

Glycoform composition is a VERSIONED INPUT: the registry below declares,
per N-glycosylation site, the glycan composition (IUPAC-condensed string +
CCD residue tree), occupancy and evidence level.  The structure prediction
layer never infers glycoforms - it consumes this registry (PRD boundary:
"从序列推断真实糖型" is explicitly out of scope).

Three sensitivity panels ship by default (assumed, no site-specific
glycoproteomics): high-mannose Man5, complex biantennary, and core-
fucosylated + terminal-sialylated biantennary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EvidenceLevel = Literal[
    "measured_site_specific",     # per-site LC-MS/MS data imported
    "literature",                 # published glycoproteomics on TROP2
    "assumed_sensitivity_panel",  # default: sensitivity-analysis hypothesis
]

# CCD residue codes for the glycan builder (PDB chemical component
# dictionary; understood by gemmi and Boltz-2)
GLYCAN_CCDS = {"NAG", "BMA", "MAN", "GAL", "FUC", "SIA"}

# standard N-linked core attachment: Asn ND2 -> GlcNAc C1 (beta)
N_LINK_PARENT_ATOM = "ND2"
N_LINK_CHILD_ATOM = "C1"


class GlycanResidue(BaseModel):
    """One residue in a glycan tree (CCD code + link to its parent)."""

    model_config = ConfigDict(extra="forbid")

    ccd: str = Field(..., description="CCD residue code, e.g. NAG/BMA/MAN/GAL/FUC/SIA")
    parent: int = Field(-1, ge=-1,
                        description="index of the parent residue (-1 = attached to Asn)")
    parent_atom: str = Field("", description="attachment atom on the parent (empty for root)")
    child_atom: str = Field("C1", description="anomeric carbon linking to the parent")

    @field_validator("ccd")
    @classmethod
    def _valid_ccd(cls, v: str) -> str:
        v = v.upper().strip()
        if v not in GLYCAN_CCDS:
            raise ValueError(f"unknown glycan CCD {v!r}; allowed: {sorted(GLYCAN_CCDS)}")
        return v


class SiteGlycoform(BaseModel):
    """Glycan composition + occupancy for ONE N-glycosylation site."""

    model_config = ConfigDict(extra="forbid")

    site: int = Field(..., ge=1, description="full-length residue number, e.g. 33")
    occupancy: float = Field(1.0, ge=0.0, le=1.0)
    iupac: str = Field(..., description="IUPAC-condensed composition string")
    residues: list[GlycanResidue] = Field(..., min_length=2,
                                          description="glycan tree, root first")


class GlycoformProfile(BaseModel):
    """One sensitivity panel: a complete assignment for all four sites."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    description: str = ""
    evidence_level: EvidenceLevel = "assumed_sensitivity_panel"
    source: str = "default sensitivity panel (no site-specific glycoproteomics)"
    sites: dict[str, SiteGlycoform] = Field(...,
                                            description="keyed by site number string")

    @model_validator(mode="after")
    def _tree_consistency(self):
        for key, sg in self.sites.items():
            if str(sg.site) != key:
                raise ValueError(f"sites key {key!r} must equal site number {sg.site}")
            seen = set()
            for i, res in enumerate(sg.residues):
                if i in seen:
                    continue
                seen.add(i)
                if res.parent >= i:
                    raise ValueError(
                        f"profile {self.profile_id} site {sg.site}: residue {i} "
                        f"parent index {res.parent} must be < {i} (roots first)")
                if i == 0 and res.parent != -1:
                    raise ValueError("first residue must attach to the Asn (parent=-1)")
        return self


class GlycoformRegistry(BaseModel):
    """Versioned collection of glycoform profiles."""

    model_config = ConfigDict(extra="forbid")

    registry_id: str
    version: str = "1"
    n_sites_expected: int = 4
    profiles: list[GlycoformProfile]

    @staticmethod
    def from_yaml(path: Path) -> "GlycoformRegistry":
        import yaml

        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return GlycoformRegistry.model_validate(raw)

    def profile(self, profile_id: str) -> GlycoformProfile:
        for p in self.profiles:
            if p.profile_id == profile_id:
                return p
        raise KeyError(f"glycoform profile {profile_id!r} not in registry "
                       f"{self.registry_id}; have {[p.profile_id for p in self.profiles]}")


# ---------------------------------------------------------------- bundle --

class TargetState(BaseModel):
    """One representative structure inside a published target bundle."""

    model_config = ConfigDict(extra="forbid")

    target_state_id: str
    glycoform_profile_id: str
    file: str                          # glycosylated_states/<id>.cif
    protein_only_view: str             # protein_only_views/<id>.cif
    md_cluster_weight: float = Field(..., ge=0.0, le=1.0)
    confidence: dict = Field(default_factory=dict)


class TargetBundleManifest(BaseModel):
    """Immutable downstream contract (PRD v1.1 section 7.3).

    target_bundle_id is derived from template hash + cleavage config +
    glycoform registry hash + software version + seed set; published once
    and never rewritten (new inputs => new bundle_id).
    """

    model_config = ConfigDict(extra="forbid")

    target_bundle_id: str
    target_bundle_version: str = "1.1"
    glycoform_registry_id: str
    profile_ids: list[str]
    evidence_level: EvidenceLevel
    glycan_site_occupancy: dict[str, dict] = Field(
        default_factory=dict,
        description="per-site occupancy per profile, e.g. "
                    "{'high_mannose_man5': {'33': 1.0, ...}}")
    states: list[TargetState]
    # per-state QC booleans are enforced upstream; the manifest carries the
    # aggregate verdicts
    cleavage_topology_pass: bool = False
    terminal_state_pass: bool = False
    disulfide_pass: bool = False
    glycan_topology_pass: bool = False
    target_uncertainty: dict = Field(default_factory=dict)

    @property
    def state_ids(self) -> list[str]:
        return [s.target_state_id for s in self.states]
