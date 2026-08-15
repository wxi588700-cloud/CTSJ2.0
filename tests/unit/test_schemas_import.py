"""Schema validation tests (M00) + AC-04 import validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from trop2_design.generation.generate import (
    import_fasta_candidates, stable_candidate_id,
)
from trop2_design.schemas.project import (
    CleavageConfig, ProjectConfig, parse_residue_id,
)


class TestSchemas:
    def test_unknown_field_rejected(self):
        with pytest.raises(Exception):
            CleavageConfig.model_validate({
                "left_residue": "R87", "right_residue": "T88",
                "totally_unknown_field": 1,
            })

    def test_invalid_residue_id_rejected(self):
        with pytest.raises(ValueError):
            parse_residue_id("87")
        with pytest.raises(ValueError):
            parse_residue_id("X87")
        with pytest.raises(ValueError):
            parse_residue_id("R8a7")

    def test_cleavage_adjacency(self):
        cfg = CleavageConfig(left_residue="R87", right_residue="T88")
        assert cfg.site == (87, 88)
        with pytest.raises(Exception):
            CleavageConfig(left_residue="R87", right_residue="T89")

    def test_disulfides_must_be_cys(self):
        with pytest.raises(Exception):
            CleavageConfig(left_residue="R87", right_residue="T88",
                           preserve_disulfides=[["A73", "C108"]])

    def test_config_roundtrip(self, project_root: Path):
        cfg = ProjectConfig.from_yaml(project_root / "configs" / "trop2_v1.yaml")
        assert cfg.target.cleavage.left_residue == "R87"
        assert cfg.target.cleavage.preserve_disulfides == [("C73", "C108")]
        assert cfg.ranking.hard_filter_profile == "v1_strict"
        assert cfg.ranking.export_top_n >= 12  # PRD: 12-24 shortlist


class TestImportValidation:
    """AC-04: legal candidates get stable IDs, illegal ones are rejected
    with recorded reasons."""

    def test_valid_fasta_accepted_with_stable_id(self, tmp_path: Path):
        fasta = tmp_path / "cands.fasta"
        seq = "AEAAKAAEAAKAAEAAKAAEAAKAAEAAKAAEAAKAAEAAKAAEAAK" + \
              "AEAAKAAEAAKAAEAAKAAEAAKAAEAAK"  # 76 aa
        fasta.write_text(f">cand1 legal\n{seq}\n")
        acc, rej = import_fasta_candidates(fasta, (60, 120))
        assert len(acc) == 1 and not rej
        assert acc[0]["candidate_id"].startswith("CAND-")
        # stable: same input -> same id
        acc2, _ = import_fasta_candidates(fasta, (60, 120))
        assert acc2[0]["candidate_id"] == acc[0]["candidate_id"]

    def test_invalid_chars_rejected(self, tmp_path: Path):
        fasta = tmp_path / "bad.fasta"
        seq = "AEAAK" * 12 + "J" + "AEAAK" * 12  # invalid J
        fasta.write_text(f">bad\n{seq}\n")
        acc, rej = import_fasta_candidates(fasta, (60, 120))
        assert not acc and len(rej) == 1
        assert any("invalid characters" in e for e in rej[0]["errors"])

    def test_length_out_of_range_rejected(self, tmp_path: Path):
        fasta = tmp_path / "short.fasta"
        fasta.write_text(">short\nAEAAKAAEAA\n")
        acc, rej = import_fasta_candidates(fasta, (60, 120))
        assert not acc
        assert any("length" in e for e in rej[0]["errors"])

    def test_low_complexity_rejected(self, tmp_path: Path):
        fasta = tmp_path / "lc.fasta"
        fasta.write_text(">lc\n" + "A" * 80 + "\n")
        _, rej = import_fasta_candidates(fasta, (60, 120))
        assert any("low-complexity" in e for e in rej[0]["errors"])

    def test_stable_candidate_id_deterministic(self):
        a = stable_candidate_id("src", "x", 1)
        b = stable_candidate_id("src", "x", 1)
        c = stable_candidate_id("src", "x", 2)
        assert a == b and a != c
