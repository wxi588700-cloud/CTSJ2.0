"""M06 T88-contact logic + M09 developability + M01 ingest unit tests."""
from __future__ import annotations

import numpy as np
import pytest

from trop2_design.io import first_protein_chain, read_structure
from trop2_design.scoring.binding import t88_terminal_evidence
from trop2_design.scoring.developability import (
    aggregation_hotspots, camsol_like_score, isoelectric_point,
    mhc2_risk_peptides, molecular_weight, net_charge,
    sequence_liabilities,
)
from trop2_design.target_builder.ingest import align_identity, qc_structure


class TestT88TerminalContact:
    """AC-07: recognise a pre-placed T88 terminus contact; AC-08: contact
    evidence is a cleaved-state-only concept."""

    def test_contact_detected_when_close(self, mini_target):
        st = read_structure(mini_target)
        chains = {"A": [r for r in first_protein_chain(st)]}
        t88 = next(r for r in chains["A"] if r.seqid.num == 61)
        n = t88.find_atom("N", "*")
        n_pos = np.array([n.pos.x, n.pos.y, n.pos.z])
        pose = n_pos + np.array([[2.0, 0, 0], [3.5, 0.5, 0.0], [6.0, 0, 0]])
        ev = t88_terminal_evidence(chains, 61, pose)
        assert ev["contacted"] is True
        assert ev["n_contacts"] == 2
        assert ev["min_distance"] == pytest.approx(2.0, abs=0.05)

    def test_no_contact_when_far(self, mini_target):
        st = read_structure(mini_target)
        chains = {"A": [r for r in first_protein_chain(st)]}
        pose = np.array([[50.0, 50.0, 50.0], [51.0, 50.0, 50.0]])
        ev = t88_terminal_evidence(chains, 61, pose)
        assert ev["contacted"] is False
        assert ev["min_distance"] > 4.5

    def test_missing_t88_raises(self, mini_target):
        # audit fix: a cleaved state without its T88 anchor is corrupt and
        # must fail fast (was: silent no-contact masking the corruption)
        st = read_structure(mini_target)
        chains = {"A": [r for r in first_protein_chain(st)]}
        with pytest.raises(ValueError, match="not found in cleaved-state chains"):
            t88_terminal_evidence(chains, 99999, np.zeros((5, 3)))

    def test_empty_pose_still_benign_no_contact(self, mini_target):
        st = read_structure(mini_target)
        chains = {"A": [r for r in first_protein_chain(st)]}
        ev = t88_terminal_evidence(chains, 61, np.zeros((0, 3)))
        assert ev["contacted"] is False
        assert ev["min_distance"] == 99.0


class TestDevelopability:
    def test_mw_sanity(self):
        # single residue + water: GLY
        assert molecular_weight("G") == pytest.approx(57.02146 + 18.010565, rel=1e-3)

    def test_pi_acidic_vs_basic(self):
        assert isoelectric_point("DDDDDDDDDD") < 5.0
        assert isoelectric_point("KKKKKKKKKK") > 9.0

    def test_net_charge_signs(self):
        assert net_charge("EEEE", 7.4) < 0
        assert net_charge("KKKK", 7.4) > 0

    def test_liabilities_detected(self):
        seq = "QCNGTNDGDSKRRMWWWMET" + "A" * 45  # NG, DG/DS, KR/RR, many M/W, N-term Q
        flags = sequence_liabilities(seq)
        names = {f["liability"] for f in flags}
        assert "deamidation_NG" in names
        assert "protease_KR" in names or "protease_RR" in names
        assert "nterm_cyclisation" in names
        assert "oxidation_MW" in names
        assert "nglycosylation_motif" in names  # N-G-T motif
        assert "rapid_renal_clearance_risk" in names  # 60-120 aa info flag

    def test_unpaired_cys_flagged(self):
        flags = sequence_liabilities("C" + "A" * 69)
        assert any(f["liability"] == "unpaired_cys" for f in flags)

    def test_mhc2_risk_ranking(self):
        seq = "AAAEAAKAAEAAKAAFWYLVLVLAAAEEEKKKAAAEEAAAKEA" + "A" * 40
        peptides = mhc2_risk_peptides(seq)
        assert peptides, "expected peptides for 60+ aa sequence"
        assert peptides[0]["propensity"] >= peptides[-1]["propensity"]

    def test_solubility_extremes(self):
        hydrophobic = camsol_like_score("LLLLLLLFFFFIIII" * 5)
        charged = camsol_like_score("EKEKEKEKEKEKEK" * 5)
        assert charged > hydrophobic

    def test_aggregation_hotspots(self):
        risk, hot = aggregation_hotspots("IIIIIII" + "E" * 53)
        assert risk > 0
        assert 1 in hot


class TestIngestHelpers:
    def test_align_identity_exact(self):
        assert align_identity("ACDEFGH", "ACDEFGH") == [(i, i) for i in range(7)]

    def test_align_identity_with_gap(self):
        pairs = align_identity("ACDEFGH", "ACEFGH")
        # positions of ACDEFGH covered: A,C,E,F,G,H
        assert (2, 2) not in pairs  # D has no partner
        mapped = {a for a, _ in pairs}
        assert {0, 1, 3, 4, 5, 6} == mapped

    def test_qc_detects_duplicates_and_mutations(self, mini_target, mini_seq, tmp_path):
        st = read_structure(mini_target)
        qc = qc_structure(st, None, mini_seq)
        assert qc["status"] in ("ok", "warning")
        # now corrupt: mutate residue 30 (A -> V)
        ch = first_protein_chain(st)
        for r in ch:
            if r.seqid.num == 30:
                r.name = "VAL"
        qc2 = qc_structure(st, None, mini_seq)
        assert qc2["status"] == "warning"
        assert qc2["mutations_vs_uniprot"]
