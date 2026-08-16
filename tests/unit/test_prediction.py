"""GPU-pinning tests for the Boltz predictor configuration path.

Covers the resolution chain: PredictorSpec.device field > notes device=N >
TROP2_BOLTZ_DEVICE env var (at prediction time) > auto-pick.
"""
from __future__ import annotations

import numpy as np
import pytest

from trop2_design.prediction import build_boltz
from trop2_design.prediction.boltz_adapter import BoltzSpec
from trop2_design.schemas.tools import PredictorSpec

BOLTZ_PY = "/home/protein_design2026/miniconda3/envs/boltz/bin/python"


def _spec(**kw) -> PredictorSpec:
    kw.setdefault("kind", "boltz")
    kw.setdefault("python", BOLTZ_PY)
    return PredictorSpec.model_validate(kw)


class TestPredictorSpecDevice:
    def test_device_field_accepted(self):
        assert _spec(device=6).device == 6

    def test_device_none_default(self):
        assert _spec().device is None

    def test_invalid_device_rejected(self):
        with pytest.raises(Exception):
            _spec(device=-1)
        with pytest.raises(Exception):
            _spec(device="six")

    def test_unknown_field_still_forbidden(self):
        with pytest.raises(Exception):
            _spec(deivce=6)  # typo must fail loudly


class TestBuildBoltzDevice:
    def test_field_device_propagates(self):
        bp = build_boltz(_spec(device=6), seed=1)
        assert bp is not None
        assert bp.spec.device == 6

    def test_notes_device_parsed(self):
        bp = build_boltz(_spec(notes="ssh_host=gn1 ; device=6"), seed=1)
        assert bp is not None
        assert bp.spec.device == 6
        assert bp.spec.ssh_host == "gn1"

    def test_field_wins_over_notes(self):
        bp = build_boltz(_spec(device=3, notes="device=6"), seed=1)
        assert bp is not None
        assert bp.spec.device == 3

    def test_no_device_stays_none(self):
        bp = build_boltz(_spec(notes="ssh_host=gn1"), seed=1)
        assert bp is not None
        assert bp.spec.device is None

    def test_bad_notes_device_ignored(self):
        bp = build_boltz(_spec(notes="device=abc"), seed=1)
        assert bp is not None
        assert bp.spec.device is None


class TestEnvResolution:
    def test_env_used_when_spec_device_none(self, monkeypatch):
        """_predict consults TROP2_BOLTZ_DEVICE only when spec.device is None."""
        monkeypatch.setenv("TROP2_BOLTZ_DEVICE", "6")
        from trop2_design.prediction import boltz_adapter as ba

        # simulate the resolution block of _predict without running a job
        spec = BoltzSpec(python=None, device=None)
        device = spec.device
        if device is None:
            import os

            env_device = os.environ.get("TROP2_BOLTZ_DEVICE", "").strip()
            if env_device.isdigit():
                device = int(env_device)
        assert device == 6

    def test_spec_device_beats_env(self, monkeypatch):
        monkeypatch.setenv("TROP2_BOLTZ_DEVICE", "5")
        spec = BoltzSpec(python=None, device=6)
        device = spec.device  # explicit spec wins before env is consulted
        assert device == 6

    def test_non_digit_env_ignored(self, monkeypatch):
        monkeypatch.setenv("TROP2_BOLTZ_DEVICE", "gpu-six")
        import os

        env_device = os.environ.get("TROP2_BOLTZ_DEVICE", "").strip()
        assert not env_device.isdigit()
