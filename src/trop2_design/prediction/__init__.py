"""Predictor adapter registry: builds Boltz/ColabFold adapters from the
tools.yaml ``predictors`` configuration."""
from .boltz_adapter import BoltzPredictor, BoltzResult, BoltzSpec, pick_free_gpu

__all__ = ["BoltzPredictor", "BoltzResult", "BoltzSpec", "pick_free_gpu"]


def build_boltz(predictor_spec, seed: int) -> BoltzPredictor | None:
    """Build a BoltzPredictor from a PredictorSpec (tools.yaml)."""
    if predictor_spec is None:
        return None
    spec = BoltzSpec(
        python=predictor_spec.python,
        seed=seed,
    )
    # optional ssh dispatch via notes field "ssh_host=gn1"
    if predictor_spec.notes:
        for part in str(predictor_spec.notes).replace(";", ",").split(","):
            if part.strip().startswith("ssh_host="):
                spec.ssh_host = part.split("=", 1)[1].strip()
    if not spec.available()[0]:
        return None
    return BoltzPredictor(spec)
