"""Predictor adapter registry: builds Boltz/ColabFold adapters from the
tools.yaml ``predictors`` configuration."""
from .boltz_adapter import BoltzPredictor, BoltzResult, BoltzSpec, pick_free_gpu

__all__ = ["BoltzPredictor", "BoltzResult", "BoltzSpec", "pick_free_gpu"]


def build_boltz(predictor_spec, seed: int) -> BoltzPredictor | None:
    """Build a BoltzPredictor from a PredictorSpec (tools.yaml).

    GPU pinning resolution (highest first):
      1. ``predictors.boltz.device`` field (e.g. ``device: 6``)
      2. ``notes`` entry ``device=6`` (same line style as ``ssh_host=gn1``)
      3. environment variable ``TROP2_BOLTZ_DEVICE`` (read at prediction time)
      4. auto-pick the GPU with most free VRAM
    """
    if predictor_spec is None:
        return None
    spec = BoltzSpec(
        python=predictor_spec.python,
        seed=seed,
        device=getattr(predictor_spec, "device", None),
    )
    if predictor_spec.notes:
        for part in str(predictor_spec.notes).replace(";", ",").split(","):
            part = part.strip()
            if part.startswith("ssh_host="):
                spec.ssh_host = part.split("=", 1)[1].strip()
            elif part.startswith("device="):
                try:
                    if spec.device is None:  # explicit field wins over notes
                        spec.device = int(part.split("=", 1)[1])
                except ValueError:
                    pass
    if not spec.available()[0]:
        return None
    return BoltzPredictor(spec)
