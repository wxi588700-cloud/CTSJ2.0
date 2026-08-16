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
    # running ON the dispatch host itself -> execute locally (no ssh loopback)
    if spec.ssh_host and _is_same_host(spec.ssh_host):
        spec.ssh_host = None
    if not spec.available()[0]:
        return None
    return BoltzPredictor(spec)


def _is_same_host(host: str) -> bool:
    """True when ``host`` refers to the machine we are already on."""
    import socket

    try:
        target = socket.gethostbyname(host)
        local = socket.gethostbyname(socket.gethostname())
        if target == local:
            return True
        # also accept 127.0.0.1 / localhost aliases
        return target.startswith("127.")
    except Exception:
        return False
