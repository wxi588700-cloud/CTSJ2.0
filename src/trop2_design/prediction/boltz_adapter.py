"""Boltz-2 structure prediction adapter (GPU, optional SSH remote).

Turns the platform's proxy confidence metrics (ipTM/pLDDT/PAE) into
MEASURED values by running the locally installed Boltz-2 model through
its CLI:

    boltz predict input.yaml --cache ~/.boltz --accelerator gpu --seed N

Verified output layout (boltz 2.0.3):
    <out>/boltz_results_<name>/predictions/<name>/
        <name>_model_0.cif                  structure (B-factor = plddt)
        confidence_<name>_model_0.json      flat keys: iptm/ptm/complex_plddt/...
        plddt_<name>_model_0.npz            per-residue plddt, 0-1
        pae_<name>_model_0.npz              full PAE matrix (N,N)

Design decisions
----------------
*   Input: minimal YAML, one protein entry per chain with a "self-only"
    single-line a3m as MSA (Boltz requires one) - fully offline and
    deterministic, no MSA server needed.
*   Execution: local subprocess on GPU hosts; on CPU-only hosts the run
    can be dispatched to a GPU node via SSH (``ssh_host``) - the cluster's
    NFS-shared home makes inputs/outputs visible on both sides.
*   Scale conventions: platform metrics use plddt 0-100 and iptm 0-1;
    Boltz reports both as 0-1, so plddt is multiplied by 100.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------- data --


@dataclass
class BoltzResult:
    ok: bool
    structure: Path | None = None
    plddt: float | None = None          # chain-mean pLDDT on 0-100 scale
    iptm: float | None = None
    ptm: float | None = None
    interface_pae: float | None = None  # mean cross-chain PAE (A)
    log: str = ""
    reason: str = ""


@dataclass
class BoltzSpec:
    python: Path | None = None          # interpreter of the boltz env
    boltz_bin: Path | None = None       # explicit boltz binary (optional)
    cache: Path = Path("~/.boltz").expanduser()
    accelerator: str = "gpu"
    device: int | None = None           # CUDA_VISIBLE_DEVICES index
    sampling_steps: int = 200
    recycling_steps: int = 3
    seed: int = 20260816
    ssh_host: str | None = None         # e.g. "gn1"; None = local run
    timeout_s: int = 3600

    def available(self) -> tuple[bool, str]:
        if self._boltz_cmd() is None:
            return False, ("boltz CLI not found - set predictors.boltz.python "
                           "in configs/tools.yaml")
        return True, "ok"

    def _boltz_cmd(self) -> list[str] | None:
        if self.boltz_bin is not None and Path(self.boltz_bin).exists():
            return [str(self.boltz_bin)]
        if self.python is not None:
            candidate = Path(self.python).expanduser().parent / "boltz"
            if candidate.exists():
                return [str(candidate)]
            return [str(Path(self.python).expanduser()), "-m", "boltz.main"]
        return None


# ------------------------------------------------------------------ helpers --

def write_self_msa(directory: Path, chain_id: str, sequence: str) -> Path:
    """Boltz requires an MSA per sequence; a single-line self-alignment
    keeps the run offline and deterministic."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"self_{chain_id}.a3m"
    path.write_text(f">{chain_id}\n{sequence}\n")
    return path


def write_boltz_yaml(sequences: dict[str, str], out_path: Path,
                     workdir: Path | None = None) -> Path:
    """Write a Boltz-2 input YAML with self-only MSAs (offline mode).

    The MSA paths are stored RELATIVE to the YAML's directory so the whole
    work directory stays portable across (NFS-shared) hosts.
    """
    import yaml

    workdir = workdir or out_path.parent
    entries = []
    for chain_id, seq in sequences.items():
        write_self_msa(workdir / "msa", chain_id, seq)
        entries.append({
            "protein": {
                "id": chain_id,
                "sequence": seq,
                "msa": f"msa/self_{chain_id}.a3m",
            }
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        yaml.safe_dump({"sequences": entries}, fh, sort_keys=False)
    return out_path


def pick_free_gpu(ssh_host: str | None = None, min_free_mb: int = 12000) -> int | None:
    """Index of the GPU with the most free memory (via nvidia-smi)."""
    cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
    try:
        if ssh_host:
            out = subprocess.run(["ssh", ssh_host, cmd], capture_output=True,
                                 text=True, timeout=30).stdout
        else:
            out = subprocess.run(cmd.split(), capture_output=True,
                                 text=True, timeout=30).stdout
        best, best_free = None, min_free_mb
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            free = int(parts[1])
            if free > best_free:
                best, best_free = int(parts[0]), free
        return best
    except Exception:
        return None


# ------------------------------------------------------------------ parsing --

def _find_file(directory: Path, pattern: str, retries: int = 5,
               delay_s: float = 3.0) -> Path | None:
    """Find a file by fnmatch pattern with retries.

    Predictions are written by the GPU node over NFS; on the submitting
    host stat-based checks (Path.is_file / glob) can fail for up to the
    NFS actimeo (~60 s) right after a run, while directory READS
    (readdir) are immediately consistent.  We therefore match purely on
    the readdir listing and never stat the entries.
    """
    import fnmatch
    import os
    import time

    for attempt in range(retries):
        try:
            for name in os.listdir(directory):
                if fnmatch.fnmatch(name, pattern):
                    return directory / name
        except (FileNotFoundError, NotADirectoryError):
            pass
        if attempt < retries - 1:
            time.sleep(delay_s)
    return None


def _prediction_dir(out_dir: Path, name: str) -> Path | None:
    """Locate predictions/<name>/ (boltz nests one directory per input)."""
    import os
    import time

    for _ in range(5):
        for candidate in (out_dir / f"boltz_results_{name}" / "predictions" / name,
                          out_dir / f"boltz_results_{name}" / "predictions"):
            try:
                inner = [n for n in os.listdir(candidate)
                         if n.endswith(".cif") or n.endswith(".json")]
                if inner:
                    return candidate
            except (FileNotFoundError, NotADirectoryError):
                continue
        time.sleep(3.0)
    return None


def parse_boltz_result(pred_dir: Path, chain_lengths: dict[str, int],
                       target_chain: str | None = None) -> BoltzResult:
    """Parse a boltz prediction directory into platform metrics."""
    cif = _find_file(pred_dir, "*_model_*.cif") or _find_file(pred_dir, "*.cif")
    if cif is None:
        return BoltzResult(ok=False, reason="no output cif")

    iptm = ptm = None
    conf_file = _find_file(pred_dir, "confidence_*.json")
    if conf_file is not None:
        try:
            conf = json.loads(conf_file.read_text())
            iptm = conf.get("iptm")
            ptm = conf.get("ptm")
        except Exception:
            pass

    # per-residue plddt (0-1) -> chain mean on 0-100 scale
    plddt = None
    plddt_npz = _find_file(pred_dir, "plddt_*.npz")
    if plddt_npz is not None:
        import numpy as np

        arr = np.load(str(plddt_npz))["plddt"].reshape(-1)
        if target_chain is not None and chain_lengths:
            start = 0
            for cid in sorted(chain_lengths):  # boltz orders chains A, B, C
                length = chain_lengths[cid]
                if cid == target_chain:
                    plddt = round(float(arr[start:start + length].mean()) * 100, 1)
                    break
                start += length
        if plddt is None:
            plddt = round(float(arr.mean()) * 100, 1)

    # mean cross-chain PAE between the first two chains
    interface_pae = None
    pae_npz = _find_file(pred_dir, "pae_*.npz")
    if pae_npz is not None and len(chain_lengths) >= 2:
        import numpy as np

        pae = np.load(str(pae_npz))["pae"]
        ids = sorted(chain_lengths)[:2]
        len_a, len_b = chain_lengths[ids[0]], chain_lengths[ids[1]]
        if pae.shape[0] >= len_a + len_b:
            cross = pae[:len_a, len_a:len_a + len_b]
            interface_pae = round(float(np.nanmean(cross)), 2)

    return BoltzResult(ok=True, structure=cif, plddt=plddt, iptm=iptm,
                       ptm=ptm, interface_pae=interface_pae)


# ----------------------------------------------------------------- executor --

class BoltzPredictor:
    def __init__(self, spec: BoltzSpec):
        self.spec = spec

    def predict_complex(self, sequences: dict[str, str], name: str,
                        workdir: Path) -> BoltzResult:
        return self._predict(sequences, name, workdir,
                             target_chain=max(sequences,
                                              key=lambda k: -len(sequences[k]))
                             if len(sequences) > 1 else list(sequences)[0])

    def predict_monomer(self, sequence: str, name: str,
                        workdir: Path) -> BoltzResult:
        return self._predict({"A": sequence}, name, workdir, target_chain="A")

    def _predict(self, sequences: dict[str, str], name: str,
                 workdir: Path, target_chain: str | None) -> BoltzResult:
        ok, why = self.spec.available()
        if not ok:
            return BoltzResult(ok=False, reason=why)

        # SSH dispatch requires a workdir on the SHARED home filesystem
        # (/tmp is node-local); mirror non-shared paths into ~/.trop2_boltz
        workdir = Path(workdir).expanduser().resolve()
        if self.spec.ssh_host:
            home = Path.home()
            if not str(workdir).startswith(str(home)):
                import hashlib

                digest = hashlib.sha256(str(workdir).encode()).hexdigest()[:10]
                workdir = home / ".trop2_boltz" / digest / Path(workdir).name
        workdir.mkdir(parents=True, exist_ok=True)

        yaml_path = write_boltz_yaml(sequences, workdir / f"{name}.yaml",
                                     workdir=workdir)
        out_dir = workdir / "boltz_out"
        # GPU pinning: explicit spec.device (tools.yaml device field or
        # notes device=N) > TROP2_BOLTZ_DEVICE env var > auto-pick the GPU
        # with most free VRAM.  The env var also applies to SSH dispatch
        # because we inject the prefix into the remote command ourselves.
        device = self.spec.device
        if device is None:
            env_device = os.environ.get("TROP2_BOLTZ_DEVICE", "").strip()
            if env_device.isdigit():
                device = int(env_device)
        if device is None:
            device = pick_free_gpu(self.spec.ssh_host)
        cmd = self.spec._boltz_cmd() + [
            "predict", f"{name}.yaml",          # relative: we always cd first
            "--out_dir", "boltz_out",
            "--cache", str(self.spec.cache),
            "--accelerator", self.spec.accelerator,
            "--sampling_steps", str(self.spec.sampling_steps),
            "--recycling_steps", str(self.spec.recycling_steps),
            "--seed", str(self.spec.seed),
        ]
        if self.spec.ssh_host:
            prefix = f"CUDA_VISIBLE_DEVICES={device} " if device is not None else ""
            argv = ["ssh", self.spec.ssh_host,
                    f"cd {workdir} && {prefix}" + " ".join(cmd)]
        else:
            env_prefix = [f"CUDA_VISIBLE_DEVICES={device}"] if device is not None else []
            argv = ["env", *env_prefix, "bash", "-c",
                    f"cd {workdir} && " + " ".join(cmd)]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=self.spec.timeout_s)
        except FileNotFoundError as exc:
            return BoltzResult(ok=False, reason=f"launch failed: {exc}")
        except subprocess.TimeoutExpired:
            return BoltzResult(ok=False, reason="timeout")

        pred_dir = _prediction_dir(out_dir, name)
        if proc.returncode != 0 or pred_dir is None:
            return BoltzResult(ok=False, reason=f"exit {proc.returncode}",
                               log=proc.stdout[-1500:] + proc.stderr[-1500:])
        chain_lengths = {cid: len(seq) for cid, seq in sequences.items()}
        result = parse_boltz_result(pred_dir, chain_lengths, target_chain)
        result.log = proc.stdout[-1500:]
        return result
