"""GPU free-memory wait guard for ssh-dispatched adapters.

Mirrors the trop2-binder production guard: on a shared card (allocation
violations observed in practice) a GPU stage waits for enough free memory
instead of OOM-crashing or silently degrading.
"""
from __future__ import annotations

import subprocess
import time


def wait_gpu_free(ssh_host: str, device: str, min_free_mb: int,
                  max_wait_min: int, label: str = "gpu-stage",
                  poll_s: int = 60, log=print) -> bool:
    """Wait until the card has >= min_free_mb free; True=proceed, False=give up."""
    deadline = time.time() + max_wait_min * 60
    waited = 0
    probe_failures = 0
    while time.time() < deadline:
        try:
            out = subprocess.run(
                ["ssh", ssh_host,
                 f"nvidia-smi -i {device} --query-gpu=memory.free "
                 f"--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30)
            if out.returncode != 0:
                raise RuntimeError(f"probe rc={out.returncode}")
            lines = out.stdout.strip().splitlines()
            if not lines:
                raise RuntimeError("probe returned no data")
            free = int(lines[0])
            probe_failures = 0
        except Exception:
            # borrowed fail-fast idea (optimized build), hardened with a
            # 3-strike tolerance: transient ssh blips retry, persistent
            # misconfiguration aborts instead of hanging for hours
            probe_failures += 1
            if probe_failures >= 3:
                log(f"[{label}] GPU probe failed {probe_failures}x on "
                    f"{ssh_host} - refusing to wait on an unavailable host")
                return False
            free = 0
        if free >= min_free_mb:
            if waited:
                log(f"[{label}] GPU{device} free {free}MiB >= {min_free_mb}MiB "
                    f"after waiting {waited}min - proceeding")
            return True
        if waited % 5 == 0:
            log(f"[{label}] GPU{device} only {free}MiB free "
                f"(need {min_free_mb}MiB) - waiting... ({waited}/{max_wait_min}min)")
        time.sleep(poll_s)
        waited += poll_s // 60
    return False
