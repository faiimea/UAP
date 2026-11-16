"""Slurm execution backend for paper reproduction.

This backend is intended to run on an HPC login node. It submits the
experiment with sbatch and waits for the allocated compute job; it never
executes the training workload on the login node.
"""
from __future__ import annotations

import os
import subprocess
import time

from .spec import PaperSpec
from .state import ReproState


class SlurmError(RuntimeError):
    pass


def _call(args: list[str], *, cwd: str | None = None, timeout: float = 60) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise SlurmError(f"required Slurm command is not available: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SlurmError(f"Slurm command timed out: {' '.join(args)}") from exc
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _job_state(job_id: str) -> str:
    rc, out = _call(["squeue", "-h", "-j", job_id, "-o", "%T"])
    if rc == 0 and out.strip():
        return out.strip().splitlines()[0].strip()

    rc, out = _call(["sacct", "-X", "-n", "-P", "-j", job_id, "--format=State,ExitCode"])
    if rc != 0:
        return "UNKNOWN"
    for line in out.splitlines():
        line = line.strip()
        if line:
            return line.split("|", 1)[0].strip()
    return "UNKNOWN"


def run(spec: PaperSpec, state: ReproState) -> dict:
    """Submit the spec's Slurm script and wait for its final state."""
    if not spec.slurm_script:
        raise SlurmError("Slurm backend requires run.slurm_script in the paper spec")

    script = spec.resolve(spec.slurm_script)
    if not os.path.isfile(script):
        raise SlurmError(f"Slurm script not found: {script}")

    cwd = spec.resolve(spec.context)
    if not os.path.isdir(cwd):
        raise SlurmError(f"Slurm working directory not found: {cwd}")

    log_dir = state.paper_dir("slurm_logs")
    output_template = os.path.join(log_dir, "job-%j.out")
    args = ["sbatch", "--parsable", "--output", output_template]
    if spec.slurm_account:
        args += ["--account", spec.slurm_account]
    if spec.slurm_partition:
        args += ["--partition", spec.slurm_partition]
    if spec.slurm_qos:
        args += ["--qos", spec.slurm_qos]
    if spec.slurm_time:
        args += ["--time", spec.slurm_time]
    args.append(script)

    rc, out = _call(args, cwd=cwd)
    if rc != 0:
        raise SlurmError(f"sbatch failed:\n{out[-4000:]}")

    job_id = out.strip().splitlines()[-1].split(";", 1)[0].strip()
    if not job_id.isdigit():
        raise SlurmError(f"could not parse job ID from sbatch output: {out!r}")

    state.data.setdefault("slurm_jobs", {})[job_id] = {
        "script": script,
        "cwd": cwd,
        "submitted": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output": output_template.replace("%j", job_id),
    }
    state.save()

    deadline = time.monotonic() + spec.timeout_minutes * 60
    poll_seconds = float(os.environ.get("SLURM_POLL_SECONDS", "30"))
    final_state = "UNKNOWN"
    while time.monotonic() < deadline:
        final_state = _job_state(job_id)
        if final_state in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}:
            break
        time.sleep(max(1.0, poll_seconds))
    else:
        final_state = "TIMEOUT"

    output_path = output_template.replace("%j", job_id)
    output = ""
    if os.path.exists(output_path):
        with open(output_path, errors="replace") as f:
            output = f.read()

    metrics_path = ""
    metrics_text = ""
    if spec.slurm_metrics_file:
        metrics_path = spec.resolve(spec.slurm_metrics_file.format(job_id=job_id))
        if os.path.exists(metrics_path):
            with open(metrics_path, errors="replace") as f:
                metrics_text = f.read()

    source = metrics_text or output
    return {
        "job_id": job_id,
        "state": final_state,
        "returncode": 0 if final_state == "COMPLETED" else 1,
        "output": source,
        "job_log": output_path,
        "metrics_file": metrics_path,
    }
