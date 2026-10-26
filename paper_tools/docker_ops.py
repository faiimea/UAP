"""Thin wrappers over the docker CLI. to facilitate better LLM usage"""
from __future__ import annotations

import shutil
import subprocess
from typing import Optional

OUTPUTS_MOUNT = "/repro_out"


class DockerError(RuntimeError):
    pass


def docker_bin() -> str:
    path = shutil.which("docker")
    if not path:
        raise DockerError(
            "docker CLI not found. Prompt the user to install the docker CLI and ensure `docker info` works for this user.")
    return path


def _run(args: list, timeout: Optional[float], log_path: Optional[str] = None) -> tuple:
    """Run a command, streaming combined stdout/stderr to log_path. Returns (rc, output)."""
    if log_path:
        with open(log_path, "w") as log:
            try:
                proc = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                log.write(f"\n[repro] TIMEOUT after {timeout}s\n")
                rc = -1
        with open(log_path, errors="replace") as log:
            return rc, log.read()
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return -1, f"[repro] TIMEOUT after {timeout}s"


def build(dockerfile: str, context: str, tag: str, log_path: str, timeout: float = 3 * 3600) -> tuple:
    return _run([docker_bin(), "build", "-f", dockerfile, "-t", tag, context], timeout, log_path)


def is_running_container(name: str) -> bool:
    rc, out = _run([docker_bin(), "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"], 30)
    return rc == 0 and name in out.split()


def image_exists(tag: str) -> bool:
    rc, _ = _run([docker_bin(), "image", "inspect", tag], 30)
    return rc == 0


def run(container: str, command: str, *, log_path: str, outputs_dir: str, gpu: bool = True,
        env: Optional[dict] = None, volumes: Optional[dict] = None, workdir: str = "",
        shm_size: str = "8g", timeout: Optional[float] = None) -> tuple:
    """Run `command` in a running container (docker exec) or a fresh one from an image (docker run --rm)."""
    env = env or {}
    if is_running_container(container):
        args = [docker_bin(), "exec"]
        for k, v in env.items():
            args += ["-e", f"{k}={v}"]
        if workdir:
            args += ["-w", workdir]
        args += [container, "bash", "-lc", command]
    else:
        if not image_exists(container):
            raise DockerError(f"'{container}' is neither a running container nor a local image")
        args = [docker_bin(), "run", "--rm", "--shm-size", shm_size,
                "-v", f"{outputs_dir}:{OUTPUTS_MOUNT}"]
        if gpu:
            args += ["--gpus", "all"]
        for host, inside in (volumes or {}).items():
            args += ["-v", f"{host}:{inside}"]
        for k, v in env.items():
            args += ["-e", f"{k}={v}"]
        if workdir:
            args += ["-w", workdir]
        args += [container, "bash", "-lc", command]
    return _run(args, timeout, log_path)
