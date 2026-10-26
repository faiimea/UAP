"""Reproduce a paper metric by running its command in a container.

Pass the same ReproState instance to all reproduction tools in an agent. Loading a
spec (load_paper_spec) must happen first; validation (validate_metric) reads the value
this tool stores in state.data["reproduced"].
"""
import json
import os
import time

from langchain_core.tools import tool

from . import docker_ops
from .extract import extract_value
from .state import ReproState

LOG_TAIL = 4000


def _tail(text: str, n: int = LOG_TAIL) -> str:
    return text if len(text) <= n else "...[truncated]...\n" + text[-n:]


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


def do_reproduce_metric(state: ReproState, metric: str, container: str = "") -> dict:
    spec = state.spec
    if metric not in spec.metrics:
        return {"status": "ERROR", "message": f"unknown metric '{metric}'", "available": list(spec.metrics)}
    m = spec.metrics[metric]
    container = container or spec.image_tag()
    command = m.command or spec.run_command
    if not command:
        return {"status": "ERROR", "message": "no run command in spec (run.command or metric.command)"}

    outputs = state.paper_dir("outputs")
    cached = state.cached_run(container, command)
    if cached:
        log_path, reused = cached["log"], True
        with open(log_path, errors="replace") as f:
            out = f.read()
    else:
        reused = False
        log_path = os.path.join(state.paper_dir("logs"), f"run-{state.run_key(container, command)}.log")
        volumes = {spec.resolve(h): c for h, c in spec.volumes.items()}
        t0 = time.time()
        try:
            rc, out = docker_ops.run(container, command, log_path=log_path, outputs_dir=outputs,
                                     gpu=spec.gpu, env=spec.env, volumes=volumes, workdir=spec.workdir,
                                     shm_size=spec.shm_size, timeout=spec.timeout_minutes * 60)
        except docker_ops.DockerError as e:
            return {"status": "ERROR", "message": str(e)}
        state.record_run(container, command, log_path, rc, time.time() - t0)
        if rc != 0:
            return {"status": "RUN_FAILED", "returncode": rc, "command": command, "log": log_path,
                    "log_tail": _tail(out)}

    if m.file:
        fpath = os.path.join(outputs, m.file)
        if not os.path.exists(fpath):
            return {"status": "ERROR", "message": f"metric file not produced: {fpath}", "log_tail": _tail(out)}
        with open(fpath, errors="replace") as f:
            source_text = f.read()
    else:
        source_text = out

    value = extract_value(source_text, m)
    if value is None:
        return {"status": "NOT_FOUND", "message": f"pattern {m.pattern!r} (after={m.after!r}) matched nothing",
                "command": command, "log": log_path, "log_tail": _tail(source_text)}
    state.data["reproduced"][state.key(metric)] = {"value": value, "container": container, "command": command,
                                                  "log": log_path, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    state.save()
    return {"status": "OK", "metric": metric, "value": value, "reused_previous_run": reused, "log": log_path}


def make_tools(state: ReproState) -> list:
    """Create reproduce_metric bound to shared session state."""
    @tool
    def reproduce_metric(metric: str, container: str = "") -> str:
        """Run the active paper's command inside `container` (a running container name, or an image
        tag - defaults to the spec's image) and parse `metric` from the output.
        Runs are cached: metrics produced by the same command reuse the previous run.
        May take hours for training runs. Returns the parsed value or the failing log tail."""
        return _dump(do_reproduce_metric(state, metric, container))

    return [reproduce_metric]
