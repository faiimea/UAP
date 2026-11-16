import json
import os
import time

from langchain_core.tools import tool

from . import docker_ops
from .state import ReproState

LOG_TAIL = 4000


def _tail(text: str, n: int = LOG_TAIL) -> str:
    return text if len(text) <= n else "...[truncated]...\n" + text[-n:]


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)

def do_docker_build(state: ReproState, dockerfile: str = "", tag: str = "", context: str = "") -> dict:
    spec = state.spec if state.data.get("spec_path") else None
    if spec and not dockerfile:
        dockerfile = spec.dockerfile
    df = spec.resolve(dockerfile) if spec else os.path.abspath(os.path.expanduser(dockerfile))
    if not os.path.isfile(df):
        return {"status": "ERROR", "message": f"Dockerfile not found: {df}"}
    if context:
        ctx = spec.resolve(context) if spec else os.path.abspath(context)
    elif spec and df == spec.resolve(spec.dockerfile):
        ctx = spec.resolve(spec.context)
    else:
        ctx = os.path.dirname(df)
    tag = tag or (spec.image_tag() if spec else f"repro-{os.path.basename(os.path.dirname(df)).lower()}:latest")
    log = os.path.join(state.paper_dir("logs") if spec else state.workspace, f"build-{tag.replace(':', '_').replace('/', '_')}.log")
    t0 = time.time()
    try:
        rc, out = docker_ops.build(df, ctx, tag, log)
    except docker_ops.DockerError as e:
        return {"status": "ERROR", "message": str(e)}
    state.data["images"][tag] = {"dockerfile": df, "context": ctx, "ok": rc == 0, "log": log,
                                 "seconds": round(time.time() - t0, 1)}
    state.save()
    if rc != 0:
        return {"status": "FAILED", "image": tag, "log": log, "log_tail": _tail(out)}
    return {"status": "OK", "image": tag, "seconds": round(time.time() - t0, 1), "log": log}

def make_tools(state: ReproState) -> list:
    """Create docker_build bound to the shared reproduction state.

    Keeping the state binding here matches reproduce_tools.py and prevents the
    tool from depending on an undefined module-level ``state`` variable.
    """
    @tool
    def docker_build(dockerfile: str = "", tag: str = "", context: str = "") -> str:
        """Build the active paper's Docker image.

        When a paper spec is active, its declared Dockerfile and context are
        authoritative; the optional dockerfile argument is only used when no
        spec has been loaded.
        """
        try:
            if state.data.get("spec_path"):
                dockerfile = state.spec.dockerfile
            return _dump(do_docker_build(state, dockerfile, tag, context))
        except Exception as e:
            return _dump({"status": "ERROR", "message": f"{type(e).__name__}: {e}"})

    return [docker_build]
