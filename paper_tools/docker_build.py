import json
import os
import re

from langchain_core.tools import tool

from . import docker_ops
from .state import ReproState

def do_docker_build(state: ReproState, dockerfile: str, tag: str = "", context: str = "") -> dict:
    spec = state.spec if state.data.get("spec_path") else None
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

@tool
def docker_build(dockerfile: str, tag: str = "", context: str = "") -> str:
    """Build a Docker image from the Dockerfile path provided. Paths are relative to the current spec's folder.
    `tag` defaults to the spec's image tag; `context` defaults to the spec's build context.
    Returns status OK with the image tag, or FAILED with the tail of the build log."""
    return _dump(do_docker_build(state, dockerfile, tag, context))
