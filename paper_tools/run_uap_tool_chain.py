"""Run the UAP paper through the reproduction-tool chain.

This produces a durable evidence directory containing state.json and the
Docker build/run logs. The first metric reproduction runs the full UAP command;
the remaining metrics reuse that cached run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Allow `python paper_tools/run_uap_tool_chain.py` from the repository root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paper_tools.docker_build import do_docker_build
from paper_tools.reproduce_tools import do_reproduce_metric
from paper_tools.spec_validation_tools import do_load_spec, do_validate_metric
from paper_tools.state import ReproState


def emit(label: str, value: dict) -> None:
    print(f"{label}: {json.dumps(value, indent=2, default=str)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="papers/uap/spec.yaml")
    parser.add_argument("--workspace", default="inspection-results/uap-tool-run")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    state = ReproState(args.workspace)
    emit("LOAD", do_load_spec(state, args.spec))

    if not args.skip_build and state.spec.backend in {"docker", "auto"}:
        build = do_docker_build(state, state.spec.dockerfile)
        emit("BUILD", build)
        if build.get("status") != "OK" and state.spec.backend != "auto":
            return 2
    elif state.spec.backend == "slurm":
        emit("BUILD", {"status": "SKIPPED", "reason": "active spec uses Slurm, not Docker"})

    failed = False
    for metric in state.spec.metrics:
        reproduced = do_reproduce_metric(state, metric)
        emit(f"REPRODUCE {metric}", reproduced)
        if reproduced.get("status") != "OK":
            failed = True
            continue
        validation = do_validate_metric(state, metric)
        emit(f"VALIDATE {metric}", validation)
        failed |= validation.get("status") not in {"PASS", "NO_RECORD"}

    print(f"Evidence directory: {os.path.abspath(args.workspace)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
