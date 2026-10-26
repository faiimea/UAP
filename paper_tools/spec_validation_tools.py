"""Load and validate paper specifications, ported from paper-repro/repro/tools.py.

Pass the same ReproState instance to all reproduction tools in an agent.
"""
import json
from langchain_core.tools import tool
from .extract import compare
from .state import ReproState

def _dump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)



def do_load_spec(state: ReproState, spec_path: str) -> dict:
    spec = state.set_spec(spec_path)
    return {
        "paper": spec.name, "repo": spec.repo, "image": spec.image_tag(),
        "dockerfile": spec.resolve(spec.dockerfile), "run_command": spec.run_command,
        "gpu": spec.gpu, "volumes": spec.volumes, "notes": spec.notes,
        "metrics": {n: {"recorded": m.recorded, "tolerance": f"{m.tolerance} ({m.tolerance_type})",
                        "source": m.source, "command": m.command or "(paper run command)"}
                    for n, m in spec.metrics.items()},
    }



def do_validate_metric(state: ReproState, metric: str) -> dict:
    spec = state.spec
    if metric not in spec.metrics:
        return {"status": "ERROR", "message": f"unknown metric '{metric}'", "available": list(spec.metrics)}
    rep = state.data["reproduced"].get(state.key(metric))
    if rep is None:
        return {"status": "ERROR", "message": f"'{metric}' has not been reproduced yet - call reproduce_metric first"}
    result = {"metric": metric, **compare(rep["value"], spec.metrics[metric])}
    state.data["validations"][state.key(metric)] = result
    state.save()
    return result



def make_tools(state: ReproState) -> list:
    """Create load_paper_spec and validate_metric bound to shared session state."""
    @tool
    def load_paper_spec(spec_path: str) -> str:
        """Load a paper's reproduction spec (YAML) and make it the active paper.
        Returns the paper's image tag, Dockerfile, run command and metrics with their recorded values.
        Call this before docker_build / reproduce_metric / validate_metric."""
        try:
            return _dump(do_load_spec(state, spec_path))
        except Exception as e:  # spec errors are for the agent to fix
            return _dump({"status": "ERROR", "message": f"{type(e).__name__}: {e}"})


    @tool
    def validate_metric(metric: str) -> str:
        """Compare the reproduced value of `metric` with its recorded value from the spec, within
        the spec's tolerance. Returns PASS / FAIL / NO_RECORD with the difference."""
        return _dump(do_validate_metric(state, metric))


    return [load_paper_spec, validate_metric]
