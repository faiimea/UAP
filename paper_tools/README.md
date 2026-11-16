# Paper tools

`spec_validation_tools.py` ports `load_paper_spec` and `validate_metric` from
`paper-repro/repro/tools.py` on the `paper-repro` branch. The shared `spec.py`,
`state.py`, and `extract.py` modules are copied from that branch without changes.
This contribution integrates the existing implementation into `main`.

Install the dependencies from the repository root:

```sh
python -m pip install langchain-core PyYAML pytest
python -m pytest -q paper_tools/tests/test_spec_validation_tools.py
```

Create the tools with one shared session state:

```python
from paper_tools.state import ReproState
from paper_tools.spec_validation_tools import make_tools

state = ReproState("workspace")
tools = make_tools(state)
# Append these two tools to the agent's other tools.
tools[0].invoke({"spec_path": "/path/to/paper/spec.yaml"})
```

`load_paper_spec` loads a YAML experiment configuration, persists the active paper,
and returns its settings and expected metrics. `validate_metric` compares a
previously reproduced measurement with the reference and tolerance, then persists
PASS, FAIL, or NO_RECORD. A missing measurement or unknown metric returns ERROR.
For example, 89.40 versus 88.88 with an absolute tolerance of 1.0 passes because
the difference is 0.52 percentage points.

`reproduce_tools.py` adds `reproduce_metric`. It runs the active paper's command
inside a container (or a fresh container from an image), parses the metric from the
run output with `extract.py`, and stores the measured value where `validate_metric`
reads it. Runs are cached by `(container, command)`, so a metric produced by an
already-completed run reuses that run's log instead of executing again. Running a
command needs the Docker CLI, which `docker_ops.py` wraps.

```python
from paper_tools.state import ReproState
from paper_tools.reproduce_tools import make_tools

state = ReproState("workspace")
reproduce_metric = make_tools(state)[0]
# load_paper_spec must run on the same state first.
reproduce_metric.invoke({"metric": "accuracy"})
```

Other reproduction tools must use the same `ReproState` instance and store measured
values in `state.data["reproduced"][state.key(metric)]` before validation. The
tests seed these values to exercise validation without Docker or model training;
`test_reproduce_tools.py` seeds a cached run so `reproduce_metric` is exercised the
same way (it reuses a recorded log instead of calling Docker).

This transfer does not include the agent or paper configurations. The remaining
branch contents need separate integration before the team removes `paper-repro`.
