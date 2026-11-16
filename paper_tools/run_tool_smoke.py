"""Fast Docker plumbing test for the UAP-shaped metric tool chain."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paper_tools.reproduce_tools import do_reproduce_metric
from paper_tools.spec_validation_tools import do_load_spec, do_validate_metric
from paper_tools.state import ReproState


def main() -> int:
    state = ReproState("inspection-results/uap-tool-smoke")
    print(json.dumps(do_load_spec(state, "papers/uap/tool_smoke_spec.yaml"), indent=2))
    failed = False
    for metric in ("view_auc", "model_auc", "piracy_detection_confidence"):
        reproduced = do_reproduce_metric(state, metric)
        print(f"REPRODUCE {metric}: {json.dumps(reproduced, indent=2)}")
        if reproduced.get("status") != "OK":
            failed = True
            continue
        validated = do_validate_metric(state, metric)
        print(f"VALIDATE {metric}: {json.dumps(validated, indent=2)}")
        failed |= validated.get("status") != "PASS"
    print("Evidence directory:", os.path.abspath(state.workspace))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
