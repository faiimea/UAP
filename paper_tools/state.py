"""Session state shared by the tools, persisted to <workspace>/state.json."""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Optional

from .spec import PaperSpec, load_spec


class ReproState:
    def __init__(self, workspace: str):
        self.workspace = os.path.abspath(workspace)
        os.makedirs(self.workspace, exist_ok=True)
        self.path = os.path.join(self.workspace, "state.json")
        self.data = {"spec_path": None, "images": {}, "runs": {}, "reproduced": {}, "validations": {}}
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.data.update(json.load(f))
        self._spec: Optional[PaperSpec] = None

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)

    # ---- active spec ----
    def set_spec(self, spec_path: str) -> PaperSpec:
        self._spec = load_spec(spec_path)
        self.data["spec_path"] = self._spec.path
        self.save()
        return self._spec

    @property
    def spec(self) -> PaperSpec:
        if self._spec is None:
            if not self.data.get("spec_path"):
                raise RuntimeError("no paper spec loaded - call load_paper_spec first")
            self._spec = load_spec(self.data["spec_path"])
        return self._spec

    def paper_dir(self, *parts) -> str:
        d = os.path.join(self.workspace, self.spec.name, *parts)
        os.makedirs(d, exist_ok=True)
        return d

    def key(self, metric: str) -> str:
        return f"{self.spec.name}/{metric}"

    # ---- run cache: one container run can yield several metrics ----
    @staticmethod
    def run_key(container: str, command: str) -> str:
        return hashlib.sha1(f"{container}\0{command}".encode()).hexdigest()[:12]

    def cached_run(self, container: str, command: str) -> Optional[dict]:
        r = self.data["runs"].get(self.run_key(container, command))
        return r if r and r.get("returncode") == 0 and os.path.exists(r["log"]) else None

    def record_run(self, container: str, command: str, log: str, rc: int, seconds: float):
        self.data["runs"][self.run_key(container, command)] = {
            "container": container, "command": command, "log": log,
            "returncode": rc, "seconds": round(seconds, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.save()
