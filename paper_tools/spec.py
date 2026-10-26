"""Paper reproduction specs.

A spec (YAML) describes one paper: where its code lives, how to build an image
for it, how to run it, and which metrics to read from the run and compare
against the values the paper (or another trusted record) reports.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Union

import yaml


@dataclass
class MetricSpec:
    name: str
    pattern: str                         # regex with one capture group holding the number
    recorded: Optional[float] = None     # value reported by the paper / trusted record
    tolerance: float = 0.0
    tolerance_type: str = "abs"          # "abs" or "rel"
    command: Optional[str] = None        # overrides the paper-level run command
    after: Optional[str] = None          # regex anchor: only search text after its last match
    occurrence: Union[str, int] = "last" # "first", "last", or an index into the matches
    scale: float = 1.0                   # multiply the parsed value (e.g. 100 for fraction -> %)
    file: Optional[str] = None           # read from this file under the outputs dir instead of stdout
    source: str = ""                     # where the recorded value comes from, e.g. "Table 2"


@dataclass
class PaperSpec:
    name: str
    path: str                            # the spec file itself
    repo: str = ""
    dockerfile: str = "Dockerfile"
    context: str = "."
    image: str = ""
    run_command: str = ""
    workdir: str = ""
    gpu: bool = True
    shm_size: str = "8g"
    timeout_minutes: int = 24 * 60
    env: dict = field(default_factory=dict)
    volumes: dict = field(default_factory=dict)   # host path -> container path
    notes: str = ""
    metrics: dict = field(default_factory=dict)   # name -> MetricSpec

    @property
    def base_dir(self) -> str:
        return os.path.dirname(os.path.abspath(self.path))

    def resolve(self, p: str) -> str:
        p = os.path.expanduser(p)
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(self.base_dir, p))

    def image_tag(self) -> str:
        return self.image or f"repro-{self.name.lower()}:latest"


def load_spec(path: str) -> PaperSpec:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    metrics = {}
    for name, m in (raw.pop("metrics", None) or {}).items():
        if "pattern" not in m and "file" not in m:
            raise ValueError(f"metric '{name}' needs a 'pattern'")
        metrics[name] = MetricSpec(name=name, **m)
    run = raw.pop("run", {}) or {}
    spec = PaperSpec(
        name=raw.pop("name"),
        path=os.path.abspath(path),
        run_command=run.get("command", ""),
        workdir=run.get("workdir", ""),
        gpu=run.get("gpu", True),
        shm_size=run.get("shm_size", "8g"),
        timeout_minutes=run.get("timeout_minutes", 24 * 60),
        env=run.get("env", {}) or {},
        volumes=run.get("volumes", {}) or {},
        metrics=metrics,
        **{k: v for k, v in raw.items() if k in {"repo", "dockerfile", "context", "image", "notes"}},
    )
    unknown = set(raw) - {"repo", "dockerfile", "context", "image", "notes"}
    if unknown:
        raise ValueError(f"unknown spec keys: {sorted(unknown)}")
    return spec
