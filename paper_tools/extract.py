"""Pull a metric value out of run output and compare it with a recorded value."""
from __future__ import annotations

import re
from typing import Optional

from .spec import MetricSpec

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def extract_value(text: str, m: MetricSpec) -> Optional[float]:
    text = ANSI.sub("", text)
    if m.after:
        anchors = list(re.finditer(m.after, text, re.MULTILINE))
        if not anchors:
            return None
        text = text[anchors[-1].end():]
    matches = re.findall(m.pattern, text, re.MULTILINE)
    if not matches:
        return None
    if m.occurrence == "first":
        hit = matches[0]
    elif m.occurrence == "last":
        hit = matches[-1]
    else:
        idx = int(m.occurrence)
        if not -len(matches) <= idx < len(matches):
            return None
        hit = matches[idx]
    if isinstance(hit, tuple):
        hit = hit[0]
    return float(hit) * m.scale


def compare(reproduced: float, m: MetricSpec) -> dict:
    if m.recorded is None:
        return {"status": "NO_RECORD", "reproduced": reproduced,
                "message": f"metric '{m.name}' has no recorded value in the spec"}
    diff = reproduced - m.recorded
    if m.tolerance_type == "rel":
        allowed = abs(m.recorded) * m.tolerance
    else:
        allowed = m.tolerance
    ok = abs(diff) <= allowed + 1e-12
    return {
        "status": "PASS" if ok else "FAIL",
        "reproduced": reproduced,
        "recorded": m.recorded,
        "difference": round(diff, 6),
        "allowed_difference": round(allowed, 6),
        "tolerance": f"{m.tolerance} ({m.tolerance_type})",
        "source": m.source,
    }
