"""Update an existing metric in a JSON specification (Python 3.10+)."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import re
import tempfile


_UNSET = object()
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("must be a finite number")
    if not math.isfinite(value):
        raise ValueError("must be a finite number")


def _validate_reader(pattern, after, log):
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("pattern must be a nonempty regex")
    reader = re.compile(pattern)
    if after is not None and not isinstance(after, str):
        raise ValueError("after must be a regex string or null")
    anchor = re.compile(after) if after is not None else None
    if log is None:
        raise ValueError("a real run_log_path is required to change pattern/after")
    region = log
    if anchor is not None:
        match = anchor.search(region)
        if match is None:
            raise ValueError("after did not match the run log")
        region = region[match.end():]
    for match in reader.finditer(region):
        # Prefer a named value group; otherwise the first capture or whole match.
        value = (match.group("value") if "value" in reader.groupindex
                 else match.group(1) if reader.groups else match.group(0))
        if value is not None and _NUMBER.fullmatch(value.strip()):
            if math.isfinite(float(value)):
                return
    raise ValueError("pattern did not extract a finite number from the run log")


def update_metric(spec_path, metric, *, recorded=_UNSET, source=_UNSET,
                  tolerance=_UNSET, pattern=_UNSET, after=_UNSET,
                  run_log_path=None):
    """Change only supplied fields; validate pattern/after as one atomic pair.

    Schema: {"metrics": {"accuracy": {"recorded": 0.9, ...}}}.
    `after` restricts matching to text after its first regex match; null clears it.
    Returns status, fields_saved, fields_not_changed, and optional error.
    A PARTIAL response persists valid paper fields while refusing reader changes.
    Single-writer API: callers must serialize concurrent updates to the same file.
    """
    result = {"status": "ERROR", "fields_saved": {}, "fields_not_changed": {}}
    supplied = {k: v for k, v in dict(recorded=recorded, source=source,
                tolerance=tolerance, pattern=pattern, after=after).items()
                if v is not _UNSET}
    try:
        path = Path(spec_path)
        spec = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(spec, dict) or not isinstance(spec.get("metrics"), dict):
            raise ValueError("spec must contain a metrics object")
        if metric not in spec["metrics"]:
            raise ValueError(f"unknown metric: {metric}")
        current = spec["metrics"][metric]
        if not isinstance(current, dict):
            raise ValueError("metric entry must be an object")
        candidate = copy.deepcopy(current)
        accepted = {}
        for key in ("recorded", "source", "tolerance"):
            if key not in supplied:
                continue
            value = supplied[key]
            try:
                if key in ("recorded", "tolerance"):
                    _number(value)
                    if key == "tolerance" and value < 0:
                        raise ValueError("must be nonnegative")
                elif not isinstance(value, str) or not value.strip():
                    raise ValueError("must be a nonempty table/row/column reference")
                candidate[key] = value
                accepted[key] = value
            except ValueError as exc:
                result["fields_not_changed"][key] = str(exc)

        reader_fields = {k: supplied[k] for k in ("pattern", "after") if k in supplied}
        if reader_fields:
            try:
                log = (Path(run_log_path).read_text(encoding="utf-8")
                       if run_log_path is not None else None)
                _validate_reader(reader_fields.get("pattern", current.get("pattern")),
                                 reader_fields.get("after", current.get("after")), log)
                candidate.update(reader_fields)
                accepted.update(reader_fields)
            except (OSError, UnicodeError, ValueError, re.error) as exc:
                for key in reader_fields:
                    result["fields_not_changed"][key] = str(exc)

        if accepted:
            spec["metrics"][metric] = candidate
            # Replace only after a complete write, so failures don't truncate the spec.
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                        dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                        delete=False) as handle:
                    temp_path = Path(handle.name)
                    json.dump(spec, handle, indent=2, ensure_ascii=False, allow_nan=False)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, path)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
            result["fields_saved"] = accepted
        result["status"] = ("PARTIAL" if accepted else "ERROR") if result["fields_not_changed"] else "OK"
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        result["error"] = str(exc)
        for key in supplied:
            result["fields_not_changed"].setdefault(key, "not saved: " + str(exc))
    return result


# Preserve the spelling shown in the requirement image.
Update_metric = update_metric
