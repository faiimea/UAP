# Update_metric

Dependency-free Python 3.10+ implementation of the attached requirement.

Create `spec.json`:

```json
{
  "metrics": {
    "accuracy": {
      "recorded": 0.9,
      "source": "Table 2 / baseline row / accuracy column",
      "tolerance": 0.01,
      "pattern": "accuracy=([0-9.]+)",
      "after": null
    }
  }
}
```

Use from Python:

```python
from update_metric import Update_metric

result = Update_metric(
    "spec.json", "accuracy",
    recorded=0.93,
    source="Table 2 / proposed row / accuracy column",
    pattern=r"accuracy=(?P<value>\d+\.\d+)",
    after=r"FINAL RESULTS",
    run_log_path="run.log",
)
print(result)
```

For this example `run.log` must contain real run output with `FINAL RESULTS`
followed by a value such as `accuracy=0.93`.

Only explicitly supplied fields are updated. Existing metrics must already exist.
`recorded` must be finite; `tolerance` must be finite and nonnegative; `source`
must be a nonempty string. Omit a field to preserve it. Pass `after=None` to clear
the anchor.

Regex assumptions: `after` selects text after its first match. `pattern` extracts
the named `value` group, otherwise its first capture, otherwise its entire match.
The extracted text must be a finite number. Regex changes are accepted together
only after checking them against a UTF-8 run log. The caller is responsible for
providing an authentic log; this function cannot establish its provenance.
Logs and regexes should come from trusted callers: Python regex evaluation has
no built-in timeout.

Results contain `status`, `fields_saved`, and `fields_not_changed` with reasons.
`OK` means all requested fields were accepted (including an empty request).
`PARTIAL` means some fields were saved and some refused. `ERROR` means no requested
changes could be saved, or reading/writing failed. Storage errors include `error`.
Unrelated JSON data is preserved; the file is reformatted with two-space indentation.
Writes use an atomic replacement. Serialize callers updating the same spec.

Run tests from this directory:

```sh
python -m unittest -v test_update_metric.py
```
