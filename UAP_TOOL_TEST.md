# UAP tool-chain evidence

This is the reproducibility-tool test for the UAP paper. It is separate from
the direct Gilbreth Slurm run.

## What to run

From the repository root on a machine with a working Docker engine:

```powershell
$env:Path="C:\Program Files\Docker\Docker\resources\bin;$env:Path"
docker info
python -m pytest -q paper_tools\tests paper_intake\test_paper_intake.py test_update_metric.py
python paper_tools\run_uap_tool_chain.py
```

The first metric reproduction runs the full UAP command. Later metrics reuse
the cached container run.

## What proves success

The console must contain:

```text
BUILD: {"status": "OK" ...}
REPRODUCE view_auc: {"status": "OK" ...}
VALIDATE view_auc: {"status": "PASS" ...}
VALIDATE model_auc: {"status": "PASS" ...}
```

The evidence is saved under:

```text
inspection-results/uap-tool-run/state.json
inspection-results/uap-tool-run/UAP/logs/
inspection-results/uap-tool-run/UAP/outputs/
```

The intake evidence is already saved under the latest
`inspection-results/run-*/manifest.json`; it records the UAP repository,
commit, paper page count, and completion status. The active UAP tool spec is
`papers/uap/spec.yaml`.

The reduced Docker end-to-end test requires the 21 checkpoint files under
`demo/checkpoint/` on the Docker host. That path is useful for a quick
plumbing check, but it is not the paper-faithful experiment.

## Preparing the real Docker reproduction

The original scratch checkpoints use a different naming/layout convention.
Train the exact Docker input model zoo on Gilbreth with:

```bash
sbatch -A zghodsi -p a100-80gb --qos=normal demo/gilbreth_train_for_docker.sbatch
```

After the reduced job completes, copy `demo/checkpoint/*.pth` from Gilbreth to
the same path on the Docker host. Confirm that there are 21 files before using
the reduced spec. For the actual paper reproduction, use
`papers/uap/paper_faithful_from_scratch_spec.yaml` (or its auto alias); it
trains and validates its own 81-model artifact set under
`result/paper_full_cifar10/` and does not reuse the reduced checkpoints.

## Fast Docker tool-chain smoke test

To test `reproduce_metric` and `validate_metric` without running the long UAP
experiment, use the UAP-shaped smoke spec:

```powershell
python paper_tools\run_tool_smoke.py
```

Success is demonstrated by three `REPRODUCE ... status OK` lines and three
`VALIDATE ... status PASS` lines. This validates Docker execution, output
parsing, state persistence, and comparison logic; it is not a scientific UAP
metric reproduction.

## Full reproduction through the LangChain agent

The agent must use the provided tools to execute and validate the experiment.
It must not only describe the steps. The runner preloads the selected spec,
then the tool sequence is:

~~~text
docker_build (local Docker/auto backend)
reproduce_metric (one metric at a time)
validate_metric (one metric at a time)
~~~

For automatic local execution with a Gilbreth fallback, first verify that the
local Docker image can execute a real CUDA operation. Then run:

~~~powershell
python -m paper_tools.langchain_agent `
  --spec papers\uap\auto_from_scratch_spec.yaml `
  --workspace inspection-results\local-uap-agent-run `
  --prompt "Use the active paper-faithful specification. Build the image, run the complete 81-model UAP reproduction, reproduce every metric, and validate every result. Do not substitute the reduced demo."
~~~

The run is successful only when the evidence state contains a completed run,
reproduced values, and PASS validations:

~~~powershell
Get-Content inspection-results\local-uap-agent-run\state.json
~~~
