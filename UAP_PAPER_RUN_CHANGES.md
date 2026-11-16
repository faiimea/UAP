# UAP paper reproduction: required changes and runbook

This document records the changes required to run the CVPR 2022 UAP paper
experiment in this repository and to execute it through the LangChain
reproduction tools.

It is intended to be read alongside the source code, not treated as a claim
that every unpublished author-side detail is known. The public paper and
repository do not specify every seed and training detail needed for a
bit-identical run. The implementation below therefore targets a reproducible,
paper-faithful CIFAR-10 protocol and records the exact configuration in a
manifest.

## 1. What the run reproduces

The paper-faithful path is the 81-model CIFAR-10 experiment:

- one ResNet-18 victim model;
- ten piracy and ten independently trained homologous models for each of
  ResNet-34, VGG-16, DenseNet-121, and GoogLeNet;
- 0.17 perturbation bound (`delta`), corresponding to the paper's UAP target
  of at least 80% fooling;
- UAP generation followed by 100 fingerprint points and 200 transformed
  views per point;
- supervised-contrastive encoder training on five piracy and five homologous
  models, with batch size 512;
- held-out model evaluation and validation of view-level AUC, model-level AUC,
  piracy-detection confidence, UAP fooling rate, target attainment, and model
  count.

The reduced 21-model demo remains useful for a quick plumbing test, but it is
not the full paper experiment.

## 2. Source-code changes

### Model construction

`train/model_structure.py` was updated so the model names used by the full
driver construct the repository's CIFAR-10 implementations correctly:

- `vgg16` uses `VGG('VGG16')`;
- `densenet121` uses `DenseNet121()`;
- `googlenet` uses `GoogLeNet()`.

Without these changes, the full model zoo cannot be constructed reliably from
the public repository.

### Device portability and UAP termination

`preparation/uap.py` and `encoder_trainingdata_preparation.py` use an explicit
CUDA/CPU device instead of assuming a particular GPU or calling `.cuda()` on
every tensor. The UAP routine also accepts `max_rounds`, and the full driver
sets a finite maximum of 1,000 rounds. This prevents a failed or unusually
hard UAP run from looping forever.

### Encoder stability and paths

`encoder_training.py` was changed to:

- save the encoder to `result/model.pth` instead of an author-specific
  absolute path;
- use an epsilon in cosine-similarity denominators so zero feature vectors do
  not produce `NaN` during evaluation;
- move models, fingerprint tensors, and batches to the selected device.

The reduced demo's encoder labels were also corrected so the victim and
piracy models form the same contrastive class while independently trained
models form the comparison class.

### Full experiment driver

`demo/paper_full_cifar10.py` is the authoritative full-run entry point. It
adds the missing orchestration around the research scripts:

1. download and split CIFAR-10 into disjoint victim and attacker halves;
2. train the ResNet-18 victim;
3. train piracy models by iterative adversarial soft-label extraction rather
   than by silently fine-tuning victim weights;
4. train independent homologous models;
5. generate and validate the UAP;
6. select fingerprint points and generate views;
7. train the supervised-contrastive encoder;
8. evaluate held-out models and print machine-readable metrics.

The driver is resumable at artifact boundaries. It writes its configuration,
checkpoints, metrics, and protocol manifest under
`result/paper_full_cifar10/`.

## 3. Docker changes

`Dockerfile.uap` was added for the current laptop GPU and for repeatable
tool execution.

Important changes from the old dependency setup:

- base image: `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`, which supports
  RTX 5080-class Blackwell CUDA kernels;
- `numpy<2`, because the original PyTorch/research dependencies can fail with
  NumPy 2.x binary-compatibility warnings or errors;
- install only the required runtime packages instead of the conflicting legacy
  pins in `requirements.txt`;
- install the LangChain/tool dependencies needed by the agent;
- do not hard-code reduced-demo values such as 150 points or 25 encoder
  epochs in the image. The active YAML spec and the full driver own those
  parameters.

The `.dockerignore` file keeps large generated datasets, checkpoints, result
folders, and inspection logs out of the Docker build context.

Build and verify the image from the repository root:

```powershell
cd C:\Users\samue\adarsh-UAP-compare
docker build -f Dockerfile.uap -t repro-uap:latest .
docker run --rm repro-uap:latest python -c "import torch, numpy, sklearn; print('CONTAINER IMPORTS OK')"
docker run --rm --gpus all repro-uap:latest python -c "import torch; assert torch.cuda.is_available(); x=torch.randn(64,64,device='cuda'); (x@x).sum().item(); torch.cuda.synchronize(); print(torch.cuda.get_device_name(0)); print('CUDA PROBE OK')"
```

The second command requires Docker Desktop GPU integration. If it fails, the
agent's `auto` backend records the failure and can use the configured Gilbreth
Slurm fallback instead.

## 4. Specification changes

`papers/uap/paper_faithful_from_scratch_spec.yaml` declares the full protocol:

- the authoritative command: `python demo/paper_full_cifar10.py`;
- the Dockerfile and image;
- the Gilbreth fallback script and Slurm resource settings;
- metric extraction patterns and recorded paper values;
- tolerances for each comparison.

`papers/uap/auto_from_scratch_spec.yaml` selects local Docker only when its
real CUDA probe succeeds; otherwise it selects the Slurm fallback. This is
different from merely checking that `torch.cuda.is_available()` is true:
Docker must be able to execute a CUDA kernel in the actual image.

The full spec validates these metrics:

| Metric | Recorded value | Acceptance rule |
|---|---:|---|
| View-level ROC AUC | 1.0 | absolute difference <= 0.05 |
| Model-level ROC AUC | 1.0 | absolute difference <= 0.05 |
| Minimum piracy detection confidence | 99.99% | absolute difference <= 0.01 |
| UAP fooling rate | 0.80 | absolute difference <= 0.05 |
| UAP target met | 1 | exact |
| Model count | 81 | exact |

These recorded values are the comparison targets declared in the spec; they
are not silently replaced by a convenient run result.

## 5. LangChain/tool changes

The LangChain layer now uses the same bound tool objects and shared
`ReproState` as the standalone tools.

### `paper_tools/langchain_agent.py`

- preloads the exact CLI-selected YAML spec before the model is called;
- prevents the model from inventing a different Windows path or spec;
- tells the model to build, reproduce, and validate rather than merely narrate;
- requires every metric to finish with `OK` and `PASS`/`NO_RECORD`;
- refuses to substitute the reduced demo for the paper-faithful spec;
- finishes deterministically with the bound tools if a model response only
  describes a tool call without actually making one.

### Docker and backend tools

- `paper_tools/docker_build.py` now binds the active spec correctly and
  returns structured status/log information;
- `paper_tools/docker_ops.py` adds the real CUDA probe;
- `paper_tools/reproduce_tools.py` supports Docker, automatic local-versus-
  Slurm selection, serialized expensive runs, cached results, and Slurm log
  extraction;
- `paper_tools/spec.py` and `spec_validation_tools.py` understand backend and
  Slurm fields;
- `paper_tools/state.py` writes state atomically and serializes operations so
  parallel LangChain calls do not launch duplicate runs or corrupt
  `state.json`.

## 6. Run locally through LangChain

Set the API key in the current PowerShell session without pasting it into a
file or committing it:

```powershell
$env:OPENAI_API_KEY = '<your key>'
```

Then run the full paper-faithful experiment:

```powershell
cd C:\Users\samue\adarsh-UAP-compare
python -m paper_tools.langchain_agent `
  --spec papers\uap\paper_faithful_from_scratch_spec.yaml `
  --workspace inspection-results\paper-faithful-run `
  --prompt "Use the active paper-faithful specification. Build the image, run the complete 81-model CIFAR-10 UAP reproduction, reproduce every metric, and validate every result. Do not substitute the reduced demo."
```

To watch the direct Docker container while it runs, use a second PowerShell
window:

```powershell
docker ps
docker logs -f <container-name-or-id>
```

The LangChain evidence directory is:

```text
C:\Users\samue\adarsh-UAP-compare\inspection-results\paper-faithful-run
```

The important files are `state.json`, the run log under the spec's `logs`
folder, and the generated manifest under
`result/paper_full_cifar10/manifest.json`.

## 7. Run on Gilbreth

The Slurm fallback script is `demo/gilbreth_paper_full.sbatch`. It uses one
GPU, eight CPUs, 64 GB RAM, and a seven-day wall-time limit. From the Gilbreth
repository copy:

```bash
cd ~/UAP
sbatch -A zghodsi -p a100-80gb --qos=normal demo/gilbreth_paper_full.sbatch
```

Monitor and inspect the job:

```bash
squeue --me
squeue --start -j JOB_ID
scontrol show job JOB_ID | grep -E "JobState|Reason|StartTime|ReqTRES"
tail -f uap_paper_full_JOB_ID.out
```

Do not run the long training command on the Gilbreth login node. The full
script must run inside the allocated GPU job.

## 8. Evidence from the completed local run

The completed paper-faithful local run returned code 0 and produced these
validated values:

| Metric | Reproduced | Validation |
|---|---:|---|
| Model count | 81 | PASS |
| View-level ROC AUC | 0.99061 | PASS |
| Model-level ROC AUC | 0.995918 | PASS |
| Piracy detection confidence | 99.999889% | PASS |
| UAP fooling rate | 0.8375 | PASS |
| UAP target met | 1 | PASS |

Evidence is stored at:

```text
C:\Users\samue\adarsh-UAP-compare\inspection-results\paper-faithful-run\state.json
C:\Users\samue\adarsh-UAP-compare\inspection-results\paper-faithful-run\UAP-paper-faithful-CIFAR10\logs\run-88b48e5ad887.log
C:\Users\samue\adarsh-UAP-compare\result\paper_full_cifar10\manifest.json
```

The run took approximately 101,654 seconds, or 28.2 hours, on the configured
local GPU setup. The older output with AUC near 0.54/0.48 was from the reduced
or incorrectly configured path and should not be presented as the final
paper-faithful result.

## 9. Tool-test evidence versus scientific evidence

Keep these claims separate in a report or poster:

1. **Tool tests:** the repository's unit/spec/intake tests passed, including
   the Docker build/import smoke path and metric parsing/validation logic.
2. **Scientific reproduction:** the paper-faithful command trained the full
   declared model zoo, generated the UAP and fingerprints, computed every
   declared metric, and validated the values against the YAML record.

The smoke test proves that the tools can build, run, parse, persist, and
compare. It does not prove the UAP paper's scientific result. The completed
81-model run and its evidence files provide that second layer of evidence.
