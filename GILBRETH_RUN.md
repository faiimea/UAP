# Running the UAP reproduction on Gilbreth

Gilbreth is an HPC cluster. Use Slurm for the GPU work; do not run the full
experiment on the login node. The paper-faithful script below trains the
complete 81-model CIFAR-10 baseline, performs adversarial soft-label model
extraction, generates the UAP and fingerprints, trains the encoder, and
prints every metric used by the LangChain validation tools.

## 1. Log in

From Windows PowerShell:

```powershell
ssh YOUR_PURDUE_USERNAME@gilbreth.rcac.purdue.edu
```

Approve the Purdue MFA prompt when requested.

## 2. Put the repository on Gilbreth

From the Gilbreth login node, clone the public repository, then copy the
updated files from this checkout if the changes have not been pushed upstream:

```bash
cd "$HOME"
git clone https://github.com/adarsh-siva/UAP.git UAP
cd "$HOME/UAP"
```

The files that must be present from this checkout are:

```text
demo/gilbreth_paper_full.sbatch
demo/paper_full_cifar10.py
demo/gilbreth_setup.sh
Dockerfile.uap                 # needed only for local Docker validation
papers/uap/paper_faithful_from_scratch_spec.yaml
```

The paper-faithful script trains and stores its own artifacts under
`result/paper_full_cifar10/`; it does not reuse the legacy `demo/checkpoint`
files.

## 3. Create the Python environment once

From the repository root:

```bash
bash demo/gilbreth_setup.sh
```

The setup script creates `$HOME/uap-env` and installs CUDA-enabled PyTorch plus
the scientific Python dependencies.

## 4. Verify the environment before submitting

```bash
$HOME/uap-env/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

## 5. Submit the GPU job

Use the Slurm account and partition assigned to your Purdue research group:

```bash
sbatch -A YOUR_ACCOUNT --partition=a100-80gb --qos=normal demo/gilbreth_paper_full.sbatch
```

The job requests one GPU and up to seven days. It is safe to leave it queued;
the training runs on the allocated compute node, never on the login node.

## 6. Monitor and inspect the result

```bash
squeue --me
tail -f uap_paper_full_JOBID.out
```

When it finishes, the final values are in:

```text
result/paper_full_cifar10/metrics_JOBID.log
result/paper_full_cifar10/manifest.json
result/paper_full_cifar10/checkpoints/*.pth
result/paper_full_cifar10/v_opt.npy
result/paper_full_cifar10/fingerprint_points.npy
result/paper_full_cifar10/encoder.pth
```

The LangChain spec extracts the two ROC AUC lines, the confidence line, the
UAP target/fooling-rate lines, and the 81-model count from the job log.

## Docker/tool validation on a local machine

The cluster path is the direct Slurm path above. The Docker path is for
validating the paper-reproduction tools on a machine with Docker Desktop and
NVIDIA Container Toolkit:

```powershell
cd C:\Users\samue\adarsh-UAP-compare
docker build -f Dockerfile.uap -t repro-uap:latest .
docker run --rm --gpus all repro-uap:latest nvidia-smi
```

The paper spec used by the tools is
`papers/uap/paper_faithful_from_scratch_spec.yaml`. It trains its own model
zoo and does not require checkpoints under `demo/checkpoint`.
