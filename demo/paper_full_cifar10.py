"""Paper-faithful CIFAR-10 reproduction driver.

This is the execution target for the LangChain reproduction tools.  It is
deliberately separate from the old ``demo/step*.py`` reduced experiment:

* ResNet18 is the victim architecture.
* ResNet34, VGG16, DenseNet121 and GoogLeNet each contribute 10 piracy and
  10 homologous models (81 models total including the victim).
* Piracy models use iterative adversarial soft-label extraction from the
  victim; they are not fine-tuned copies of the victim.
* Homologous models are trained independently on sampled CIFAR-10 labels.
* UAP generation uses the repository DeepFool/UAP implementation and the
  paper's 83% target (delta=0.17).
* Fingerprints use 100 points and 200 views.
* The encoder uses five piracy and five homologous training models at batch
  size 512.  The remaining models are held out for metrics.

The script is resumable at every artifact boundary.  It writes a manifest
with the exact protocol and refuses to silently reuse artifacts made with a
different configuration.
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset, TensorDataset


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "train"))
sys.path.insert(0, str(REPO / "preparation"))

from model_structure import get_model  # noqa: E402
from uap import compute_fooling_rate, universal_perturbation  # noqa: E402


SEED = int(os.environ.get("UAP_FULL_SEED", "0"))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
WORKERS = int(os.environ.get("UAP_WORKERS", "4"))
PIN_MEMORY = DEVICE.type == "cuda"

VICTIM_ARCH = "resnet18"
OTHER_ARCHS = ("resnet34", "vgg16", "densenet121", "googlenet")
MODELS_PER_ARCH = int(os.environ.get("UAP_MODELS_PER_ARCH", "10"))
VICTIM_EPOCHS = int(os.environ.get("UAP_FULL_VICTIM_EPOCHS", "100"))
HOMO_EPOCHS = int(os.environ.get("UAP_FULL_HOMO_EPOCHS", "100"))
PIRACY_EPOCHS = int(os.environ.get("UAP_FULL_PIRACY_EPOCHS", "100"))
PIRACY_QUERY_FRACTION = float(os.environ.get("UAP_PIRACY_QUERY_FRACTION", "0.15"))
PIRACY_TEMPERATURE = float(os.environ.get("UAP_PIRACY_TEMPERATURE", "1.0"))
PIRACY_MODE = os.environ.get("UAP_PIRACY_MODE", "adversarial").lower()
PIRACY_ROUNDS = int(os.environ.get("UAP_PIRACY_ROUNDS", "4"))
PIRACY_ADV_STEPS = int(os.environ.get("UAP_PIRACY_ADV_STEPS", "5"))
PIRACY_ADV_EPS = float(os.environ.get("UAP_PIRACY_ADV_EPS", "0.17"))
PIRACY_ADV_ALPHA = float(os.environ.get("UAP_PIRACY_ADV_ALPHA", "0.034"))
TRAIN_BATCH = int(os.environ.get("UAP_FULL_TRAIN_BATCH", "128"))
POINTS = int(os.environ.get("UAP_POINTS", "100"))
VIEWS = int(os.environ.get("UAP_VIEWS", "200"))
UAP_FIT_IMAGES = int(os.environ.get("UAP_FIT_IMAGES", "2000"))
UAP_VAL_IMAGES = int(os.environ.get("UAP_VAL_IMAGES", "2000"))
UAP_DELTA = float(os.environ.get("UAP_DELTA", "0.17"))
UAP_XI = float(os.environ.get("UAP_XI", "10"))
UAP_MAX_DF = int(os.environ.get("UAP_MAX_ITER_DF", "50"))
UAP_MAX_ROUNDS = int(os.environ.get("UAP_MAX_ROUNDS", "1000"))
ENCODER_TRAIN_MODELS = int(os.environ.get("UAP_ENCODER_MODELS", "5"))
ENCODER_EPOCHS = int(os.environ.get("UAP_ENC_EPOCHS", "10"))
ENCODER_BATCH = int(os.environ.get("UAP_ENC_BATCH", "512"))
ENCODER_LR = float(os.environ.get("UAP_ENC_LR", "5e-4"))
ENCODER_TAU = float(os.environ.get("UAP_ENC_TAU", "0.2"))

MEAN = 0.49116027
STD = 0.24728487

DATA_DIR = REPO / "data"
ARTIFACT_DIR = Path(os.environ.get("UAP_ARTIFACT_DIR", str(REPO / "result" / "paper_full_cifar10")))
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"
FEATURE_DIR = ARTIFACT_DIR / "features"
SPLIT_PKL = DATA_DIR / "CIFAR10_sub_train_split.pkl"
HOMO_INDEX_FILE = ARTIFACT_DIR / "homologous_indices.json"
CONFIG_FILE = ARTIFACT_DIR / "config.json"
MANIFEST_FILE = ARTIFACT_DIR / "manifest.json"
UAP_FILE = ARTIFACT_DIR / "v_opt.npy"
POINT_FILE = ARTIFACT_DIR / "fingerprint_points.npy"
ENCODER_FILE = ARTIFACT_DIR / "encoder.pth"


class ArrayDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = np.asarray(labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        from PIL import Image
        image = self.images[index]
        if not hasattr(image, "mode"):
            image = Image.fromarray(np.asarray(image))
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index])


class SoftLabelDataset(Dataset):
    def __init__(self, images: torch.Tensor, targets: torch.Tensor):
        self.images = images
        self.targets = targets

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.targets[index]


class SupervisedContrastiveLoss(nn.Module):
    """Equation (7) from the paper, vectorized over one mini-batch."""

    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z = nn.functional.normalize(features, dim=1)
        logits = torch.matmul(z, z.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        n = labels.shape[0]
        eye = torch.eye(n, device=labels.device, dtype=torch.bool)
        positive = labels[:, None].eq(labels[None, :]) & ~eye
        log_prob = logits - torch.logsumexp(logits.masked_fill(eye, -torch.inf), dim=1, keepdim=True)
        positive_count = positive.sum(dim=1)
        valid = positive_count > 0
        if not bool(valid.any()):
            raise RuntimeError("contrastive batch has no positive pairs")
        mean_log_prob_pos = torch.where(positive, log_prob, torch.zeros_like(log_prob)).sum(dim=1)[valid] / positive_count[valid]
        return -mean_log_prob_pos.mean()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def loader(dataset, *, batch_size=TRAIN_BATCH, shuffle=False):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": WORKERS,
        "pin_memory": PIN_MEMORY,
    }
    if WORKERS:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def make_model(architecture: str) -> nn.Module:
    return get_model(architecture, False, 10).to(DEVICE)


def checkpoint_path(name: str) -> Path:
    return CHECKPOINT_DIR / f"{name}.pth"


def save_checkpoint(net: nn.Module, name: str) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), checkpoint_path(name))


def load_checkpoint(name: str, architecture: str) -> nn.Module:
    path = checkpoint_path(name)
    if not path.exists():
        raise FileNotFoundError(f"missing {name} checkpoint: {path}")
    net = make_model(architecture)
    net.load_state_dict(torch.load(path, map_location=DEVICE), strict=True)
    net.eval()
    return net


@torch.no_grad()
def accuracy(net: nn.Module, data_loader: DataLoader) -> float:
    net.eval()
    good = total = 0
    for images, labels in data_loader:
        output = net(images.to(DEVICE, non_blocking=True))
        good += int((output.argmax(1).cpu() == labels).sum())
        total += labels.numel()
    return good / max(1, total)


def train_test_loader():
    data = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR / "CIFAR10"), train=False, download=True, transform=eval_transform()
    )
    return loader(data, batch_size=256, shuffle=False)


def milestones(epochs: int) -> list[int]:
    return [max(1, int(epochs * 0.4)), max(2, int(epochs * 0.7))]


def train_hard(net: nn.Module, dataset: Dataset, epochs: int, name: str) -> float:
    train_loader = loader(dataset, shuffle=True)
    optimizer = SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones(epochs), gamma=0.2)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        net.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        schedule.step()
        print(f"[{name}] epoch {epoch}/{epochs} avg loss {total_loss / max(1, len(train_loader)):.5f}", flush=True)
    return accuracy(net, train_test_loader())


def train_soft(
    net: nn.Module,
    dataset: SoftLabelDataset,
    epochs: int,
    name: str,
    *,
    evaluate: bool = True,
) -> float | None:
    train_loader = loader(dataset, shuffle=True)
    optimizer = SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones(epochs), gamma=0.2)
    criterion = nn.KLDivLoss(reduction="batchmean")
    temperature = PIRACY_TEMPERATURE
    for epoch in range(1, epochs + 1):
        net.train()
        total_loss = 0.0
        for images, target_probs in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            target_probs = target_probs.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            student_log_probs = nn.functional.log_softmax(net(images) / temperature, dim=1)
            loss = criterion(student_log_probs, target_probs) * (temperature ** 2)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        schedule.step()
        print(f"[{name}] epoch {epoch}/{epochs} avg soft loss {total_loss / max(1, len(train_loader)):.5f}", flush=True)
    return accuracy(net, train_test_loader()) if evaluate else None


def normalized_bounds() -> tuple[torch.Tensor, torch.Tensor]:
    low = torch.tensor([(0.0 - MEAN) / STD] * 3, device=DEVICE).view(1, 3, 1, 1)
    high = torch.tensor([(1.0 - MEAN) / STD] * 3, device=DEVICE).view(1, 3, 1, 1)
    return low, high


def craft_extraction_adversarial(net: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Craft bounded untargeted adversarial queries against the substitute.

    The official repository's extraction path augments natural queries with
    adversarial examples before querying the victim for soft labels.  This
    native PyTorch implementation keeps that behavior inside the reproducible
    driver, avoiding the original script's hard-coded GPU index and paths.
    Inputs are already normalized, so clipping uses normalized CIFAR-10 bounds.
    """
    if PIRACY_ADV_STEPS <= 0 or PIRACY_ADV_EPS <= 0:
        return images.detach().clone()
    net.eval()
    low, high = normalized_bounds()
    with torch.no_grad():
        labels = net(images).argmax(dim=1)
    adv = images.detach().clone()
    for _ in range(PIRACY_ADV_STEPS):
        adv.requires_grad_(True)
        logits = net(adv)
        loss = nn.functional.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, adv, only_inputs=True)[0]
        adv = adv.detach() + PIRACY_ADV_ALPHA * grad.sign()
        adv = torch.maximum(torch.minimum(adv, images + PIRACY_ADV_EPS), images - PIRACY_ADV_EPS)
        adv = torch.maximum(torch.minimum(adv, high), low)
    return adv.detach()


def teacher_soft_labels(teacher: nn.Module, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.softmax(teacher(images) / PIRACY_TEMPERATURE, dim=1).cpu()


def train_adversarial_extraction(
    teacher: nn.Module,
    student: nn.Module,
    query_images: torch.Tensor,
    epochs: int,
    name: str,
) -> float:
    """Train one piracy model using iterative adversarial soft-label queries.

    Each extraction round generates adversarial examples against the current
    substitute, queries the victim on clean and adversarial inputs, and trains
    on both sets.  The total training budget is split across rounds so the
    default remains resumable and comparable across the ten copies per
    architecture.
    """
    if PIRACY_MODE == "clean":
        target_probs = teacher_soft_labels(teacher, query_images.to(DEVICE))
        return train_soft(student, SoftLabelDataset(query_images.cpu(), target_probs), epochs, name)
    if PIRACY_MODE != "adversarial":
        raise ValueError("UAP_PIRACY_MODE must be 'adversarial' (paper path) or 'clean' (debug only)")

    rounds = max(1, PIRACY_ROUNDS)
    epochs_per_round = max(1, int(np.ceil(epochs / rounds)))
    for round_index in range(rounds):
        clean_probs_chunks = []
        adversarial_images_chunks = []
        adversarial_probs_chunks = []
        for start in range(0, len(query_images), TRAIN_BATCH):
            end = min(start + TRAIN_BATCH, len(query_images))
            clean_batch = query_images[start:end].to(DEVICE, non_blocking=True)
            clean_probs_chunks.append(teacher_soft_labels(teacher, clean_batch))
            adversarial_batch = craft_extraction_adversarial(student, clean_batch)
            adversarial_images_chunks.append(adversarial_batch.cpu())
            adversarial_probs_chunks.append(teacher_soft_labels(teacher, adversarial_batch))
        clean_probs = torch.cat(clean_probs_chunks)
        adversarial_images = torch.cat(adversarial_images_chunks)
        adversarial_probs = torch.cat(adversarial_probs_chunks)
        combined_images = torch.cat((query_images, adversarial_images), dim=0)
        combined_probs = torch.cat((clean_probs, adversarial_probs), dim=0)
        train_soft(
            student,
            SoftLabelDataset(combined_images, combined_probs),
            epochs_per_round,
            f"{name}/extraction-round-{round_index + 1}",
            evaluate=False,
        )
        predictions = []
        for start in range(0, len(query_images), TRAIN_BATCH):
            end = min(start + TRAIN_BATCH, len(query_images))
            with torch.no_grad():
                predictions.append(student(query_images[start:end].to(DEVICE)).argmax(1).cpu())
        agreement = float((torch.cat(predictions) == clean_probs.argmax(1)).float().mean())
        print(f"[{name}] extraction round {round_index + 1}/{rounds} query agreement {agreement:.4f}", flush=True)
    return accuracy(student, train_test_loader())


def protocol_config() -> dict:
    return {
        "paper": "Peng et al., CVPR 2022, CIFAR-10 baseline",
        "victim_architecture": VICTIM_ARCH,
        "other_architectures": list(OTHER_ARCHS),
        "models_per_architecture_and_type": MODELS_PER_ARCH,
        "expected_model_count": 1 + 2 * len(OTHER_ARCHS) * MODELS_PER_ARCH,
        "victim_epochs": VICTIM_EPOCHS,
        "homologous_epochs": HOMO_EPOCHS,
        "piracy_epochs": PIRACY_EPOCHS,
        "piracy_query_fraction": PIRACY_QUERY_FRACTION,
        "piracy_mode": PIRACY_MODE,
        "piracy_extraction_rounds": PIRACY_ROUNDS,
        "piracy_adversarial_steps": PIRACY_ADV_STEPS,
        "piracy_adversarial_epsilon": PIRACY_ADV_EPS,
        "piracy_adversarial_alpha": PIRACY_ADV_ALPHA,
        "fingerprint_points": POINTS,
        "fingerprint_views": VIEWS,
        "uap_fit_images": UAP_FIT_IMAGES,
        "uap_validation_images": UAP_VAL_IMAGES,
        "uap_delta": UAP_DELTA,
        "uap_xi_linf": UAP_XI,
        "uap_deepfool_max_iter": UAP_MAX_DF,
        "encoder_training_models_per_type": ENCODER_TRAIN_MODELS,
        "encoder_batch_size": ENCODER_BATCH,
        "encoder_epochs": ENCODER_EPOCHS,
        "encoder_temperature": ENCODER_TAU,
        "seed": SEED,
    }


def ensure_dirs_and_config() -> None:
    for path in (ARTIFACT_DIR, CHECKPOINT_DIR, FEATURE_DIR):
        path.mkdir(parents=True, exist_ok=True)
    current = protocol_config()
    if CONFIG_FILE.exists():
        old = json.loads(CONFIG_FILE.read_text())
        if old != current:
            raise RuntimeError(
                f"{CONFIG_FILE} was created with a different protocol. "
                "Use a new paper_full_cifar10 artifact directory instead of mixing runs."
            )
    else:
        CONFIG_FILE.write_text(json.dumps(current, indent=2) + "\n")


def load_cifar_train() -> tuple[np.ndarray, np.ndarray]:
    ds = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR / "CIFAR10"), train=True, download=True, transform=None
    )
    return np.asarray(ds.data), np.asarray(ds.targets)


def ensure_split(images: np.ndarray, labels: np.ndarray):
    if not SPLIT_PKL.exists():
        rng = np.random.RandomState(SEED)
        victim_idx = rng.choice(len(images), len(images) // 2, replace=False)
        attacker_idx = np.setdiff1d(np.arange(len(images)), victim_idx)
        with open(SPLIT_PKL, "wb") as f:
            pickle.dump(((images[victim_idx], labels[victim_idx]), (images[attacker_idx], labels[attacker_idx])), f)
    with open(SPLIT_PKL, "rb") as f:
        (victim_x, victim_y), (attacker_x, attacker_y) = pickle.load(f)
    if len(victim_x) != len(images) // 2 or len(attacker_x) != len(images) // 2:
        raise RuntimeError("CIFAR-10 split is not two equal halves")
    return np.asarray(victim_x), np.asarray(victim_y), np.asarray(attacker_x), np.asarray(attacker_y)


def ensure_homologous_indices(total: int, n: int) -> dict[str, list[int]]:
    if HOMO_INDEX_FILE.exists():
        return json.loads(HOMO_INDEX_FILE.read_text())
    rng = np.random.RandomState(SEED + 1)
    result = {}
    for arch in OTHER_ARCHS:
        for i in range(MODELS_PER_ARCH):
            result[f"homo_{arch}_{i}"] = rng.choice(total, n, replace=False).tolist()
    HOMO_INDEX_FILE.write_text(json.dumps(result, indent=2) + "\n")
    return result


def train_model_zoo(images, labels, victim_x, victim_y, homo_indices):
    expected = 1 + 2 * len(OTHER_ARCHS) * MODELS_PER_ARCH
    print(f"=== model preparation: paper-faithful {expected}-model CIFAR-10 baseline ===", flush=True)
    if not checkpoint_path("victim").exists():
        seed_everything(SEED)
        victim = make_model(VICTIM_ARCH)
        acc = train_hard(victim, ArrayDataset(victim_x, victim_y, train_transform()), VICTIM_EPOCHS, "victim")
        save_checkpoint(victim, "victim")
        print(f"[victim] test accuracy {acc:.4f}", flush=True)
    else:
        print("[victim] checkpoint exists, skipping", flush=True)

    for arch in OTHER_ARCHS:
        for i in range(MODELS_PER_ARCH):
            name = f"homo_{arch}_{i}"
            if checkpoint_path(name).exists():
                print(f"[{name}] checkpoint exists, skipping", flush=True)
                continue
            seed_everything(SEED + 1000 + i + 100 * OTHER_ARCHS.index(arch))
            idx = np.asarray(homo_indices[name], dtype=np.int64)
            net = make_model(arch)
            acc = train_hard(net, ArrayDataset(images[idx], labels[idx], train_transform()), HOMO_EPOCHS, name)
            save_checkpoint(net, name)
            print(f"[{name}] test accuracy {acc:.4f}", flush=True)
            del net
            gc.collect()

    query_count = max(1, int(round(PIRACY_QUERY_FRACTION * len(images))))
    teacher = load_checkpoint("victim", VICTIM_ARCH)
    for arch in OTHER_ARCHS:
        for i in range(MODELS_PER_ARCH):
            name = f"piracy_{arch}_{i}"
            if checkpoint_path(name).exists():
                print(f"[{name}] checkpoint exists, skipping", flush=True)
                continue
            seed = SEED + 2000 + i + 100 * OTHER_ARCHS.index(arch)
            seed_everything(seed)
            idx = np.random.RandomState(seed).choice(len(images), query_count, replace=False)
            query_ds = ArrayDataset(images[idx], labels[idx], eval_transform())
            query_images = torch.stack([query_ds[j][0] for j in range(len(query_ds))])
            net = make_model(arch)
            acc = train_adversarial_extraction(teacher, net, query_images, PIRACY_EPOCHS, name)
            predictions = []
            teacher_labels = []
            for start in range(0, len(query_images), TRAIN_BATCH):
                end = min(start + TRAIN_BATCH, len(query_images))
                with torch.no_grad():
                    batch = query_images[start:end].to(DEVICE)
                    predictions.append(net(batch).argmax(1).cpu())
                    teacher_labels.append(teacher(batch).argmax(1).cpu())
            agreement = float((torch.cat(predictions) == torch.cat(teacher_labels)).float().mean())
            save_checkpoint(net, name)
            print(f"[{name}] test accuracy {acc:.4f}; query agreement {agreement:.4f}", flush=True)
            del net, query_images
            gc.collect()

    actual = len(list(CHECKPOINT_DIR.glob("*.pth")))
    if actual != expected:
        raise RuntimeError(f"expected {expected} checkpoints, found {actual} in {CHECKPOINT_DIR}")
    print(f"MODEL_COUNT: {actual}", flush=True)


def generate_uap(victim, victim_x, victim_y, test_x, test_y) -> torch.Tensor:
    validation = ArrayDataset(test_x[:UAP_VAL_IMAGES], test_y[:UAP_VAL_IMAGES], eval_transform())
    if UAP_FILE.exists():
        print(f"UAP exists, reusing {UAP_FILE}", flush=True)
        array = np.asarray(np.load(UAP_FILE), dtype=np.float32)
        rate = float(compute_fooling_rate(victim, loader(validation, batch_size=128, shuffle=False), array))
        target_met = rate >= 1 - UAP_DELTA
        print(f"UAP_TARGET_MET: {int(target_met)}", flush=True)
        print(f"UAP_FINAL_FOOLING_RATE: {rate:.6f}", flush=True)
        if not target_met:
            raise RuntimeError(f"reused UAP has fooling rate {rate:.4f}; required {1 - UAP_DELTA:.4f}")
        return torch.from_numpy(array).float().to(DEVICE)
    fit = ArrayDataset(victim_x[:UAP_FIT_IMAGES], victim_y[:UAP_FIT_IMAGES], eval_transform())
    print(f"=== UAP generation: {len(fit)} fit images, target fooling rate {1 - UAP_DELTA:.2%} ===", flush=True)
    start = time.time()
    v_opt, fooling_rates, _ = universal_perturbation(
        fit,
        loader(validation, batch_size=128, shuffle=False),
        victim,
        delta=UAP_DELTA,
        xi=UAP_XI,
        p=np.inf,
        num_classes=10,
        max_iter_df=UAP_MAX_DF,
        max_rounds=UAP_MAX_ROUNDS,
    )
    best = max(fooling_rates) if fooling_rates else 0.0
    if best < 1 - UAP_DELTA:
        raise RuntimeError(f"UAP stopped at fooling rate {best:.4f}; required {1 - UAP_DELTA:.4f}")
    array = np.asarray(v_opt, dtype=np.float32)
    np.save(UAP_FILE, array)
    print("UAP_TARGET_MET: 1", flush=True)
    print(f"UAP_FINAL_FOOLING_RATE: {best:.6f}", flush=True)
    print(f"UAP generation took {(time.time() - start) / 60:.2f} min; Linf={np.abs(array).max():.6f}", flush=True)
    return torch.from_numpy(array).float().to(DEVICE)


@torch.no_grad()
def victim_embeddings(victim, victim_x, victim_y):
    values, labels = [], []
    dl = loader(ArrayDataset(victim_x, victim_y, eval_transform()), batch_size=256, shuffle=False)
    for images, y in dl:
        values.append(victim(images.to(DEVICE)).cpu())
        labels.append(y)
    return torch.cat(values).numpy(), torch.cat(labels).numpy()


def select_fingerprint_points(victim, victim_x, victim_y) -> np.ndarray:
    if POINT_FILE.exists():
        points = np.load(POINT_FILE, mmap_mode="r")
        if points.shape == (VIEWS, POINTS, 3, 32, 32):
            print(f"fingerprint points exist, reusing {POINT_FILE}", flush=True)
            return np.asarray(points)
    print(f"=== fingerprint selection: {POINTS} points x {VIEWS} views ===", flush=True)
    embeddings, _ = victim_embeddings(victim, victim_x, victim_y)
    cluster_labels = KMeans(n_clusters=POINTS, n_init=10, random_state=SEED).fit_predict(embeddings)
    neighbors = NearestNeighbors(n_neighbors=VIEWS, algorithm="ball_tree").fit(embeddings)
    _, nearest = neighbors.kneighbors(embeddings)
    rng = np.random.RandomState(SEED)
    chosen = np.empty((POINTS, VIEWS), dtype=np.int64)
    for cluster in range(POINTS):
        members = np.flatnonzero(cluster_labels == cluster)
        if len(members) == 0:
            raise RuntimeError(f"empty KMeans cluster {cluster}")
        representative = int(rng.choice(members))
        chosen[cluster] = nearest[representative]
    from PIL import Image
    transform = eval_transform()
    points = np.stack([
        np.stack([transform(Image.fromarray(victim_x[index])).numpy() for index in chosen[:, view]])
        for view in range(VIEWS)
    ]).astype(np.float32)
    np.save(POINT_FILE, points)
    print(f"saved {POINT_FILE} with shape {points.shape}", flush=True)
    return points


@torch.no_grad()
def make_features(net, points: np.ndarray, perturbation: torch.Tensor) -> np.ndarray:
    net.eval()
    result = np.empty((len(points), 2 * 10 * POINTS), dtype=np.float32)
    for view, batch in enumerate(points):
        x = torch.from_numpy(np.ascontiguousarray(batch)).to(DEVICE)
        clean = torch.softmax(net(x), dim=1).cpu()
        perturbed = torch.softmax(net(x + perturbation), dim=1).cpu()
        result[view] = torch.cat((clean, perturbed), dim=1).reshape(-1).numpy()
    return result


def feature_names() -> list[tuple[str, str]]:
    result = [("victim", VICTIM_ARCH)]
    for kind in ("piracy", "homo"):
        for arch in OTHER_ARCHS:
            result.extend((f"{kind}_{arch}_{i}", arch) for i in range(MODELS_PER_ARCH))
    return result


def extract_features(points: np.ndarray, perturbation: torch.Tensor) -> None:
    expected = 1 + 2 * len(OTHER_ARCHS) * MODELS_PER_ARCH
    print(f"=== fingerprint feature extraction for all {expected} models ===", flush=True)
    for name, arch in feature_names():
        path = FEATURE_DIR / f"{name}.npy"
        if path.exists():
            array = np.load(path, mmap_mode="r")
            if array.shape == (VIEWS, 2 * 10 * POINTS):
                print(f"[{name}] features exist, skipping", flush=True)
                continue
        net = load_checkpoint(name, arch)
        np.save(path, make_features(net, points, perturbation))
        print(f"[{name}] saved features {path}", flush=True)
        del net
        gc.collect()


def balanced_selection(names: list[str], count: int) -> list[str]:
    if count > len(names):
        raise ValueError(f"cannot select {count} models from {len(names)}")
    by_arch = {arch: [n for n in names if f"_{arch}_" in n] for arch in OTHER_ARCHS}
    chosen = []
    for arch in OTHER_ARCHS:
        if len(chosen) < count:
            chosen.append(by_arch[arch][0])
    cursor = 1
    while len(chosen) < count:
        for arch in OTHER_ARCHS:
            if cursor < len(by_arch[arch]) and len(chosen) < count:
                chosen.append(by_arch[arch][cursor])
    return chosen


def load_features(name: str) -> torch.Tensor:
    path = FEATURE_DIR / f"{name}.npy"
    array = np.load(path)
    expected = (VIEWS, 2 * 10 * POINTS)
    if array.shape != expected:
        raise RuntimeError(f"{path} has {array.shape}, expected {expected}")
    return torch.from_numpy(array).float()


def train_encoder(piracy_names: list[str], homo_names: list[str]) -> nn.Module:
    from encoder import classifier
    if ENCODER_FILE.exists():
        net = classifier(input_channel=2 * 10 * POINTS).to(DEVICE)
        net.load_state_dict(torch.load(ENCODER_FILE, map_location=DEVICE))
        net.eval()
        print(f"encoder exists, reusing {ENCODER_FILE}", flush=True)
        return net
    print(f"=== encoder training: {len(piracy_names)} piracy + {len(homo_names)} homologous ===", flush=True)
    xs = [load_features("victim")]
    # Victim and piracy are the positive class; homologous models are the
    # negative class.  This is the label convention used by main.py upstream.
    ys = [torch.ones(len(xs[0]), dtype=torch.long)]
    for name in piracy_names:
        x = load_features(name)
        xs.append(x)
        ys.append(torch.ones(len(x), dtype=torch.long))
    for name in homo_names:
        x = load_features(name)
        xs.append(x)
        ys.append(torch.zeros(len(x), dtype=torch.long))
    dataset = TensorDataset(torch.cat(xs), torch.cat(ys))
    n_train = int(0.8 * len(dataset))
    train_set, _ = torch.utils.data.random_split(
        dataset, [n_train, len(dataset) - n_train], generator=torch.Generator().manual_seed(SEED)
    )
    train_dl = DataLoader(train_set, batch_size=ENCODER_BATCH, shuffle=True, drop_last=False)
    net = classifier(input_channel=2 * 10 * POINTS).to(DEVICE)
    optimizer = Adam(net.parameters(), lr=ENCODER_LR)
    criterion = SupervisedContrastiveLoss(ENCODER_TAU)
    for epoch in range(1, ENCODER_EPOCHS + 1):
        net.train()
        losses = []
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net.feature(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        print(f"[encoder] epoch {epoch}/{ENCODER_EPOCHS} loss {np.mean(losses):.6f}", flush=True)
    torch.save(net.state_dict(), ENCODER_FILE)
    net.eval()
    return net


@torch.no_grad()
def evaluate_encoder(net: nn.Module, piracy_all: list[str], homo_all: list[str]) -> dict:
    reference = net.feature(load_features("victim")[0:1].to(DEVICE))[0]
    similarities = {}
    for name in piracy_all + homo_all:
        z = net.feature(load_features(name).to(DEVICE))
        ref = reference.unsqueeze(0).expand_as(z)
        scores = nn.functional.cosine_similarity(ref, z, dim=1).cpu().numpy()
        similarities[name] = scores
        print(f"{name}: mean sim {scores.mean():.6f}", flush=True)
    piracy_views = np.concatenate([similarities[n] for n in piracy_all])
    homo_views = np.concatenate([similarities[n] for n in homo_all])
    view_auc = float(roc_auc_score(
        np.r_[np.ones(len(piracy_views)), np.zeros(len(homo_views))],
        np.r_[piracy_views, homo_views],
    ))
    piracy_means = np.array([similarities[n].mean() for n in piracy_all])
    homo_means = np.array([similarities[n].mean() for n in homo_all])
    model_auc = float(roc_auc_score(
        np.r_[np.ones(len(piracy_means)), np.zeros(len(homo_means))],
        np.r_[piracy_means, homo_means],
    ))
    confidences = []
    rng = np.random.default_rng(SEED)
    for name in piracy_all:
        repeated = []
        for _ in range(30):
            sample = rng.choice(similarities[name], 20, replace=False)
            _, p_value = stats.ttest_ind(sample, homo_views, equal_var=False, alternative="greater")
            repeated.append(float((1.0 - p_value) * 100.0))
        confidences.extend(repeated)
        print(
            f"{name}: detection confidence with 20 fingerprints = "
            f"{np.mean(repeated):.6f}% +/- {np.std(repeated):.6f}% "
            f"(min = {np.min(repeated):.6f}%, repeats = 30)",
            flush=True,
        )
    min_confidence = float(np.min(confidences))
    print(f"\nROC AUC (view level, {len(piracy_views)} piracy vs {len(homo_views)} homologous views): {view_auc:.6f}", flush=True)
    print(f"ROC AUC (model level, {len(piracy_means)} vs {len(homo_means)}): {model_auc:.6f}", flush=True)
    print(f"minimum piracy detection confidence with 20 fingerprints = {min_confidence:.6f}%", flush=True)
    return {
        "view_auc": view_auc,
        "model_auc": model_auc,
        "piracy_detection_confidence": min_confidence,
        "piracy_models_tested": len(piracy_all),
        "homologous_models_tested": len(homo_all),
    }


def main() -> int:
    seed_everything(SEED)
    ensure_dirs_and_config()
    print("PAPER_FAITHFUL_PROTOCOL: " + json.dumps(protocol_config(), sort_keys=True), flush=True)
    images, labels = load_cifar_train()
    victim_x, victim_y, _, _ = ensure_split(images, labels)
    test = torchvision.datasets.CIFAR10(root=str(DATA_DIR / "CIFAR10"), train=False, download=True, transform=None)
    test_x, test_y = np.asarray(test.data), np.asarray(test.targets)
    homo_indices = ensure_homologous_indices(len(images), len(victim_x))
    train_model_zoo(images, labels, victim_x, victim_y, homo_indices)
    victim = load_checkpoint("victim", VICTIM_ARCH)
    perturbation = generate_uap(victim, victim_x, victim_y, test_x, test_y)
    points = select_fingerprint_points(victim, victim_x, victim_y)
    extract_features(points, perturbation)
    piracy_all = [f"piracy_{arch}_{i}" for arch in OTHER_ARCHS for i in range(MODELS_PER_ARCH)]
    homo_all = [f"homo_{arch}_{i}" for arch in OTHER_ARCHS for i in range(MODELS_PER_ARCH)]
    piracy_train = balanced_selection(piracy_all, ENCODER_TRAIN_MODELS)
    homo_train = balanced_selection(homo_all, ENCODER_TRAIN_MODELS)
    piracy_test = [n for n in piracy_all if n not in piracy_train]
    homo_test = [n for n in homo_all if n not in homo_train]
    encoder = train_encoder(piracy_train, homo_train)
    metrics = evaluate_encoder(encoder, piracy_test, homo_test)
    manifest = {
        "protocol": protocol_config(),
        "checkpoint_count": len(list(CHECKPOINT_DIR.glob("*.pth"))),
        "feature_count": len(list(FEATURE_DIR.glob("*.npy"))),
        "encoder_training_piracy": piracy_train,
        "encoder_training_homologous": homo_train,
        "held_out_piracy": piracy_test,
        "held_out_homologous": homo_test,
        "metrics": metrics,
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2) + "\n")
    print("PAPER_FULL_REPRODUCTION_COMPLETE: true", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
