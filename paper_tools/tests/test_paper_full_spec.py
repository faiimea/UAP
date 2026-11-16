from pathlib import Path

from paper_tools.spec import load_spec


def test_uap_full_spec_is_not_the_reduced_demo():
    root = Path(__file__).resolve().parents[2]
    spec = load_spec(str(root / "papers" / "uap" / "paper_faithful_from_scratch_spec.yaml"))
    assert spec.name == "UAP-paper-faithful-CIFAR10"
    assert spec.run_command.strip() == "python demo/paper_full_cifar10.py"
    assert "81 models" in spec.notes
    assert "adversarial" in spec.notes.lower()
    assert spec.metrics["view_auc"].recorded == 1.0
    assert spec.metrics["model_auc"].recorded == 1.0
    assert spec.metrics["piracy_detection_confidence"].recorded == 99.99


def test_auto_alias_uses_the_same_full_driver():
    root = Path(__file__).resolve().parents[2]
    spec = load_spec(str(root / "papers" / "uap" / "auto_from_scratch_spec.yaml"))
    assert spec.backend == "auto"
    assert "paper_full_cifar10.py" in spec.run_command
    assert "adversarial" in spec.notes.lower()
    assert spec.slurm_script.endswith("gilbreth_paper_full.sbatch")
