from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from biolaya.cloud.drive import (
    DEFAULT_DRIVE_ROOT,
    backup_checkpoint,
    initialize_drive,
    latest_checkpoint,
    mount_google_drive,
    restore_latest_checkpoint,
)
from biolaya.cloud.env import environment_report, resolve_device
from biolaya.evaluation import evaluate


def env_main():
    p = argparse.ArgumentParser(description="Inspect BioLaya CPU/GPU environment")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()
    print(json.dumps(environment_report(args.device), indent=2))


def drive_main():
    p = argparse.ArgumentParser(description="Initialize/check BioLaya Google Drive storage")
    p.add_argument("--root", default=DEFAULT_DRIVE_ROOT)
    p.add_argument("--mount", action="store_true")
    p.add_argument("--latest", default=None, help="Run subdirectory under checkpoints to inspect")
    args = p.parse_args()
    if args.mount:
        mount_google_drive()
    layout = initialize_drive(args.root)
    print(f"BioLaya Drive root: {layout.root}")
    if args.latest:
        print(f"Latest checkpoint: {latest_checkpoint(layout.checkpoints / args.latest)}")


def eval_main():
    p = argparse.ArgumentParser(description="Evaluate a prepared BioLaya decision dataset")
    p.add_argument("dataset_file")
    p.add_argument("--model", default="english")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--output-dir", default="results/eval")
    args = p.parse_args()
    print(json.dumps(evaluate(args.dataset_file, model=args.model, device=args.device, max_examples=args.max_examples, output_dir=args.output_dir), indent=2))


def data_main():
    from biolaya.datasets.registry import DEFAULT, prepare_dataset

    p = argparse.ArgumentParser(description="Prepare BioLaya biomedical decision datasets")
    p.add_argument("datasets", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--root", default="data/processed")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    names = DEFAULT if args.all else args.datasets
    if not names:
        p.error("Specify dataset names or --all")
    root = Path(args.root)
    for name in names:
        print(f"[prepare] {name}")
        counts = prepare_dataset(name, root, args.seed)
        print("  " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))


def _structural_smoke(device: str) -> dict:
    from biolaya.schemas import DecisionExample

    resolved = resolve_device(device)
    example = DecisionExample(
        id="smoke-1",
        dataset="synthetic",
        split="test",
        state="A randomized study reports that treatment A reduced blood pressure.",
        question_id="nli",
        type="choice",
        instructions="Choose the relationship: treatment A reduced blood pressure.",
        criteria={
            "entailment": "The statement is supported by the state.",
            "contradiction": "The statement conflicts with the state.",
        },
        gold="entailment",
    )
    with tempfile.TemporaryDirectory(prefix="biolaya-smoke-") as td:
        root = Path(td)
        layout = initialize_drive(root / "drive")
        ckpt = root / "local" / "checkpoint-10"
        ckpt.mkdir(parents=True)
        (ckpt / "trainer_state.json").write_text('{"global_step": 10}', encoding="utf-8")
        backed = backup_checkpoint(ckpt, layout.checkpoints / "smoke")
        restored = restore_latest_checkpoint(layout.checkpoints / "smoke", root / "restored")
        assert backed.exists() and restored is not None and restored.exists()
    return {
        "ok": True,
        "mode": "structural",
        "resolved_device": resolved,
        "question": example.laya_questions(),
        "drive_backup_restore": True,
    }


def smoke_main():
    p = argparse.ArgumentParser(description="BioLaya local/Colab smoke test")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--model", default="english")
    p.add_argument("--skip-model", action="store_true", help="Validate package/device/Drive without downloading Laya")
    args = p.parse_args()
    report = _structural_smoke(args.device)
    if not args.skip_model:
        from biolaya.models import LayaDecisionModel
        from biolaya.schemas import DecisionExample

        example = DecisionExample(
            id="smoke-model-1",
            dataset="synthetic",
            split="test",
            state="A randomized study reports that treatment A reduced blood pressure compared with placebo.",
            question_id="nli",
            type="choice",
            instructions="Determine whether the hypothesis is entailed or contradicted. Hypothesis: Treatment A reduced blood pressure.",
            criteria={
                "entailment": "The hypothesis is supported by the state.",
                "contradiction": "The hypothesis conflicts with the state.",
            },
            gold="entailment",
        )
        runner = LayaDecisionModel(model=args.model, device=args.device)
        pred = runner.predict_one(example)
        report.update(mode="model", prediction=pred.prediction, confidence=pred.confidence, probabilities=pred.probabilities)
    print(json.dumps(report, indent=2, default=str))
