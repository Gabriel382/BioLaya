from __future__ import annotations

import argparse
import json
from pathlib import Path

from biolaya.cloud.drive import DEFAULT_DRIVE_ROOT, initialize_drive, latest_checkpoint, mount_google_drive
from biolaya.cloud.env import environment_report
from biolaya.evaluation import evaluate


def env_main():
    print(json.dumps(environment_report(), indent=2))


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
    p.add_argument("--device", default="auto")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--output-dir", default="results/eval")
    args = p.parse_args()
    print(json.dumps(evaluate(args.dataset_file, model=args.model, device=args.device, max_examples=args.max_examples, output_dir=args.output_dir), indent=2))
