from __future__ import annotations

import argparse
from pathlib import Path
from biolaya.datasets.registry import DEFAULT, prepare_dataset


def main():
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


if __name__ == "__main__":
    main()
