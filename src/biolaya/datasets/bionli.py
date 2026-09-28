from __future__ import annotations

from pathlib import Path
from datasets import load_dataset

from biolaya.datasets.common import nli_example, save_splits, stratified_split


def prepare(root: Path, seed: int = 42) -> dict[str, int]:
    ds = load_dataset("presencesw/bionli", split="train")
    rows = []
    for i, row in enumerate(ds):
        premise = row.get("premise") or row.get("sentence1") or row.get("text_a")
        hypothesis = row.get("hypothesis") or row.get("sentence2") or row.get("text_b")
        label = row.get("gold_label", row.get("label"))
        if premise is None or hypothesis is None or label is None:
            raise KeyError(f"Unexpected BioNLI columns: {sorted(row.keys())}")
        if isinstance(label, int):
            # Common NLI convention used by this mirror when integer encoded.
            label = {0: "entailment", 1: "neutral", 2: "contradiction"}[label]
        rows.append(nli_example(row_id=f"bionli-{i}", dataset="bionli", split="train", premise=str(premise), hypothesis=str(hypothesis), label=str(label), allowed_labels=("entailment", "contradiction")))
    return save_splits("bionli", stratified_split(rows, seed=seed), root)
