from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path

from biolaya.schemas import DecisionExample
from biolaya.utils.io import write_jsonl

NLI_CRITERIA = {
    "entailment": "The hypothesis is supported by the evidence in the state.",
    "contradiction": "The hypothesis conflicts with the evidence in the state.",
    "neutral": "The evidence does not establish either entailment or contradiction.",
}


def normalize_label(value: str) -> str:
    v = str(value).strip().lower()
    aliases = {"entailment": "entailment", "entails": "entailment", "contradiction": "contradiction", "contradictory": "contradiction", "neutral": "neutral"}
    if v not in aliases:
        raise ValueError(f"Unsupported NLI label: {value!r}")
    return aliases[v]


def nli_example(*, row_id: str, dataset: str, split: str, premise: str, hypothesis: str, label: str, allowed_labels: tuple[str, ...] = ("entailment", "contradiction", "neutral")) -> DecisionExample:
    gold = normalize_label(label)
    criteria = {k: NLI_CRITERIA[k] for k in allowed_labels}
    if gold not in criteria:
        raise ValueError(f"Gold label {gold!r} is not in allowed labels {allowed_labels}")
    return DecisionExample(
        id=row_id,
        dataset=dataset,
        split=split,
        state=premise.strip(),
        question_id="nli",
        type="choice",
        instructions=(
            "Determine the relationship between the hypothesis and the biomedical evidence in the state. "
            f"Hypothesis: {hypothesis.strip()}"
        ),
        criteria=criteria,
        gold=gold,
        metadata={"hypothesis": hypothesis.strip()},
    )


def stratified_split(rows: list[DecisionExample], seed: int = 42, ratios=(0.8, 0.1, 0.1)) -> dict[str, list[DecisionExample]]:
    rng = random.Random(seed)
    groups: dict[str, list[DecisionExample]] = defaultdict(list)
    for row in rows:
        groups[str(row.gold)].append(row)
    out = {"train": [], "dev": [], "test": []}
    for group in groups.values():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * ratios[0])
        n_dev = int(n * ratios[1])
        parts = (group[:n_train], group[n_train:n_train+n_dev], group[n_train+n_dev:])
        for split, part in zip(out, parts):
            out[split].extend(x.model_copy(update={"split": split}) for x in part)
    for values in out.values():
        rng.shuffle(values)
    return out


def stable_id(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode()).hexdigest()[:16]


def save_splits(dataset: str, splits: dict[str, list[DecisionExample]], root: Path) -> dict[str, int]:
    counts = {}
    for split, rows in splits.items():
        write_jsonl(root / dataset / f"{split}.jsonl", rows)
        counts[split] = len(rows)
    return counts
