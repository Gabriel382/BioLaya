from __future__ import annotations

import json
from pathlib import Path
from datasets import load_dataset

from biolaya.datasets.common import nli_example, save_splits


def _trial_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        chunks = []
        for key, content in value.items():
            if isinstance(content, list):
                content = "\n".join(str(x) for x in content)
            chunks.append(f"{key}: {content}")
        return "\n".join(chunks)
    return json.dumps(value, ensure_ascii=False)


def _convert(row, split: str, i: int):
    primary = _trial_text(row.get("Primary_ct", ""))
    secondary = _trial_text(row.get("Secondary_ct", ""))
    premise = primary if not secondary else f"PRIMARY TRIAL\n{primary}\n\nSECONDARY TRIAL\n{secondary}"
    return nli_example(
        row_id=f"nli4ct-{split}-{i}", dataset="nli4ct", split=split,
        premise=premise, hypothesis=str(row["Statement"]), label=str(row["Label"]), allowed_labels=("entailment", "contradiction"),
    )


def prepare(root: Path, seed: int = 42) -> dict[str, int]:
    ds = load_dataset("tasksource/nli4ct")
    splits = {"train": [_convert(r, "train", i) for i, r in enumerate(ds["train"])]}
    if "validation" in ds:
        splits["dev"] = [_convert(r, "dev", i) for i, r in enumerate(ds["validation"])]
    return save_splits("nli4ct", splits, root)
