from __future__ import annotations

from pathlib import Path
from datasets import load_dataset

from biolaya.datasets.common import save_splits, stable_id
from biolaya.schemas import DecisionExample

SPECS = {
    "chemprot": ("bigbio/chemprot", "chemprot_bigbio_kb"),
    "ddi2013": ("bigbio/ddi_corpus", "ddi_corpus_bigbio_kb"),
    "biored": ("bigbio/biored", "biored_bigbio_kb"),
}


def _load(name: str):
    repo, config = SPECS[name]
    try:
        return load_dataset(repo, config)
    except Exception:
        # Older datasets releases need explicit permission for the BigBio loader scripts.
        return load_dataset(repo, config, trust_remote_code=True)


def _text(doc) -> str:
    chunks = []
    for passage in doc.get("passages", []):
        value = passage.get("text", [])
        chunks.extend(value if isinstance(value, list) else [value])
    return "\n".join(str(x) for x in chunks if x)


def _entity_map(doc) -> dict[str, str]:
    result = {}
    for entity in doc.get("entities", []):
        text = entity.get("text", "")
        if isinstance(text, list):
            text = text[0] if text else ""
        result[str(entity["id"])] = str(text)
    return result


def prepare(name: str, root: Path, seed: int = 42) -> dict[str, int]:
    ds = _load(name)
    labels = sorted({str(rel["type"]) for split in ds for doc in ds[split] for rel in doc.get("relations", [])})
    criteria = {label: f"The annotated biomedical relation is {label}." for label in labels}
    splits: dict[str, list[DecisionExample]] = {}
    for source_split in ds:
        target_split = "dev" if source_split == "validation" else source_split
        rows = []
        for doc in ds[source_split]:
            context = _text(doc)
            entities = _entity_map(doc)
            for rel in doc.get("relations", []):
                arg1 = str(rel.get("arg1_id", rel.get("arg1", "")))
                arg2 = str(rel.get("arg2_id", rel.get("arg2", "")))
                e1, e2 = entities.get(arg1, arg1), entities.get(arg2, arg2)
                gold = str(rel["type"])
                rows.append(DecisionExample(
                    id=f"{name}-{stable_id(str(doc.get('document_id', doc.get('id',''))), arg1, arg2, gold)}",
                    dataset=name,
                    split=target_split,
                    state=context,
                    question_id="relation",
                    type="choice",
                    instructions=f"What is the annotated biomedical relation between entity 1 ({e1}) and entity 2 ({e2})?",
                    criteria=criteria,
                    gold=gold,
                    metadata={"entity1": e1, "entity2": e2, "arg1_id": arg1, "arg2_id": arg2},
                ))
        splits[target_split] = rows
    return save_splits(name, splits, root)
