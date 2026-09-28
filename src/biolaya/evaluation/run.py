from __future__ import annotations

from pathlib import Path
from tqdm import tqdm

from biolaya.evaluation.metrics import classification_metrics
from biolaya.models import LayaDecisionModel
from biolaya.schemas import DecisionExample
from biolaya.utils.io import read_jsonl, write_json, write_jsonl


def evaluate(dataset_file: str | Path, *, model: str = "english", device: str = "auto", max_examples: int | None = None, output_dir: str | Path = "results") -> dict:
    examples = read_jsonl(dataset_file, DecisionExample)
    if max_examples:
        examples = examples[:max_examples]
    runner = LayaDecisionModel(model=model, device=device)
    records = []
    for ex in tqdm(examples, desc="BioLaya eval"):
        pred = runner.predict_one(ex)
        records.append({
            "id": ex.id, "dataset": ex.dataset, "split": ex.split,
            "gold": ex.gold, "prediction": pred.prediction,
            "confidence": pred.confidence, "probabilities": pred.probabilities,
        })
    metrics = classification_metrics(
        [str(x["gold"]) for x in records], [str(x["prediction"]) for x in records],
        [x["confidence"] for x in records],
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "predictions.jsonl", records)
    write_json(out / "metrics.json", metrics)
    return metrics
