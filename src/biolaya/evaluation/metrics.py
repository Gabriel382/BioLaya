from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def expected_calibration_error(gold, pred, confidence, bins: int = 15) -> float | None:
    if not confidence or any(x is None for x in confidence):
        return None
    gold = np.asarray(gold)
    pred = np.asarray(pred)
    conf = np.asarray(confidence, dtype=float)
    correct = (gold == pred).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def classification_metrics(gold, pred, confidence=None) -> dict:
    result = {
        "n": len(gold),
        "accuracy": float(accuracy_score(gold, pred)),
        "f1_macro": float(f1_score(gold, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(gold, pred, average="weighted", zero_division=0)),
    }
    ece = expected_calibration_error(gold, pred, confidence or [])
    if ece is not None:
        result["ece"] = ece
    return result
