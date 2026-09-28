from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from biolaya.schemas import DecisionExample
from biolaya.cloud.env import resolve_device


@dataclass
class Prediction:
    prediction: str | float | bool
    confidence: float | None
    probabilities: dict[str, float] | None
    raw: dict[str, Any]


class LayaDecisionModel:
    def __init__(self, model: str = "english", device: str | None = "auto"):
        from laya import Router
        self.device = resolve_device(device)
        self.router = Router(device=self.device)
        self.model = model

    def predict_one(self, example: DecisionExample) -> Prediction:
        result = self.router.predict(example.state, example.laya_questions(), model=self.model)
        answer = result["answers"][example.question_id]
        if example.type == "choice":
            prediction = answer.get("choice")
            probs = answer.get("probabilities") or answer.get("distribution")
        elif example.type == "noul":
            probability = float(answer.get("noul"))
            prediction = probability >= 0.5
            probs = {"false": 1.0 - probability, "true": probability}
        else:
            prediction = answer.get("score")
            probs = answer.get("probabilities") or answer.get("distribution")
        confidence = answer.get("confidence")
        if confidence is None and isinstance(probs, dict) and probs:
            confidence = max(float(x) for x in probs.values())
        return Prediction(prediction=prediction, confidence=None if confidence is None else float(confidence), probabilities=probs, raw=result)
