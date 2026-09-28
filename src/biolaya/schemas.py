from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


DecisionType = Literal["choice", "score", "noul"]


class DecisionExample(BaseModel):
    id: str
    dataset: str
    split: str
    state: str
    question_id: str
    type: DecisionType = "choice"
    instructions: str
    criteria: dict[str, str] | list[str] | None = None
    gold: str | float | bool
    metadata: dict[str, Any] = Field(default_factory=dict)

    def laya_questions(self) -> dict[str, dict[str, Any]]:
        q: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            q["criteria"] = self.criteria
        return {self.question_id: q}
