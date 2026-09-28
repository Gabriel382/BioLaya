from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def load_dapt_preset(name: str) -> dict[str, Any]:
    filename = name if name.endswith(".yaml") else f"{name}.yaml"
    resource = files("biolaya").joinpath("configs", "dapt", filename)
    if not resource.is_file():
        available = sorted(p.name.removesuffix(".yaml") for p in files("biolaya").joinpath("configs", "dapt").iterdir() if p.name.endswith(".yaml"))
        raise KeyError(f"Unknown DAPT preset {name!r}. Available: {', '.join(available)}")
    return yaml.safe_load(resource.read_text(encoding="utf-8"))
