from __future__ import annotations

from pathlib import Path

from biolaya.datasets import bionli, nli4ct
from biolaya.datasets.bigbio_re import SPECS, prepare as prepare_re

DEFAULT = ["bionli", "nli4ct", "chemprot", "ddi2013", "biored"]


def prepare_dataset(name: str, root: Path, seed: int = 42):
    if name == "bionli":
        return bionli.prepare(root, seed)
    if name == "nli4ct":
        return nli4ct.prepare(root, seed)
    if name in SPECS:
        return prepare_re(name, root, seed)
    raise KeyError(f"Unknown BioLaya dataset: {name}")
