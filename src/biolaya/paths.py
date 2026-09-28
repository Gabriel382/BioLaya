from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.getenv("BIOLAYA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd().resolve()


def data_root() -> Path:
    return Path(os.getenv("BIOLAYA_DATA", project_root() / "data")).expanduser().resolve()


def output_root() -> Path:
    return Path(os.getenv("BIOLAYA_OUTPUTS", project_root() / "outputs")).expanduser().resolve()
