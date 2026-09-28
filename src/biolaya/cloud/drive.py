from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from biolaya.utils.io import ensure_dir

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/BioLaya"


def in_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or Path("/content").exists() and "google.colab" in os.sys.modules


def mount_google_drive(mountpoint: str = "/content/drive", force_remount: bool = False) -> Path:
    try:
        from google.colab import drive  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Google Drive mounting is only available inside Google Colab.") from exc
    drive.mount(mountpoint, force_remount=force_remount)
    return Path(mountpoint)


@dataclass(frozen=True)
class DriveLayout:
    root: Path
    checkpoints: Path
    logs: Path
    results: Path
    final_models: Path
    manifests: Path


def initialize_drive(root: str | Path = DEFAULT_DRIVE_ROOT) -> DriveLayout:
    root = ensure_dir(root)
    layout = DriveLayout(
        root=root,
        checkpoints=ensure_dir(root / "checkpoints"),
        logs=ensure_dir(root / "logs"),
        results=ensure_dir(root / "results"),
        final_models=ensure_dir(root / "final_models"),
        manifests=ensure_dir(root / "manifests"),
    )
    (root / "README.txt").write_text(
        "BioLaya persistent Colab storage. Keep checkpoints/results here; train from /content for speed.\n",
        encoding="utf-8",
    )
    return layout


def _checkpoint_step(path: Path) -> int:
    name = path.name
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("-", 1)[1])
        except ValueError:
            return -1
    meta = path / "trainer_state.json"
    if meta.exists():
        try:
            return int(json.loads(meta.read_text(encoding="utf-8")).get("global_step", -1))
        except Exception:
            return -1
    return -1


def latest_checkpoint(root: str | Path) -> Path | None:
    root = Path(root)
    if not root.exists():
        return None
    candidates = [p for p in root.rglob("checkpoint-*") if p.is_dir()]
    return max(candidates, key=lambda p: (_checkpoint_step(p), p.stat().st_mtime), default=None)


def atomic_copytree(source: str | Path, destination: str | Path) -> Path:
    source, destination = Path(source), Path(destination)
    ensure_dir(destination.parent)
    tmp = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        shutil.rmtree(tmp)
        shutil.copytree(source, tmp)
        if destination.exists():
            shutil.rmtree(destination)
        tmp.replace(destination)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return destination


def backup_checkpoint(local_checkpoint: str | Path, drive_run_dir: str | Path, keep: int = 2) -> Path:
    local_checkpoint = Path(local_checkpoint)
    if not local_checkpoint.is_dir():
        raise FileNotFoundError(local_checkpoint)
    drive_run_dir = ensure_dir(drive_run_dir)
    destination = drive_run_dir / local_checkpoint.name
    atomic_copytree(local_checkpoint, destination)
    checkpoints = sorted(
        [p for p in drive_run_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: (_checkpoint_step(p), p.stat().st_mtime),
        reverse=True,
    )
    for old in checkpoints[max(1, keep):]:
        shutil.rmtree(old, ignore_errors=True)
    marker = {
        "latest": destination.name,
        "step": _checkpoint_step(destination),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (drive_run_dir / "latest.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return destination


def restore_latest_checkpoint(drive_run_dir: str | Path, local_run_dir: str | Path) -> Path | None:
    latest = latest_checkpoint(drive_run_dir)
    if latest is None:
        return None
    local_run_dir = ensure_dir(local_run_dir)
    destination = local_run_dir / latest.name
    return atomic_copytree(latest, destination)
