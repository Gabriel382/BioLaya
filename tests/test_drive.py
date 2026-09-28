from pathlib import Path

from biolaya.cloud.drive import backup_checkpoint, latest_checkpoint, restore_latest_checkpoint


def _make_checkpoint(root: Path, step: int):
    p = root / f"checkpoint-{step}"
    p.mkdir(parents=True)
    (p / "weights.bin").write_bytes(str(step).encode())
    return p


def test_drive_checkpoint_rotation_and_restore(tmp_path):
    local = tmp_path / "local"
    drive = tmp_path / "drive"
    local.mkdir()
    for step in [100, 200, 300]:
        backup_checkpoint(_make_checkpoint(local, step), drive, keep=2)
    assert latest_checkpoint(drive).name == "checkpoint-300"
    assert not (drive / "checkpoint-100").exists()
    restored = restore_latest_checkpoint(drive, tmp_path / "restore")
    assert restored.name == "checkpoint-300"
    assert (restored / "weights.bin").read_bytes() == b"300"
