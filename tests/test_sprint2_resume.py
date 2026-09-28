from pathlib import Path

from biolaya.cloud.drive import backup_checkpoint
from biolaya.training.dapt import _resolve_resume


def test_auto_resume_restores_drive_checkpoint(tmp_path: Path):
    source = tmp_path / "source" / "checkpoint-120"
    source.mkdir(parents=True)
    (source / "trainer_state.json").write_text('{"global_step": 120}', encoding="utf-8")
    drive = tmp_path / "drive"
    backup_checkpoint(source, drive)
    local = tmp_path / "local"
    restored = _resolve_resume("auto", drive, local)
    assert restored is not None
    assert Path(restored).name == "checkpoint-120"
    assert Path(restored).exists()
