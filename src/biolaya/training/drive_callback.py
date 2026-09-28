from __future__ import annotations

from pathlib import Path

from biolaya.cloud.drive import backup_checkpoint


def make_drive_backup_callback(drive_run_dir: str | Path, keep: int = 2):
    from transformers import TrainerCallback

    class DriveBackupCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if checkpoint.is_dir():
                destination = backup_checkpoint(checkpoint, drive_run_dir, keep=keep)
                print(f"[Drive] checkpoint backed up: {destination}")
            return control

    return DriveBackupCallback()
