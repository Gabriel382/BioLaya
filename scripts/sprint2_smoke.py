#!/usr/bin/env python
from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path
from biolaya.cloud.drive import backup_checkpoint, restore_latest_checkpoint
from biolaya.config import load_dapt_preset
from biolaya.training.dapt import plan_dapt, resolve_runtime

p = argparse.ArgumentParser(description="Local structural smoke test for BioLaya Sprint 2")
p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
p.add_argument("--preset", default="smoke")
a = p.parse_args()
cfg = load_dapt_preset(a.preset)
runtime = resolve_runtime(a.device, cfg["training"].get("mixed_precision", "auto"), cfg["training"].get("gradient_checkpointing", True))
plan = plan_dapt(cfg)
with tempfile.TemporaryDirectory(prefix="biolaya-s2-") as td:
    root = Path(td)
    ckpt = root / "local" / "checkpoint-50"
    ckpt.mkdir(parents=True)
    (ckpt / "trainer_state.json").write_text('{"global_step": 50}', encoding="utf-8")
    backup_checkpoint(ckpt, root / "drive", keep=2)
    restored = restore_latest_checkpoint(root / "drive", root / "restored")
    assert restored and restored.exists()
print(json.dumps({"ok": True, "runtime": runtime.__dict__, "plan": plan, "checkpoint_resume": True}, indent=2))
