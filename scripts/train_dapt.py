#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from biolaya.config import load_dapt_preset, load_yaml
from biolaya.training.dapt import train_dapt

p = argparse.ArgumentParser(description="Train BioLaya Sprint-2 biomedical ModernBERT backbone")
g = p.add_mutually_exclusive_group(required=True)
g.add_argument("--preset")
g.add_argument("--config")
p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
p.add_argument("--resume", default="auto", help="auto, none, or checkpoint path")
p.add_argument("--drive-run-dir", default=None)
a = p.parse_args()
cfg = load_dapt_preset(a.preset) if a.preset else load_yaml(a.config)
print(json.dumps(train_dapt(cfg, device=a.device, resume=a.resume, drive_run_dir=a.drive_run_dir), indent=2, default=str))
