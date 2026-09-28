#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from biolaya.config import load_dapt_preset, load_yaml
from biolaya.training.dapt import plan_dapt

p = argparse.ArgumentParser(description="Plan BioLaya Sprint-2 biomedical MLM DAPT")
g = p.add_mutually_exclusive_group(required=True)
g.add_argument("--preset")
g.add_argument("--config")
a = p.parse_args()
cfg = load_dapt_preset(a.preset) if a.preset else load_yaml(a.config)
print(json.dumps(plan_dapt(cfg), indent=2))
