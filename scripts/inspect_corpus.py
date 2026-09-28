#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from biolaya.config import load_dapt_preset, load_yaml
from biolaya.corpus.biomedical_stream import preflight_sources

p = argparse.ArgumentParser(description="Inspect one document per BioLaya Sprint-2 corpus source")
g = p.add_mutually_exclusive_group(required=True)
g.add_argument("--preset")
g.add_argument("--config")
p.add_argument("--partition", default="train", choices=["train", "validation"])
a = p.parse_args()
cfg = load_dapt_preset(a.preset) if a.preset else load_yaml(a.config)
print(json.dumps(preflight_sources(cfg, a.partition), indent=2))
