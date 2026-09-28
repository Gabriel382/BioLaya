from __future__ import annotations

import argparse
from biolaya.cloud.drive import DEFAULT_DRIVE_ROOT, initialize_drive, latest_checkpoint

p = argparse.ArgumentParser()
p.add_argument("--root", default=DEFAULT_DRIVE_ROOT)
p.add_argument("--run", default=None)
a = p.parse_args()
layout = initialize_drive(a.root)
print(f"Drive root: {layout.root}")
print(f"Checkpoints: {layout.checkpoints}")
if a.run:
    print(f"Latest for {a.run}: {latest_checkpoint(layout.checkpoints / a.run)}")
