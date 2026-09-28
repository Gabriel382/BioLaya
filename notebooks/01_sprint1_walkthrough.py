# %% [markdown]
# # BioLaya Sprint 1 — benchmark foundation
# Run after `00_colab_setup`. This creates the same biomedical benchmark family as BioJev, expressed as Laya typed decisions.

# %%
import os, pathlib, subprocess
os.chdir("/content/BioLaya") if pathlib.Path("/content/BioLaya").exists() else None
subprocess.run(["python", "scripts/download_datasets.py", "bionli", "nli4ct"], check=True)

# %%
import subprocess
subprocess.run(["python", "scripts/evaluate.py", "--dataset-file", "data/processed/bionli/test.jsonl", "--model", "english", "--device", "auto", "--max-examples", "50", "--output-dir", "results/laya_base/bionli_smoke"], check=True)

# %%
from pathlib import Path
from biolaya.cloud.drive import initialize_drive
import shutil
layout = initialize_drive("/content/drive/MyDrive/BioLaya")
src = Path("results/laya_base/bionli_smoke")
dst = layout.results / "sprint1" / "laya_base_bionli_smoke"
if dst.exists(): shutil.rmtree(dst)
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(src, dst)
print("saved:", dst)

# %% [markdown]
# After this smoke test, prepare all five datasets with `python scripts/download_datasets.py --all`. Relation datasets can take longer because they expand annotated entity-pair relations into decision examples.
