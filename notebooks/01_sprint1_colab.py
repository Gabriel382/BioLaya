# %% [markdown]
# # BioLaya Sprint 1 — Google Colab + Google Drive
# Canonical Sprint 1 notebook companion. The .ipynb is the file intended to be
# copied into Google Drive and opened with Colab.

# %%
DEVICE = "auto"  # "auto", "cpu", or "cuda"
DRIVE_ROOT = "/content/drive/MyDrive/BioLaya"
GITHUB_REPO = "https://github.com/Gabriel382/BioLaya.git"

# %%
# Colab-only mount.
from google.colab import drive
drive.mount("/content/drive")

# %%
# Install the current GitHub version. In the .ipynb this is shown as a %pip cell.
import subprocess, sys
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "biolaya[train] @ git+https://github.com/Gabriel382/BioLaya.git",
    ],
    check=True,
)

# %%
import json
from biolaya.cloud.env import environment_report, resolve_device
from biolaya.cloud.drive import initialize_drive
print(json.dumps(environment_report(DEVICE), indent=2))
RESOLVED_DEVICE = resolve_device(DEVICE)
layout = initialize_drive(DRIVE_ROOT)
print("resolved device:", RESOLVED_DEVICE)
print("drive root:", layout.root)

# %%
subprocess.run(["biolaya-smoke", "--device", DEVICE, "--skip-model"], check=True)

# %%
from pathlib import Path
DATA_ROOT = Path("/content/biolaya_data/processed")
DATA_ROOT.mkdir(parents=True, exist_ok=True)
subprocess.run(["biolaya-data", "bionli", "nli4ct", "--root", str(DATA_ROOT)], check=True)

# %%
subprocess.run(["biolaya-smoke", "--device", DEVICE], check=True)

# %%
RESULT_LOCAL = "/content/biolaya_results/sprint1/laya_base_bionli_smoke"
subprocess.run(
    [
        "biolaya-eval",
        str(DATA_ROOT / "bionli" / "test.jsonl"),
        "--model", "english",
        "--device", DEVICE,
        "--max-examples", "50",
        "--output-dir", RESULT_LOCAL,
    ],
    check=True,
)

# %%
import shutil
src = Path(RESULT_LOCAL)
dst = layout.results / "sprint1" / "laya_base_bionli_smoke"
if dst.exists():
    shutil.rmtree(dst)
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(src, dst)
print("saved:", dst)
