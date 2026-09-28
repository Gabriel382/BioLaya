# %% [markdown]
# # BioLaya — Colab setup
# This notebook treats `/content` as fast disposable storage and Google Drive as persistent checkpoint/recovery storage.

# %%
from google.colab import drive
drive.mount("/content/drive")

# %%
import os, subprocess, pathlib
REPO_URL = "https://github.com/Gabriel382/BioLaya.git"
REPO_DIR = pathlib.Path("/content/BioLaya")
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
os.chdir(REPO_DIR)
print("repo:", pathlib.Path.cwd())

# %%
import subprocess
subprocess.run(["python", "-m", "pip", "install", "-q", "-e", ".[train]"], check=True)

# %%
import json
from biolaya.cloud.env import environment_report
print(json.dumps(environment_report(), indent=2))

# %%
from biolaya.cloud.drive import initialize_drive
layout = initialize_drive("/content/drive/MyDrive/BioLaya")
print(layout)

# %% [markdown]
# The environment is ready when `cuda_available` is `true`. Checkpoints should be backed up to `layout.checkpoints`; live training should stay under `/content`.
