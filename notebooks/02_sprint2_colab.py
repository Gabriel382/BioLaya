# %% [markdown]
# # BioLaya Sprint 2 — BioModernBERT on Google Colab + Google Drive
# 
# **This is the canonical Sprint-2 notebook.** Download/copy `02_sprint2_colab.ipynb` into Google Drive, open it with Google Colab, and run it there. It installs BioLaya directly from `https://github.com/Gabriel382/BioLaya`; no repository clone is required.
# 
# Sprint 2 performs biomedical masked-language-model domain adaptation of the ModernBERT-large backbone. The final output is **BioModernBERT**, which Sprint 3 will combine with Laya's decision-model training.

# %%
DEVICE = "cuda"   # strict: fail instead of accidentally training on CPU
PRESET = "10m"     # "1m" for throughput test; "10m" substantial dev; "100m" paper candidate
DRIVE_ROOT = "/content/drive/MyDrive/BioLaya"
GITHUB_REPO = "https://github.com/Gabriel382/BioLaya.git"
PUSH_TO_HUB = False
HF_REPO = "Gabriel382/BioLaya-BioModernBERT"

# %%
from google.colab import drive
drive.mount("/content/drive")

# %% [markdown]
# Install the current BioLaya directly from GitHub. The `colab` extra pins compatible `gcsfs`/`fsspec` versions for the Colab runtime.

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", 'biolaya[train,colab] @ git+https://github.com/Gabriel382/BioLaya.git'], check=True)

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
import fsspec, gcsfs
print("fsspec:", fsspec.__version__)
print("gcsfs:", gcsfs.__version__)

# %%
import json
from pathlib import Path
from biolaya.cloud.env import environment_report, resolve_device
from biolaya.cloud.drive import initialize_drive
print(json.dumps(environment_report(DEVICE), indent=2))
RESOLVED_DEVICE = resolve_device(DEVICE)
layout = initialize_drive(DRIVE_ROOT)
DRIVE_RUN_DIR = layout.checkpoints / "sprint2" / PRESET
DRIVE_RUN_DIR.mkdir(parents=True, exist_ok=True)
print("resolved device:", RESOLVED_DEVICE)
print("persistent run dir:", DRIVE_RUN_DIR)

# %% [markdown]
# Structural validation first. This checks the packaged preset and device path before any large model is downloaded.

# %%
subprocess.run(["biolaya-sprint2-smoke", "--device", DEVICE, "--preset", PRESET], check=True)
subprocess.run(["biolaya-plan-dapt", "--preset", PRESET], check=True)

# %% [markdown]
# Corpus preflight. It streams one usable document from **each** configured source (PubMed and PMC) before loading ModernBERT.

# %%
subprocess.run(["biolaya-corpus", "--preset", PRESET, "--partition", "train"], check=True)

# %% [markdown]
# ## Train / resume
# 
# Training happens on the Colab VM (`/content`) for speed. Every Trainer checkpoint is copied to Google Drive. If the session dies, reopen this notebook and run the cells again: `--resume auto` restores the newest Drive checkpoint and continues.

# %%
subprocess.run([
    "biolaya-dapt",
    "--preset", PRESET,
    "--device", DEVICE,
    "--resume", "auto",
    "--drive-run-dir", str(DRIVE_RUN_DIR),
], check=True)

# %% [markdown]
# Inspect the persistent final export and manifest. This survives Colab VM deletion.

# %%
FINAL_DIR = DRIVE_RUN_DIR / "final"
MANIFEST = DRIVE_RUN_DIR / "training_manifest.json"
print("final exists:", FINAL_DIR.exists(), FINAL_DIR)
print("manifest exists:", MANIFEST.exists(), MANIFEST)
if MANIFEST.exists():
    print(MANIFEST.read_text()[:4000])

# %% [markdown]
# Optional: upload the finished Sprint-2 backbone to Hugging Face. Keep this disabled until you intentionally want to publish/store the intermediate model.

# %%
if PUSH_TO_HUB:
    from huggingface_hub import HfApi, login
    login()
    api = HfApi()
    api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=HF_REPO, repo_type="model", folder_path=str(FINAL_DIR))
    print("uploaded:", f"https://huggingface.co/{HF_REPO}")
else:
    print("PUSH_TO_HUB=False; final model remains in Google Drive.")
