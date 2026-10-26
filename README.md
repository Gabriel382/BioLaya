# BioLaya

> **Current project status:** Sprint 1 and 2 foundation retained; reproducible Standard NLI and native Laya decision training/evaluation pipeline added. **No trained BioLaya SystemOne checkpoints or measured results are contained in this source release.** For native Ollama export see the converter gate and runtime requirements in [the full pipeline guide](docs/BIOLAYA_FULL_PIPELINE.md).

## Standard & SystemOne workflows

```bash
python -m pip install -e ".[train,systemone,dev]"
python scripts/prepare_comparable_data.py \
  --biojev-train ../BioJev/data/sprint8/systemone_4b/train.jsonl \
  --biojev-dev ../BioJev/data/sprint8/systemone_4b/dev.jsonl
# If the source train/dev contain overlaps, first inspect:
# python scripts/prepare_comparable_data.py --audit-only
# Then explicitly sanitize only training rows:
# python scripts/prepare_comparable_data.py --overlap-policy drop-train
python scripts/prepare_standard.py --source bionli --out data/standard/bionli
python scripts/train_systemone.py --dry-run
python scripts/train_systemone.py
python scripts/eval_comparable.py --backend laya --limit 55
```

**Dataset leakage audit:** [Split contamination and safe conversion](docs/SPLIT_LEAKAGE_AUDIT.md). The source BioJev Sprint8 data may contain overlapping training and development content; strict conversion defaults to rejecting overlaps.

**Details:** [BioLaya full protocol](docs/BIOLAYA_FULL_PIPELINE.md). Existing biomedical MLM DAPT, Sprint 1 and Sprint 2 notebooks remain available below.


BioLaya is a biomedical specialization of the open-source **Laya** decision model. The project mirrors BioJev's seven-sprint scientific protocol while keeping Laya's native non-autoregressive typed-decision architecture.

## Infrastructure

- **GitHub** — source code, configs, notebooks, tests and paper scripts: `https://github.com/Gabriel382/BioLaya`
- **Google Colab** — disposable GPU compute.
- **Google Drive** — persistent checkpoints, results and final intermediate models for interrupted Colab sessions.
- **Hugging Face** — final/publishable model artifacts.

## Install

Python 3.10+:

```bash
python -m pip install -e ".[train]"
```

Google Colab installs the current GitHub version directly:

```python
%pip install -q --upgrade "biolaya[train,colab] @ git+https://github.com/Gabriel382/BioLaya.git"
```

BioLaya pins `laya==0.3.20`. The Colab extra also pins compatible `fsspec`/`gcsfs` versions for the current Colab + Hugging Face Datasets stack.

## Canonical Colab notebooks

The `.ipynb` files live in this repository and in the release ZIP. **Download/copy the notebook into Google Drive, open it with Google Colab, and run it there.** The notebooks do not require cloning the repository into the VM.

- **Sprint 1:** `notebooks/01_sprint1_colab.ipynb`
- **Sprint 2:** `notebooks/02_sprint2_colab.ipynb`

Each notebook installs BioLaya directly from GitHub.

## Local validation before Colab

CPU-only validation is always supported:

```powershell
python -m pip install -e ".[train,dev]"
python scripts\check_env.py --device cpu
python scripts\local_smoke.py --device cpu --skip-model
python scripts\sprint2_smoke.py --device cpu --preset smoke
python scripts\plan_dapt.py --preset 10m
pytest -q
```

A machine with CUDA can run the same checks with `--device cuda`. Explicit CUDA mode is strict and fails instead of silently falling back to CPU. `--device auto` selects CUDA when available and CPU otherwise.

See `docs/LOCAL_TESTING.md`.

# Sprint 1 — Benchmark foundation

Sprint 1 converts the BioJev benchmark family into Laya-native typed decisions:

- BioNLI
- NLI4CT
- ChemProt
- DDI2013
- BioRED

Prepare datasets:

```powershell
python scripts\download_datasets.py --all
```

Evaluate base Laya:

```powershell
python scripts\evaluate.py `
  --dataset-file data\processed\bionli\test.jsonl `
  --model english `
  --device auto `
  --output-dir results\laya_base\bionli
```

See `docs/SPRINT1.md`.

# Sprint 2 — Biomedical domain adaptation

Sprint 2 adapts the **ModernBERT-large backbone family used by Laya** with masked-language-model continued pretraining over an 80/20 PubMed/PMC streaming mixture.

The result is an intermediate **BioModernBERT** backbone. Sprint 3 will compare a no-DAPT Laya route against the BioModernBERT/DAPT route when constructing the actual BioLaya decision model.

Available presets:

```text
smoke  100K tokens
1m       1M tokens
10m     10M tokens
100m   100M tokens
```

Plan a run without loading a model:

```powershell
python scripts\plan_dapt.py --preset 10m
```

Preflight PubMed + PMC without loading ModernBERT:

```powershell
python scripts\inspect_corpus.py --preset smoke
```

Real local training:

```powershell
python scripts\train_dapt.py `
  --preset smoke `
  --device cuda `
  --resume none
```

On Colab, use **`notebooks/02_sprint2_colab.ipynb`**. It trains from `/content`, copies resumable checkpoints to Google Drive, restores the newest checkpoint with `--resume auto`, and persists the final BioModernBERT model to Drive.

See `docs/SPRINT2.md`.

## Scientific roadmap

See `docs/ROADMAP.md` for Sprints 1–7.

## Upstream

BioLaya depends on Laya rather than vendoring it. Laya is Apache-2.0 and distributed as an installable Python package. BioLaya records the pinned upstream version in `pyproject.toml` for reproducibility.
