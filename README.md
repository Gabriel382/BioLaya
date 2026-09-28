# BioLaya

BioLaya is a biomedical specialization of the open-source **Laya** decision model. The project mirrors BioJev's seven-sprint scientific protocol while retaining Laya's native non-autoregressive typed-decision architecture.

## Storage / reproducibility

- **GitHub:** source code, configs, notebooks, tests and paper scripts.
- **Google Colab:** disposable GPU compute.
- **Google Drive:** persistent checkpoints/logs/results for interrupted Colab sessions.
- **Hugging Face:** final BioLaya checkpoints/model cards (and publishable processed datasets if appropriate).

## Install

Python 3.10+:

```bash
python -m pip install -e ".[train]"
python scripts/check_env.py
```

BioLaya pins `laya==0.3.20` so experiments do not silently change when upstream Laya changes.

## Sprint 1

Prepare the same benchmark family used by BioJev:

```bash
python scripts/download_datasets.py --all
```

Evaluate base Laya on a prepared split:

```bash
python scripts/evaluate.py \
  --dataset-file data/processed/bionli/test.jsonl \
  --model english \
  --device auto \
  --output-dir results/laya_base/bionli
```

For a first Colab smoke test, add `--max-examples 50`.

## Google Colab — canonical Sprint notebook

For Sprint 1 use **`notebooks/01_sprint1_colab.ipynb`**. Download/copy that notebook to Google Drive and open it with Google Colab. It mounts Drive and installs BioLaya directly from:

`https://github.com/Gabriel382/BioLaya`

No repository clone is required inside the Colab VM. See `docs/COLAB.md`.

## Local validation before Colab

BioLaya supports `auto`, `cpu`, and strict `cuda` device modes. A no-GPU machine can validate the project with:

```bash
python -m pip install -e ".[train,dev]"
python scripts/check_env.py --device cpu
python scripts/local_smoke.py --device cpu --skip-model
pytest -q
```

For one real Laya inference on CPU run `python scripts/local_smoke.py --device cpu`. On a CUDA machine replace `cpu` with `cuda`. See `docs/LOCAL_TESTING.md`.

## Roadmap

See `docs/ROADMAP.md` for Sprints 1–7.

## Upstream

BioLaya depends on Laya rather than vendoring/forking it. Laya is Apache-2.0 and distributed as an installable Python package. BioLaya remains a separate research repository and records the pinned upstream version in `pyproject.toml`.


## Colab dependency compatibility

For Google Colab, install the dedicated extra so `gcsfs` and `fsspec` remain compatible with the Hugging Face Datasets version used by Sprint 1:

```python
%pip install -q --upgrade "biolaya[train,colab] @ git+https://github.com/Gabriel382/BioLaya.git"
```

The Colab extra currently pins `fsspec==2025.3.0` and `gcsfs==2025.3.0`. This is intentionally Colab-specific; local CPU/GPU installs do not require those pins.
