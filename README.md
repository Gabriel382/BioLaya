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

## Colab

Use `notebooks/00_colab_setup.ipynb` first, then `01_sprint1_walkthrough.ipynb`. See `docs/COLAB.md`.

## Roadmap

See `docs/ROADMAP.md` for Sprints 1–7.

## Upstream

BioLaya depends on Laya rather than vendoring/forking it. Laya is Apache-2.0 and distributed as an installable Python package. BioLaya remains a separate research repository and records the pinned upstream version in `pyproject.toml`.
